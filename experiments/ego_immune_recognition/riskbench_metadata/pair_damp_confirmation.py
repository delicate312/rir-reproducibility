"""Run the one-time frozen pair-specific DAMP confirmation evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .complete_structure_ablation import BACKBONES, _checkpoint_path
from .core_hybrid_evaluation import (
    _usable_frames,
    _wilson_interval,
    event_alert,
    event_score,
    threshold_at_event_fpr,
)
from .pair_aligned_damp_evaluation import (
    _attach_aligned_geometry,
    _attach_pair_activation,
    _attach_physical_percentile,
    _load_or_score_pair_frames,
    _traces,
)
from .pair_damp_freeze import (
    DEFAULT_FREEZE_PATH,
    DEFAULT_OUTPUT_ROOT as DEFAULT_INDEX_ROOT,
    FOLD_COUNT,
    FROZEN_CODE_PATHS,
    FROZEN_METHOD,
    FPR_BUDGETS,
    PROJECT_ROOT,
    sha256_file,
    sha256_payload,
)
from .persistent_guarded_fusion import ALERT_PERCENTILE
from .relation_geometry_rescue import (
    _geometry_from_observations,
    _observations_by_frame,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_confirmation_v1"
)
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _verify_frozen_inputs(freeze_path: Path, index_root: Path) -> dict[str, Any]:
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    receipt = str(frozen.pop("freeze_receipt_sha256"))
    if sha256_payload(frozen) != receipt:
        raise ValueError("Freeze receipt payload hash no longer matches.")
    frozen["freeze_receipt_sha256"] = receipt
    if frozen["method"] != FROZEN_METHOD:
        raise ValueError("Runtime method constants differ from the frozen method.")
    if sha256_payload(FROZEN_METHOD) != frozen["method_sha256"]:
        raise ValueError("Frozen method hash verification failed.")
    for relative in FROZEN_CODE_PATHS:
        actual = sha256_file(PROJECT_ROOT / relative)
        if actual != frozen["code_hashes"][relative]:
            raise ValueError(f"Frozen code changed: {relative}")
    backbone = str(FROZEN_METHOD["backbone"])
    for fold in range(FOLD_COUNT):
        index_path = index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
        if sha256_file(index_path) != frozen["holdout"]["index_hashes"][str(fold)]:
            raise ValueError(f"Frozen holdout index changed for fold {fold}.")
        checkpoint = _checkpoint_path(BACKBONES[backbone], fold)
        if sha256_file(checkpoint) != frozen["checkpoints"][str(fold)]["sha256"]:
            raise ValueError(f"Frozen checkpoint changed for fold {fold}.")
    for path_string, expected in frozen["holdout"]["selected_state_hashes"].items():
        if sha256_file(Path(path_string)) != expected:
            raise ValueError(f"Frozen state input changed: {path_string}")
    return frozen


def _cpa_trace(event_frames: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
    return [
        (
            frame,
            max(
                (
                    float(values["cpa_only"])
                    for values in frame.get("aligned_geometry_relations", {}).values()
                ),
                default=0.0,
            ),
        )
        for frame in _usable_frames(event_frames)
    ]


def _event_traces(
    frames: dict[str, list[dict[str, Any]]],
    *,
    method: str,
) -> dict[str, dict[str, list[tuple[dict[str, Any], float]]]]:
    if method == "frozen_pair_damp":
        return _traces(frames, aligned=True)
    if method == "ttc":
        return _traces(frames, aligned=False)
    if method != "cpa_peak":
        raise ValueError(method)
    return {
        split: {
            str(event_uid): _cpa_trace(event_frames)
            for event_uid, event_frames in _group_dual_frames_by_event(split_frames).items()
        }
        for split, split_frames in frames.items()
    }


def _evaluate_fold(
    *,
    fold: int,
    index_root: Path,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    backbone = str(FROZEN_METHOD["backbone"])
    frames = _load_or_score_pair_frames(
        backbone=backbone,
        fold=fold,
        index_root=index_root,
        cache_root=cache_root,
    )
    _attach_physical_percentile(frames)
    calibration = _attach_pair_activation(frames)
    index_path = index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    observations = _observations_by_frame(
        index_path,
        splits=("validation_normal", "validation_positive", "test_normal", "test_positive"),
    )
    geometry = _geometry_from_observations(
        observations,
        horizon_seconds=float(FROZEN_METHOD["cpa_horizon_seconds"]),
        clearance_m=float(FROZEN_METHOD["cpa_clearance_m"]),
    )
    alignment = _attach_aligned_geometry(frames, geometry=geometry)
    traces = {
        method: _event_traces(frames, method=method)
        for method in ("ttc", "cpa_peak", "frozen_pair_damp")
    }
    ttc_flags = {
        split: {
            event_uid: event_alert(trace, ALERT_PERCENTILE)
            for event_uid, trace in events.items()
        }
        for split, events in traces["ttc"].items()
    }
    metric_rows = []
    event_rows = []
    validation_ttc_negative = {
        event_uid
        for event_uid, (detected, _) in ttc_flags["validation_normal"].items()
        if not detected
    }
    for budget in FPR_BUDGETS:
        thresholds = {"ttc": (ALERT_PERCENTILE, 0.0)}
        for method in ("cpa_peak", "frozen_pair_damp"):
            thresholds[method] = threshold_at_event_fpr(
                [
                    event_score(traces[method]["validation_normal"][event_uid])
                    for event_uid in validation_ttc_negative
                ],
                budget,
            )
        for method in ("ttc", "cpa_peak", "frozen_pair_damp"):
            threshold, inner_fpr = thresholds[method]
            fold_events = []
            for split, positive in (("test_positive", True), ("test_normal", False)):
                for event_uid, trace in traces[method][split].items():
                    ttc_detected, ttc_lead = ttc_flags[split][event_uid]
                    detected, lead = event_alert(trace, threshold)
                    rescue = bool(method != "ttc" and not ttc_detected and detected)
                    row = {
                        "fold": fold,
                        "fpr_budget": budget,
                        "method": method,
                        "event_uid": event_uid,
                        "split": split,
                        "positive": int(positive),
                        "threshold": threshold,
                        "inner_rescue_fpr": inner_fpr,
                        "method_score": event_score(trace),
                        "ttc_detected": int(ttc_detected),
                        "method_detected": int(detected),
                        "blind_rescue": int(rescue),
                        "final_detected": int(ttc_detected or rescue),
                        "ttc_lead_seconds": "" if ttc_lead is None else ttc_lead,
                        "method_lead_seconds": "" if lead is None else lead,
                    }
                    event_rows.append(row)
                    fold_events.append(row)
            metric_rows.append(_summarize_events(fold_events, fold=fold, budget=budget, method=method, threshold=threshold, inner_fpr=inner_fpr))
    audit = {
        "fold": fold,
        "backbone": backbone,
        **alignment,
        "activation_reference_count": calibration["activation_reference_count"],
    }
    return metric_rows, event_rows, audit


def _summarize_events(
    rows: Sequence[dict[str, Any]],
    *,
    fold: int | str,
    budget: float,
    method: str,
    threshold: float | str = "",
    inner_fpr: float | str = "",
) -> dict[str, Any]:
    positives = [row for row in rows if int(row["positive"])]
    normals = [row for row in rows if not int(row["positive"])]
    blind = [row for row in positives if not int(row["ttc_detected"])]
    ttc_negative_normal = [row for row in normals if not int(row["ttc_detected"])]
    rescues = sum(int(row["blind_rescue"]) for row in blind)
    false_rescues = sum(int(row["blind_rescue"]) for row in ttc_negative_normal)
    hits = sum(int(row["final_detected"]) for row in positives)
    normal_alerts = sum(int(row["final_detected"]) for row in normals)
    leads = [
        float(row["method_lead_seconds"])
        for row in blind
        if int(row["blind_rescue"])
    ]
    ci = _wilson_interval(rescues, len(blind))
    return {
        "fold": fold,
        "fpr_budget": budget,
        "method": method,
        "threshold": threshold,
        "inner_rescue_fpr": inner_fpr,
        "positive_hits": hits,
        "positive_event_count": len(positives),
        "positive_recall": hits / len(positives),
        "ttc_blind_rescues": rescues,
        "ttc_blind_positive_count": len(blind),
        "ttc_blind_rescue_recall": 0.0 if not blind else rescues / len(blind),
        "ttc_blind_rescue_ci_low": ci[0],
        "ttc_blind_rescue_ci_high": ci[1],
        "hard_safe_alerts": normal_alerts,
        "hard_safe_event_count": len(normals),
        "hard_safe_event_fpr": normal_alerts / len(normals),
        "false_rescues": false_rescues,
        "ttc_negative_hard_safe_count": len(ttc_negative_normal),
        "incremental_rescue_fpr": 0.0 if not ttc_negative_normal else false_rescues / len(ttc_negative_normal),
        "mean_rescue_lead_seconds": "" if not leads else float(np.mean(leads)),
        "median_rescue_lead_seconds": "" if not leads else float(np.median(leads)),
    }


def _paired_comparison(
    event_rows: Sequence[dict[str, Any]],
    *,
    budget: float,
) -> dict[str, Any]:
    methods = {}
    for method in ("cpa_peak", "frozen_pair_damp"):
        methods[method] = {
            (int(row["fold"]), str(row["event_uid"]), int(row["positive"])): int(row["blind_rescue"])
            for row in event_rows
            if row["method"] == method and float(row["fpr_budget"]) == budget
        }
    keys = sorted(set(methods["cpa_peak"]) & set(methods["frozen_pair_damp"]))

    def compare(positive: int) -> dict[str, Any]:
        pairs = np.asarray(
            [
                (methods["cpa_peak"][key], methods["frozen_pair_damp"][key])
                for key in keys
                if key[2] == positive
            ],
            dtype=int,
        )
        counts = Counter(
            "both" if left and right else "cpa_only" if left else "damp_only" if right else "neither"
            for left, right in pairs
        )
        if len(pairs):
            rng = np.random.default_rng(BOOTSTRAP_SEED + positive)
            indices = rng.integers(0, len(pairs), size=(BOOTSTRAP_SAMPLES, len(pairs)))
            sampled = pairs[indices]
            deltas = sampled[:, :, 1].mean(axis=1) - sampled[:, :, 0].mean(axis=1)
            ci = [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]
        else:
            ci = [0.0, 0.0]
        discordant = int(counts["cpa_only"] + counts["damp_only"])
        smaller = min(int(counts["cpa_only"]), int(counts["damp_only"]))
        p_value = 1.0 if not discordant else min(1.0, 2.0 * sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2.0**discordant))
        return {
            "event_count": len(pairs),
            "counts": dict(counts),
            "damp_minus_cpa_rate": 0.0 if not len(pairs) else float(pairs[:, 1].mean() - pairs[:, 0].mean()),
            "paired_bootstrap_ci95": ci,
            "mcnemar_exact_p_value": p_value,
        }

    return {
        "fpr_budget": budget,
        "ttc_blind_positive_rescue": compare(1),
        "ttc_negative_hard_safe_false_rescue": compare(0),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report(rows: Sequence[dict[str, Any]], paired: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Frozen Pair-DAMP Confirmation",
        "",
        "This is a sealed unseen-exact-variant confirmation. Each event is scored",
        "by the fold model that excluded its scenario family during training.",
        "The same families were observed during method development, so this is not",
        "an external unseen-family evaluation.",
        "",
    ]
    for budget in FPR_BUDGETS:
        lines.extend([
            f"## Incremental rescue FPR budget: {budget:.0%}",
            "",
            "| Method | Collision recall | TTC-blind rescue | Hard-safe FPR | Incremental FPR | Mean lead |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            if float(row["fpr_budget"]) != budget:
                continue
            lead = row["mean_rescue_lead_seconds"]
            lines.append(
                f"| `{row['method']}` | {row['positive_hits']}/{row['positive_event_count']} ({float(row['positive_recall']):.1%}) | "
                f"{row['ttc_blind_rescues']}/{row['ttc_blind_positive_count']} ({float(row['ttc_blind_rescue_recall']):.1%}) | "
                f"{row['hard_safe_alerts']}/{row['hard_safe_event_count']} ({float(row['hard_safe_event_fpr']):.1%}) | "
                f"{row['false_rescues']}/{row['ttc_negative_hard_safe_count']} ({float(row['incremental_rescue_fpr']):.1%}) | "
                f"{'n/a' if lead == '' else f'{float(lead):.2f} s'} |"
            )
        comparison = next(item for item in paired if float(item["fpr_budget"]) == budget)
        positive = comparison["ttc_blind_positive_rescue"]
        normal = comparison["ttc_negative_hard_safe_false_rescue"]
        lines.extend([
            "",
            f"Paired DAMP minus CPA rescue-rate delta: {float(positive['damp_minus_cpa_rate']):+.1%} "
            f"(95% bootstrap CI {positive['paired_bootstrap_ci95'][0]:+.1%} to {positive['paired_bootstrap_ci95'][1]:+.1%}).",
            f"Paired DAMP minus CPA false-rescue-rate delta: {float(normal['damp_minus_cpa_rate']):+.1%} "
            f"(95% bootstrap CI {normal['paired_bootstrap_ci95'][0]:+.1%} to {normal['paired_bootstrap_ci95'][1]:+.1%}).",
            "",
        ])
    return "\n".join(lines)


def run_confirmation(
    *,
    freeze_path: Path = DEFAULT_FREEZE_PATH,
    index_root: Path = DEFAULT_INDEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    resume: bool = False,
) -> dict[str, Any]:
    freeze_path = freeze_path.expanduser().resolve()
    index_root = index_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    receipt_path = output_root / "evaluation_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "complete":
            raise FileExistsError("Frozen confirmation was already completed.")
        if not resume:
            raise FileExistsError("An interrupted receipt exists; rerun with --resume.")
    frozen = _verify_frozen_inputs(freeze_path, index_root)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "protocol": "riskbench-frozen-pair-damp-confirmation-v1",
        "freeze_receipt_sha256": frozen["freeze_receipt_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    metrics = []
    events = []
    audits = []
    cache_root = output_root / "pair_frame_cache"
    for fold in range(FOLD_COUNT):
        fold_metrics, fold_events, audit = _evaluate_fold(
            fold=fold,
            index_root=index_root,
            cache_root=cache_root,
        )
        metrics.extend(fold_metrics)
        events.extend(fold_events)
        audits.append(audit)
    aggregate = []
    for budget in FPR_BUDGETS:
        for method in ("ttc", "cpa_peak", "frozen_pair_damp"):
            aggregate.append(
                _summarize_events(
                    [row for row in events if float(row["fpr_budget"]) == budget and row["method"] == method],
                    fold="all",
                    budget=budget,
                    method=method,
                )
            )
    paired = [_paired_comparison(events, budget=budget) for budget in FPR_BUDGETS]
    metrics_path = output_root / "frozen_fold_metrics.csv"
    events_path = output_root / "frozen_events.csv"
    table_path = output_root / "frozen_main_table.csv"
    audit_path = output_root / "frozen_alignment_audit.csv"
    report_path = output_root / "FROZEN_CONFIRMATION_REPORT.md"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    _write_csv(table_path, aggregate)
    _write_csv(audit_path, audits)
    report_path.write_text(_report(aggregate, paired), encoding="utf-8")
    result = {
        "protocol": "riskbench-frozen-pair-damp-confirmation-v1",
        "holdout_tier": frozen["holdout"]["tier"],
        "unseen_family_claim_allowed": False,
        "model_retrained_after_freeze": False,
        "positive_events_used_for_fitting": 0,
        "method": frozen["method"],
        "freeze_receipt_sha256": frozen["freeze_receipt_sha256"],
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "alignment_audit": audits,
        "artifacts": {
            "fold_metrics": str(metrics_path),
            "events": str(events_path),
            "main_table": str(table_path),
            "alignment_audit": str(audit_path),
            "report": str(report_path),
        },
        "status": "sealed_unseen_variant_confirmation_complete",
    }
    summary_path = output_root / "frozen_confirmation_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    receipt.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary_sha256": sha256_file(summary_path),
            "status": "complete",
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_confirmation(
                freeze_path=args.freeze_path,
                index_root=args.index_root,
                output_root=args.output_root,
                resume=args.resume,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
