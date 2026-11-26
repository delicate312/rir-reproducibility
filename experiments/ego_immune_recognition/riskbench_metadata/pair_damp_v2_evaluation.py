"""Run the one-time Pair-DAMP v2 evaluation on the second holdout."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .core_hybrid_evaluation import _wilson_interval
from .frozen_onset_shock_rescue import _fold_rows
from .pair_damp_freeze import FOLD_COUNT, FPR_BUDGETS, PROJECT_ROOT, sha256_file
from .pair_damp_v2_freeze import (
    DEFAULT_FREEZE_PATH,
    DEFAULT_OUTPUT_ROOT as DEFAULT_INDEX_ROOT,
    V2_METHOD,
    verify_v2_freeze,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_pair_damp_v2_holdout"
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260804
PRIMARY_CONTINUOUS_FPR_BUDGET = 0.05
METHODS = ("ttc", "baseline", "onset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _binary_metrics(
    rows: Sequence[dict[str, Any]],
    score_key: str,
) -> dict[str, float | None]:
    labels = np.asarray([int(row["positive"]) for row in rows], dtype=int)
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=float)
    if not len(labels) or len(np.unique(labels)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _bootstrap_mean(values: np.ndarray, *, seed: int) -> list[float] | None:
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _lift_summary(
    rows: Sequence[dict[str, Any]],
    key: str,
    *,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not len(values):
        return {
            "event_count": 0,
            "mean_lift": None,
            "median_lift": None,
            "win_rate": None,
            "tie_rate": None,
            "mean_lift_bootstrap_ci95": None,
        }
    return {
        "event_count": len(values),
        "mean_lift": float(np.mean(values)),
        "median_lift": float(np.median(values)),
        "win_rate": float(np.mean(values > 0.0)),
        "tie_rate": float(np.mean(np.isclose(values, 0.0, atol=1e-12))),
        "mean_lift_bootstrap_ci95": _bootstrap_mean(values, seed=seed),
    }


def _net_lift(
    positive_rows: Sequence[dict[str, Any]],
    normal_rows: Sequence[dict[str, Any]],
    key: str,
    *,
    seed: int,
) -> dict[str, Any]:
    positive = np.asarray([float(row[key]) for row in positive_rows], dtype=float)
    normal = np.asarray([float(row[key]) for row in normal_rows], dtype=float)
    if not len(positive) or not len(normal):
        return {"mean_difference": None, "bootstrap_ci95": None}
    rng = np.random.default_rng(seed)
    positive_indices = rng.integers(
        0,
        len(positive),
        size=(BOOTSTRAP_SAMPLES, len(positive)),
    )
    normal_indices = rng.integers(
        0,
        len(normal),
        size=(BOOTSTRAP_SAMPLES, len(normal)),
    )
    sampled = positive[positive_indices].mean(axis=1) - normal[
        normal_indices
    ].mean(axis=1)
    return {
        "mean_difference": float(positive.mean() - normal.mean()),
        "bootstrap_ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
    }


def summarize_continuous_metrics(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize fixed score operators on one row per holdout event."""
    ttc_negative = [row for row in rows if not int(row["ttc_detected"])]
    blind_positive = [row for row in ttc_negative if int(row["positive"])]
    hard_safe = [row for row in ttc_negative if not int(row["positive"])]
    output = []
    for width_index, width in enumerate((1, 5, 10)):
        metrics = {
            method: _binary_metrics(ttc_negative, f"{method}_pool_{width}")
            for method in METHODS
        }
        onset_positive = _lift_summary(
            blind_positive,
            f"onset_vs_ttc_lift_{width}",
            seed=BOOTSTRAP_SEED + width_index * 20,
        )
        onset_normal = _lift_summary(
            hard_safe,
            f"onset_vs_ttc_lift_{width}",
            seed=BOOTSTRAP_SEED + width_index * 20 + 1,
        )
        baseline_positive = _lift_summary(
            blind_positive,
            f"baseline_vs_ttc_lift_{width}",
            seed=BOOTSTRAP_SEED + width_index * 20 + 2,
        )
        onset_vs_baseline_key = f"onset_vs_baseline_lift_{width}"
        for row in rows:
            row[onset_vs_baseline_key] = (
                float(row[f"onset_pool_{width}"])
                - float(row[f"baseline_pool_{width}"])
            )
        output.append(
            {
                "pooling_frames": width,
                "pooling_seconds": width * float(V2_METHOD["timestep_seconds"]),
                "pooling_name": (
                    "instantaneous_peak"
                    if width == 1
                    else "maximum_contiguous_mean"
                ),
                "ttc_negative_event_count": len(ttc_negative),
                "ttc_blind_positive_count": len(blind_positive),
                "ttc_negative_hard_safe_count": len(hard_safe),
                "ttc_negative_metrics": metrics,
                "onset_vs_ttc_positive": onset_positive,
                "baseline_vs_ttc_positive": baseline_positive,
                "onset_vs_ttc_hard_safe": onset_normal,
                "net_onset_vs_ttc_lift": _net_lift(
                    blind_positive,
                    hard_safe,
                    f"onset_vs_ttc_lift_{width}",
                    seed=BOOTSTRAP_SEED + width_index * 20 + 3,
                ),
                "onset_vs_baseline_positive": _lift_summary(
                    blind_positive,
                    onset_vs_baseline_key,
                    seed=BOOTSTRAP_SEED + width_index * 20 + 4,
                ),
            }
        )
    return output


def summarize_events(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for budget in FPR_BUDGETS:
        selected = [row for row in rows if float(row["fpr_budget"]) == budget]
        positives = [row for row in selected if int(row["positive"])]
        normals = [row for row in selected if not int(row["positive"])]
        blind_positive = [row for row in positives if not int(row["ttc_detected"])]
        blind_normal = [row for row in normals if not int(row["ttc_detected"])]
        for method in METHODS:
            if method == "ttc":
                hits = sum(int(row["ttc_detected"]) for row in positives)
                alerts = sum(int(row["ttc_detected"]) for row in normals)
                rescues = 0
                false_rescues = 0
                leads: list[float] = []
            else:
                detected_key = f"{method}_detected"
                lead_key = f"{method}_lead_seconds"
                hits = sum(
                    int(row["ttc_detected"] or row[detected_key])
                    for row in positives
                )
                alerts = sum(
                    int(row["ttc_detected"] or row[detected_key])
                    for row in normals
                )
                rescues = sum(int(row[detected_key]) for row in blind_positive)
                false_rescues = sum(int(row[detected_key]) for row in blind_normal)
                leads = [
                    float(row[lead_key])
                    for row in blind_positive
                    if int(row[detected_key]) and row[lead_key] != ""
                ]
            rescue_ci = _wilson_interval(rescues, len(blind_positive))
            output.append(
                {
                    "fpr_budget": budget,
                    "method": method,
                    "positive_hits": hits,
                    "positive_event_count": len(positives),
                    "positive_recall": hits / max(len(positives), 1),
                    "ttc_blind_rescues": rescues,
                    "ttc_blind_positive_count": len(blind_positive),
                    "ttc_blind_rescue_recall": rescues
                    / max(len(blind_positive), 1),
                    "ttc_blind_rescue_ci_low": rescue_ci[0],
                    "ttc_blind_rescue_ci_high": rescue_ci[1],
                    "hard_safe_alerts": alerts,
                    "hard_safe_event_count": len(normals),
                    "hard_safe_event_fpr": alerts / max(len(normals), 1),
                    "false_rescues": false_rescues,
                    "ttc_negative_hard_safe_count": len(blind_normal),
                    "incremental_rescue_fpr": false_rescues
                    / max(len(blind_normal), 1),
                    "mean_rescue_lead_seconds": (
                        "" if not leads else float(np.mean(leads))
                    ),
                    "median_rescue_lead_seconds": (
                        "" if not leads else float(np.median(leads))
                    ),
                }
            )
    return output


def _continuous_table(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for summary in summaries:
        metrics = summary["ttc_negative_metrics"]
        lift = summary["onset_vs_ttc_positive"]
        hard_safe = summary["onset_vs_ttc_hard_safe"]
        net = summary["net_onset_vs_ttc_lift"]
        baseline = summary["baseline_vs_ttc_positive"]
        onset_delta = summary["onset_vs_baseline_positive"]
        ci = lift["mean_lift_bootstrap_ci95"]
        net_ci = net["bootstrap_ci95"]
        output.append(
            {
                "pooling_frames": summary["pooling_frames"],
                "pooling_seconds": summary["pooling_seconds"],
                "onset_auroc": metrics["onset"]["auroc"],
                "baseline_auroc": metrics["baseline"]["auroc"],
                "ttc_auroc": metrics["ttc"]["auroc"],
                "onset_ap": metrics["onset"]["average_precision"],
                "baseline_ap": metrics["baseline"]["average_precision"],
                "ttc_ap": metrics["ttc"]["average_precision"],
                "onset_vs_ttc_mean_lift": lift["mean_lift"],
                "onset_vs_ttc_median_lift": lift["median_lift"],
                "onset_vs_ttc_win_rate": lift["win_rate"],
                "onset_vs_ttc_ci95_low": None if ci is None else ci[0],
                "onset_vs_ttc_ci95_high": None if ci is None else ci[1],
                "baseline_vs_ttc_mean_lift": baseline["mean_lift"],
                "onset_vs_baseline_mean_lift": onset_delta["mean_lift"],
                "hard_safe_onset_vs_ttc_mean_lift": hard_safe["mean_lift"],
                "net_onset_vs_ttc_lift": net["mean_difference"],
                "net_onset_vs_ttc_ci95_low": (
                    None if net_ci is None else net_ci[0]
                ),
                "net_onset_vs_ttc_ci95_high": (
                    None if net_ci is None else net_ci[1]
                ),
            }
        )
    return output


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):+.1f} pp"


def _report(
    event_rows: Sequence[dict[str, Any]],
    continuous_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Pair-DAMP v2 Second Holdout",
        "",
        "This is a sealed second unseen-exact-variant holdout. Each scenario",
        "family was excluded from its assigned fold model, but the families",
        "were seen during method development; this is not unseen-family evidence.",
        "No holdout positive label was used to fit the model, threshold, or onset rule.",
        "",
        "## Event Detection",
        "",
        "The budget controls incremental rescue FPR on validation-normal",
        "TTC-negative events. Reported holdout FPR is measured, not forced.",
        "",
        "| Budget | Method | Collision recall | TTC-blind rescue | Hard-safe FPR | Incremental FPR | Mean rescue lead |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in event_rows:
        lead = row["mean_rescue_lead_seconds"]
        lines.append(
            f"| {float(row['fpr_budget']):.0%} | `{row['method']}` "
            f"| {row['positive_hits']}/{row['positive_event_count']} "
            f"({float(row['positive_recall']):.1%}) "
            f"| {row['ttc_blind_rescues']}/{row['ttc_blind_positive_count']} "
            f"({float(row['ttc_blind_rescue_recall']):.1%}) "
            f"| {row['hard_safe_alerts']}/{row['hard_safe_event_count']} "
            f"({float(row['hard_safe_event_fpr']):.1%}) "
            f"| {row['false_rescues']}/{row['ttc_negative_hard_safe_count']} "
            f"({float(row['incremental_rescue_fpr']):.1%}) "
            f"| {'n/a' if lead == '' else f'{float(lead):.2f} s'} |"
        )
    lines.extend(
        [
            "",
            "## Continuous TTC-blind Evidence",
            "",
            "Scores are validation-normal percentiles, not calibrated danger",
            "probabilities. Lift is measured on TTC-blind collision events; AUROC",
            "and AP include TTC-negative hard-safe events to test discrimination.",
            "",
            "| Pooling | v2 AUROC | Baseline AUROC | TTC AUROC | v2 AP | TTC AP | v2-TTC lift | Win rate | v2-baseline lift | Hard-safe lift | Net lift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in continuous_rows:
        width = int(row["pooling_frames"])
        label = "Peak" if width == 1 else f"Sustain-{width}"
        lines.append(
            f"| {label} ({float(row['pooling_seconds']):.2f} s) "
            f"| {float(row['onset_auroc']):.3f} "
            f"| {float(row['baseline_auroc']):.3f} "
            f"| {float(row['ttc_auroc']):.3f} "
            f"| {float(row['onset_ap']):.3f} "
            f"| {float(row['ttc_ap']):.3f} "
            f"| {_pct(row['onset_vs_ttc_mean_lift'])} "
            f"| {float(row['onset_vs_ttc_win_rate']):.1%} "
            f"| {_pct(row['onset_vs_baseline_mean_lift'])} "
            f"| {_pct(row['hard_safe_onset_vs_ttc_mean_lift'])} "
            f"| {_pct(row['net_onset_vs_ttc_lift'])} |"
        )
    return "\n".join(lines)


def run_evaluation(
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
            raise FileExistsError("Pair-DAMP v2 holdout was already evaluated.")
        if not resume:
            raise FileExistsError("Interrupted receipt exists; rerun with --resume.")
    frozen = verify_v2_freeze(freeze_path, index_root)
    if frozen["method"] != V2_METHOD:
        raise ValueError("Runtime Pair-DAMP v2 method differs from sealed method.")
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "protocol": "riskbench-pair-damp-v2-second-holdout",
        "freeze_receipt_sha256": frozen["freeze_receipt_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    rows = []
    audits = []
    cache_root = output_root / "pair_frame_cache"
    for fold in range(FOLD_COUNT):
        fold_rows, audit = _fold_rows(
            fold=fold,
            index_root=index_root,
            cache_root=cache_root,
        )
        rows.extend(fold_rows)
        audits.append(audit)

    event_summary = summarize_events(rows)
    continuous_source = [
        row
        for row in rows
        if float(row["fpr_budget"]) == PRIMARY_CONTINUOUS_FPR_BUDGET
    ]
    continuous_summary = summarize_continuous_metrics(continuous_source)
    continuous_table = _continuous_table(continuous_summary)

    events_path = output_root / "v2_holdout_events.csv"
    event_table_path = output_root / "v2_event_table.csv"
    continuous_path = output_root / "v2_continuous_metrics.csv"
    audit_path = output_root / "v2_fold_audit.csv"
    report_path = output_root / "PAIR_DAMP_V2_HOLDOUT_REPORT.md"
    _write_csv(events_path, rows)
    _write_csv(event_table_path, event_summary)
    _write_csv(continuous_path, continuous_table)
    _write_csv(audit_path, audits)
    report_path.write_text(
        _report(event_summary, continuous_table),
        encoding="utf-8",
    )
    result = {
        "protocol": "riskbench-pair-damp-v2-second-holdout",
        "evidence_tier": "second_unseen_exact_variant_family_excluded_model",
        "unseen_family_claim_allowed": False,
        "model_retrained_after_freeze": False,
        "positive_events_used_for_fitting": 0,
        "method": frozen["method"],
        "freeze_receipt_sha256": frozen["freeze_receipt_sha256"],
        "continuous_metric_fpr_budget": PRIMARY_CONTINUOUS_FPR_BUDGET,
        "event_summary": event_summary,
        "continuous_summary": continuous_summary,
        "fold_audits": audits,
        "artifacts": {
            "events": str(events_path),
            "event_table": str(event_table_path),
            "continuous_metrics": str(continuous_path),
            "fold_audit": str(audit_path),
            "report": str(report_path),
        },
        "status": "sealed_pair_damp_v2_second_holdout_complete",
    }
    summary_path = output_root / "v2_holdout_summary.json"
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
            run_evaluation(
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
