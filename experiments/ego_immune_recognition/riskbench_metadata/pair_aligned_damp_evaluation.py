"""Evaluate exact-pair CPA and learned DAMP alignment on frozen CV models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..danger_signal.diagnostics import ACTIVATION_WEIGHTS, score_sequences
from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .complete_structure_ablation import BACKBONES, _checkpoint_path
from .core_hybrid_evaluation import (
    DEFAULT_CLEARANCE_M,
    DEFAULT_FPR_BUDGETS,
    DEFAULT_HORIZON_SECONDS,
    _usable_frames,
    _wilson_interval,
    event_alert,
    event_score,
    threshold_at_event_fpr,
)
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT
from .persistent_guarded_fusion import ALERT_PERCENTILE
from .relation_geometry_rescue import (
    _geometry_from_observations,
    _observations_by_frame,
    _percentile,
)
from .stratified_eval import _load_model_and_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_aligned_damp_v1"
)
DEFAULT_PAIR_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "pair_frame_cache"
PAIR_BACKBONES = ("graph_free_one_class", "full_gat_one_class")
RAW_COMPONENTS = ("stress_raw", "damage_raw", "persistence_raw", "recovery_raw")
SEVERE_CPA_PERCENTILE = 0.999
PAIR_SUPPORT_FRAMES = 3
RECOVERY_BETA = 0.10
PERCENTILE_WARMUP_FRAMES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--pair-cache-root",
        type=Path,
        default=DEFAULT_PAIR_CACHE_ROOT,
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--fpr-budgets", default="0.05,0.10")
    parser.add_argument(
        "--horizon-seconds", type=float, default=DEFAULT_HORIZON_SECONDS
    )
    parser.add_argument("--clearance-m", type=float, default=DEFAULT_CLEARANCE_M)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_or_score_pair_frames(
    *,
    backbone: str,
    fold: int,
    index_root: Path,
    cache_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    cache_path = cache_root / backbone / f"fold_{fold:02d}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    index_path = (
        index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    )
    checkpoint_path = _checkpoint_path(BACKBONES[backbone], fold)
    model, normalized_splits, _, device = _load_model_and_sequences(
        index_path,
        checkpoint_path,
    )
    frames = {
        split: score_sequences(
            model,
            normalized_splits[split],
            device,
            split=split,
            group_count=4,
            recovery_initialization_predictions=3,
            include_relation_components=True,
        )
        for split in (
            "validation_normal",
            "validation_positive",
            "test_normal",
            "test_positive",
        )
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return frames


def _outgoing_components(
    frames: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        values
        for frame in frames
        for values in frame.get("relation_components", {}).values()
        if int(values.get("ego_outgoing", 0))
    ]


def _component_calibration(
    validation_frames: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    relations = _outgoing_components(validation_frames)
    calibration = {}
    for key in RAW_COMPONENTS:
        values = np.asarray([float(row[key]) for row in relations], dtype=float)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        upper_scale = float(np.quantile(values, 0.90) - center)
        calibration[key] = {
            "center": center,
            "scale": max(1.4826 * mad, upper_scale, 1e-3),
        }
    return calibration


def _normalized_component(
    value: float,
    stats: dict[str, float],
) -> float:
    return max((float(value) - stats["center"]) / stats["scale"], 0.0)


def _attach_pair_activation(
    frames: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    calibration = _component_calibration(frames["validation_normal"])
    for split_frames in frames.values():
        for frame in split_frames:
            for values in frame.get("relation_components", {}).values():
                if not int(values.get("ego_outgoing", 0)):
                    continue
                normalized = {
                    key.removesuffix("_raw"): _normalized_component(
                        float(values[key]), calibration[key]
                    )
                    for key in RAW_COMPONENTS
                }
                activation = (
                    ACTIVATION_WEIGHTS["danger_stress"] * normalized["stress"]
                    + ACTIVATION_WEIGHTS["danger_damage"] * normalized["damage"]
                    + ACTIVATION_WEIGHTS["danger_persistence"]
                    * normalized["persistence"]
                )
                values.update(normalized)
                values["pair_activation"] = max(
                    activation - RECOVERY_BETA * normalized["recovery"],
                    0.0,
                )
    reference = np.sort(
        np.asarray(
            [
                float(values["pair_activation"])
                for values in _outgoing_components(frames["validation_normal"])
            ],
            dtype=float,
        )
    )
    for split_frames in frames.values():
        for frame in split_frames:
            for values in frame.get("relation_components", {}).values():
                if int(values.get("ego_outgoing", 0)):
                    values["pair_damp_percentile"] = _percentile(
                        reference,
                        float(values["pair_activation"]),
                    )
    return {
        "components": calibration,
        "activation_reference_count": int(reference.size),
    }


def _attach_physical_percentile(
    frames: dict[str, list[dict[str, Any]]],
) -> int:
    reference = np.sort(
        np.asarray(
            [
                float(frame["aggregations"]["inverse_ttc_ego_max"])
                for frame in frames["validation_normal"]
                if int(frame["frame_position"]) >= PERCENTILE_WARMUP_FRAMES
            ],
            dtype=float,
        )
    )
    if reference.size == 0:
        raise ValueError("No validation-normal TTC values for calibration.")
    for split_frames in frames.values():
        for frame in split_frames:
            frame["aggregations"]["physical_percentile"] = _percentile(
                reference,
                float(frame["aggregations"]["inverse_ttc_ego_max"]),
            )
    return int(reference.size)


def _component_by_target(frame: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(values["target_id"]): values
        for values in frame.get("relation_components", {}).values()
        if int(values.get("ego_outgoing", 0))
    }


def _attach_aligned_geometry(
    frames: dict[str, list[dict[str, Any]]],
    *,
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
) -> dict[str, int]:
    cpa_reference = np.sort(
        np.asarray(
            [
                float(values["raw"])
                for frame in frames["validation_normal"]
                for values in geometry.get(
                    (str(frame["event_uid"]), int(frame["frame_position"])),
                    {},
                ).values()
            ],
            dtype=float,
        )
    )
    matched = 0
    missing = 0
    for split_frames in frames.values():
        for frame in split_frames:
            components = _component_by_target(frame)
            attached = {}
            for target_id, values in geometry.get(
                (str(frame["event_uid"]), int(frame["frame_position"])),
                {},
            ).items():
                component = components.get(str(target_id))
                cpa = _percentile(cpa_reference, float(values["raw"]))
                if component is None:
                    missing += 1
                    continue
                matched += 1
                damp = float(component["pair_damp_percentile"])
                aligned = math.sqrt(max(cpa * damp, 0.0))
                attached[str(target_id)] = {
                    **values,
                    "cpa_only": cpa,
                    "pair_damp_percentile": damp,
                    "pair_aligned_geomean": aligned,
                }
            frame["aligned_geometry_relations"] = attached
    return {"matched_relation_frames": matched, "missing_relation_frames": missing}


def _aligned_trace(
    event_frames: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], float]]:
    """Severe CPA bypass plus three-frame exact-pair DAMP confirmation."""
    histories: dict[str, list[tuple[int, float]]] = defaultdict(list)
    output = []
    for frame in _usable_frames(event_frames):
        position = int(frame["frame_position"])
        best = 0.0
        for relation_id, values in frame.get(
            "aligned_geometry_relations", {}
        ).items():
            cpa = float(values["cpa_only"])
            aligned = float(values["pair_aligned_geomean"])
            history = histories[str(relation_id)]
            if history and position != history[-1][0] + 1:
                history.clear()
            history.append((position, aligned))
            history[:] = history[-PAIR_SUPPORT_FRAMES:]
            if cpa >= SEVERE_CPA_PERCENTILE:
                candidate = cpa
            elif len(history) == PAIR_SUPPORT_FRAMES:
                candidate = float(np.mean([value for _, value in history]))
            else:
                candidate = 0.0
            best = max(best, candidate)
        output.append((frame, best))
    return output


def _ttc_trace(
    event_frames: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], float]]:
    return [
        (frame, float(frame["aggregations"]["physical_percentile"]))
        for frame in _usable_frames(event_frames)
    ]


def _traces(
    frames: dict[str, list[dict[str, Any]]],
    *,
    aligned: bool,
) -> dict[str, dict[str, list[tuple[dict[str, Any], float]]]]:
    trace_fn = _aligned_trace if aligned else _ttc_trace
    return {
        split: {
            str(event_uid): trace_fn(event_frames)
            for event_uid, event_frames in _group_dual_frames_by_event(
                split_frames
            ).items()
        }
        for split, split_frames in frames.items()
    }


def run_fold(
    *,
    fold: int,
    index_root: Path,
    pair_cache_root: Path,
    horizon_seconds: float,
    clearance_m: float,
    fpr_budgets: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = (
        index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    )
    observations = _observations_by_frame(
        index_path,
        splits=(
            "validation_normal",
            "validation_positive",
            "test_normal",
            "test_positive",
        ),
    )
    geometry = _geometry_from_observations(
        observations,
        horizon_seconds=horizon_seconds,
        clearance_m=clearance_m,
    )
    traces_by_method = {}
    audit_rows = []
    ttc_traces = None
    for backbone in PAIR_BACKBONES:
        frames = _load_or_score_pair_frames(
            backbone=backbone,
            fold=fold,
            index_root=index_root,
            cache_root=pair_cache_root,
        )
        physical_reference_count = _attach_physical_percentile(frames)
        calibration = _attach_pair_activation(frames)
        alignment = _attach_aligned_geometry(frames, geometry=geometry)
        method = (
            "pair_aligned_graph_free"
            if backbone == "graph_free_one_class"
            else "pair_aligned_gat"
        )
        traces_by_method[method] = _traces(frames, aligned=True)
        if ttc_traces is None:
            ttc_traces = _traces(frames, aligned=False)
        audit_rows.append(
            {
                "fold": fold,
                "backbone": backbone,
                **alignment,
                "activation_reference_count": calibration[
                    "activation_reference_count"
                ],
                "physical_reference_count": physical_reference_count,
            }
        )
    assert ttc_traces is not None
    ttc_flags = {
        split: {
            event_uid: event_alert(trace, ALERT_PERCENTILE)
            for event_uid, trace in events.items()
        }
        for split, events in ttc_traces.items()
    }
    metrics = []
    event_rows = []
    for budget in fpr_budgets:
        validation_ttc_negative = {
            event_uid
            for event_uid, (detected, _) in ttc_flags["validation_normal"].items()
            if not detected
        }
        for method, split_traces in traces_by_method.items():
            threshold, inner_fpr = threshold_at_event_fpr(
                [
                    event_score(split_traces["validation_normal"][event_uid])
                    for event_uid in validation_ttc_negative
                ],
                budget,
            )
            fold_events = []
            for split, positive in (("test_positive", True), ("test_normal", False)):
                for event_uid, trace in split_traces[split].items():
                    ttc_detected, ttc_lead = ttc_flags[split][event_uid]
                    detected, lead = event_alert(trace, threshold)
                    rescue = bool(not ttc_detected and detected)
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
                    fold_events.append(row)
                    event_rows.append(row)
            positives = [row for row in fold_events if int(row["positive"])]
            normals = [row for row in fold_events if not int(row["positive"])]
            blind = [row for row in positives if not int(row["ttc_detected"])]
            ttc_negative_normal = [
                row for row in normals if not int(row["ttc_detected"])
            ]
            rescues = sum(int(row["blind_rescue"]) for row in blind)
            false_rescues = sum(
                int(row["blind_rescue"]) for row in ttc_negative_normal
            )
            leads = [
                float(row["method_lead_seconds"])
                for row in blind
                if int(row["blind_rescue"])
            ]
            metrics.append(
                {
                    "fold": fold,
                    "fpr_budget": budget,
                    "method": method,
                    "threshold": threshold,
                    "inner_rescue_fpr": inner_fpr,
                    "positive_hits": sum(
                        int(row["final_detected"]) for row in positives
                    ),
                    "positive_event_count": len(positives),
                    "ttc_blind_rescues": rescues,
                    "ttc_blind_positive_count": len(blind),
                    "hard_safe_alerts": sum(
                        int(row["final_detected"]) for row in normals
                    ),
                    "hard_safe_event_count": len(normals),
                    "false_rescues": false_rescues,
                    "ttc_negative_hard_safe_count": len(ttc_negative_normal),
                    "mean_rescue_lead_seconds": (
                        "" if not leads else float(np.mean(leads))
                    ),
                }
            )
    return metrics, event_rows, audit_rows


def _aggregate(event_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    methods = sorted({str(row["method"]) for row in event_rows})
    budgets = sorted({float(row["fpr_budget"]) for row in event_rows})
    for budget in budgets:
        for method in methods:
            rows = [
                row
                for row in event_rows
                if float(row["fpr_budget"]) == budget and row["method"] == method
            ]
            positives = [row for row in rows if int(row["positive"])]
            normals = [row for row in rows if not int(row["positive"])]
            blind = [row for row in positives if not int(row["ttc_detected"])]
            ttc_negative_normal = [
                row for row in normals if not int(row["ttc_detected"])
            ]
            rescues = sum(int(row["blind_rescue"]) for row in blind)
            false_rescues = sum(
                int(row["blind_rescue"]) for row in ttc_negative_normal
            )
            positive_hits = sum(int(row["final_detected"]) for row in positives)
            normal_alerts = sum(int(row["final_detected"]) for row in normals)
            leads = [
                float(row["method_lead_seconds"])
                for row in blind
                if int(row["blind_rescue"])
            ]
            ci = _wilson_interval(rescues, len(blind))
            output.append(
                {
                    "fpr_budget": budget,
                    "method": method,
                    "positive_hits": positive_hits,
                    "positive_event_count": len(positives),
                    "positive_recall": positive_hits / len(positives),
                    "ttc_blind_rescues": rescues,
                    "ttc_blind_positive_count": len(blind),
                    "ttc_blind_rescue_recall": rescues / len(blind),
                    "ttc_blind_rescue_ci_low": ci[0],
                    "ttc_blind_rescue_ci_high": ci[1],
                    "hard_safe_alerts": normal_alerts,
                    "hard_safe_event_count": len(normals),
                    "hard_safe_event_fpr": normal_alerts / len(normals),
                    "false_rescues": false_rescues,
                    "ttc_negative_hard_safe_count": len(ttc_negative_normal),
                    "incremental_rescue_fpr": false_rescues
                    / len(ttc_negative_normal),
                    "mean_rescue_lead_seconds": (
                        "" if not leads else float(np.mean(leads))
                    ),
                    "median_rescue_lead_seconds": (
                        "" if not leads else float(np.median(leads))
                    ),
                }
            )
    return output


def _report(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Pair-Aligned DAMP Evaluation",
        "",
        "Exact ego-target DAMP evidence is aligned with the same target's CPA.",
        "Severe CPA (>= 99th normal percentile) bypasses temporal confirmation;",
        "moderate CPA requires a three-frame pair-aligned signal.",
        "",
    ]
    for budget in sorted({float(row["fpr_budget"]) for row in rows}):
        lines.extend(
            [
                f"## Incremental rescue FPR budget: {budget:.0%}",
                "",
                "| Method | Collision recall | TTC-blind rescue | Hard-safe FPR | Incremental FPR | Mean lead |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if float(row["fpr_budget"]) != budget:
                continue
            lead = row["mean_rescue_lead_seconds"]
            lines.append(
                f"| `{row['method']}` | {row['positive_hits']}/{row['positive_event_count']} "
                f"({float(row['positive_recall']):.1%}) | "
                f"{row['ttc_blind_rescues']}/{row['ttc_blind_positive_count']} "
                f"({float(row['ttc_blind_rescue_recall']):.1%}) | "
                f"{row['hard_safe_alerts']}/{row['hard_safe_event_count']} "
                f"({float(row['hard_safe_event_fpr']):.1%}) | "
                f"{row['false_rescues']}/{row['ttc_negative_hard_safe_count']} "
                f"({float(row['incremental_rescue_fpr']):.1%}) | "
                f"{'n/a' if lead == '' else f'{float(lead):.2f} s'} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(
    *,
    folds: Sequence[int],
    fpr_budgets: Sequence[float] = DEFAULT_FPR_BUDGETS,
    index_root: Path = DEFAULT_INDEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pair_cache_root: Path = DEFAULT_PAIR_CACHE_ROOT,
    horizon_seconds: float = DEFAULT_HORIZON_SECONDS,
    clearance_m: float = DEFAULT_CLEARANCE_M,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics = []
    events = []
    audits = []
    for fold in folds:
        fold_metrics, fold_events, fold_audits = run_fold(
            fold=fold,
            index_root=index_root.expanduser().resolve(),
            pair_cache_root=pair_cache_root.expanduser().resolve(),
            horizon_seconds=horizon_seconds,
            clearance_m=clearance_m,
            fpr_budgets=fpr_budgets,
        )
        metrics.extend(fold_metrics)
        events.extend(fold_events)
        audits.extend(fold_audits)
    aggregate = _aggregate(events)
    metrics_path = output_root / "pair_aligned_fold_metrics.csv"
    events_path = output_root / "pair_aligned_events.csv"
    table_path = output_root / "pair_aligned_table.csv"
    audit_path = output_root / "pair_alignment_audit.csv"
    report_path = output_root / "PAIR_ALIGNED_DAMP_REPORT.md"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    _write_csv(table_path, aggregate)
    _write_csv(audit_path, audits)
    report_path.write_text(_report(aggregate), encoding="utf-8")
    result = {
        "protocol": "riskbench-pair-aligned-damp-v1",
        "model_retrained": False,
        "positive_events_used_for_threshold_fitting": 0,
        "configuration_status": (
            "Development-selected after pair-alignment sensitivity analysis; "
            "freeze severe CPA and support settings before any new holdout."
        ),
        "folds": list(folds),
        "fpr_budgets": list(fpr_budgets),
        "geometry": {
            "horizon_seconds": horizon_seconds,
            "clearance_m": clearance_m,
        },
        "gate": {
            "severe_cpa_percentile": SEVERE_CPA_PERCENTILE,
            "moderate_pair_support_frames": PAIR_SUPPORT_FRAMES,
            "pair_score": "sqrt(CPA percentile * exact-pair DAMP percentile)",
        },
        "aggregate": aggregate,
        "alignment_audit": audits,
        "artifacts": {
            "fold_metrics": str(metrics_path),
            "events": str(events_path),
            "table": str(table_path),
            "alignment_audit": str(audit_path),
            "report": str(report_path),
        },
        "status": "pair_aligned_damp_evaluation_complete",
    }
    summary_path = output_root / "pair_aligned_damp_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    folds = sorted(
        {int(value.strip()) for value in args.folds.split(",") if value.strip()}
    )
    budgets = sorted(
        {
            float(value.strip())
            for value in args.fpr_budgets.split(",")
            if value.strip()
        }
    )
    print(
        json.dumps(
            run(
                folds=folds,
                fpr_budgets=budgets,
                index_root=args.index_root,
                output_root=args.output_root,
                pair_cache_root=args.pair_cache_root,
                horizon_seconds=args.horizon_seconds,
                clearance_m=args.clearance_m,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
