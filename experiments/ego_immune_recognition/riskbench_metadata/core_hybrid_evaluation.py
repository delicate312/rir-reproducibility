"""Unified TTC/CPA/learned-DAMP event evaluation on frozen CV folds.

The evaluator uses a TTC-preserving rescue protocol.  Each rescue threshold is
fit only on TTC-negative validation hard-safe events, then applied once to the
outer positive and hard-safe events.  No model is retrained.
"""

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

from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .complete_structure_ablation import DEFAULT_CACHE_ROOT
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT, WINDOW_FRAMES
from .persistent_guarded_fusion import ALERT_PERCENTILE
from .relation_geometry_rescue import (
    _attach_relation_signals,
    _geometry_from_observations,
    _observations_by_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_core_hybrid_evaluation_v1"
)
DEFAULT_HORIZON_SECONDS = 2.0
DEFAULT_CLEARANCE_M = 3.5
DEFAULT_PERSISTENCE_FRAMES = 5
DEFAULT_FPR_BUDGETS = (0.05, 0.10)

METHOD_SPECS = {
    "ttc": {
        "backbone": "full_gat_one_class",
        "signal": "physical_percentile",
        "pooling_frames": 1,
        "learned": False,
    },
    "cpa_peak": {
        "backbone": "full_gat_one_class",
        "signal": "cpa_only",
        "pooling_frames": 1,
        "learned": False,
    },
    "cpa_rule_persistence": {
        "backbone": "full_gat_one_class",
        "signal": "cpa_only",
        "pooling_frames": DEFAULT_PERSISTENCE_FRAMES,
        "learned": False,
    },
    "cpa_graph_free_damp": {
        "backbone": "graph_free_one_class",
        "signal": "cpa_relation_geomean",
        "pooling_frames": DEFAULT_PERSISTENCE_FRAMES,
        "learned": True,
    },
    "cpa_gat_damp": {
        "backbone": "full_gat_one_class",
        "signal": "cpa_relation_geomean",
        "pooling_frames": DEFAULT_PERSISTENCE_FRAMES,
        "learned": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--fpr-budgets", default="0.05,0.10")
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=DEFAULT_HORIZON_SECONDS,
    )
    parser.add_argument(
        "--clearance-m",
        type=float,
        default=DEFAULT_CLEARANCE_M,
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_backbone_frames(
    cache_root: Path,
    backbone: str,
    fold: int,
) -> dict[str, list[dict[str, Any]]]:
    path = cache_root / backbone / f"fold_{fold:02d}.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def _usable_frames(
    event_frames: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        frame
        for frame in sorted(
            event_frames,
            key=lambda item: int(item["frame_position"]),
        )
        if 0
        < int(frame.get("eval_frames_to_anchor", frame["frames_to_anchor"]))
        <= WINDOW_FRAMES
    ]


def _rolling_relation_trace(
    event_frames: Sequence[dict[str, Any]],
    *,
    signal: str,
    pooling_frames: int,
) -> list[tuple[dict[str, Any], float]]:
    """Return a causal same-relation rolling-mean trace."""
    histories: dict[str, list[tuple[int, float]]] = defaultdict(list)
    output = []
    for frame in _usable_frames(event_frames):
        position = int(frame["frame_position"])
        best = 0.0
        for relation_id, values in frame.get("geometry_relations", {}).items():
            history = histories[str(relation_id)]
            if history and position != history[-1][0] + 1:
                history.clear()
            history.append((position, float(values[signal])))
            history[:] = history[-pooling_frames:]
            # Require a complete temporal window for persistence methods.
            if len(history) == pooling_frames:
                best = max(
                    best,
                    float(np.mean([value for _, value in history])),
                )
        output.append((frame, best))
    return output


def event_trace(
    event_frames: Sequence[dict[str, Any]],
    *,
    signal: str,
    pooling_frames: int,
) -> list[tuple[dict[str, Any], float]]:
    if signal.startswith("cpa_"):
        return _rolling_relation_trace(
            event_frames,
            signal=signal,
            pooling_frames=pooling_frames,
        )
    return [
        (frame, float(frame["aggregations"][signal]))
        for frame in _usable_frames(event_frames)
    ]


def event_score(trace: Sequence[tuple[dict[str, Any], float]]) -> float:
    return max((score for _, score in trace), default=0.0)


def event_alert(
    trace: Sequence[tuple[dict[str, Any], float]],
    threshold: float,
) -> tuple[bool, float | None]:
    hits = [
        frame
        for frame, score in trace
        if float(score) >= float(threshold)
    ]
    if not hits:
        return False, None
    lead = max(
        int(frame.get("eval_frames_to_anchor", frame["frames_to_anchor"]))
        for frame in hits
    )
    return True, lead * RISK_BENCH_TIMESTEP_SECONDS


def threshold_at_event_fpr(
    scores: Sequence[float],
    target_fpr: float,
) -> tuple[float, float]:
    """Choose the least strict observed threshold that stays under target."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return math.inf, 0.0
    candidates = [
        float(np.nextafter(np.max(values), math.inf)),
        *sorted({float(value) for value in values}, reverse=True),
    ]
    eligible = [
        threshold
        for threshold in candidates
        if float(np.mean(values >= threshold)) <= target_fpr + 1e-12
    ]
    threshold = min(eligible)
    return threshold, float(np.mean(values >= threshold))


def _fold_event_traces(
    frames: dict[str, list[dict[str, Any]]],
    *,
    signal: str,
    pooling_frames: int,
) -> dict[str, dict[str, list[tuple[dict[str, Any], float]]]]:
    return {
        split: {
            str(event_uid): event_trace(
                event_frames,
                signal=signal,
                pooling_frames=pooling_frames,
            )
            for event_uid, event_frames in _group_dual_frames_by_event(
                split_frames
            ).items()
        }
        for split, split_frames in frames.items()
    }


def _ttc_flags(
    traces: dict[str, dict[str, list[tuple[dict[str, Any], float]]]],
) -> dict[str, dict[str, tuple[bool, float | None]]]:
    return {
        split: {
            event_uid: event_alert(trace, ALERT_PERCENTILE)
            for event_uid, trace in events.items()
        }
        for split, events in traces.items()
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(center - half, 0.0), min(center + half, 1.0)


def run_fold(
    *,
    fold: int,
    index_root: Path,
    cache_root: Path,
    horizon_seconds: float,
    clearance_m: float,
    fpr_budgets: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    backbones = {
        str(spec["backbone"])
        for spec in METHOD_SPECS.values()
    }
    frames_by_backbone = {
        backbone: _load_backbone_frames(cache_root, backbone, fold)
        for backbone in backbones
    }
    index_path = (
        index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    )
    splits = tuple(next(iter(frames_by_backbone.values())))
    observations = _observations_by_frame(index_path, splits=splits)
    geometry = _geometry_from_observations(
        observations,
        horizon_seconds=horizon_seconds,
        clearance_m=clearance_m,
    )
    for frames in frames_by_backbone.values():
        _attach_relation_signals(frames, geometry=geometry)

    traces_by_method = {
        method: _fold_event_traces(
            frames_by_backbone[str(spec["backbone"])],
            signal=str(spec["signal"]),
            pooling_frames=int(spec["pooling_frames"]),
        )
        for method, spec in METHOD_SPECS.items()
    }
    ttc_flags = _ttc_flags(traces_by_method["ttc"])
    metric_rows = []
    event_rows = []
    for budget in fpr_budgets:
        thresholds = {"ttc": (ALERT_PERCENTILE, 0.0)}
        validation_ttc_negative = {
            event_uid
            for event_uid, (detected, _) in ttc_flags["validation_normal"].items()
            if not detected
        }
        for method in METHOD_SPECS:
            if method == "ttc":
                continue
            scores = [
                event_score(
                    traces_by_method[method]["validation_normal"][event_uid]
                )
                for event_uid in validation_ttc_negative
            ]
            thresholds[method] = threshold_at_event_fpr(scores, budget)

        for method, spec in METHOD_SPECS.items():
            threshold, inner_fpr = thresholds[method]
            positive_rows = []
            normal_rows = []
            for split, positive in (("test_positive", True), ("test_normal", False)):
                for event_uid, trace in traces_by_method[method][split].items():
                    ttc_detected, ttc_lead = ttc_flags[split][event_uid]
                    method_detected, method_lead = event_alert(trace, threshold)
                    rescue = bool(method != "ttc" and not ttc_detected and method_detected)
                    final = bool(ttc_detected or rescue)
                    row = {
                        "fold": fold,
                        "fpr_budget": budget,
                        "method": method,
                        "backbone": spec["backbone"],
                        "signal": spec["signal"],
                        "pooling_frames": spec["pooling_frames"],
                        "event_uid": event_uid,
                        "split": split,
                        "positive": int(positive),
                        "threshold": threshold,
                        "inner_rescue_fpr": inner_fpr,
                        "method_score": event_score(trace),
                        "ttc_detected": int(ttc_detected),
                        "method_detected": int(method_detected),
                        "blind_rescue": int(rescue),
                        "final_detected": int(final),
                        "ttc_lead_seconds": "" if ttc_lead is None else ttc_lead,
                        "method_lead_seconds": (
                            "" if method_lead is None else method_lead
                        ),
                    }
                    event_rows.append(row)
                    (positive_rows if positive else normal_rows).append(row)

            blind_positive = [
                row for row in positive_rows if not int(row["ttc_detected"])
            ]
            ttc_negative_normal = [
                row for row in normal_rows if not int(row["ttc_detected"])
            ]
            blind_rescues = sum(int(row["blind_rescue"]) for row in blind_positive)
            false_rescues = sum(
                int(row["blind_rescue"]) for row in ttc_negative_normal
            )
            final_hits = sum(int(row["final_detected"]) for row in positive_rows)
            normal_alerts = sum(int(row["final_detected"]) for row in normal_rows)
            rescue_leads = [
                float(row["method_lead_seconds"])
                for row in blind_positive
                if int(row["blind_rescue"])
            ]
            metric_rows.append(
                {
                    "fold": fold,
                    "fpr_budget": budget,
                    "method": method,
                    "threshold": threshold,
                    "inner_rescue_fpr": inner_fpr,
                    "positive_event_count": len(positive_rows),
                    "positive_hits": final_hits,
                    "positive_recall": final_hits / len(positive_rows),
                    "ttc_blind_positive_count": len(blind_positive),
                    "ttc_blind_rescues": blind_rescues,
                    "ttc_blind_rescue_recall": (
                        0.0
                        if not blind_positive
                        else blind_rescues / len(blind_positive)
                    ),
                    "hard_safe_event_count": len(normal_rows),
                    "hard_safe_alerts": normal_alerts,
                    "hard_safe_event_fpr": normal_alerts / len(normal_rows),
                    "ttc_negative_hard_safe_count": len(ttc_negative_normal),
                    "false_rescues": false_rescues,
                    "incremental_rescue_fpr": (
                        0.0
                        if not ttc_negative_normal
                        else false_rescues / len(ttc_negative_normal)
                    ),
                    "mean_rescue_lead_seconds": (
                        "" if not rescue_leads else float(np.mean(rescue_leads))
                    ),
                }
            )
    return metric_rows, event_rows


def aggregate(
    metric_rows: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    budgets = sorted({float(row["fpr_budget"]) for row in metric_rows})
    for budget in budgets:
        for method in METHOD_SPECS:
            events = [
                row
                for row in event_rows
                if float(row["fpr_budget"]) == budget
                and row["method"] == method
            ]
            positives = [row for row in events if int(row["positive"])]
            normals = [row for row in events if not int(row["positive"])]
            blind = [row for row in positives if not int(row["ttc_detected"])]
            ttc_negative_normal = [
                row for row in normals if not int(row["ttc_detected"])
            ]
            positive_hits = sum(int(row["final_detected"]) for row in positives)
            rescues = sum(int(row["blind_rescue"]) for row in blind)
            normal_alerts = sum(int(row["final_detected"]) for row in normals)
            false_rescues = sum(
                int(row["blind_rescue"]) for row in ttc_negative_normal
            )
            rescue_ci = _wilson_interval(rescues, len(blind))
            leads = [
                float(row["method_lead_seconds"])
                for row in blind
                if int(row["blind_rescue"])
            ]
            output.append(
                {
                    "fpr_budget": budget,
                    "method": method,
                    "positive_hits": positive_hits,
                    "positive_event_count": len(positives),
                    "positive_recall": positive_hits / len(positives),
                    "ttc_blind_rescues": rescues,
                    "ttc_blind_positive_count": len(blind),
                    "ttc_blind_rescue_recall": (
                        0.0 if not blind else rescues / len(blind)
                    ),
                    "ttc_blind_rescue_ci_low": rescue_ci[0],
                    "ttc_blind_rescue_ci_high": rescue_ci[1],
                    "hard_safe_alerts": normal_alerts,
                    "hard_safe_event_count": len(normals),
                    "hard_safe_event_fpr": normal_alerts / len(normals),
                    "false_rescues": false_rescues,
                    "ttc_negative_hard_safe_count": len(ttc_negative_normal),
                    "incremental_rescue_fpr": (
                        0.0
                        if not ttc_negative_normal
                        else false_rescues / len(ttc_negative_normal)
                    ),
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
        "# Core Hybrid Evaluation",
        "",
        "Frozen five-fold outer evaluation. Rescue thresholds use only",
        "TTC-negative validation hard-safe events; positive events do not fit",
        "thresholds. CPA horizon is 2 s and generic clearance is 3.5 m.",
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
            lead_text = "n/a" if lead == "" else f"{float(lead):.2f} s"
            lines.append(
                f"| `{row['method']}` | "
                f"{row['positive_hits']}/{row['positive_event_count']} "
                f"({float(row['positive_recall']):.1%}) | "
                f"{row['ttc_blind_rescues']}/{row['ttc_blind_positive_count']} "
                f"({float(row['ttc_blind_rescue_recall']):.1%}) | "
                f"{row['hard_safe_alerts']}/{row['hard_safe_event_count']} "
                f"({float(row['hard_safe_event_fpr']):.1%}) | "
                f"{row['false_rescues']}/{row['ttc_negative_hard_safe_count']} "
                f"({float(row['incremental_rescue_fpr']):.1%}) | {lead_text} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "This is a frozen-cache structural diagnostic. Geometry and temporal",
            "pooling are fixed globally, but a completely untouched external",
            "holdout is still required before making a final paper claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    folds: Sequence[int],
    fpr_budgets: Sequence[float] = DEFAULT_FPR_BUDGETS,
    index_root: Path = DEFAULT_INDEX_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    horizon_seconds: float = DEFAULT_HORIZON_SECONDS,
    clearance_m: float = DEFAULT_CLEARANCE_M,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    event_rows = []
    for fold in folds:
        fold_metrics, fold_events = run_fold(
            fold=fold,
            index_root=index_root.expanduser().resolve(),
            cache_root=cache_root.expanduser().resolve(),
            horizon_seconds=horizon_seconds,
            clearance_m=clearance_m,
            fpr_budgets=fpr_budgets,
        )
        metric_rows.extend(fold_metrics)
        event_rows.extend(fold_events)
    aggregate_rows = aggregate(metric_rows, event_rows)
    metrics_path = output_root / "core_hybrid_fold_metrics.csv"
    events_path = output_root / "core_hybrid_events.csv"
    aggregate_path = output_root / "core_hybrid_table.csv"
    report_path = output_root / "CORE_HYBRID_REPORT.md"
    _write_csv(metrics_path, metric_rows)
    _write_csv(events_path, event_rows)
    _write_csv(aggregate_path, aggregate_rows)
    report_path.write_text(_report(aggregate_rows), encoding="utf-8")
    result = {
        "protocol": "riskbench-core-hybrid-event-evaluation-v1",
        "model_retrained": False,
        "folds": list(folds),
        "fpr_budgets": list(fpr_budgets),
        "geometry": {
            "horizon_seconds": horizon_seconds,
            "clearance_m": clearance_m,
        },
        "calibration": (
            "Per-fold event thresholds fitted only on TTC-negative validation "
            "hard-safe events. Positive events used for threshold fitting: 0."
        ),
        "methods": METHOD_SPECS,
        "aggregate": aggregate_rows,
        "artifacts": {
            "fold_metrics": str(metrics_path),
            "events": str(events_path),
            "table": str(aggregate_path),
            "report": str(report_path),
        },
        "status": "core_hybrid_evaluation_complete",
    }
    summary_path = output_root / "core_hybrid_summary.json"
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
                cache_root=args.cache_root,
                output_root=args.output_root,
                horizon_seconds=args.horizon_seconds,
                clearance_m=args.clearance_m,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
