"""Evaluate a supported-peak relation gate without retraining the backbone."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from ..danger_signal.diagnostics import false_alarm_morphology
from ..temporal_relation_train_eval import (
    _evaluate_dual_track,
    _group_dual_frames_by_event,
)
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAME_PATH = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_guarded_epoch_sweep_v1"
    / "best_epoch_full_evaluation"
    / "persistent_guarded_frames.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_peak_or_sustain_gate_v1"
)

METHODS = {
    "ttc_percentile": "physical_percentile",
    "persistent_guarded": "persistent_guarded",
    "peak_or_sustain_guarded": "peak_or_sustain_guarded",
}


@dataclass(frozen=True)
class PeakOrSustainConfig:
    """Mechanistic supported-peak gate fixed before aggregate evaluation."""

    peak_relation_percentile: float = 0.95
    peak_persistence_percentile: float = 0.85
    support_relation_percentile: float = 0.80
    support_window_frames: int = 14
    minimum_support_frames: int = 4
    minimum_relation_rise: float = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_frame_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    integer_fields = {
        "event_index",
        "frame_position",
        "frames_to_anchor",
        "original_alert",
        "physical",
    }
    metadata_fields = {"split", "event_uid"} | integer_fields
    by_split: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            frame = {
                "split": str(raw["split"]),
                "event_uid": str(raw["event_uid"]),
                **{key: int(raw[key]) for key in integer_fields},
                "aggregations": {
                    key: float(value)
                    for key, value in raw.items()
                    if key not in metadata_fields and value not in {"", None}
                },
            }
            by_split.setdefault(frame["split"], []).append(frame)
    return by_split


def attach_peak_or_sustain_response(
    frames: list[dict[str, Any]],
    config: PeakOrSustainConfig,
) -> None:
    """Add a causal supported-peak branch while preserving the frozen gate."""

    for event_frames in _group_dual_frames_by_event(frames).values():
        event_frames.sort(key=lambda frame: int(frame["frame_position"]))
        relation_history: list[float] = []
        for frame in event_frames:
            values = frame["aggregations"]
            relation = float(values["relation_percentile"])
            persistence = float(values["persistence_percentile"])
            physical = float(values["physical_percentile"])
            relation_history.append(relation)

            window = relation_history[-config.support_window_frames :]
            support_count = sum(
                value >= config.support_relation_percentile
                for value in window
            )
            relation_rise = relation - min(window)
            supported_peak = (
                relation >= config.peak_relation_percentile
                and persistence >= config.peak_persistence_percentile
                and support_count >= config.minimum_support_frames
                and relation_rise >= config.minimum_relation_rise
            )
            frozen_sustain = bool(values["persistent_relation_gate"])
            relation_gate = frozen_sustain or supported_peak

            gated_relation = (
                relation
                if relation_gate
                else min(relation, ALERT_PERCENTILE - 1e-6)
            )
            guarded = max(physical, gated_relation)
            if (
                bool(values["physical_inhibition_gate"])
                and not bool(values["severe_physical_guard"])
            ):
                guarded = min(guarded, ALERT_PERCENTILE - 1e-6)

            values.update(
                {
                    "peak_support_count": float(support_count),
                    "peak_relation_rise": float(relation_rise),
                    "supported_peak_gate": float(supported_peak),
                    "peak_or_sustain_relation_gate": float(relation_gate),
                    "peak_or_sustain_guarded": float(guarded),
                }
            )


def _event_alert(
    event_frames: Sequence[dict[str, Any]],
    score_key: str,
    *,
    positive_window_only: bool,
) -> bool:
    return any(
        (
            not positive_window_only
            or int(frame["original_alert"]) == 1
        )
        and float(frame["aggregations"][score_key]) >= ALERT_PERCENTILE
        for frame in event_frames
    )


def _event_rows(
    frames: list[dict[str, Any]],
    *,
    split: str,
    positive_window_only: bool,
) -> list[dict[str, Any]]:
    rows = []
    for event_frames in _group_dual_frames_by_event(frames).values():
        event_frames.sort(key=lambda frame: int(frame["frame_position"]))
        detections = {
            method: _event_alert(
                event_frames,
                score_key,
                positive_window_only=positive_window_only,
            )
            for method, score_key in METHODS.items()
        }
        rows.append(
            {
                "split": split,
                "event_uid": str(event_frames[0]["event_uid"]),
                **{
                    f"{method}_detected": int(detected)
                    for method, detected in detections.items()
                },
                "peak_rescued_vs_ttc": int(
                    detections["peak_or_sustain_guarded"]
                    and not detections["ttc_percentile"]
                ),
                "peak_added_vs_persistent": int(
                    detections["peak_or_sustain_guarded"]
                    and not detections["persistent_guarded"]
                ),
                "ttc_suppressed_by_peak_gate": int(
                    detections["ttc_percentile"]
                    and not detections["peak_or_sustain_guarded"]
                ),
                "supported_peak_frame_count": sum(
                    int(frame["aggregations"]["supported_peak_gate"])
                    for frame in event_frames
                ),
            }
        )
    return rows


def _metrics(
    *,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "method": method,
            **_evaluate_dual_track(
                normal_frames=normal_frames,
                positive_frames=positive_frames,
                aggregation=score_key,
                label_window="original",
                threshold=ALERT_PERCENTILE,
                timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
            ),
        }
        for method, score_key in METHODS.items()
    ]


def _method_delta(
    metrics: Sequence[dict[str, Any]],
    *,
    split: str,
    left: str,
    right: str,
) -> dict[str, float | None]:
    rows = {
        str(row["method"]): row
        for row in metrics
        if str(row["split"]) == split
    }
    result: dict[str, float | None] = {}
    for key in [
        "frame_auroc",
        "event_recall_at_threshold",
        "normal_false_alarm_rate_at_threshold",
        "mean_lead_seconds",
    ]:
        left_value = rows[left].get(key)
        right_value = rows[right].get(key)
        result[key] = (
            None
            if left_value is None or right_value is None
            else float(left_value) - float(right_value)
        )
    return result


def run_peak_or_sustain_gate(
    *,
    frame_path: Path = DEFAULT_FRAME_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: PeakOrSustainConfig | None = None,
) -> dict[str, Any]:
    config = config or PeakOrSustainConfig()
    frame_path = frame_path.expanduser().resolve()
    frames = _load_frame_rows(frame_path)
    for split_frames in frames.values():
        attach_peak_or_sustain_response(split_frames, config)

    evaluations = [
        (
            "development_validation",
            "development_validation_normal",
            "development_validation_positive",
        ),
        (
            "development_test",
            "development_test_normal",
            "development_test_positive",
        ),
        ("confirmation", "confirmation_normal", "confirmation_positive"),
    ]
    metrics: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    false_alarms: list[dict[str, Any]] = []
    event_counts: dict[str, Any] = {}
    for split, normal_key, positive_key in evaluations:
        metrics.extend(
            _metrics(
                normal_frames=frames[normal_key],
                positive_frames=frames[positive_key],
                split=split,
            )
        )
        positive_rows = _event_rows(
            frames[positive_key],
            split=f"{split}_positive",
            positive_window_only=True,
        )
        normal_rows = _event_rows(
            frames[normal_key],
            split=f"{split}_normal",
            positive_window_only=False,
        )
        events.extend(positive_rows)
        events.extend(normal_rows)
        event_counts[split] = {
            "positive_peak_rescued_vs_ttc": int(
                sum(row["peak_rescued_vs_ttc"] for row in positive_rows)
            ),
            "positive_peak_added_vs_persistent": int(
                sum(row["peak_added_vs_persistent"] for row in positive_rows)
            ),
            "hard_safe_peak_added_vs_persistent": int(
                sum(row["peak_added_vs_persistent"] for row in normal_rows)
            ),
            "ttc_vs_peak_pairs": dict(
                Counter(
                    (
                        "both"
                        if row["ttc_percentile_detected"]
                        and row["peak_or_sustain_guarded_detected"]
                        else "ttc_only"
                        if row["ttc_percentile_detected"]
                        else "peak_only"
                        if row["peak_or_sustain_guarded_detected"]
                        else "neither"
                    )
                    for row in positive_rows
                )
            ),
        }
        for method, score_key in METHODS.items():
            false_alarms.append(
                {
                    "split": split,
                    "method": method,
                    **false_alarm_morphology(
                        frames[normal_key],
                        score_key=score_key,
                        threshold=ALERT_PERCENTILE,
                    ),
                }
            )

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "peak_or_sustain_metrics.csv"
    events_path = output_root / "peak_or_sustain_events.csv"
    false_alarm_path = output_root / "peak_or_sustain_false_alarms.csv"
    frame_output_path = output_root / "peak_or_sustain_frames.csv"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    _write_csv(false_alarm_path, false_alarms)

    frame_rows = []
    for split, split_frames in frames.items():
        for frame in split_frames:
            frame_rows.append(
                {
                    "split": split,
                    "event_index": frame["event_index"],
                    "event_uid": frame["event_uid"],
                    "frame_position": frame["frame_position"],
                    "frames_to_anchor": frame["frames_to_anchor"],
                    "original_alert": frame["original_alert"],
                    **frame["aggregations"],
                }
            )
    _write_csv(frame_output_path, frame_rows)

    deltas = {
        split: {
            "peak_minus_ttc": _method_delta(
                metrics,
                split=split,
                left="peak_or_sustain_guarded",
                right="ttc_percentile",
            ),
            "peak_minus_persistent": _method_delta(
                metrics,
                split=split,
                left="peak_or_sustain_guarded",
                right="persistent_guarded",
            ),
        }
        for split, _, _ in evaluations
    }
    summary = {
        "protocol": "riskbench-peak-or-sustain-gate-v1",
        "model_retrained": False,
        "source_frame_path": str(frame_path),
        "config": asdict(config),
        "config_status": (
            "mechanistic exploratory ablation motivated by inspected misses; "
            "requires a fresh untouched holdout before paper-facing promotion"
        ),
        "metrics": metrics,
        "deltas": deltas,
        "event_counts": event_counts,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "events_csv": str(events_path),
            "false_alarms_csv": str(false_alarm_path),
            "frames_csv": str(frame_output_path),
        },
        "status": "peak_or_sustain_no_retraining_ablation_complete",
    }
    summary_path = output_root / "peak_or_sustain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    result = run_peak_or_sustain_gate(
        frame_path=args.frame_path,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
