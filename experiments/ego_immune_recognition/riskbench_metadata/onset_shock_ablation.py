"""Evaluate relation-onset shock as a positive no-training alert branch."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

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
    / "ego_immune_riskbench_onset_shock_ablation_v1"
    / "source"
    / "peak"
    / "peak_or_sustain_frames.csv"
)
DEFAULT_AUDIT_CSV = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_ttc_blind_manual_audit_v1"
    / "ttc_blind_manual_review.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_onset_shock_ablation_v1"
    / "evaluation"
)
ONSET_QUANTILES = (0.95, 0.975, 0.99, 0.995, 0.999)
BASELINE_SCORE = "peak_or_sustain_guarded"
ONSET_RAW_SCORE = "danger_onset_shock_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    }
    metadata_fields = {"split", "event_uid"} | integer_fields | {"physical"}
    by_split: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            frame = {
                "split": str(raw["split"]),
                "event_uid": str(raw["event_uid"]),
                **{key: int(raw[key]) for key in integer_fields},
                "physical": int(raw.get("physical") or 0),
                "aggregations": {
                    key: float(value)
                    for key, value in raw.items()
                    if key not in metadata_fields and value not in {"", None}
                },
            }
            by_split.setdefault(frame["split"], []).append(frame)
    return by_split


def _positive_reference(frames: Sequence[dict[str, Any]]) -> np.ndarray:
    values = np.asarray(
        [
            float(frame["aggregations"][ONSET_RAW_SCORE])
            for frame in frames
            if float(frame["aggregations"][ONSET_RAW_SCORE]) > 0.0
        ],
        dtype=float,
    )
    if not values.size:
        raise ValueError("No positive onset-shock values in calibration frames.")
    return np.sort(values)


def _positive_percentile(reference: np.ndarray, value: float) -> float:
    if value <= 0.0:
        return 0.0
    return float(np.searchsorted(reference, value, side="right") / len(reference))


def _score_name(quantile: float) -> str:
    return f"peak_plus_onset_q{int(round(quantile * 1000)):03d}"


def _attach_onset_scores(
    frames_by_split: dict[str, list[dict[str, Any]]],
    reference: np.ndarray,
) -> None:
    for frames in frames_by_split.values():
        for frame in frames:
            values = frame["aggregations"]
            percentile = _positive_percentile(
                reference,
                float(values[ONSET_RAW_SCORE]),
            )
            values["onset_shock_positive_percentile"] = percentile
            for quantile in ONSET_QUANTILES:
                onset_gate = 1.0 if percentile >= quantile else 0.0
                values[_score_name(quantile)] = max(
                    float(values[BASELINE_SCORE]),
                    onset_gate,
                )


def _metric_rows(
    frames: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for split, normal_key, positive_key in (
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
    ):
        for quantile in (None, *ONSET_QUANTILES):
            score = BASELINE_SCORE if quantile is None else _score_name(quantile)
            rows.append(
                {
                    "split": split,
                    "method": (
                        "baseline_peak_or_sustain"
                        if quantile is None
                        else score
                    ),
                    "onset_quantile": "" if quantile is None else quantile,
                    **_evaluate_dual_track(
                        normal_frames=frames[normal_key],
                        positive_frames=frames[positive_key],
                        aggregation=score,
                        label_window="original",
                        threshold=ALERT_PERCENTILE,
                        timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
                    ),
                    **{
                        f"morphology_{key}": value
                        for key, value in false_alarm_morphology(
                            frames[normal_key],
                            score_key=score,
                            threshold=ALERT_PERCENTILE,
                        ).items()
                        if key not in {"score", "frame_count"}
                    },
                }
            )
    return rows


def _event_detected(
    frames: Sequence[dict[str, Any]],
    score: str,
    *,
    positive_window_only: bool,
) -> bool:
    return any(
        (
            not positive_window_only
            or int(frame["original_alert"]) == 1
        )
        and float(frame["aggregations"][score]) >= ALERT_PERCENTILE
        for frame in frames
    )


def _event_rows(
    frames_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for split, frames in frames_by_split.items():
        positive = split.endswith("_positive")
        for event_frames in _group_dual_frames_by_event(frames).values():
            uid = str(event_frames[0]["event_uid"])
            baseline = _event_detected(
                event_frames,
                BASELINE_SCORE,
                positive_window_only=positive,
            )
            alert_frames = [
                frame
                for frame in event_frames
                if not positive or int(frame["original_alert"]) == 1
            ]
            max_onset = max(
                (
                    float(
                        frame["aggregations"][
                            "onset_shock_positive_percentile"
                        ]
                    )
                    for frame in alert_frames
                ),
                default=0.0,
            )
            row = {
                "split": split,
                "event_uid": uid,
                "baseline_detected": int(baseline),
                "max_onset_shock_positive_percentile": max_onset,
            }
            for quantile in ONSET_QUANTILES:
                detected = _event_detected(
                    event_frames,
                    _score_name(quantile),
                    positive_window_only=positive,
                )
                row[f"detected_q{int(round(quantile * 1000)):03d}"] = int(
                    detected
                )
                row[f"rescued_q{int(round(quantile * 1000)):03d}"] = int(
                    detected and not baseline
                )
            rows.append(row)
    return rows


def _audit_rows(
    event_rows: Sequence[dict[str, Any]],
    audit_csv: Path,
) -> list[dict[str, Any]]:
    events = {
        (row["split"], row["event_uid"]): row
        for row in event_rows
    }
    output = []
    for audit in _read_csv(audit_csv):
        event = events[(audit["split"], audit["event_uid"])]
        output.append(
            {
                "audit_index": int(audit["audit_index"]),
                "split": audit["split"],
                "event_uid": audit["event_uid"],
                "target_visible_alert_rate": float(
                    audit["target_visible_alert_rate"]
                ),
                **{
                    key: value
                    for key, value in event.items()
                    if key not in {"split", "event_uid"}
                },
            }
        )
    return output


def run_onset_shock_ablation(
    *,
    frame_path: Path = DEFAULT_FRAME_PATH,
    audit_csv: Path = DEFAULT_AUDIT_CSV,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    frames = _load_frame_rows(frame_path.expanduser().resolve())
    reference = _positive_reference(
        frames["development_validation_normal"]
    )
    _attach_onset_scores(frames, reference)
    metrics = _metric_rows(frames)
    events = _event_rows(frames)
    audit = _audit_rows(events, audit_csv)
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "onset_shock_metrics.csv"
    events_path = output_root / "onset_shock_events.csv"
    audit_path = output_root / "onset_shock_audit20.csv"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    _write_csv(audit_path, audit)

    event_counts = {}
    for quantile in ONSET_QUANTILES:
        suffix = f"q{int(round(quantile * 1000)):03d}"
        event_counts[suffix] = {
            split: {
                "positive_rescued": sum(
                    int(row[f"rescued_{suffix}"])
                    for row in events
                    if row["split"] == f"{split}_positive"
                ),
                "hard_safe_added": sum(
                    int(row[f"rescued_{suffix}"])
                    for row in events
                    if row["split"] == f"{split}_normal"
                ),
            }
            for split in (
                "development_validation",
                "development_test",
                "confirmation",
            )
        }
    audit16 = next(row for row in audit if int(row["audit_index"]) == 16)
    result = {
        "protocol": "riskbench-onset-shock-positive-branch-v1",
        "model_retrained": False,
        "source_frame_path": str(frame_path.expanduser().resolve()),
        "onset_definition": (
            "recovery-shaped residual/damage drop within the first three "
            "valid predictions of an ego-incident directed relation"
        ),
        "calibration": {
            "source": "development_validation_hard_safe_positive_onset_values",
            "positive_value_count": int(len(reference)),
            "raw_q50": float(np.quantile(reference, 0.50)),
            "raw_q95": float(np.quantile(reference, 0.95)),
            "raw_q99": float(np.quantile(reference, 0.99)),
        },
        "event_counts": event_counts,
        "audit16": audit16,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "events_csv": str(events_path),
            "audit20_csv": str(audit_path),
        },
        "status": "onset_shock_exploratory_ablation_complete",
    }
    summary_path = output_root / "onset_shock_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(summary_path)
    return result


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_onset_shock_ablation(
                frame_path=args.frame_path,
                audit_csv=args.audit_csv,
                output_root=args.output_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
