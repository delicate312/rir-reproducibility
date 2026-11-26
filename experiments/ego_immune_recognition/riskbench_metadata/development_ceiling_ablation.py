"""Fit and report a clearly labeled TTC-blind development ceiling."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from ..danger_signal.diagnostics import false_alarm_morphology
from ..temporal_relation_train_eval import (
    _evaluate_dual_track,
    _group_dual_frames_by_event,
)
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .audited_gate_ablation import _attach_candidate_scores
from .early_window_sensitivity import (
    _attach_window_scores,
)
from .onset_shock_ablation import (
    DEFAULT_FRAME_PATH,
    _load_frame_rows,
    _positive_reference,
)
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_development_ceiling_v1"
)
METHODS = {
    "ttc": "physical_percentile",
    "development_ceiling": "development_ceiling",
}
WINDOWS = {"primary_1s": 20, "extended_1p5s": 30}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _attach_development_ceiling(
    frames_by_split: dict[str, list[dict[str, Any]]],
) -> None:
    for frames in frames_by_split.values():
        for event_frames in _group_dual_frames_by_event(frames).values():
            relation_history: list[float] = []
            onset_history: list[float] = []
            recovery_history: list[float] = []
            for frame in event_frames:
                values = frame["aggregations"]
                relation_history.append(values["relation_percentile"])
                onset_history.append(
                    values["window_onset_positive_percentile"]
                )
                recovery_history.append(values["recovery_percentile"])

                temporal_confirmed_onset = (
                    values["window_onset_positive_percentile"] >= 0.975
                    and max(relation_history[-5:]) >= 0.90
                )
                short_history_onset_recovery = (
                    int(frame["frame_position"]) <= 10
                    and max(onset_history[-5:]) >= 0.825
                    and values["recovery_percentile"] >= 0.825
                    and values["danger_stress"] >= 0.80
                )
                repeated_recovery_memory = (
                    values["peak_or_sustain_guarded"] >= 0.90
                    and sum(
                        value >= 0.65
                        for value in recovery_history[-31:]
                    )
                    >= 6
                    and values["danger_persistence"] >= 0.50
                )
                values.update(
                    {
                        "temporal_confirmed_onset_gate": float(
                            temporal_confirmed_onset
                        ),
                        "short_history_onset_recovery_gate": float(
                            short_history_onset_recovery
                        ),
                        "repeated_recovery_memory_gate": float(
                            repeated_recovery_memory
                        ),
                        "development_ceiling": max(
                            values["candidate_all"],
                            values["window_onset_q999"],
                            float(temporal_confirmed_onset),
                            float(short_history_onset_recovery),
                            float(repeated_recovery_memory),
                        ),
                    }
                )


def _metric_rows(
    frames: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for split in (
        "development_validation",
        "development_test",
        "confirmation",
    ):
        normal = frames[f"{split}_normal"]
        positive = frames[f"{split}_positive"]
        for window, lead_frames in WINDOWS.items():
            for method, score_key in METHODS.items():
                morphology = false_alarm_morphology(
                    normal,
                    score_key=score_key,
                    threshold=ALERT_PERCENTILE,
                )
                rows.append(
                    {
                        "split": split,
                        "window": window,
                        "method": method,
                        **_evaluate_dual_track(
                            normal_frames=normal,
                            positive_frames=positive,
                            aggregation=score_key,
                            label_window=f"pre_anchor_{lead_frames}",
                            threshold=ALERT_PERCENTILE,
                            timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
                        ),
                        "normal_event_false_alarm_rate": morphology[
                            "event_false_alarm_rate"
                        ],
                    }
                )
    return rows


def _confirmation_ttc_blind_rows(
    frames: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for event_frames in _group_dual_frames_by_event(
        frames["confirmation_positive"]
    ).values():
        usable = [
            frame
            for frame in event_frames
            if 0 < int(frame["frames_to_anchor"]) <= 30
        ]
        ttc_detected = any(
            frame["aggregations"]["physical_percentile"]
            >= ALERT_PERCENTILE
            for frame in usable
        )
        if ttc_detected:
            continue
        model_hits = [
            frame
            for frame in usable
            if frame["aggregations"]["development_ceiling"]
            >= ALERT_PERCENTILE
        ]
        first = (
            None
            if not model_hits
            else max(
                model_hits,
                key=lambda frame: int(frame["frames_to_anchor"]),
            )
        )
        rows.append(
            {
                "event_uid": event_frames[0]["event_uid"],
                "model_rescued": int(first is not None),
                "first_alert_frames_to_anchor": (
                    "" if first is None else first["frames_to_anchor"]
                ),
                "first_alert_seconds": (
                    ""
                    if first is None
                    else int(first["frames_to_anchor"])
                    * RISK_BENCH_TIMESTEP_SECONDS
                ),
                "short_history_gate": (
                    ""
                    if first is None
                    else first["aggregations"][
                        "short_history_onset_recovery_gate"
                    ]
                ),
                "repeated_recovery_gate": (
                    ""
                    if first is None
                    else first["aggregations"][
                        "repeated_recovery_memory_gate"
                    ]
                ),
            }
        )
    return rows


def run_development_ceiling(
    *,
    frame_path: Path = DEFAULT_FRAME_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    frame_path = frame_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    frames = _load_frame_rows(frame_path)
    reference = _positive_reference(
        frames["development_validation_normal"]
    )
    _attach_window_scores(frames, reference)
    _attach_candidate_scores(frames)
    _attach_development_ceiling(frames)

    metrics = _metric_rows(frames)
    blind_events = _confirmation_ttc_blind_rows(frames)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "development_ceiling_metrics.csv"
    blind_path = output_root / "confirmation_ttc_blind_events.csv"
    _write_csv(metrics_path, metrics)
    _write_csv(blind_path, blind_events)

    confirmation_extended = {
        row["method"]: row
        for row in metrics
        if row["split"] == "confirmation"
        and row["window"] == "extended_1p5s"
    }
    result = {
        "protocol": "riskbench-development-ceiling-v1",
        "paper_facing": False,
        "holdout_valid": False,
        "warning": (
            "Rules were developed after inspecting confirmation misses. "
            "This is a development ceiling, not an independent result."
        ),
        "ttc_blind_confirmation_event_count": len(blind_events),
        "ttc_blind_rescued_count": sum(
            int(row["model_rescued"]) for row in blind_events
        ),
        "ttc_blind_rescue_rate": sum(
            int(row["model_rescued"]) for row in blind_events
        )
        / len(blind_events),
        "confirmation_extended_1p5s": confirmation_extended,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "ttc_blind_events_csv": str(blind_path),
        },
        "status": "development_ceiling_complete",
    }
    summary_path = output_root / "development_ceiling_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(summary_path)
    return result


def main() -> None:
    args = parse_args()
    result = run_development_ceiling(
        frame_path=args.frame_path,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
