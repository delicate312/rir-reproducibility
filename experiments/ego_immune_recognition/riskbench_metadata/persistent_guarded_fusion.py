"""Evaluate persistent relation rescue and guarded TTC inhibition on RiskBench."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..danger_signal.diagnostics import (
    SELF_CONTEXT_WARMUP_FRAMES,
    apply_danger_calibration,
    false_alarm_morphology,
    robust_component_calibration,
    score_sequences,
)
from ..temporal_relation_train_eval import (
    _NormalizedSequenceView,
    _evaluate_dual_track,
    _group_dual_frames_by_event,
)
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .lazy_dataset import load_lazy_package
from .stratified_eval import _load_model_and_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEVELOPMENT_INDEX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_riskbench_metadata_lazy_v2_no_gt"
    / "riskbench_metadata_lazy_index.json"
)
DEFAULT_CONFIRMATION_INDEX = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_riskbench_metadata_confirmation_v2_no_gt"
    / "riskbench_confirmation_index.json"
)
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_metadata_lazy_v2_no_gt_temporal"
    / "riskbench_temporal_immune_localizer.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_persistent_guarded_fusion_v2"
)
DEFAULT_SEED = 20260721

ALERT_PERCENTILE = 0.95
SEVERE_PHYSICAL_PERCENTILE = 0.99
RELATION_PERSISTENCE_PERCENTILE = 0.90
RECOVERY_PERCENTILE = 0.75
PERSISTENT_RELATION_FRAMES = 3
RECOVERY_TREND_FRAMES = 3

METHODS = {
    "ttc_percentile": "physical_percentile",
    "unconstrained_max": "unconstrained_max",
    "persistent_rescue": "persistent_rescue",
    "persistent_guarded": "persistent_guarded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-index",
        type=Path,
        default=DEFAULT_DEVELOPMENT_INDEX,
    )
    parser.add_argument(
        "--confirmation-index",
        type=Path,
        default=DEFAULT_CONFIRMATION_INDEX,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--group-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--recovery-initialization-predictions",
        type=int,
        default=0,
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percentile_reference(
    frames: Sequence[dict[str, Any]],
    key: str,
) -> np.ndarray:
    values = [
        float(frame["aggregations"][key])
        for frame in frames
        if int(frame["frame_position"]) >= SELF_CONTEXT_WARMUP_FRAMES
    ]
    if not values:
        raise ValueError(f"No calibration values for {key!r}.")
    return np.sort(np.asarray(values, dtype=float))


def _percentile(reference: np.ndarray, value: float) -> float:
    return float(
        np.searchsorted(reference, float(value), side="right")
        / len(reference)
    )


def _attach_percentile_inputs(
    frames_by_split: dict[str, list[dict[str, Any]]],
    validation_normal: list[dict[str, Any]],
    *,
    relation_source: str = "danger_activation_inhibited",
) -> dict[str, dict[str, float]]:
    keys = {
        "physical_percentile": "inverse_ttc_ego_max",
        "relation_percentile": relation_source,
        "persistence_percentile": "danger_persistence",
        "recovery_percentile": "danger_recovery",
    }
    references = {
        output_key: _percentile_reference(validation_normal, source_key)
        for output_key, source_key in keys.items()
    }
    for frames in frames_by_split.values():
        for frame in frames:
            values = frame["aggregations"]
            for output_key, source_key in keys.items():
                values[output_key] = _percentile(
                    references[output_key],
                    float(values[source_key]),
                )
    return {
        key: {
            "count": int(len(reference)),
            "q50": float(np.quantile(reference, 0.50)),
            "q75": float(np.quantile(reference, 0.75)),
            "q90": float(np.quantile(reference, 0.90)),
            "q95": float(np.quantile(reference, 0.95)),
            "q99": float(np.quantile(reference, 0.99)),
        }
        for key, reference in references.items()
    }


def _attach_guarded_response(frames: list[dict[str, Any]]) -> None:
    for event_frames in _group_dual_frames_by_event(frames).values():
        relation_run = 0
        physical_recovery_run = 0
        previous_inverse_ttc: float | None = None
        for frame in event_frames:
            values = frame["aggregations"]
            physical = float(values["physical_percentile"])
            relation = float(values["relation_percentile"])
            persistence = float(values["persistence_percentile"])
            recovery = float(values["recovery_percentile"])
            inverse_ttc = float(values["inverse_ttc_ego_max"])

            relation_active = (
                relation >= ALERT_PERCENTILE
                and persistence >= RELATION_PERSISTENCE_PERCENTILE
            )
            relation_run = relation_run + 1 if relation_active else 0
            persistent_relation = relation_run >= PERSISTENT_RELATION_FRAMES

            physical_recovering = (
                previous_inverse_ttc is not None
                and inverse_ttc < previous_inverse_ttc - 1e-6
            )
            physical_recovery_run = (
                physical_recovery_run + 1 if physical_recovering else 0
            )
            previous_inverse_ttc = inverse_ttc

            severe_physical = physical >= SEVERE_PHYSICAL_PERCENTILE
            marginal_physical = (
                ALERT_PERCENTILE
                <= physical
                < SEVERE_PHYSICAL_PERCENTILE
            )
            inhibition = (
                marginal_physical
                and physical_recovery_run >= RECOVERY_TREND_FRAMES
                and recovery >= RECOVERY_PERCENTILE
            )

            gated_relation = (
                relation
                if persistent_relation
                else min(relation, ALERT_PERCENTILE - 1e-6)
            )
            unconstrained = max(physical, relation)
            rescue = max(physical, gated_relation)
            guarded = rescue
            if inhibition and not severe_physical:
                guarded = min(guarded, ALERT_PERCENTILE - 1e-6)

            values.update(
                {
                    "unconstrained_max": unconstrained,
                    "persistent_relation_run": float(relation_run),
                    "physical_recovery_run": float(physical_recovery_run),
                    "persistent_relation_gate": float(persistent_relation),
                    "physical_inhibition_gate": float(inhibition),
                    "severe_physical_guard": float(severe_physical),
                    "persistent_rescue": rescue,
                    "persistent_guarded": guarded,
                }
            )


def _event_alert(
    event_frames: Sequence[dict[str, Any]],
    score_key: str,
    *,
    positive_window_only: bool,
) -> bool:
    usable = [
        frame
        for frame in event_frames
        if not positive_window_only or int(frame["original_alert"]) == 1
    ]
    return any(
        float(frame["aggregations"][score_key]) >= ALERT_PERCENTILE
        for frame in usable
    )


def _event_diagnostics(
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
                "relation_rescued_vs_ttc": int(
                    detections["persistent_guarded"]
                    and not detections["ttc_percentile"]
                ),
                "ttc_event_suppressed": int(
                    detections["ttc_percentile"]
                    and not detections["persistent_guarded"]
                ),
                "persistent_gate_frame_count": sum(
                    int(
                        frame["aggregations"][
                            "persistent_relation_gate"
                        ]
                    )
                    for frame in event_frames
                ),
                "inhibition_gate_frame_count": sum(
                    int(
                        frame["aggregations"][
                            "physical_inhibition_gate"
                        ]
                    )
                    for frame in event_frames
                ),
                "severe_guard_frame_count": sum(
                    int(
                        frame["aggregations"]["severe_physical_guard"]
                    )
                    for frame in event_frames
                ),
            }
        )
    return rows


def _method_metrics(
    *,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for method, score_key in METHODS.items():
        rows.append(
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
        )
    return rows


def _decision_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        key: int(sum(int(row[key]) for row in rows))
        for key in [
            "relation_rescued_vs_ttc",
            "ttc_event_suppressed",
        ]
    }


def _load_confirmation_splits(
    confirmation_index: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    package = load_lazy_package(confirmation_index)
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    normalization = checkpoint["normalization"]
    return {
        split: _NormalizedSequenceView(
            sequences,
            normalization["x_mean"],
            normalization["x_std"],
            normalization["edge_mean"],
            normalization["edge_std"],
        )
        for split, sequences in package["splits"].items()
    }


def _score_all_splits(
    *,
    model: torch.nn.Module,
    device: torch.device,
    group_count: int,
    development_splits: dict[str, Any],
    confirmation_splits: dict[str, Any],
    recovery_initialization_predictions: int,
) -> dict[str, list[dict[str, Any]]]:
    requested = {
        "development_validation_normal": development_splits[
            "validation_normal"
        ],
        "development_validation_positive": development_splits[
            "validation_positive"
        ],
        "development_test_normal": development_splits["test_normal"],
        "development_test_positive": development_splits["test_positive"],
        "confirmation_normal": confirmation_splits["test_normal"],
        "confirmation_positive": confirmation_splits["test_positive"],
    }
    return {
        split: score_sequences(
            model,
            sequences,
            device,
            split=split,
            group_count=group_count,
            recovery_initialization_predictions=(
                recovery_initialization_predictions
            ),
        )
        for split, sequences in requested.items()
    }


def run_persistent_guarded_fusion(
    *,
    development_index: Path = DEFAULT_DEVELOPMENT_INDEX,
    confirmation_index: Path = DEFAULT_CONFIRMATION_INDEX,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    group_count: int = 4,
    seed: int = DEFAULT_SEED,
    recovery_initialization_predictions: int = 0,
) -> dict[str, Any]:
    if recovery_initialization_predictions < 0:
        raise ValueError(
            "recovery_initialization_predictions must be non-negative"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, development_splits, _, device = _load_model_and_sequences(
        development_index,
        checkpoint_path,
    )
    confirmation_splits = _load_confirmation_splits(
        confirmation_index,
        checkpoint_path,
    )
    frames = _score_all_splits(
        model=model,
        device=device,
        group_count=group_count,
        development_splits=development_splits,
        confirmation_splits=confirmation_splits,
        recovery_initialization_predictions=(
            recovery_initialization_predictions
        ),
    )

    component_calibration = robust_component_calibration(
        frames["development_validation_normal"]
    )
    for split_frames in frames.values():
        apply_danger_calibration(split_frames, component_calibration)
    percentile_references = _attach_percentile_inputs(
        frames,
        frames["development_validation_normal"],
    )
    for split_frames in frames.values():
        _attach_guarded_response(split_frames)

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
    metrics = []
    event_rows = []
    false_alarm_rows = []
    event_counts = {}
    for split, normal_key, positive_key in evaluations:
        metrics.extend(
            _method_metrics(
                normal_frames=frames[normal_key],
                positive_frames=frames[positive_key],
                split=split,
            )
        )
        positive_events = _event_diagnostics(
            frames[positive_key],
            split=f"{split}_positive",
            positive_window_only=True,
        )
        normal_events = _event_diagnostics(
            frames[normal_key],
            split=f"{split}_normal",
            positive_window_only=False,
        )
        event_rows.extend(positive_events)
        event_rows.extend(normal_events)
        event_counts[split] = {
            "positive": _decision_counts(positive_events),
            "hard_safe": _decision_counts(normal_events),
            "pair_counts": dict(
                Counter(
                    (
                        "both"
                        if int(row["ttc_percentile_detected"])
                        and int(row["persistent_guarded_detected"])
                        else "ttc_only"
                        if int(row["ttc_percentile_detected"])
                        else "guarded_only"
                        if int(row["persistent_guarded_detected"])
                        else "neither"
                    )
                    for row in positive_events
                )
            ),
        }
        for method, score_key in METHODS.items():
            false_alarm_rows.append(
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
    metrics_path = output_root / "persistent_guarded_metrics.csv"
    event_path = output_root / "persistent_guarded_events.csv"
    false_alarm_path = output_root / "persistent_guarded_false_alarms.csv"
    frame_path = output_root / "persistent_guarded_frames.csv"
    _write_csv(metrics_path, metrics)
    _write_csv(event_path, event_rows)
    _write_csv(false_alarm_path, false_alarm_rows)

    frame_rows = []
    frame_keys = [
        "inverse_ttc_ego_max",
        "danger_onset_shock_raw",
        "danger_stress",
        "danger_damage",
        "danger_persistence",
        "danger_recovery",
        "danger_activation_inhibited",
        "physical_percentile",
        "relation_percentile",
        "persistence_percentile",
        "recovery_percentile",
        "unconstrained_max",
        "persistent_relation_run",
        "persistent_relation_gate",
        "physical_recovery_run",
        "physical_inhibition_gate",
        "severe_physical_guard",
        "persistent_rescue",
        "persistent_guarded",
    ]
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
                    "physical": frame["physical"],
                    **{
                        key: frame["aggregations"][key]
                        for key in frame_keys
                    },
                }
            )
    _write_csv(frame_path, frame_rows)

    summary = {
        "protocol": "riskbench-persistent-guarded-fusion-v2",
        "model_retrained": False,
        "graph_input_protocol": "riskbench-metadata-no-gt-target-features-v1",
        "calibration_source": "development_validation_hard_safe_only",
        "development_index": str(development_index.expanduser().resolve()),
        "confirmation_index": str(confirmation_index.expanduser().resolve()),
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()),
        "seed": seed,
        "constants": {
            "alert_percentile": ALERT_PERCENTILE,
            "severe_physical_percentile": SEVERE_PHYSICAL_PERCENTILE,
            "relation_persistence_percentile": (
                RELATION_PERSISTENCE_PERCENTILE
            ),
            "recovery_percentile": RECOVERY_PERCENTILE,
            "persistent_relation_frames": PERSISTENT_RELATION_FRAMES,
            "persistent_relation_seconds": (
                PERSISTENT_RELATION_FRAMES * RISK_BENCH_TIMESTEP_SECONDS
            ),
            "recovery_trend_frames": RECOVERY_TREND_FRAMES,
            "recovery_trend_seconds": (
                RECOVERY_TREND_FRAMES * RISK_BENCH_TIMESTEP_SECONDS
            ),
            "recovery_initialization_predictions": (
                recovery_initialization_predictions
            ),
        },
        "component_calibration": component_calibration,
        "percentile_references": percentile_references,
        "metrics": metrics,
        "event_decision_counts": event_counts,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "events_csv": str(event_path),
            "false_alarms_csv": str(false_alarm_path),
            "frames_csv": str(frame_path),
        },
        "status": "fixed_no_training_ablation_complete",
    }
    summary_path = output_root / "persistent_guarded_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    result = run_persistent_guarded_fusion(
        development_index=args.development_index,
        confirmation_index=args.confirmation_index,
        checkpoint_path=args.checkpoint_path,
        output_root=args.output_root,
        group_count=args.group_count,
        seed=args.seed,
        recovery_initialization_predictions=(
            args.recovery_initialization_predictions
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
