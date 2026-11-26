"""Fit hard-safe-only one-class inhibition for frozen danger gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ..danger_signal.diagnostics import false_alarm_morphology
from ..temporal_relation_train_eval import (
    _evaluate_dual_track,
    _group_dual_frames_by_event,
)
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .audited_gate_ablation import _attach_candidate_scores
from .development_ceiling_ablation import _attach_development_ceiling
from .early_window_sensitivity import _attach_window_scores
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
    / "ego_immune_riskbench_hard_safe_inhibition_v1"
)
SEED = 20260727
PROFILES = {
    "strict_10": 0.10,
    "balanced_20": 0.20,
    "balanced_30": 0.30,
    "high_recall_40": 0.40,
}
FEATURES = (
    "physical_percentile",
    "relation_percentile",
    "persistence_percentile",
    "recovery_percentile",
    "danger_stress",
    "danger_damage",
    "danger_persistence",
    "danger_recovery",
    "peak_support_count",
    "peak_relation_rise",
    "window_onset_positive_percentile",
    "audited_gate_near_sustain",
    "audited_gate_supported_peak_p80",
    "audited_gate_very_high_supported",
    "audited_gate_isolated_extreme_rise",
    "temporal_confirmed_onset_gate",
    "short_history_onset_recovery_gate",
    "repeated_recovery_memory_gate",
)


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


def _feature_vector(frame: dict[str, Any]) -> np.ndarray:
    values = frame["aggregations"]
    return np.asarray([float(values[key]) for key in FEATURES], dtype=float)


def _fit_hard_safe_profiles(
    frames: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, IsolationForest], np.ndarray, np.ndarray, int]:
    reference_frames = [
        frame
        for frame in frames["development_validation_normal"]
        if frame["aggregations"]["development_ceiling"]
        >= ALERT_PERCENTILE
        and frame["aggregations"]["physical_percentile"]
        < ALERT_PERCENTILE
    ]
    reference = np.stack(
        [_feature_vector(frame) for frame in reference_frames]
    )
    center = np.median(reference, axis=0)
    scale = np.maximum(
        np.median(np.abs(reference - center), axis=0) * 1.4826,
        0.05,
    )
    normalized = (reference - center) / scale
    models = {
        profile: IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=SEED,
        ).fit(normalized)
        for profile, contamination in PROFILES.items()
    }
    return models, center, scale, len(reference_frames)


def _attach_inhibited_scores(
    frames: dict[str, list[dict[str, Any]]],
    *,
    models: dict[str, IsolationForest],
    center: np.ndarray,
    scale: np.ndarray,
) -> None:
    for split_frames in frames.values():
        for frame in split_frames:
            values = frame["aggregations"]
            relation_candidate = (
                values["development_ceiling"] >= ALERT_PERCENTILE
                and values["physical_percentile"] < ALERT_PERCENTILE
            )
            normalized = (
                (_feature_vector(frame) - center) / scale
            ).reshape(1, -1)
            for profile, model in models.items():
                allow_relation = relation_candidate and (
                    int(model.predict(normalized)[0]) == -1
                )
                values[f"inhibited_{profile}"] = max(
                    float(values["physical_percentile"]),
                    min(
                        float(values["development_ceiling"]),
                        ALERT_PERCENTILE - 1e-6,
                    ),
                    1.0 if allow_relation else 0.0,
                )


def _metric_rows(
    frames: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    methods = {
        "ttc": "physical_percentile",
        "uninhibited_ceiling": "development_ceiling",
        **{
            profile: f"inhibited_{profile}"
            for profile in PROFILES
        },
    }
    for split in (
        "development_validation",
        "development_test",
        "confirmation",
    ):
        normal = frames[f"{split}_normal"]
        positive = frames[f"{split}_positive"]
        for window, lead_frames in (
            ("primary_1s", 20),
            ("extended_1p5s", 30),
        ):
            for method, score_key in methods.items():
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


def _ttc_blind_rows(
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
        if any(
            frame["aggregations"]["physical_percentile"]
            >= ALERT_PERCENTILE
            for frame in usable
        ):
            continue
        row = {"event_uid": event_frames[0]["event_uid"]}
        for profile in PROFILES:
            score_key = f"inhibited_{profile}"
            row[f"{profile}_rescued"] = int(
                any(
                    frame["aggregations"][score_key]
                    >= ALERT_PERCENTILE
                    for frame in usable
                )
            )
        rows.append(row)
    return rows


def run_hard_safe_inhibition(
    *,
    frame_path: Path = DEFAULT_FRAME_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    frame_path = frame_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    frames = _load_frame_rows(frame_path)
    onset_reference = _positive_reference(
        frames["development_validation_normal"]
    )
    _attach_window_scores(frames, onset_reference)
    _attach_candidate_scores(frames)
    _attach_development_ceiling(frames)
    models, center, scale, reference_count = _fit_hard_safe_profiles(frames)
    _attach_inhibited_scores(
        frames,
        models=models,
        center=center,
        scale=scale,
    )

    metrics = _metric_rows(frames)
    blind_events = _ttc_blind_rows(frames)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "hard_safe_inhibition_metrics.csv"
    blind_path = output_root / "confirmation_ttc_blind_inhibition.csv"
    model_path = output_root / "hard_safe_inhibition.joblib"
    _write_csv(metrics_path, metrics)
    _write_csv(blind_path, blind_events)
    joblib.dump(
        {
            "models": models,
            "center": center,
            "scale": scale,
            "features": FEATURES,
            "profiles": PROFILES,
            "seed": SEED,
            "source": "development_validation_normal_relation_alert_frames",
        },
        model_path,
    )

    selected_profile = "high_recall_40"
    selected_metrics = {
        row["split"]: row
        for row in metrics
        if row["window"] == "extended_1p5s"
        and row["method"] == selected_profile
    }
    result = {
        "protocol": "riskbench-hard-safe-only-inhibition-v1",
        "positive_frames_used_for_fit": 0,
        "calibration_split": "development_validation_normal",
        "calibration_relation_alert_frame_count": reference_count,
        "features": list(FEATURES),
        "profiles": PROFILES,
        "provisionally_selected_profile": selected_profile,
        "selection_reason": (
            "Predeclared high-recall operating profile; selection did not "
            "use positive labels."
        ),
        "selected_profile_metrics": selected_metrics,
        "confirmation_ttc_blind_rescued": sum(
            int(row[f"{selected_profile}_rescued"])
            for row in blind_events
        ),
        "confirmation_ttc_blind_count": len(blind_events),
        "warning": (
            "Confirmation was previously audited and is development-only. "
            "The saved inhibitor must be evaluated unchanged on a new holdout."
        ),
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "ttc_blind_csv": str(blind_path),
            "model_joblib": str(model_path),
        },
        "status": "hard_safe_only_inhibition_complete",
    }
    summary_path = output_root / "hard_safe_inhibition_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(summary_path)
    return result


def main() -> None:
    args = parse_args()
    result = run_hard_safe_inhibition(
        frame_path=args.frame_path,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
