"""Causal CV/CA motion-rollout baselines with one shared risk scorer.

The adaptive model is a two-mode kinematic mixture, not an interacting multiple
model (IMM) filter and not a calibrated collision-probability estimator.  Its
score is the expected soft encounter risk under constant-velocity (CV) and
constant-acceleration (CA) rollouts.  All variants use the same horizon,
center-clearance kernel, actor aggregation, event aggregation, and matched
hard-safe calibration.  Safe-core data set only robust acceleration and
backtest-error scales; no danger labels are used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .core_hybrid_evaluation import (
    _usable_frames,
    event_alert,
    event_score,
    threshold_at_event_fpr,
)
from .family_grouped_nested_cv import WINDOW_FRAMES
from .mainstream_baseline_evaluation import _load_state
from .mainstream_baseline_evaluation_v2 import (
    DEFAULT_FPR_BUDGETS,
    DEFAULT_INDEX_ROOT,
    DEFAULT_PAIR_DAMP_ROOT,
    DEFAULT_SEEDS,
    _aggregate_seed_summary,
    _family_bootstrap_intervals,
    _formal_reference_rows,
    _load_index,
    _load_pair_frames,
    _localization_summary,
    _paired_bootstrap,
    _record_map,
    _summarize_seed,
    _write_csv,
)
from .pair_aligned_damp_evaluation import _attach_physical_percentile
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/riskbench_probabilistic_rollout_v2"
PROTOCOL = "riskbench-probabilistic-cv-ca-rollout-v2"
METHODS = {
    "cv": "rollout_cv_shared_risk",
    "ca": "rollout_ca_shared_risk",
    "mixture": "rollout_adaptive_cv_ca_shared_risk",
}
PREDICTION_STEPS = 30
CLEARANCE_M = 3.0
HISTORY_STEPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--pair-damp-root", type=Path, default=DEFAULT_PAIR_DAMP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def _consecutive_positions(
    state: dict[int, dict[str, np.ndarray]], actor: str, endpoint: int
) -> np.ndarray:
    positions: list[np.ndarray] = []
    for frame in range(endpoint, endpoint - HISTORY_STEPS, -1):
        value = state.get(frame, {}).get(actor)
        if value is None:
            break
        positions.append(np.asarray(value, dtype=np.float64))
    return np.stack(positions[::-1]) if positions else np.empty((0, 2))


def _polynomial_state(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit position over recent time and return velocity/acceleration now.

    Quadratic least squares is materially less sensitive to 20 Hz coordinate
    jitter than applying two finite differences. With fewer than ten samples,
    use a linear fit and set acceleration to zero.
    """
    if len(positions) < 2:
        return np.zeros(2), np.zeros(2)
    times = (
        np.arange(len(positions), dtype=np.float64) - (len(positions) - 1)
    ) * RISK_BENCH_TIMESTEP_SECONDS
    if len(positions) < 10:
        design = np.column_stack([np.ones(len(times)), times])
        coefficient = np.linalg.lstsq(design, positions, rcond=None)[0]
        return coefficient[1], np.zeros(2)
    design = np.column_stack([np.ones(len(times)), times, 0.5 * times * times])
    coefficient = np.linalg.lstsq(design, positions, rcond=None)[0]
    return coefficient[1], coefficient[2]


def _kinematics(positions: np.ndarray, acceleration_cap: float) -> tuple[np.ndarray, np.ndarray]:
    velocity, acceleration = _polynomial_state(positions)
    norm = float(np.linalg.norm(acceleration))
    if norm > acceleration_cap > 0:
        acceleration = acceleration * acceleration_cap / norm
    return velocity, acceleration


def _one_step_errors(positions: np.ndarray, acceleration_cap: float) -> tuple[float, float] | None:
    if len(positions) < 4:
        return None
    training = positions[:-1]
    velocity, acceleration = _kinematics(training, acceleration_cap)
    dt = RISK_BENCH_TIMESTEP_SECONDS
    cv = training[-1] + velocity * dt
    ca = cv + 0.5 * acceleration * dt * dt
    return (
        float(np.linalg.norm(positions[-1] - cv)),
        float(np.linalg.norm(positions[-1] - ca)),
    )


def fit_safe_scales(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    acceleration_norms: list[float] = []
    raw_states: list[dict[int, dict[str, np.ndarray]]] = []
    for record in records:
        state = _load_state(record["state_path"])
        raw_states.append(state)
        for endpoint, actors in state.items():
            for actor in actors:
                positions = _consecutive_positions(state, actor, endpoint)
                if len(positions) == HISTORY_STEPS:
                    _, acceleration = _polynomial_state(positions)
                    acceleration_norms.append(float(np.linalg.norm(acceleration)))
    if acceleration_norms:
        acceleration_median = float(np.median(acceleration_norms))
        acceleration_mad = float(
            np.median(np.abs(np.asarray(acceleration_norms) - acceleration_median))
        )
        # A robust three-scale upper fence prevents tracking jitter from
        # producing unconstrained quadratic rollouts. This is estimated only
        # from safe-core families and is not interpreted as a physical limit.
        acceleration_cap = acceleration_median + 3.0 * 1.4826 * acceleration_mad
    else:
        acceleration_median = 0.0
        acceleration_mad = 0.0
        acceleration_cap = 1.0
    errors: list[float] = []
    for state in raw_states:
        for endpoint, actors in state.items():
            for actor in actors:
                pair = _one_step_errors(
                    _consecutive_positions(state, actor, endpoint), acceleration_cap
                )
                if pair is not None:
                    errors.extend(pair)
    error_temperature = float(np.median(errors)) if errors else 1.0
    return {
        "acceleration_cap_mps2": max(acceleration_cap, 1e-3),
        "acceleration_median_mps2": acceleration_median,
        "acceleration_mad_mps2": acceleration_mad,
        "acceleration_cap_rule": "median + 3 * 1.4826 * MAD on full 20-frame safe fits",
        "mode_error_temperature_m": max(error_temperature, 1e-3),
        "safe_acceleration_sample_count": len(acceleration_norms),
        "safe_backtest_error_count": len(errors),
    }


def _rollouts(
    positions: np.ndarray, safe: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    velocity, acceleration = _kinematics(
        positions, safe["acceleration_cap_mps2"]
    )
    times = (
        np.arange(1, PREDICTION_STEPS + 1, dtype=np.float64)
        * RISK_BENCH_TIMESTEP_SECONDS
    )
    origin = positions[-1]
    cv = origin[None] + times[:, None] * velocity[None]
    ca = cv + 0.5 * (times[:, None] ** 2) * acceleration[None]
    errors = _one_step_errors(positions, safe["acceleration_cap_mps2"])
    if errors is None:
        weights = np.asarray([0.5, 0.5], dtype=np.float64)
    else:
        logits = -np.asarray(errors) / safe["mode_error_temperature_m"]
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
    return np.stack([cv, ca]), weights


def shared_soft_encounter_score(
    trajectories: np.ndarray, weights: np.ndarray
) -> float:
    """Expected soft center-encounter score under supplied trajectory modes."""
    distances = np.linalg.norm(trajectories, axis=-1)
    times = (
        np.arange(1, PREDICTION_STEPS + 1, dtype=np.float64)
        * RISK_BENCH_TIMESTEP_SECONDS
    )
    horizon = PREDICTION_STEPS * RISK_BENCH_TIMESTEP_SECONDS
    proximity = np.exp(-0.5 * (distances / CLEARANCE_M) ** 2)
    urgency = np.exp(-times / horizon)
    mode_scores = np.max(proximity * urgency[None], axis=1)
    return float(np.dot(weights, mode_scores))


def frame_actor_scores(
    state: dict[int, dict[str, np.ndarray]], endpoint: int, safe: dict[str, float]
) -> dict[str, dict[str, float]]:
    output = {key: {} for key in METHODS}
    for actor in state.get(endpoint, {}):
        positions = _consecutive_positions(state, str(actor), endpoint)
        if len(positions) == 0:
            continue
        trajectories, weights = _rollouts(positions, safe)
        output["cv"][str(actor)] = shared_soft_encounter_score(
            trajectories[:1], np.asarray([1.0])
        )
        output["ca"][str(actor)] = shared_soft_encounter_score(
            trajectories[1:], np.asarray([1.0])
        )
        output["mixture"][str(actor)] = shared_soft_encounter_score(
            trajectories, weights
        )
    return output


def _rank(scores: dict[str, float], targets: Sequence[str]) -> int | None:
    present = [target for target in targets if target in scores]
    if not present:
        return None
    target_score = max(scores[target] for target in present)
    return 1 + sum(value > target_score for value in scores.values())


def _event_groups(pair_frames: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, frames in pair_frames.items():
        output[split] = {}
        for event_frames in _group_dual_frames_by_event(frames).values():
            if event_frames:
                output[split][str(event_frames[0]["event_uid"])] = _usable_frames(
                    event_frames
                )
    return output


def score_fold(
    *,
    seed: int,
    fold: int,
    index_root: Path,
    pair_damp_root: Path,
    safe: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = _load_index(index_root, fold)
    records = _record_map(index)
    pair_frames = _load_pair_frames(pair_damp_root, seed, fold)
    _attach_physical_percentile(pair_frames)
    groups = _event_groups(pair_frames)
    evaluation_records = [
        record
        for split in ("validation_normal", "test_normal", "test_positive")
        for record in index["splits"][split]
    ]
    states = {
        str(record["scenario_id"]): _load_state(record["state_path"])
        for record in evaluation_records
    }
    traces: dict[str, dict[str, dict[str, list[tuple[dict[str, Any], float]]]]] = {
        method: defaultdict(dict) for method in METHODS
    }
    actor_traces: dict[str, dict[str, dict[str, list[tuple[dict[str, Any], dict[str, float]]]]]] = {
        method: defaultdict(dict) for method in METHODS
    }
    ttc_traces: dict[str, dict[str, list[tuple[dict[str, Any], float]]]] = defaultdict(dict)
    for split in ("validation_normal", "test_normal", "test_positive"):
        for uid, frames in groups[split].items():
            record = records[uid]
            state = states[str(record["scenario_id"])]
            for method in METHODS:
                traces[method][split][uid] = []
                actor_traces[method][split][uid] = []
            ttc_traces[split][uid] = []
            for frame in frames:
                endpoint = int(frame["source_frame_index"])
                scores = frame_actor_scores(state, endpoint, safe)
                for method in METHODS:
                    actor_traces[method][split][uid].append((frame, scores[method]))
                    traces[method][split][uid].append(
                        (frame, max(scores[method].values(), default=0.0))
                    )
                ttc_traces[split][uid].append(
                    (frame, float(frame["aggregations"]["physical_percentile"]))
                )
    ttc_flags = {
        split: {
            uid: event_alert(trace, ALERT_PERCENTILE)[0]
            for uid, trace in split_traces.items()
        }
        for split, split_traces in ttc_traces.items()
    }
    rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    localization: list[dict[str, Any]] = []
    for short_method, method in METHODS.items():
        for budget in DEFAULT_FPR_BUDGETS:
            calibration_uids = [
                uid
                for uid in traces[short_method]["validation_normal"]
                if not ttc_flags["validation_normal"][uid]
            ]
            threshold, observed_fpr = threshold_at_event_fpr(
                [
                    event_score(traces[short_method]["validation_normal"][uid])
                    for uid in calibration_uids
                ],
                budget,
            )
            thresholds.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "method": method,
                    "fpr_budget": budget,
                    "threshold": threshold,
                    "validation_ttc_negative_count": len(calibration_uids),
                    "validation_incremental_fpr": observed_fpr,
                }
            )
            for split in ("test_normal", "test_positive"):
                for uid, trace in traces[short_method][split].items():
                    record = records[uid]
                    detected, lead = event_alert(trace, threshold)
                    ttc_detected, ttc_lead = event_alert(
                        ttc_traces[split][uid], ALERT_PERCENTILE
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "method": method,
                            "fpr_budget": budget,
                            "event_uid": uid,
                            "scenario_group_id": record["scenario_group_id"],
                            "category": record["category"],
                            "scenario_id": record["scenario_id"],
                            "target_type": record.get("target_type_name", "none"),
                            "split": split,
                            "positive": int(split == "test_positive"),
                            "score": event_score(trace),
                            "threshold": threshold,
                            "detected": int(detected),
                            "lead_seconds": "" if lead is None else lead,
                            "ttc_score": event_score(ttc_traces[split][uid]),
                            "ttc_detected": int(ttc_detected),
                            "ttc_lead_seconds": "" if ttc_lead is None else ttc_lead,
                            "final_detected": int(ttc_detected or detected),
                            "blind_rescue": int((not ttc_detected) and detected),
                        }
                    )
            for uid, frame_scores in actor_traces[short_method]["test_positive"].items():
                record = records[uid]
                targets = [str(value) for value in record.get("risk_actor_ids", [])]
                aggregate: dict[str, float] = {}
                for _, scores in frame_scores:
                    for actor, value in scores.items():
                        aggregate[actor] = max(aggregate.get(actor, 0.0), value)
                event_rank = _rank(aggregate, targets)
                alert_rank = None
                alert_lead = None
                alerted = False
                for (frame, frame_score), (_, actor_scores) in zip(
                    traces[short_method]["test_positive"][uid], frame_scores
                ):
                    if frame_score >= threshold:
                        alerted = True
                        alert_rank = _rank(actor_scores, targets)
                        alert_lead = int(
                            frame.get("eval_frames_to_anchor", frame["frames_to_anchor"])
                        ) * RISK_BENCH_TIMESTEP_SECONDS
                        break
                localization.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "method": method,
                        "fpr_budget": budget,
                        "event_uid": uid,
                        "target_type": record.get("target_type_name", "unknown"),
                        "target_ids": "|".join(targets),
                        "event_target_rank": "" if event_rank is None else event_rank,
                        "event_top1": int(event_rank is not None and event_rank <= 1),
                        "event_top3": int(event_rank is not None and event_rank <= 3),
                        "method_alerted": int(alerted),
                        "alert_target_rank": "" if alert_rank is None else alert_rank,
                        "alert_top1": int(alert_rank is not None and alert_rank <= 1),
                        "alert_top3": int(alert_rank is not None and alert_rank <= 3),
                        "alert_lead_seconds": "" if alert_lead is None else alert_lead,
                    }
                )
    return rows, thresholds, localization


def run_evaluation(
    *,
    index_root: Path = DEFAULT_INDEX_ROOT,
    pair_damp_root: Path = DEFAULT_PAIR_DAMP_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    folds: Iterable[int] = range(5),
    seeds: Iterable[int] = DEFAULT_SEEDS,
    bootstrap_replicates: int = 5000,
) -> dict[str, Any]:
    folds = tuple(int(value) for value in folds)
    seeds = tuple(int(value) for value in seeds)
    output_root.mkdir(parents=True, exist_ok=True)
    safe_by_fold = {
        fold: fit_safe_scales(_load_index(index_root, fold)["splits"]["train_normal"])
        for fold in folds
    }
    freeze_path = output_root / "pre_outer_test_freeze.json"
    freeze_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "status": "frozen_before_outer_evaluation",
                "folds": folds,
                "seeds": seeds,
                "history_steps_for_state_estimation": HISTORY_STEPS,
                "prediction_steps": PREDICTION_STEPS,
                "timestep_seconds": RISK_BENCH_TIMESTEP_SECONDS,
                "clearance_m": CLEARANCE_M,
                "shared_score": "weighted soft center encounter with temporal urgency",
                "mixture": "adaptive CV/CA; not IMM and not calibrated collision probability",
                "positive_labels_used": False,
                "safe_fitted_parameters": safe_by_fold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    event_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    for seed in seeds:
        event_rows.extend(
            _formal_reference_rows(
                seed=seed,
                pair_damp_root=pair_damp_root,
                index_root=index_root,
                folds=folds,
            )
        )
        for fold in folds:
            rows, thresholds, localization = score_fold(
                seed=seed,
                fold=fold,
                index_root=index_root,
                pair_damp_root=pair_damp_root,
                safe=safe_by_fold[fold],
            )
            event_rows.extend(rows)
            threshold_rows.extend(thresholds)
            localization_rows.extend(localization)
            print(json.dumps({"phase": "outer_evaluation", "seed": seed, "fold": fold, "rows": len(rows)}), flush=True)
    seed_summary = _summarize_seed(event_rows)
    aggregate = _aggregate_seed_summary(seed_summary)
    intervals = _family_bootstrap_intervals(event_rows, bootstrap_replicates)
    paired = _paired_bootstrap(event_rows, bootstrap_replicates)
    localization_summary = _localization_summary(localization_rows, pair_damp_root)
    _write_csv(output_root / "all_seed_events.csv", event_rows)
    _write_csv(output_root / "all_seed_thresholds.csv", threshold_rows)
    _write_csv(output_root / "per_seed_summary.csv", seed_summary)
    _write_csv(output_root / "multi_seed_aggregate.csv", aggregate)
    _write_csv(output_root / "family_bootstrap_intervals.csv", intervals)
    _write_csv(output_root / "paired_family_bootstrap.csv", paired)
    _write_csv(output_root / "localization_events.csv", localization_rows)
    _write_csv(output_root / "localization_per_seed_summary.csv", localization_summary)
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "freeze": str(freeze_path),
        "method_contract": {
            "claim_tier": "controlled_same_input_motion_rollout",
            "positive_training": False,
            "risk_actor_training": False,
            "footprints_or_yaw_used": False,
            "published_method_reproduction": False,
        },
        "aggregate": aggregate,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        index_root=args.index_root.expanduser().resolve(),
        pair_damp_root=args.pair_damp_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        folds=[int(value) for value in args.folds.split(",") if value.strip()],
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
