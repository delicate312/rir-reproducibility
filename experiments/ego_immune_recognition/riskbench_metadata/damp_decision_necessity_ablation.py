"""Evaluate the necessity of each DAMP decision mechanism at matched FPR.

The forecast checkpoints and cached frame predictions are held fixed.  Every
learned branch receives its own event threshold fitted only on TTC-negative
``validation_normal`` events.  Positive and outer-normal events are opened
only after threshold fitting and are used solely for evaluation.

This is a post-hoc mechanistic ablation, not a new sealed holdout.  It compares
the paper's original hand-set DAMP head with a parsimonious hard-safe-only
configuration and a cumulative chain from raw residual to the full decision.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .core_hybrid_evaluation import (
    _usable_frames,
    event_alert,
    event_score,
    threshold_at_event_fpr,
)
from .damp_hard_safe_grid_selection import (
    Candidate,
    PAIR_SUPPORT_FRAMES,
    PERCENTILE_WARMUP_FRAMES,
    SEVERE_CPA_PERCENTILE,
    _calibration_from_components,
    _event_frames,
    _recompute_components,
)
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT
from .normal_only_checkpoint_audit import _attach_event_metadata
from .pair_aligned_damp_evaluation import _normalized_component, _percentile
from .pair_damp_freeze import PROJECT_ROOT, sha256_file, sha256_payload
from .relation_geometry_rescue import (
    _geometry_from_observations,
    _observations_by_frame,
)


DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_normal_only_multiseed_v1"
)
DEFAULT_SELECTION_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_damp_hard_safe_grid_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_damp_necessity_v1"
)
DEFAULT_SEEDS = (20260812, 20260813, 20260814)
DEFAULT_FPR_BUDGET = 0.05
TTC_THRESHOLD = 0.95
BOOTSTRAP_REPLICATES = 10_000

SIMPLE = Candidate(
    persistence_decay=0.50,
    weight_stress=1.0 / 3.0,
    weight_damage=1.0 / 3.0,
    weight_persistence=1.0 / 3.0,
    recovery_beta=0.0,
    recovery_warmup=0,
)
SIMPLE_RECOVERY = Candidate(
    persistence_decay=0.50,
    weight_stress=1.0 / 3.0,
    weight_damage=1.0 / 3.0,
    weight_persistence=1.0 / 3.0,
    recovery_beta=0.10,
    recovery_warmup=0,
)
SIMPLE_NO_STRESS = Candidate(
    persistence_decay=0.50,
    weight_stress=0.0,
    weight_damage=0.5,
    weight_persistence=0.5,
    recovery_beta=0.0,
    recovery_warmup=0,
)
SIMPLE_NO_DAMAGE = Candidate(
    persistence_decay=0.50,
    weight_stress=0.5,
    weight_damage=0.0,
    weight_persistence=0.5,
    recovery_beta=0.0,
    recovery_warmup=0,
)
SIMPLE_NO_PERSISTENCE = Candidate(
    persistence_decay=0.50,
    weight_stress=0.5,
    weight_damage=0.5,
    weight_persistence=0.0,
    recovery_beta=0.0,
    recovery_warmup=0,
)
PAPER_CURRENT = Candidate(
    persistence_decay=0.80,
    weight_stress=0.25,
    weight_damage=0.35,
    weight_persistence=0.40,
    recovery_beta=0.10,
    recovery_warmup=3,
)
PAPER_CURRENT_NO_RECOVERY = Candidate(
    persistence_decay=0.80,
    weight_stress=0.25,
    weight_damage=0.35,
    weight_persistence=0.40,
    recovery_beta=0.0,
    recovery_warmup=0,
)

METHOD_ORDER = (
    "ttc_only",
    "cpa_only",
    "severe_cpa_shortcut",
    "cpa_temporal_matched",
    "raw_residual",
    "directional_damage",
    "stress_damage",
    "damp_no_recovery",
    "damp_with_recovery",
    "same_pair_cpa",
    "same_pair_cpa_confirmed",
    "full_no_confirmation",
    "full_no_stress",
    "full_no_damage",
    "full_no_persistence",
    "full_with_recovery",
    "full_rir_simple",
    "rir_nested_selected",
    "paper_current_no_recovery",
    "paper_current_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--fpr-budget", type=float, default=DEFAULT_FPR_BUDGET)
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("Expected at least one integer.")
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _attach_family_metadata(
    rows: Sequence[dict[str, Any]], index_root: Path
) -> None:
    mapping: dict[tuple[int, str], str] = {}
    for fold in sorted({int(row["fold"]) for row in rows}):
        payload = json.loads(
            (
                index_root
                / f"fold_{fold:02d}"
                / "riskbench_nested_fold_index.json"
            ).read_text(encoding="utf-8")
        )
        for split in ("test_positive", "test_normal"):
            for record in payload["splits"][split]:
                uid = f"riskbench:{record['category']}:{record['scenario_id']}"
                mapping[(fold, uid)] = str(record["scenario_group_id"])
    for row in rows:
        key = (int(row["fold"]), str(row["event_uid"]))
        if key not in mapping:
            raise KeyError(f"No scenario family for {key}")
        row["scenario_group_id"] = mapping[key]


def _load_frames(source_root: Path, seed: int, fold: int) -> dict[str, list[dict[str, Any]]]:
    path = (
        source_root
        / f"seed_{seed}"
        / "pair_frame_cache"
        / "graph_free_one_class"
        / f"fold_{fold:02d}.pkl"
    )
    with path.open("rb") as handle:
        frames = pickle.load(handle)
    required = {"validation_normal", "test_normal", "test_positive"}
    if not required.issubset(frames):
        raise KeyError(f"Missing cached splits in {path}: {required - set(frames)}")
    return frames


def _component_values(
    components: dict[str, dict[int, dict[str, dict[str, float]]]],
    event_uids: Iterable[str],
    key: str,
) -> list[float]:
    return [
        float(values[key])
        for uid in event_uids
        for relations in components[uid].values()
        for values in relations.values()
    ]


def _single_calibration(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        raise ValueError("Cannot calibrate an empty component.")
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    upper = float(np.quantile(array, 0.90) - center)
    return {"center": center, "scale": max(1.4826 * mad, upper, 1e-3)}


def _causal_by_target(
    events: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[int, dict[str, float]]]:
    output: dict[str, dict[int, dict[str, float]]] = {}
    for uid, event_rows in events.items():
        output[uid] = {}
        for frame in event_rows:
            output[uid][int(frame["frame_position"])] = {
                str(values["target_id"]): float(values["causal_raw"])
                for values in frame.get("relation_components", {}).values()
                if int(values.get("ego_outgoing", 0))
            }
    return output


def _activation_fn(
    method: str,
    calibration: dict[str, dict[str, float]],
    candidate: Candidate,
) -> Callable[[dict[str, float]], float]:
    def normalized(values: dict[str, float], key: str) -> float:
        return _normalized_component(float(values[key]), calibration[key])

    if method == "raw_residual":
        return lambda values: normalized(values, "causal_raw")
    if method == "directional_damage":
        return lambda values: normalized(values, "damage_raw")
    if method == "stress_damage":
        return lambda values: 0.5 * (
            normalized(values, "stress_raw") + normalized(values, "damage_raw")
        )
    if method in {
        "damp_no_recovery",
        "damp_with_recovery",
        "same_pair_cpa",
        "same_pair_cpa_confirmed",
        "full_no_confirmation",
        "full_no_stress",
        "full_no_damage",
        "full_no_persistence",
        "full_with_recovery",
        "full_rir_simple",
        "rir_nested_selected",
        "paper_current_no_recovery",
        "paper_current_full",
    }:
        def damp(values: dict[str, float]) -> float:
            positive = (
                candidate.weight_stress * normalized(values, "stress_raw")
                + candidate.weight_damage * normalized(values, "damage_raw")
                + candidate.weight_persistence * normalized(values, "persistence_raw")
            )
            return max(
                positive
                - candidate.recovery_beta * normalized(values, "recovery_raw"),
                0.0,
            )

        return damp
    raise ValueError(method)


def _calibration(
    events: dict[str, list[dict[str, Any]]],
    components: dict[str, dict[int, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, float]]:
    output = _calibration_from_components(components, set(events))
    causal = _causal_by_target(events)
    output["causal_raw"] = _single_calibration(
        [
            value
            for uid in causal
            for by_target in causal[uid].values()
            for value in by_target.values()
        ]
    )
    return output


def _merged_components(
    events: dict[str, list[dict[str, Any]]],
    recomputed: dict[str, dict[int, dict[str, dict[str, float]]]],
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    causal = _causal_by_target(events)
    output = recomputed
    for uid, positions in output.items():
        for position, targets in positions.items():
            for target, values in targets.items():
                values["causal_raw"] = causal[uid][position][target]
    return output


def _activation_reference(
    components: dict[str, dict[int, dict[str, dict[str, float]]]],
    activation: Callable[[dict[str, float]], float],
) -> np.ndarray:
    values = np.asarray(
        [
            activation(component)
            for positions in components.values()
            for targets in positions.values()
            for component in targets.values()
        ],
        dtype=float,
    )
    if not values.size:
        raise ValueError("Empty activation reference.")
    return np.sort(values)


def _physical_reference(events: dict[str, list[dict[str, Any]]]) -> np.ndarray:
    values = np.asarray(
        [
            float(frame["aggregations"]["inverse_ttc_ego_max"])
            for frames in events.values()
            for frame in frames
            if int(frame["frame_position"]) >= PERCENTILE_WARMUP_FRAMES
        ],
        dtype=float,
    )
    if not values.size:
        raise ValueError("Empty physical reference.")
    return np.sort(values)


def _physical_trace(
    frames: Sequence[dict[str, Any]], reference: np.ndarray
) -> list[tuple[dict[str, Any], float]]:
    return [
        (
            frame,
            _percentile(
                reference,
                float(frame["aggregations"]["inverse_ttc_ego_max"]),
            ),
        )
        for frame in _usable_frames(frames)
    ]


def _relation_trace(
    frames: Sequence[dict[str, Any]],
    components: dict[int, dict[str, dict[str, float]]],
    activation: Callable[[dict[str, float]], float],
    activation_reference: np.ndarray,
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
    cpa_reference: np.ndarray,
    *,
    alignment: bool,
    confirmation: bool,
    severe_bypass: bool,
) -> list[tuple[dict[str, Any], float]]:
    histories: dict[str, list[tuple[int, float]]] = defaultdict(list)
    output: list[tuple[dict[str, Any], float]] = []
    for frame in _usable_frames(frames):
        uid = str(frame["event_uid"])
        position = int(frame["frame_position"])
        by_target = components.get(position, {})
        best = 0.0
        if not alignment:
            best = max(
                (
                    _percentile(activation_reference, activation(values))
                    for values in by_target.values()
                ),
                default=0.0,
            )
        else:
            for target, geo in geometry.get((uid, position), {}).items():
                values = by_target.get(str(target))
                if values is None:
                    continue
                damp = _percentile(activation_reference, activation(values))
                cpa = _percentile(cpa_reference, float(geo["raw"]))
                aligned = math.sqrt(max(damp * cpa, 0.0))
                if confirmation:
                    history = histories[str(target)]
                    if history and position != history[-1][0] + 1:
                        history.clear()
                    history.append((position, aligned))
                    history[:] = history[-PAIR_SUPPORT_FRAMES:]
                    candidate_score = (
                        float(np.mean([value for _, value in history]))
                        if len(history) == PAIR_SUPPORT_FRAMES
                        else 0.0
                    )
                else:
                    candidate_score = aligned
                if severe_bypass and cpa >= SEVERE_CPA_PERCENTILE:
                    candidate_score = max(candidate_score, cpa)
                best = max(best, candidate_score)
        output.append((frame, best))
    return output


def _cpa_trace(
    frames: Sequence[dict[str, Any]],
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
    cpa_reference: np.ndarray,
    *,
    confirmation: bool = False,
    severe_bypass: bool = False,
    severe_only: bool = False,
) -> list[tuple[dict[str, Any], float]]:
    histories: dict[str, list[tuple[int, float]]] = defaultdict(list)
    output = []
    for frame in _usable_frames(frames):
        position = int(frame["frame_position"])
        key = (str(frame["event_uid"]), position)
        best = 0.0
        for target, values in geometry.get(key, {}).items():
            cpa = _percentile(cpa_reference, float(values["raw"]))
            if severe_only:
                candidate = cpa if cpa >= SEVERE_CPA_PERCENTILE else 0.0
            elif confirmation:
                history = histories[str(target)]
                if history and position != history[-1][0] + 1:
                    history.clear()
                history.append((position, cpa))
                history[:] = history[-PAIR_SUPPORT_FRAMES:]
                candidate = (
                    float(np.mean([value for _, value in history]))
                    if len(history) == PAIR_SUPPORT_FRAMES
                    else 0.0
                )
            else:
                candidate = cpa
            if severe_bypass and cpa >= SEVERE_CPA_PERCENTILE:
                candidate = max(candidate, cpa)
            best = max(best, candidate)
        output.append((frame, best))
    return output


def _cpa_reference(
    events: dict[str, list[dict[str, Any]]],
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
) -> np.ndarray:
    values = np.asarray(
        [
            float(item["raw"])
            for uid, frames in events.items()
            for frame in frames
            for item in geometry.get((uid, int(frame["frame_position"])), {}).values()
        ],
        dtype=float,
    )
    if not values.size:
        raise ValueError("Empty CPA reference.")
    return np.sort(values)


def _build_traces(
    frames_by_split: dict[str, list[dict[str, Any]]],
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
    selected_candidate: Candidate,
) -> dict[str, dict[str, dict[str, list[tuple[dict[str, Any], float]]]]]:
    events = {split: _event_frames(frames) for split, frames in frames_by_split.items()}
    validation = events["validation_normal"]
    simple_components = {
        split: _merged_components(
            split_events,
            _recompute_components(
                split_events,
                persistence_decay=SIMPLE.persistence_decay,
                recovery_warmup=SIMPLE.recovery_warmup,
            ),
        )
        for split, split_events in events.items()
    }
    current_components = {
        split: _merged_components(
            split_events,
            _recompute_components(
                split_events,
                persistence_decay=PAPER_CURRENT.persistence_decay,
                recovery_warmup=PAPER_CURRENT.recovery_warmup,
            ),
        )
        for split, split_events in events.items()
    }
    selected_components = {
        split: _merged_components(
            split_events,
            _recompute_components(
                split_events,
                persistence_decay=selected_candidate.persistence_decay,
                recovery_warmup=selected_candidate.recovery_warmup,
            ),
        )
        for split, split_events in events.items()
    }
    simple_calibration = _calibration(validation, simple_components["validation_normal"])
    current_calibration = _calibration(validation, current_components["validation_normal"])
    selected_calibration = _calibration(
        validation, selected_components["validation_normal"]
    )
    physical_reference = _physical_reference(validation)
    cpa_reference = _cpa_reference(validation, geometry)

    definitions = {
        "raw_residual": (simple_components, SIMPLE, False, False, False),
        "directional_damage": (simple_components, SIMPLE, False, False, False),
        "stress_damage": (simple_components, SIMPLE, False, False, False),
        "damp_no_recovery": (simple_components, SIMPLE, False, False, False),
        "damp_with_recovery": (simple_components, SIMPLE_RECOVERY, False, False, False),
        "same_pair_cpa": (simple_components, SIMPLE, True, False, False),
        "same_pair_cpa_confirmed": (simple_components, SIMPLE, True, True, False),
        "full_no_confirmation": (simple_components, SIMPLE, True, False, True),
        "full_no_stress": (simple_components, SIMPLE_NO_STRESS, True, True, True),
        "full_no_damage": (simple_components, SIMPLE_NO_DAMAGE, True, True, True),
        "full_no_persistence": (simple_components, SIMPLE_NO_PERSISTENCE, True, True, True),
        "full_with_recovery": (simple_components, SIMPLE_RECOVERY, True, True, True),
        "full_rir_simple": (simple_components, SIMPLE, True, True, True),
        "rir_nested_selected": (
            selected_components,
            selected_candidate,
            True,
            True,
            True,
        ),
        "paper_current_no_recovery": (current_components, PAPER_CURRENT_NO_RECOVERY, True, True, True),
        "paper_current_full": (current_components, PAPER_CURRENT, True, True, True),
    }
    output: dict[str, dict[str, dict[str, list[tuple[dict[str, Any], float]]]]] = {
        method: {split: {} for split in events} for method in METHOD_ORDER
    }
    for split, split_events in events.items():
        for uid, event_rows in split_events.items():
            output["ttc_only"][split][uid] = _physical_trace(event_rows, physical_reference)
            output["cpa_only"][split][uid] = _cpa_trace(event_rows, geometry, cpa_reference)
            output["severe_cpa_shortcut"][split][uid] = _cpa_trace(
                event_rows,
                geometry,
                cpa_reference,
                severe_only=True,
            )
            output["cpa_temporal_matched"][split][uid] = _cpa_trace(
                event_rows,
                geometry,
                cpa_reference,
                confirmation=True,
                severe_bypass=True,
            )
        for method, (components_by_split, candidate, alignment, confirmation, bypass) in definitions.items():
            calibration = (
                selected_calibration
                if method == "rir_nested_selected"
                else current_calibration
                if method in {"paper_current_no_recovery", "paper_current_full"}
                else simple_calibration
            )
            activation = _activation_fn(method, calibration, candidate)
            reference = _activation_reference(
                components_by_split["validation_normal"], activation
            )
            for uid, event_rows in split_events.items():
                output[method][split][uid] = _relation_trace(
                    event_rows,
                    components_by_split[split][uid],
                    activation,
                    reference,
                    geometry,
                    cpa_reference,
                    alignment=alignment,
                    confirmation=confirmation,
                    severe_bypass=bypass,
                )
    return output


def _selected_candidate(
    selection_root: Path, *, seed: int, fold: int
) -> Candidate:
    path = selection_root / "selected_parameters.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if int(row["seed"]) == seed and int(row["fold"]) == fold
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one selected DAMP row for seed={seed}, fold={fold}; "
            f"found {len(matches)} in {path}."
        )
    row = matches[0]
    return Candidate(
        persistence_decay=float(row["persistence_decay"]),
        weight_stress=float(row["weight_stress"]),
        weight_damage=float(row["weight_damage"]),
        weight_persistence=float(row["weight_persistence"]),
        recovery_beta=float(row["recovery_beta"]),
        recovery_warmup=int(row["recovery_warmup"]),
    )


def _evaluate_fold(
    *,
    seed: int,
    fold: int,
    index_root: Path,
    source_root: Path,
    selection_root: Path,
    fpr_budget: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frames = _load_frames(source_root, seed, fold)
    index_path = index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    observations = _observations_by_frame(
        index_path,
        splits=("validation_normal", "test_normal", "test_positive"),
    )
    geometry = _geometry_from_observations(
        observations, horizon_seconds=2.0, clearance_m=3.5
    )
    traces = _build_traces(
        {key: frames[key] for key in ("validation_normal", "test_normal", "test_positive")},
        geometry,
        _selected_candidate(selection_root, seed=seed, fold=fold),
    )
    ttc_flags = {
        split: {
            uid: event_alert(trace, TTC_THRESHOLD)
            for uid, trace in traces["ttc_only"][split].items()
        }
        for split in traces["ttc_only"]
    }
    validation_negative = {
        uid
        for uid, (detected, _) in ttc_flags["validation_normal"].items()
        if not detected
    }
    thresholds: dict[str, tuple[float, float]] = {
        "ttc_only": (TTC_THRESHOLD, 0.0)
    }
    for method in METHOD_ORDER[1:]:
        validation_scores = [
            event_score(traces[method]["validation_normal"][uid])
            for uid in validation_negative
        ]
        if method == "severe_cpa_shortcut":
            thresholds[method] = (
                SEVERE_CPA_PERCENTILE,
                float(
                    np.mean(
                        np.asarray(validation_scores, dtype=float)
                        >= SEVERE_CPA_PERCENTILE
                    )
                ),
            )
        else:
            thresholds[method] = threshold_at_event_fpr(
                validation_scores,
                fpr_budget,
            )

    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        threshold, fitted_fpr = thresholds[method]
        threshold_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "method": method,
                "threshold": threshold,
                "validation_incremental_fpr": fitted_fpr,
                "validation_ttc_negative_count": len(validation_negative),
            }
        )
        for split, positive in (("test_positive", 1), ("test_normal", 0)):
            for uid, trace in traces[method][split].items():
                ttc_detected, ttc_lead = ttc_flags[split][uid]
                if method == "ttc_only":
                    detected, lead = ttc_detected, ttc_lead
                    rescue = False
                    final = ttc_detected
                else:
                    detected, lead = event_alert(trace, threshold)
                    rescue = bool(not ttc_detected and detected)
                    final = bool(ttc_detected or rescue)
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "fpr_budget": fpr_budget,
                        "method": method,
                        "event_uid": uid,
                        "split": split,
                        "positive": positive,
                        "threshold": threshold,
                        "validation_incremental_fpr": fitted_fpr,
                        "method_score": event_score(trace),
                        "ttc_score": event_score(traces["ttc_only"][split][uid]),
                        "ttc_detected": int(ttc_detected),
                        "method_detected": int(detected),
                        "blind_rescue": int(rescue),
                        "final_detected": int(final),
                        "ttc_lead_seconds": "" if ttc_lead is None else ttc_lead,
                        "method_lead_seconds": "" if lead is None else lead,
                    }
                )
    return rows, threshold_rows


def _binary_metrics(rows: Sequence[dict[str, Any]]) -> tuple[float | None, float | None]:
    labels = np.asarray([int(row["positive"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["method_score"]) for row in rows], dtype=float)
    if not labels.size or len(np.unique(labels)) < 2:
        return None, None
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _seed_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        for method in METHOD_ORDER:
            selected = [
                row for row in rows
                if int(row["seed"]) == seed and row["method"] == method
            ]
            positive = [row for row in selected if int(row["positive"])]
            normal = [row for row in selected if not int(row["positive"])]
            blind_positive = [row for row in positive if not int(row["ttc_detected"])]
            blind_normal = [row for row in normal if not int(row["ttc_detected"])]
            continuous = [row for row in selected if not int(row["ttc_detected"])]
            auroc, ap = _binary_metrics(continuous)
            leads = [
                float(row["method_lead_seconds"])
                for row in blind_positive
                if int(row["blind_rescue"]) and row["method_lead_seconds"] != ""
            ]
            output.append(
                {
                    "seed": seed,
                    "method": method,
                    "positive_event_count": len(positive),
                    "overall_recall": sum(int(row["final_detected"]) for row in positive) / max(len(positive), 1),
                    "ttc_blind_positive_count": len(blind_positive),
                    "blind_rescue_recall": sum(int(row["blind_rescue"]) for row in blind_positive) / max(len(blind_positive), 1),
                    "hard_safe_event_count": len(normal),
                    "overall_hard_safe_fpr": sum(int(row["final_detected"]) for row in normal) / max(len(normal), 1),
                    "ttc_negative_hard_safe_count": len(blind_normal),
                    "incremental_rescue_fpr": sum(int(row["blind_rescue"]) for row in blind_normal) / max(len(blind_normal), 1),
                    "blind_auroc": auroc,
                    "blind_average_precision": ap,
                    "mean_rescue_lead_seconds": None if not leads else float(np.mean(leads)),
                    "median_rescue_lead_seconds": None if not leads else float(np.median(leads)),
                    "rescue_lead_q25_seconds": None if not leads else float(np.quantile(leads, 0.25)),
                    "rescue_lead_q75_seconds": None if not leads else float(np.quantile(leads, 0.75)),
                }
            )
    return output


def _aggregate(seed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "overall_recall",
        "blind_rescue_recall",
        "incremental_rescue_fpr",
        "overall_hard_safe_fpr",
        "blind_auroc",
        "blind_average_precision",
        "mean_rescue_lead_seconds",
        "median_rescue_lead_seconds",
    )
    output = []
    for method in METHOD_ORDER:
        selected = [row for row in seed_rows if row["method"] == method]
        result: dict[str, Any] = {"method": method, "seed_count": len(selected)}
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in selected if row.get(metric) not in (None, "")],
                dtype=float,
            )
            result[f"{metric}_mean"] = None if not values.size else float(np.mean(values))
            result[f"{metric}_std"] = None if not values.size else (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        output.append(result)
    return output


def _paired_family_bootstrap(
    rows: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    output = []
    rng = np.random.default_rng(20260823)
    reference = "rir_nested_selected"
    for comparator in METHOD_ORDER:
        if comparator == reference:
            continue
        for metric, population, field in (
            ("overall_recall_delta", "positive", "final_detected"),
            ("blind_rescue_delta", "blind_positive", "blind_rescue"),
            ("incremental_fpr_delta", "blind_normal", "blind_rescue"),
        ):
            event_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            for row in rows:
                positive = bool(int(row["positive"]))
                blind = not bool(int(row["ttc_detected"]))
                include = (
                    (population == "positive" and positive)
                    or (population == "blind_positive" and positive and blind)
                    or (population == "blind_normal" and not positive and blind)
                )
                if include and row["method"] in (reference, comparator):
                    event_values[(str(row["method"]), str(row["scenario_group_id"]), str(row["event_uid"]))].append(float(row[field]))
            paired: dict[str, list[float]] = defaultdict(list)
            keys = {
                (family, uid)
                for method, family, uid in event_values
                if method == reference
            }
            for family, uid in keys:
                left = event_values.get((reference, family, uid))
                right = event_values.get((comparator, family, uid))
                if left and right:
                    paired[family].append(float(np.mean(left) - np.mean(right)))
            families = sorted(paired)
            values = [value for family in families for value in paired[family]]
            draws = np.empty(replicates, dtype=float)
            for index in range(replicates):
                sampled = rng.choice(families, size=len(families), replace=True)
                draws[index] = float(np.mean([value for family in sampled for value in paired[family]]))
            lower_tail = (float(np.sum(draws <= 0.0)) + 1.0) / (replicates + 1.0)
            upper_tail = (float(np.sum(draws >= 0.0)) + 1.0) / (replicates + 1.0)
            output.append(
                {
                    "reference": reference,
                    "comparator": comparator,
                    "metric": metric,
                    "paired_family_count": len(families),
                    "paired_event_count": len(values),
                    "estimate_reference_minus_comparator": float(np.mean(values)),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "two_sided_bootstrap_p": min(1.0, 2.0 * min(lower_tail, upper_tail)),
                    "bootstrap_replicates": replicates,
                }
            )
    return output


def _report(
    aggregate: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    fpr_budget: float,
) -> str:
    lines = [
        "# DAMP decision-mechanism necessity ablation",
        "",
        f"Every learned row is independently calibrated at a {fpr_budget:.0%} incremental false-rescue budget using validation-normal TTC-negative events only.",
        "The forecast checkpoint and outer events are identical across rows.",
        "",
        "| Method | Recall | Blind rescue | Incremental FPR | Blind AP | Mean rescue lead |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lead = row["mean_rescue_lead_seconds_mean"]
        ap = row["blind_average_precision_mean"]
        ap_text = "-" if ap is None else f"{float(ap):.3f}"
        lead_text = "-" if lead is None else f"{float(lead):.3f} s"
        lines.append(
            f"| `{row['method']}` | {float(row['overall_recall_mean']):.2%} | "
            f"{float(row['blind_rescue_recall_mean']):.2%} | "
            f"{float(row['incremental_rescue_fpr_mean']):.2%} | "
            f"{ap_text} | {lead_text} |"
        )
    lines.extend([
        "",
        "Paired family-bootstrap comparisons use `rir_nested_selected` as the reference and average repeated-seed predictions within each event before resampling families.",
        "See `paired_family_bootstrap.csv` for confidence intervals and p-values.",
        "",
        "This is post-hoc mechanistic evidence on frozen outer predictions, not a newly sealed external holdout.",
    ])
    return "\n".join(lines) + "\n"


def run(
    *,
    index_root: Path = DEFAULT_INDEX_ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    selection_root: Path = DEFAULT_SELECTION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    folds: Sequence[int] = tuple(range(5)),
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    fpr_budget: float = DEFAULT_FPR_BUDGET,
) -> dict[str, Any]:
    index_root = index_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    selection_root = selection_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection_summary = selection_root / "selection_summary.json"
    if not selection_summary.exists():
        raise FileNotFoundError("Run hard-safe DAMP selection before this ablation.")
    selection = json.loads(selection_summary.read_text(encoding="utf-8"))
    if int(selection.get("positive_or_outer_test_metrics_used", -1)) != 0:
        raise RuntimeError("Parameter-selection firewall is not clean.")

    rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    inputs = []
    for seed in seeds:
        for fold in folds:
            fold_rows, fold_thresholds = _evaluate_fold(
                seed=int(seed),
                fold=int(fold),
                index_root=index_root,
                source_root=source_root,
                selection_root=selection_root,
                fpr_budget=fpr_budget,
            )
            rows.extend(fold_rows)
            thresholds.extend(fold_thresholds)
            cache = source_root / f"seed_{seed}" / "pair_frame_cache" / "graph_free_one_class" / f"fold_{fold:02d}.pkl"
            inputs.append({
                "seed": int(seed),
                "fold": int(fold),
                "cache_sha256": sha256_file(cache),
                "index_sha256": sha256_file(index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"),
            })
            print(json.dumps({"seed": seed, "fold": fold, "status": "complete"}), flush=True)

    _attach_event_metadata(rows, index_root)
    _attach_family_metadata(rows, index_root)
    seed_summary = _seed_summary(rows)
    aggregate = _aggregate(seed_summary)
    paired = _paired_family_bootstrap(rows, bootstrap_replicates)
    _write_csv(output_root / "outer_event_predictions.csv", rows)
    _write_csv(output_root / "fold_thresholds.csv", thresholds)
    _write_csv(output_root / "seed_metrics.csv", seed_summary)
    _write_csv(output_root / "main_ablation_table.csv", aggregate)
    _write_csv(output_root / "paired_family_bootstrap.csv", paired)
    report_path = output_root / "NECESSITY_ABLATION_REPORT.md"
    report_path.write_text(_report(aggregate, paired, fpr_budget), encoding="utf-8")

    result = {
        "protocol": "damp-decision-mechanism-necessity-multiseed-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "claim_status": "posthoc_mechanistic_evaluation_on_frozen_outer_predictions",
        "seeds": [int(value) for value in seeds],
        "folds": [int(value) for value in folds],
        "fpr_budget": fpr_budget,
        "positive_labels_used_for_parameter_or_threshold_selection": 0,
        "positive_labels_used_for_outer_metrics": "yes_after_threshold_freeze",
        "hard_safe_selection_summary_sha256": sha256_file(selection_summary),
        "parameterizations": {
            "parsimonious": SIMPLE.__dict__,
            "recovery_sensitivity": SIMPLE_RECOVERY.__dict__,
            "no_stress": SIMPLE_NO_STRESS.__dict__,
            "no_damage": SIMPLE_NO_DAMAGE.__dict__,
            "no_persistence": SIMPLE_NO_PERSISTENCE.__dict__,
            "paper_current": PAPER_CURRENT.__dict__,
            "paper_current_no_recovery": PAPER_CURRENT_NO_RECOVERY.__dict__,
            "nested_selected": "seed/fold-specific hard-safe-only selection",
        },
        "method_order": list(METHOD_ORDER),
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "input_hashes": inputs,
        "artifacts": {
            "events": str(output_root / "outer_event_predictions.csv"),
            "thresholds": str(output_root / "fold_thresholds.csv"),
            "seed_metrics": str(output_root / "seed_metrics.csv"),
            "table": str(output_root / "main_ablation_table.csv"),
            "paired": str(output_root / "paired_family_bootstrap.csv"),
            "report": str(report_path),
        },
    }
    result["result_sha256"] = sha256_payload(result)
    summary_path = output_root / "necessity_ablation_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run(
        index_root=args.index_root,
        source_root=args.source_root,
        selection_root=args.selection_root,
        output_root=args.output_root,
        seeds=_parse_ints(args.seeds),
        folds=_parse_ints(args.folds),
        bootstrap_replicates=args.bootstrap_replicates,
        fpr_budget=args.fpr_budget,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
