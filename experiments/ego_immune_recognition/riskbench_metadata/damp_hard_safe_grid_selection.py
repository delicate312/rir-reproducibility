"""Select Pair-DAMP parameters using hard-safe validation events only.

This selector is deliberately separate from outer-test evaluation.  It uses
family-balanced cross-calibration inside each fold's ``validation_normal``
split and never uses validation-positive or outer-test outcomes in its
selection objective.  The selected parameters and input hashes are written as
a receipt before any parameter-selected outer evaluation is allowed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .core_hybrid_evaluation import _usable_frames, threshold_at_event_fpr
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT
from .pair_aligned_damp_evaluation import (
    PERCENTILE_WARMUP_FRAMES,
    _geometry_from_observations,
    _normalized_component,
    _observations_by_frame,
    _percentile,
)
from .pair_damp_freeze import PROJECT_ROOT, sha256_file, sha256_payload


DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_normal_only_multiseed_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_damp_hard_safe_grid_v1"
)
DEFAULT_SEEDS = (20260812, 20260813, 20260814)
PRIMARY_FPR_BUDGET = 0.05
INNER_SUBFOLDS = 5
SEVERE_CPA_PERCENTILE = 0.999
PAIR_SUPPORT_FRAMES = 3

PERSISTENCE_DECAYS = (0.50, 0.80, 0.90)
WEIGHT_CANDIDATES = (
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.25, 0.35, 0.40),
    (0.20, 0.50, 0.30),
    (0.20, 0.30, 0.50),
    (0.50, 0.25, 0.25),
    (0.00, 0.50, 0.50),
)
RECOVERY_CONFIGS = (
    (0.00, 0),
    (0.10, 0),
    (0.10, 3),
    (0.20, 0),
    (0.20, 3),
)


@dataclass(frozen=True)
class Candidate:
    persistence_decay: float
    weight_stress: float
    weight_damage: float
    weight_persistence: float
    recovery_beta: float
    recovery_warmup: int

    @property
    def candidate_id(self) -> str:
        return (
            f"lp{self.persistence_decay:.2f}_"
            f"w{self.weight_stress:.3f}-{self.weight_damage:.3f}-"
            f"{self.weight_persistence:.3f}_"
            f"bu{self.recovery_beta:.2f}_mu{self.recovery_warmup}"
        )

    @property
    def uniform_distance(self) -> float:
        return float(
            sum(
                abs(value - 1.0 / 3.0)
                for value in (
                    self.weight_stress,
                    self.weight_damage,
                    self.weight_persistence,
                )
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--inner-subfolds", type=int, default=INNER_SUBFOLDS)
    parser.add_argument("--fpr-budget", type=float, default=PRIMARY_FPR_BUDGET)
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


def candidate_grid() -> list[Candidate]:
    candidates = []
    for decay, weights, recovery in itertools.product(
        PERSISTENCE_DECAYS,
        WEIGHT_CANDIDATES,
        RECOVERY_CONFIGS,
    ):
        candidates.append(
            Candidate(
                persistence_decay=float(decay),
                weight_stress=float(weights[0]),
                weight_damage=float(weights[1]),
                weight_persistence=float(weights[2]),
                recovery_beta=float(recovery[0]),
                recovery_warmup=int(recovery[1]),
            )
        )
    return candidates


def _event_frames(frames: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        events.setdefault(str(frame["event_uid"]), []).append(frame)
    for rows in events.values():
        rows.sort(key=lambda item: int(item["frame_position"]))
    return events


def _validation_frames(cache_path: Path) -> list[dict[str, Any]]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    if "validation_normal" not in payload:
        raise KeyError(f"validation_normal missing from {cache_path}")
    # Detach the only split the selector is permitted to inspect.
    return copy.deepcopy(payload["validation_normal"])


def _event_family_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    output = {}
    for record in payload["splits"]["validation_normal"]:
        uid = f"riskbench:{record['category']}:{record['scenario_id']}"
        output[uid] = str(record["scenario_group_id"])
    return output


def _balanced_subfolds(
    event_uids: Iterable[str],
    family_by_uid: dict[str, str],
    *,
    seed: int,
    fold: int,
    count: int,
) -> list[list[str]]:
    if count < 2:
        raise ValueError("At least two inner subfolds are required.")
    missing = sorted(set(event_uids) - set(family_by_uid))
    if missing:
        raise KeyError(f"Missing family identifiers for {missing[:3]}")

    def rank(uid: str) -> str:
        family = family_by_uid[uid]
        return hashlib.sha256(
            f"{seed}:{fold}:{family}".encode("utf-8")
        ).hexdigest()

    ordered = sorted(event_uids, key=rank)
    subfolds = [[] for _ in range(count)]
    for index, uid in enumerate(ordered):
        subfolds[index % count].append(uid)
    if any(not values for values in subfolds):
        raise ValueError("Inner family subfold assignment produced an empty fold.")
    return subfolds


def _recompute_components(
    events: dict[str, list[dict[str, Any]]],
    *,
    persistence_decay: float,
    recovery_warmup: int,
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    output: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    for uid, frames in events.items():
        previous: dict[str, tuple[float, float, float]] = {}
        counts: Counter[str] = Counter()
        event_output: dict[int, dict[str, dict[str, float]]] = {}
        for frame in frames:
            frame_output: dict[str, dict[str, float]] = {}
            for edge_key, values in frame.get("relation_components", {}).items():
                if not int(values.get("ego_outgoing", 0)):
                    continue
                relation_id = str(edge_key)
                target_id = str(values["target_id"])
                residual = float(values["causal_raw"])
                damage = float(values["damage_raw"])
                prior = previous.get(relation_id)
                if prior is None:
                    stress = 0.0
                    persistence = (1.0 - persistence_decay) * damage
                    recovery = 0.0
                else:
                    prior_residual, prior_damage, prior_persistence = prior
                    stress = max(residual - prior_residual, 0.0)
                    persistence = (
                        persistence_decay * prior_persistence
                        + (1.0 - persistence_decay) * (stress + damage)
                    )
                    recovery = 0.5 * (
                        max(prior_residual - residual, 0.0)
                        + max(prior_damage - damage, 0.0)
                    )
                    if counts[relation_id] < recovery_warmup:
                        recovery = 0.0
                previous[relation_id] = (residual, damage, persistence)
                counts[relation_id] += 1
                frame_output[target_id] = {
                    "stress_raw": stress,
                    "damage_raw": damage,
                    "persistence_raw": persistence,
                    "recovery_raw": recovery,
                }
            event_output[int(frame["frame_position"])] = frame_output
        output[uid] = event_output
    return output


def _verify_cached_reproduction(
    events: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    recomputed = _recompute_components(
        events,
        persistence_decay=0.80,
        recovery_warmup=3,
    )
    maxima = {
        "stress_raw": 0.0,
        "damage_raw": 0.0,
        "persistence_raw": 0.0,
        "recovery_raw": 0.0,
    }
    comparisons = 0
    for uid, frames in events.items():
        for frame in frames:
            position = int(frame["frame_position"])
            by_target = recomputed[uid][position]
            for values in frame.get("relation_components", {}).values():
                if not int(values.get("ego_outgoing", 0)):
                    continue
                target_id = str(values["target_id"])
                if target_id not in by_target:
                    raise KeyError(f"Recomputed relation missing: {uid}/{target_id}")
                for key in maxima:
                    maxima[key] = max(
                        maxima[key],
                        abs(float(values[key]) - float(by_target[target_id][key])),
                    )
                comparisons += 1
    if comparisons == 0:
        raise ValueError("No outgoing relations were available for reproduction.")
    if max(maxima.values()) > 2e-5:
        raise ValueError(f"Cached component reproduction failed: {maxima}")
    return {"comparison_count": comparisons, **maxima}


def _raw_values(
    components: dict[str, dict[int, dict[str, dict[str, float]]]],
    event_uids: set[str],
) -> dict[str, list[float]]:
    output = {
        "stress_raw": [],
        "damage_raw": [],
        "persistence_raw": [],
        "recovery_raw": [],
    }
    for uid in event_uids:
        for relations in components[uid].values():
            for values in relations.values():
                for key in output:
                    output[key].append(float(values[key]))
    return output


def _calibration_from_components(
    components: dict[str, dict[int, dict[str, dict[str, float]]]],
    event_uids: set[str],
) -> dict[str, dict[str, float]]:
    calibration = {}
    for key, rows in _raw_values(components, event_uids).items():
        values = np.asarray(rows, dtype=float)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        upper_scale = float(np.quantile(values, 0.90) - center)
        calibration[key] = {
            "center": center,
            "scale": max(1.4826 * mad, upper_scale, 1e-3),
        }
    return calibration


def _activation(
    values: dict[str, float],
    calibration: dict[str, dict[str, float]],
    candidate: Candidate,
) -> float:
    normalized = {
        key.removesuffix("_raw"): _normalized_component(
            float(values[key]), calibration[key]
        )
        for key in calibration
    }
    positive = (
        candidate.weight_stress * normalized["stress"]
        + candidate.weight_damage * normalized["damage"]
        + candidate.weight_persistence * normalized["persistence"]
    )
    return max(positive - candidate.recovery_beta * normalized["recovery"], 0.0)


def _references(
    *,
    train_uids: set[str],
    events: dict[str, list[dict[str, Any]]],
    components: dict[str, dict[int, dict[str, dict[str, float]]]],
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
    candidate: Candidate,
) -> dict[str, Any]:
    calibration = _calibration_from_components(components, train_uids)
    activations = []
    cpa_values = []
    physical_values = []
    for uid in train_uids:
        for frame in events[uid]:
            position = int(frame["frame_position"])
            for values in components[uid][position].values():
                activations.append(_activation(values, calibration, candidate))
            cpa_values.extend(
                float(values["raw"])
                for values in geometry.get((uid, position), {}).values()
            )
            if position >= PERCENTILE_WARMUP_FRAMES:
                physical_values.append(
                    float(frame["aggregations"]["inverse_ttc_ego_max"])
                )
    if not activations or not cpa_values or not physical_values:
        raise ValueError("A calibration reference is empty.")
    return {
        "component_calibration": calibration,
        "damp": np.sort(np.asarray(activations, dtype=float)),
        "cpa": np.sort(np.asarray(cpa_values, dtype=float)),
        "physical": np.sort(np.asarray(physical_values, dtype=float)),
    }


def _event_scores(
    *,
    event_uids: set[str],
    events: dict[str, list[dict[str, Any]]],
    components: dict[str, dict[int, dict[str, dict[str, float]]]],
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
    references: dict[str, Any],
    candidate: Candidate,
    allow_severe_cpa_bypass: bool,
) -> dict[str, dict[str, float | bool]]:
    output = {}
    for uid in event_uids:
        histories: dict[str, list[tuple[int, float]]] = {}
        best_relation = 0.0
        best_physical = 0.0
        for frame in _usable_frames(events[uid]):
            position = int(frame["frame_position"])
            physical = _percentile(
                references["physical"],
                float(frame["aggregations"]["inverse_ttc_ego_max"]),
            )
            best_physical = max(best_physical, physical)
            by_target = components[uid][position]
            for target_id, geometry_values in geometry.get(
                (uid, position), {}
            ).items():
                component = by_target.get(str(target_id))
                if component is None:
                    continue
                activation = _activation(
                    component,
                    references["component_calibration"],
                    candidate,
                )
                damp = _percentile(references["damp"], activation)
                cpa = _percentile(
                    references["cpa"], float(geometry_values["raw"])
                )
                aligned = math.sqrt(max(cpa * damp, 0.0))
                history = histories.setdefault(str(target_id), [])
                if history and position != history[-1][0] + 1:
                    history.clear()
                history.append((position, aligned))
                history[:] = history[-PAIR_SUPPORT_FRAMES:]
                if allow_severe_cpa_bypass and cpa >= SEVERE_CPA_PERCENTILE:
                    score = cpa
                elif len(history) == PAIR_SUPPORT_FRAMES:
                    score = float(np.mean([value for _, value in history]))
                else:
                    score = 0.0
                best_relation = max(best_relation, score)
        output[uid] = {
            "ttc_detected": bool(best_physical >= 0.95),
            "ttc_score": best_physical,
            "relation_score": best_relation,
        }
    return output


def _cross_calibrate_candidate(
    *,
    candidate: Candidate,
    events: dict[str, list[dict[str, Any]]],
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
    subfolds: Sequence[Sequence[str]],
    target_fpr: float,
) -> dict[str, Any]:
    components = _recompute_components(
        events,
        persistence_decay=candidate.persistence_decay,
        recovery_warmup=candidate.recovery_warmup,
    )
    all_uids = set(events)
    rates = []
    denominators = []
    false_rescue_counts = []
    thresholds = []
    train_rates = []
    for heldout_values in subfolds:
        heldout = set(heldout_values)
        train = all_uids - heldout
        references = _references(
            train_uids=train,
            events=events,
            components=components,
            geometry=geometry,
            candidate=candidate,
        )
        train_scores = _event_scores(
            event_uids=train,
            events=events,
            components=components,
            geometry=geometry,
            references=references,
            candidate=candidate,
            allow_severe_cpa_bypass=False,
        )
        train_negative = [
            float(values["relation_score"])
            for values in train_scores.values()
            if not bool(values["ttc_detected"])
        ]
        threshold, achieved = threshold_at_event_fpr(train_negative, target_fpr)
        heldout_scores = _event_scores(
            event_uids=heldout,
            events=events,
            components=components,
            geometry=geometry,
            references=references,
            candidate=candidate,
            allow_severe_cpa_bypass=False,
        )
        heldout_negative = [
            values
            for values in heldout_scores.values()
            if not bool(values["ttc_detected"])
        ]
        false_rescues = sum(
            float(values["relation_score"]) >= threshold
            for values in heldout_negative
        )
        rate = false_rescues / max(len(heldout_negative), 1)
        rates.append(float(rate))
        denominators.append(len(heldout_negative))
        false_rescue_counts.append(int(false_rescues))
        thresholds.append(float(threshold))
        train_rates.append(float(achieved))
    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    pooled_rate = sum(false_rescue_counts) / max(sum(denominators), 1)
    calibration_error = abs(pooled_rate - target_fpr)
    return {
        **asdict(candidate),
        "candidate_id": candidate.candidate_id,
        "heldout_fpr_mean": mean_rate,
        "heldout_fpr_std": std_rate,
        "heldout_fpr_pooled": pooled_rate,
        "calibration_error": calibration_error,
        "selection_objective": calibration_error,
        "within_budget": int(pooled_rate <= target_fpr + 1e-12),
        "subfold_fprs": json.dumps(rates),
        "subfold_false_rescue_counts": json.dumps(false_rescue_counts),
        "subfold_ttc_negative_counts": json.dumps(denominators),
        "subfold_thresholds": json.dumps(thresholds),
        "subfold_train_achieved_fprs": json.dumps(train_rates),
        "uniform_weight_distance": candidate.uniform_distance,
    }


def _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["calibration_error"]),
        float(row["heldout_fpr_std"]),
        float(row["persistence_decay"]),
        float(row["uniform_weight_distance"]),
        int(float(row["recovery_beta"]) > 0.0),
        float(row["recovery_beta"]),
        int(row["recovery_warmup"]),
        str(row["candidate_id"]),
    )


def select_fold(
    *,
    seed: int,
    fold: int,
    index_root: Path,
    source_root: Path,
    output_root: Path,
    inner_subfolds: int,
    target_fpr: float,
) -> dict[str, Any]:
    index_path = index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    cache_path = (
        source_root
        / f"seed_{seed}"
        / "pair_frame_cache"
        / "graph_free_one_class"
        / f"fold_{fold:02d}.pkl"
    )
    checkpoint_path = source_root / f"seed_{seed}" / f"fold_{fold:02d}" / "best_guarded_checkpoint.pt"
    frames = _validation_frames(cache_path)
    events = _event_frames(frames)
    family_by_uid = _event_family_map(index_path)
    subfolds = _balanced_subfolds(
        events,
        family_by_uid,
        seed=seed,
        fold=fold,
        count=inner_subfolds,
    )
    observations = _observations_by_frame(index_path, splits=("validation_normal",))
    geometry = _geometry_from_observations(
        observations,
        horizon_seconds=2.0,
        clearance_m=3.5,
    )
    reproduction = _verify_cached_reproduction(events)

    rows = []
    for index, candidate in enumerate(candidate_grid(), start=1):
        row = _cross_calibrate_candidate(
            candidate=candidate,
            events=events,
            geometry=geometry,
            subfolds=subfolds,
            target_fpr=target_fpr,
        )
        rows.append(row)
        if index % 15 == 0:
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "fold": fold,
                        "candidate": index,
                        "candidate_count": len(candidate_grid()),
                    }
                ),
                flush=True,
            )
    eligible = [row for row in rows if int(row["within_budget"])]
    selected = min(eligible or rows, key=_selection_key)
    selected_candidate = Candidate(
        persistence_decay=float(selected["persistence_decay"]),
        weight_stress=float(selected["weight_stress"]),
        weight_damage=float(selected["weight_damage"]),
        weight_persistence=float(selected["weight_persistence"]),
        recovery_beta=float(selected["recovery_beta"]),
        recovery_warmup=int(selected["recovery_warmup"]),
    )
    full_components = _recompute_components(
        events,
        persistence_decay=selected_candidate.persistence_decay,
        recovery_warmup=selected_candidate.recovery_warmup,
    )
    references = _references(
        train_uids=set(events),
        events=events,
        components=full_components,
        geometry=geometry,
        candidate=selected_candidate,
    )
    full_scores = _event_scores(
        event_uids=set(events),
        events=events,
        components=full_components,
        geometry=geometry,
        references=references,
        candidate=selected_candidate,
        allow_severe_cpa_bypass=True,
    )
    ttc_negative_scores = [
        float(values["relation_score"])
        for values in full_scores.values()
        if not bool(values["ttc_detected"])
    ]
    threshold, achieved = threshold_at_event_fpr(ttc_negative_scores, target_fpr)

    fold_root = output_root / f"seed_{seed}" / f"fold_{fold:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)
    _write_csv(fold_root / "candidate_cross_calibration.csv", rows)
    receipt = {
        "protocol": "damp-hard-safe-only-grid-selection-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replicate_seed": seed,
        "outer_fold": fold,
        "selection_split": "validation_normal_only",
        "positive_or_outer_test_metrics_used": 0,
        "target_incremental_fpr": target_fpr,
        "inner_family_subfold_count": inner_subfolds,
        "validation_event_count": len(events),
        "family_count": len(set(family_by_uid.values())),
        "candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "selection_rule": (
            "cross-calibrate the moderate-CPA learned branch with the extreme-CPA "
            "bypass disabled because DAMP parameters cannot affect that bypass; "
            "minimize the absolute pooled held-out false-rescue-rate error, "
            "constrained to the pooled budget; ties minimize the across-subfold "
            "sample SD and then prefer shorter memory, weights nearer uniform, "
            "and no recovery"
        ),
        "selected": selected,
        "full_validation_refit": {
            "threshold": threshold,
            "achieved_incremental_fpr": achieved,
            "ttc_negative_event_count": len(ttc_negative_scores),
        },
        "component_reproduction_audit": reproduction,
        "input_hashes": {
            "index": sha256_file(index_path),
            "validation_container": sha256_file(cache_path),
            "checkpoint": sha256_file(checkpoint_path),
        },
        "inspected_splits": ["validation_normal"],
        "status": "selected_before_parameterized_outer_evaluation",
    }
    receipt["receipt_sha256"] = sha256_payload(receipt)
    receipt_path = fold_root / "selection_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def run(
    *,
    seeds: Sequence[int],
    folds: Sequence[int],
    index_root: Path,
    source_root: Path,
    output_root: Path,
    inner_subfolds: int,
    target_fpr: float,
) -> dict[str, Any]:
    index_root = index_root.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    receipts = []
    for seed in seeds:
        for fold in folds:
            receipts.append(
                select_fold(
                    seed=seed,
                    fold=fold,
                    index_root=index_root,
                    source_root=source_root,
                    output_root=output_root,
                    inner_subfolds=inner_subfolds,
                    target_fpr=target_fpr,
                )
            )
    selected_rows = [
        {
            "seed": int(receipt["replicate_seed"]),
            "fold": int(receipt["outer_fold"]),
            **receipt["selected"],
            **{
                f"refit_{key}": value
                for key, value in receipt["full_validation_refit"].items()
            },
        }
        for receipt in receipts
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "selected_parameters.csv", selected_rows)
    selection_counts = Counter(row["candidate_id"] for row in selected_rows)
    result = {
        "protocol": "damp-hard-safe-only-grid-selection-multiseed-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [int(value) for value in seeds],
        "folds": [int(value) for value in folds],
        "selection_count": len(selected_rows),
        "candidate_grid": [asdict(candidate) for candidate in candidate_grid()],
        "selected_candidate_counts": dict(selection_counts),
        "positive_or_outer_test_metrics_used": 0,
        "status": "all_fold_parameters_selected_and_receipted",
    }
    result["summary_sha256"] = sha256_payload(result)
    summary_path = output_root / "selection_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run(
        seeds=_parse_ints(args.seeds),
        folds=_parse_ints(args.folds),
        index_root=args.index_root,
        source_root=args.source_root,
        output_root=args.output_root,
        inner_subfolds=args.inner_subfolds,
        target_fpr=args.fpr_budget,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
