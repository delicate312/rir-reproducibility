"""Evaluate a same-relation CPA rescue channel for TTC-blind danger.

This is an exploratory, no-retraining ablation. It preserves the frozen TTC
decision and only allows a relation episode to rescue frames before TTC has
alerted. Configuration selection is constrained by the event false-positive
rate among TTC-negative hard-safe events.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..dataset_io import load_graph_dataset
from ..schema import EDGE_FEATURE_NAMES
from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .complete_structure_ablation import DEFAULT_CACHE_ROOT
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT, WINDOW_FRAMES
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_relation_geometry_rescue_v1"
)
HORIZONS_SECONDS = (1.5, 2.0, 3.0)
GENERIC_CLEARANCES_M = (1.5, 2.5, 3.5)
SIGNALS = (
    "cpa_only",
    "cpa_relation_geomean",
    "cpa_immune_geomean",
)
THRESHOLD_QUANTILES = (
    0.80,
    0.85,
    0.875,
    0.90,
    0.925,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
)
SUPPORT_RULES = ((1, 1), (2, 3), (2, 5), (3, 5), (3, 7))
MINIMUM_RISES = (0.0, 0.02)
DEFAULT_RESCUE_EVENT_FPR_BUDGET = 0.10


@dataclass(frozen=True)
class GeometryGateConfig:
    horizon_seconds: float
    clearance_m: float
    signal: str
    threshold_quantile: float
    score_threshold: float
    support_frames: int
    window_frames: int
    minimum_rise: float
    max_recovery_drop: float = math.inf
    minimum_current_delta: float = -math.inf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--rescue-event-fpr-budget",
        type=float,
        default=DEFAULT_RESCUE_EVENT_FPR_BUDGET,
    )
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Aggregate existing fold summaries without recomputing geometry.",
    )
    return parser.parse_args()


def closest_approach(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    horizon_seconds: float,
) -> tuple[float, float, bool]:
    """Return TCPA, DCPA, and whether the closest approach is in the future."""
    position = np.asarray(relative_position, dtype=float)
    velocity = np.asarray(relative_velocity, dtype=float)
    speed_sq = float(np.dot(velocity, velocity))
    if speed_sq <= 1e-8:
        return float(horizon_seconds), float(np.linalg.norm(position)), False
    raw_tcpa = -float(np.dot(position, velocity)) / speed_sq
    future = 0.0 <= raw_tcpa <= horizon_seconds
    tcpa = float(np.clip(raw_tcpa, 0.0, horizon_seconds))
    dcpa = float(np.linalg.norm(position + tcpa * velocity))
    return tcpa, dcpa, future


def cpa_risk(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    horizon_seconds: float,
    clearance_m: float,
) -> dict[str, float]:
    """Continuous collision-course evidence without actor-class metadata."""
    tcpa, dcpa, future = closest_approach(
        relative_position,
        relative_velocity,
        horizon_seconds=horizon_seconds,
    )
    distance = max(float(np.linalg.norm(relative_position)), 1e-6)
    cross = abs(
        float(
            relative_position[0] * relative_velocity[1]
            - relative_position[1] * relative_velocity[0]
        )
    )
    bearing_rate = cross / (distance * distance)
    tangential_speed = cross / distance
    if not future:
        raw = 0.0
    else:
        clearance_scale = max(float(clearance_m), 0.5)
        clearance_support = math.exp(
            -max(dcpa - clearance_m, 0.0) / clearance_scale
        )
        urgency = math.exp(-tcpa / max(horizon_seconds, 1e-6))
        raw = clearance_support * urgency
    return {
        "raw": float(raw),
        "tcpa_seconds": tcpa,
        "dcpa_m": dcpa,
        "bearing_rate_radps": float(bearing_rate),
        "tangential_speed_mps": float(tangential_speed),
        "future_cpa": float(future),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fold_frames(cache_root: Path, fold: int) -> dict[str, list[dict[str, Any]]]:
    path = (
        cache_root
        / "full_gat_one_class"
        / f"fold_{fold:02d}.pkl"
    )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _outgoing_ego_relations(graph: Any) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    edge_attr = getattr(graph, "raw_edge_attr", graph.edge_attr).detach().cpu()
    edge_index = graph.edge_index.detach().cpu()
    node_ids = [str(value) for value in graph.node_ids]
    ego_id = str(graph.ego_agent_id)
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for edge_position, (source, target) in enumerate(edge_index.t().tolist()):
        if node_ids[int(source)] != ego_id:
            continue
        target_id = node_ids[int(target)]
        values = edge_attr[edge_position]
        position = np.asarray(
            [
                float(values[EDGE_FEATURE_NAMES.index("relative_x_m")]),
                float(values[EDGE_FEATURE_NAMES.index("relative_y_m")]),
            ],
            dtype=float,
        )
        velocity = np.asarray(
            [
                float(values[EDGE_FEATURE_NAMES.index("relative_vx_mps")]),
                float(values[EDGE_FEATURE_NAMES.index("relative_vy_mps")]),
            ],
            dtype=float,
        )
        output[target_id] = (position, velocity)
    return output


def _observations_by_frame(
    index_path: Path,
    *,
    splits: Sequence[str],
) -> dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]]:
    package = load_graph_dataset(index_path)
    output: dict[
        tuple[str, int],
        dict[str, tuple[np.ndarray, np.ndarray]],
    ] = {}
    for split in splits:
        for sequence in package["splits"][split]:
            for graph in sequence:
                key = (
                    str(graph.event_uid),
                    int(graph.frame_position.item()),
                )
                output[key] = _outgoing_ego_relations(graph)
    return output


def _geometry_from_observations(
    observations: dict[
        tuple[str, int],
        dict[str, tuple[np.ndarray, np.ndarray]],
    ],
    *,
    horizon_seconds: float,
    clearance_m: float,
) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    return {
        key: {
            relation_id: cpa_risk(
                position,
                velocity,
                horizon_seconds=horizon_seconds,
                clearance_m=clearance_m,
            )
            for relation_id, (position, velocity) in relations.items()
        }
        for key, relations in observations.items()
    }


def _percentile(reference: np.ndarray, value: float) -> float:
    if reference.size == 0:
        return 0.0
    return float(np.searchsorted(reference, value, side="right") / reference.size)


def _attach_relation_signals(
    frames: dict[str, list[dict[str, Any]]],
    *,
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
) -> None:
    reference = np.sort(
        np.asarray(
            [
                values["raw"]
                for frame in frames["validation_normal"]
                for values in geometry.get(
                    (str(frame["event_uid"]), int(frame["frame_position"])),
                    {},
                ).values()
            ],
            dtype=float,
        )
    )
    for split_frames in frames.values():
        for frame in split_frames:
            relation_context = float(
                frame["aggregations"]["relation_percentile"]
            )
            immune_context = float(frame["aggregations"]["immune_rank_raw"])
            attached = {}
            for relation_id, values in geometry.get(
                (str(frame["event_uid"]), int(frame["frame_position"])),
                {},
            ).items():
                cpa_percentile = _percentile(reference, float(values["raw"]))
                attached[relation_id] = {
                    **values,
                    "cpa_percentile": cpa_percentile,
                    "cpa_only": cpa_percentile,
                    "cpa_relation_geomean": math.sqrt(
                        max(cpa_percentile * relation_context, 0.0)
                    ),
                    "cpa_immune_geomean": math.sqrt(
                        max(cpa_percentile * immune_context, 0.0)
                    ),
                }
            frame["geometry_relations"] = attached


def _relation_alert_sequences(
    frames: Sequence[dict[str, Any]],
    *,
    config: GeometryGateConfig,
) -> dict[str, list[tuple[dict[str, Any], bool, str]]]:
    output = {}
    for event_uid, event_frames in _group_dual_frames_by_event(frames).items():
        histories: dict[str, list[tuple[int, float]]] = defaultdict(list)
        sequence = []
        for frame in sorted(
            event_frames,
            key=lambda item: int(item["frame_position"]),
        ):
            position = int(frame["frame_position"])
            triggered_relation = ""
            for relation_id, values in frame.get(
                "geometry_relations",
                {},
            ).items():
                score = float(values[config.signal])
                history = histories[relation_id]
                if history and position != history[-1][0] + 1:
                    history.clear()
                history.append((position, score))
                history[:] = history[-config.window_frames :]
                scores = [value for _, value in history]
                support = sum(
                    value >= config.score_threshold for value in scores
                )
                recovery_drop = max(scores) - score
                current_delta = (
                    math.inf if len(scores) < 2 else score - scores[-2]
                )
                trigger = (
                    score >= config.score_threshold
                    and support >= config.support_frames
                    and score - min(scores) >= config.minimum_rise
                    and recovery_drop
                    <= getattr(config, "max_recovery_drop", math.inf)
                    and current_delta
                    >= getattr(config, "minimum_current_delta", -math.inf)
                )
                if trigger:
                    triggered_relation = relation_id
                    break
            sequence.append((frame, bool(triggered_relation), triggered_relation))
        output[str(event_uid)] = sequence
    return output


def _ttc_sequences(
    frames: Sequence[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], bool]]]:
    return {
        str(event_uid): [
            (
                frame,
                float(frame["aggregations"]["physical_percentile"])
                >= ALERT_PERCENTILE,
            )
            for frame in sorted(
                event_frames,
                key=lambda item: int(item["frame_position"]),
            )
        ]
        for event_uid, event_frames in _group_dual_frames_by_event(frames).items()
    }


def _rescue_metrics(
    frames: Sequence[dict[str, Any]],
    *,
    config: GeometryGateConfig,
    positive: bool,
) -> dict[str, Any]:
    ttc_sequences = _ttc_sequences(frames)
    relation_sequences = _relation_alert_sequences(frames, config=config)
    rows = []
    ttc_hits = 0
    rescue_hits = 0
    final_hits = 0
    leads = []
    for event_uid, ttc_sequence in ttc_sequences.items():
        relation_by_position = {
            int(frame["frame_position"]): (alert, relation_id)
            for frame, alert, relation_id in relation_sequences[event_uid]
        }
        usable = [
            (frame, alert)
            for frame, alert in ttc_sequence
            if 0
            < int(
                frame.get(
                    "eval_frames_to_anchor",
                    frame["frames_to_anchor"],
                )
            )
            <= WINDOW_FRAMES
        ]
        ttc_seen = False
        event_ttc = False
        event_rescue = False
        rescue_lead = None
        rescue_relation = ""
        for frame, ttc_alert in usable:
            relation_alert, relation_id = relation_by_position[
                int(frame["frame_position"])
            ]
            if relation_alert and not ttc_seen and not ttc_alert:
                event_rescue = True
                rescue_relation = rescue_relation or relation_id
                if positive:
                    lead = int(frame["frames_to_anchor"])
                    rescue_lead = (
                        lead if rescue_lead is None else max(rescue_lead, lead)
                    )
            event_ttc = event_ttc or bool(ttc_alert)
            ttc_seen = ttc_seen or bool(ttc_alert)
        event_final = event_ttc or event_rescue
        ttc_hits += int(event_ttc)
        rescue_hits += int(event_rescue and not event_ttc)
        final_hits += int(event_final)
        if rescue_lead is not None:
            leads.append(rescue_lead * RISK_BENCH_TIMESTEP_SECONDS)
        source_event_uid = (
            str(usable[0][0]["event_uid"]) if usable else str(event_uid)
        )
        rows.append(
            {
                "event_uid": source_event_uid,
                "event_index": event_uid,
                "positive": int(positive),
                "ttc_detected": int(event_ttc),
                "rescue_detected": int(event_rescue),
                "blind_rescue": int(event_rescue and not event_ttc),
                "final_detected": int(event_final),
                "rescue_relation_id": rescue_relation,
                "rescue_lead_seconds": (
                    "" if rescue_lead is None
                    else rescue_lead * RISK_BENCH_TIMESTEP_SECONDS
                ),
            }
        )
    event_count = len(rows)
    ttc_negative_count = event_count - ttc_hits
    return {
        "event_count": event_count,
        "ttc_event_hits": ttc_hits,
        "ttc_negative_event_count": ttc_negative_count,
        "blind_rescue_count": rescue_hits,
        "blind_rescue_rate": (
            0.0 if not ttc_negative_count else rescue_hits / ttc_negative_count
        ),
        "final_event_hits": final_hits,
        "final_event_rate": 0.0 if not event_count else final_hits / event_count,
        "mean_rescue_lead_seconds": (
            None if not leads else float(np.mean(leads))
        ),
        "median_rescue_lead_seconds": (
            None if not leads else float(np.median(leads))
        ),
        "events": rows,
    }


def _candidate_configs(
    frames: dict[str, list[dict[str, Any]]],
    *,
    horizon_seconds: float,
    clearance_m: float,
) -> list[GeometryGateConfig]:
    output = []
    for signal in SIGNALS:
        reference = np.asarray(
            [
                float(values[signal])
                for frame in frames["validation_normal"]
                for values in frame.get("geometry_relations", {}).values()
            ],
            dtype=float,
        )
        for quantile in THRESHOLD_QUANTILES:
            threshold = float(
                np.quantile(reference, quantile, method="higher")
            )
            for support, window in SUPPORT_RULES:
                for minimum_rise in MINIMUM_RISES:
                    output.append(
                        GeometryGateConfig(
                            horizon_seconds=horizon_seconds,
                            clearance_m=clearance_m,
                            signal=signal,
                            threshold_quantile=quantile,
                            score_threshold=threshold,
                            support_frames=support,
                            window_frames=window,
                            minimum_rise=minimum_rise,
                        )
                    )
    return output


def _selection_key(
    positive: dict[str, Any],
    normal: dict[str, Any],
    config: GeometryGateConfig,
) -> tuple[Any, ...]:
    lead = positive["mean_rescue_lead_seconds"]
    return (
        int(positive["blind_rescue_count"]),
        -float(normal["blind_rescue_rate"]),
        0.0 if lead is None else float(lead),
        -config.support_frames,
        config.score_threshold,
    )


def run_fold(
    *,
    fold: int,
    index_root: Path,
    cache_root: Path,
    output_root: Path,
    rescue_event_fpr_budget: float,
) -> dict[str, Any]:
    frames = _fold_frames(cache_root, fold)
    index_path = (
        index_root
        / f"fold_{fold:02d}"
        / "riskbench_nested_fold_index.json"
    )
    observations = _observations_by_frame(
        index_path,
        splits=tuple(frames),
    )
    evaluated = []
    candidate_rows = []
    for horizon_seconds in HORIZONS_SECONDS:
        for clearance_m in GENERIC_CLEARANCES_M:
            geometry = _geometry_from_observations(
                observations,
                horizon_seconds=horizon_seconds,
                clearance_m=clearance_m,
            )
            _attach_relation_signals(frames, geometry=geometry)
            for config in _candidate_configs(
                frames,
                horizon_seconds=horizon_seconds,
                clearance_m=clearance_m,
            ):
                normal = _rescue_metrics(
                    frames["validation_normal"],
                    config=config,
                    positive=False,
                )
                positive = _rescue_metrics(
                    frames["validation_positive"],
                    config=config,
                    positive=True,
                )
                eligible = (
                    normal["blind_rescue_rate"]
                    <= rescue_event_fpr_budget + 1e-12
                )
                candidate_rows.append(
                    {
                        "fold": fold,
                        **asdict(config),
                        "eligible": int(eligible),
                        "normal_rescue_event_fpr": normal[
                            "blind_rescue_rate"
                        ],
                        "positive_ttc_blind_count": positive[
                            "ttc_negative_event_count"
                        ],
                        "positive_blind_rescues": positive[
                            "blind_rescue_count"
                        ],
                        "positive_blind_rescue_rate": positive[
                            "blind_rescue_rate"
                        ],
                        "positive_mean_rescue_lead_seconds": positive[
                            "mean_rescue_lead_seconds"
                        ],
                    }
                )
                if eligible:
                    evaluated.append((config, normal, positive))
    if not evaluated:
        raise ValueError(
            f"Fold {fold} has no config under rescue event FPR budget."
        )
    selected, inner_normal, inner_positive = max(
        evaluated,
        key=lambda item: _selection_key(item[2], item[1], item[0]),
    )

    geometry = _geometry_from_observations(
        observations,
        horizon_seconds=selected.horizon_seconds,
        clearance_m=selected.clearance_m,
    )
    _attach_relation_signals(frames, geometry=geometry)
    outer_normal = _rescue_metrics(
        frames["test_normal"],
        config=selected,
        positive=False,
    )
    outer_positive = _rescue_metrics(
        frames["test_positive"],
        config=selected,
        positive=True,
    )
    fold_root = output_root / f"fold_{fold:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)
    _write_csv(fold_root / "candidate_configs.csv", candidate_rows)
    _write_csv(
        fold_root / "outer_events.csv",
        [
            {"fold": fold, **row}
            for row in outer_positive["events"] + outer_normal["events"]
        ],
    )
    result = {
        "fold": fold,
        "selected_config": asdict(selected),
        "inner_normal": {
            key: value for key, value in inner_normal.items() if key != "events"
        },
        "inner_positive": {
            key: value for key, value in inner_positive.items() if key != "events"
        },
        "outer_normal": {
            key: value for key, value in outer_normal.items() if key != "events"
        },
        "outer_positive": {
            key: value for key, value in outer_positive.items() if key != "events"
        },
    }
    path = fold_root / "summary.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(path)
    return result


def _aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positive_count = sum(
        int(result["outer_positive"]["event_count"]) for result in results
    )
    ttc_hits = sum(
        int(result["outer_positive"]["ttc_event_hits"]) for result in results
    )
    ttc_blind = positive_count - ttc_hits
    rescues = sum(
        int(result["outer_positive"]["blind_rescue_count"]) for result in results
    )
    normal_count = sum(
        int(result["outer_normal"]["event_count"]) for result in results
    )
    normal_ttc_hits = sum(
        int(result["outer_normal"]["ttc_event_hits"]) for result in results
    )
    normal_ttc_negative = normal_count - normal_ttc_hits
    false_rescues = sum(
        int(result["outer_normal"]["blind_rescue_count"]) for result in results
    )
    leads = [
        float(result["outer_positive"]["mean_rescue_lead_seconds"])
        for result in results
        if result["outer_positive"]["mean_rescue_lead_seconds"] is not None
    ]
    return {
        "positive_event_count": positive_count,
        "ttc_positive_hits": ttc_hits,
        "ttc_blind_positive_count": ttc_blind,
        "ttc_blind_rescue_count": rescues,
        "ttc_blind_rescue_recall": (
            0.0 if not ttc_blind else rescues / ttc_blind
        ),
        "final_positive_hits": ttc_hits + rescues,
        "final_positive_recall": (
            0.0 if not positive_count else (ttc_hits + rescues) / positive_count
        ),
        "hard_safe_event_count": normal_count,
        "ttc_hard_safe_alerts": normal_ttc_hits,
        "ttc_negative_hard_safe_count": normal_ttc_negative,
        "rescue_false_alert_count": false_rescues,
        "rescue_event_fpr_on_ttc_negative_hard_safe": (
            0.0
            if not normal_ttc_negative
            else false_rescues / normal_ttc_negative
        ),
        "mean_fold_rescue_lead_seconds": (
            None if not leads else float(np.mean(leads))
        ),
    }


def run(
    *,
    folds: Sequence[int],
    index_root: Path = DEFAULT_INDEX_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    rescue_event_fpr_budget: float = DEFAULT_RESCUE_EVENT_FPR_BUDGET,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        run_fold(
            fold=fold,
            index_root=index_root.expanduser().resolve(),
            cache_root=cache_root.expanduser().resolve(),
            output_root=output_root,
            rescue_event_fpr_budget=rescue_event_fpr_budget,
        )
        for fold in folds
    ]
    result = {
        "protocol": "riskbench-relation-geometry-rescue-exploratory-v1",
        "folds": list(folds),
        "ttc_preserved_by_construction": True,
        "rescue_fpr_denominator": "TTC-negative hard-safe events",
        "inner_rescue_event_fpr_budget": rescue_event_fpr_budget,
        "positive_selection_status": (
            "inner validation positives used for exploratory configuration "
            "selection; freeze before a new holdout"
        ),
        "aggregate": _aggregate(results),
        "fold_summaries": [result["summary_path"] for result in results],
    }
    path = output_root / "relation_geometry_rescue_summary.json"
    result["summary_path"] = str(path)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_folds(value: str) -> list[int]:
    return sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )


def main() -> None:
    args = parse_args()
    folds = _parse_folds(args.folds)
    if args.aggregate_existing:
        output_root = args.output_root.expanduser().resolve()
        fold_results = [
            json.loads(
                (
                    output_root
                    / f"fold_{fold:02d}"
                    / "summary.json"
                ).read_text(encoding="utf-8")
            )
            for fold in folds
        ]
        result = {
            "protocol": "riskbench-relation-geometry-rescue-exploratory-v1",
            "folds": folds,
            "ttc_preserved_by_construction": True,
            "rescue_fpr_denominator": "TTC-negative hard-safe events",
            "inner_rescue_event_fpr_budget": args.rescue_event_fpr_budget,
            "positive_selection_status": (
                "inner validation positives used for exploratory "
                "configuration selection; freeze before a new holdout"
            ),
            "aggregate": _aggregate(fold_results),
            "fold_summaries": [
                str(
                    output_root
                    / f"fold_{fold:02d}"
                    / "summary.json"
                )
                for fold in folds
            ],
        }
        path = output_root / "relation_geometry_rescue_summary.json"
        result["summary_path"] = str(path)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = run(
            folds=folds,
            index_root=args.index_root,
            cache_root=args.cache_root,
            output_root=args.output_root,
            rescue_event_fpr_budget=args.rescue_event_fpr_budget,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
