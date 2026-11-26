"""Matched-FPR ablations for the temporal relation model and decision head.

This script intentionally keeps the historical ablation outputs untouched.
Unlike the legacy evaluator, calibration and test metrics use the same
pre-anchor window for hard-safe and collision events.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from ..danger_signal.diagnostics import (
    ACTIVATION_WEIGHTS,
    score_sequences,
)
from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .family_grouped_nested_cv import (
    DEFAULT_INDEX_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_FULL_ROOT,
    DEFAULT_SEED,
    WINDOW_FRAMES,
    _evaluate_fold_frames,
    _fpr,
    _matched_fpr_threshold,
)
from .persistent_guarded_fusion import ALERT_PERCENTILE
from .stratified_eval import _load_model_and_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_FREE_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_graph_free_ablation_v1"
)
DEFAULT_NO_PAIR_MEMORY_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_no_pair_memory_ablation_v1"
)
DEFAULT_NO_NODE_MEMORY_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_no_node_memory_ablation_v1"
)
DEFAULT_NO_TEMPORAL_MEMORY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_no_temporal_memory_ablation_v1"
)
DEFAULT_CAUSAL_FUTURE_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_causal_future_ablation_v1"
)
DEFAULT_HYBRID_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_hybrid_objective_ablation_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_complete_ablation_v1"
)
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "scored_frame_cache"

BACKBONES = {
    "full_gat_one_class": DEFAULT_FULL_ROOT,
    "graph_free_one_class": DEFAULT_GRAPH_FREE_ROOT,
    "no_node_memory_one_class": DEFAULT_NO_NODE_MEMORY_ROOT,
    "no_pair_memory_one_class": DEFAULT_NO_PAIR_MEMORY_ROOT,
    "no_temporal_memory_one_class": DEFAULT_NO_TEMPORAL_MEMORY_ROOT,
    "full_gat_causal_future": DEFAULT_CAUSAL_FUTURE_ROOT,
    "full_gat_hybrid_reconstruction": DEFAULT_HYBRID_ROOT,
}

DECISION_METHODS = {
    "ttc": "physical_percentile",
    "causal_raw": "baseline_causal_raw",
    "causal_ewma": "baseline_causal_ewma",
    "stress_only": "danger_stress",
    "damage_only": "danger_damage",
    "persistence_only": "danger_persistence",
    "stress_damage": "ablation_stress_damage",
    "stress_persistence": "ablation_stress_persistence",
    "damage_persistence": "ablation_damage_persistence",
    "damp_no_inhibition": "danger_activation",
    "damp_recovery_inhibition": "danger_activation_recovery_only",
    "damp_self_context": "danger_activation_inhibited",
    "relation_only": "relation_percentile",
    "ttc_relation_max": "unconstrained_max",
    "ttc_persistent_relation": "persistent_rescue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--backbones",
        default=",".join(BACKBONES),
        help="Comma-separated backbone names; unavailable checkpoints are skipped.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_path(root: Path, fold: int) -> Path:
    direct = root / f"fold_{fold:02d}" / "best_guarded_checkpoint.pt"
    if direct.exists():
        return direct
    nested = (
        root
        / f"fold_{fold:02d}"
        / "training"
        / "best_guarded_checkpoint.pt"
    )
    return nested


def _in_window(frame: dict[str, Any]) -> bool:
    frames_to_anchor = int(
        frame.get("eval_frames_to_anchor", frame["frames_to_anchor"])
    )
    return 0 < frames_to_anchor <= WINDOW_FRAMES


def _windowed(frames: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame for frame in frames if _in_window(frame)]


def _attach_decision_ablation_scores(
    frames_by_split: dict[str, list[dict[str, Any]]],
) -> None:
    stress_weight = float(ACTIVATION_WEIGHTS["danger_stress"])
    damage_weight = float(ACTIVATION_WEIGHTS["danger_damage"])
    persistence_weight = float(ACTIVATION_WEIGHTS["danger_persistence"])
    for frames in frames_by_split.values():
        for frame in frames:
            values = frame["aggregations"]
            stress = float(values["danger_stress"])
            damage = float(values["danger_damage"])
            persistence = float(values["danger_persistence"])
            values["ablation_stress_damage"] = (
                stress_weight * stress + damage_weight * damage
            )
            values["ablation_stress_persistence"] = (
                stress_weight * stress + persistence_weight * persistence
            )
            values["ablation_damage_persistence"] = (
                damage_weight * damage + persistence_weight * persistence
            )


def _event_detection(
    frames: Sequence[dict[str, Any]],
    score_key: str,
    threshold: float,
) -> dict[str, int]:
    detections = {}
    for event_uid, event_frames in _group_dual_frames_by_event(
        list(frames)
    ).items():
        detections[str(event_uid)] = int(
            any(
                float(frame["aggregations"][score_key]) >= threshold
                for frame in event_frames
                if _in_window(frame)
            )
        )
    return detections


def _safe_auc(
    normal_scores: Sequence[float],
    positive_scores: Sequence[float],
) -> float | None:
    if not normal_scores or not positive_scores:
        return None
    labels = np.asarray(
        [0] * len(normal_scores) + [1] * len(positive_scores),
        dtype=int,
    )
    scores = np.asarray([*normal_scores, *positive_scores], dtype=float)
    return float(roc_auc_score(labels, scores))


def _evaluate_symmetric_window(
    *,
    normal_frames: Sequence[dict[str, Any]],
    positive_frames: Sequence[dict[str, Any]],
    score_key: str,
    threshold: float,
) -> dict[str, Any]:
    normal_window = _windowed(normal_frames)
    positive_window = _windowed(positive_frames)
    normal_scores = [
        float(frame["aggregations"][score_key]) for frame in normal_window
    ]
    positive_scores = [
        float(frame["aggregations"][score_key]) for frame in positive_window
    ]
    event_leads = []
    positive_detections = _event_detection(
        positive_frames,
        score_key,
        threshold,
    )
    for event_frames in _group_dual_frames_by_event(
        list(positive_frames)
    ).values():
        hits = [
            frame
            for frame in event_frames
            if _in_window(frame)
            and float(frame["aggregations"][score_key]) >= threshold
        ]
        if hits:
            event_leads.append(
                max(
                    int(
                        frame.get(
                            "eval_frames_to_anchor",
                            frame["frames_to_anchor"],
                        )
                    )
                    for frame in hits
                )
                * RISK_BENCH_TIMESTEP_SECONDS
            )
    return {
        "frame_auroc": _safe_auc(normal_scores, positive_scores),
        "normal_frame_count": len(normal_scores),
        "positive_frame_count": len(positive_scores),
        "normal_false_alarm_rate_at_threshold": _fpr(
            normal_scores,
            threshold,
        ),
        "positive_alert_recall_at_threshold": _fpr(
            positive_scores,
            threshold,
        ),
        "event_count": len(positive_detections),
        "event_recall_at_threshold": (
            float(np.mean(list(positive_detections.values())))
            if positive_detections
            else None
        ),
        "mean_lead_seconds": (
            float(np.mean(event_leads)) if event_leads else None
        ),
    }


def _load_scored_frames(
    *,
    fold: int,
    index_root: Path,
    checkpoint_path: Path,
    seed: int,
    cache_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    index_path = (
        index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    )
    model, normalized_splits, _, device = _load_model_and_sequences(
        index_path,
        checkpoint_path,
    )
    frames = {
        split: score_sequences(
            model,
            normalized_splits[split],
            device,
            split=split,
            group_count=4,
            recovery_initialization_predictions=3,
        )
        for split in (
            "validation_normal",
            "validation_positive",
            "test_normal",
            "test_positive",
        )
    }
    _evaluate_fold_frames(frames, fold=fold, seed=seed + fold)
    _attach_decision_ablation_scores(frames)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return frames


def _calibrated_threshold(
    *,
    validation_frames: Sequence[dict[str, Any]],
    score_key: str,
    target_fpr: float,
) -> tuple[float, float]:
    return _matched_fpr_threshold(
        reference_scores=[
            float(frame["aggregations"][score_key])
            for frame in validation_frames
            if _in_window(frame)
        ],
        target_fpr=target_fpr,
    )


def evaluate_backbone_fold(
    *,
    backbone: str,
    fold: int,
    root: Path,
    index_root: Path,
    seed: int,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_path = _checkpoint_path(root, fold)
    if not checkpoint_path.exists():
        return [], []
    frames = _load_scored_frames(
        fold=fold,
        index_root=index_root,
        checkpoint_path=checkpoint_path,
        seed=seed,
        cache_path=cache_root / backbone / f"fold_{fold:02d}.pkl",
    )
    validation_window = _windowed(frames["validation_normal"])
    target_fpr = _fpr(
        [
            float(frame["aggregations"]["physical_percentile"])
            for frame in validation_window
        ],
        ALERT_PERCENTILE,
    )
    method_items = (
        DECISION_METHODS.items()
        if backbone == "full_gat_one_class"
        else (("ttc_persistent_relation", "persistent_rescue"),)
    )
    metrics = []
    events: list[dict[str, Any]] = []
    for method, score_key in method_items:
        if method == "ttc":
            threshold = ALERT_PERCENTILE
            achieved_fpr = target_fpr
        else:
            threshold, achieved_fpr = _calibrated_threshold(
                validation_frames=frames["validation_normal"],
                score_key=score_key,
                target_fpr=target_fpr,
            )
        metrics.append(
            {
                "fold": fold,
                "backbone": backbone,
                "method": method,
                "score_key": score_key,
                "threshold": threshold,
                "inner_target_fpr": target_fpr,
                "inner_achieved_fpr": achieved_fpr,
                **_evaluate_symmetric_window(
                    normal_frames=frames["test_normal"],
                    positive_frames=frames["test_positive"],
                    score_key=score_key,
                    threshold=threshold,
                ),
            }
        )
        for positive, split in (
            (True, "test_positive"),
            (False, "test_normal"),
        ):
            for event_uid, detected in _event_detection(
                frames[split],
                score_key,
                threshold,
            ).items():
                events.append(
                    {
                        "fold": fold,
                        "backbone": backbone,
                        "method": method,
                        "split": split,
                        "event_uid": event_uid,
                        "positive": int(positive),
                        "detected": detected,
                    }
                )
    return metrics, events


def _aggregate(
    metrics: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    keys = sorted(
        {(row["backbone"], row["method"]) for row in metrics}
    )
    for backbone, method in keys:
        label = f"{backbone}:{method}"
        rows = [
            row
            for row in metrics
            if row["backbone"] == backbone and row["method"] == method
        ]
        event_rows = [
            row
            for row in events
            if row["backbone"] == backbone and row["method"] == method
        ]
        positive = [row for row in event_rows if int(row["positive"]) == 1]
        normal = [row for row in event_rows if int(row["positive"]) == 0]
        summary[label] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in rows])),
                "std": float(np.std([float(row[metric]) for row in rows])),
            }
            for metric in (
                "frame_auroc",
                "normal_false_alarm_rate_at_threshold",
                "positive_alert_recall_at_threshold",
                "event_recall_at_threshold",
                "mean_lead_seconds",
            )
            if all(row[metric] is not None for row in rows)
        }
        summary[label].update(
            {
                "fold_count": len(rows),
                "positive_hits": sum(
                    int(row["detected"]) for row in positive
                ),
                "positive_event_count": len(positive),
                "normal_alerts": sum(
                    int(row["detected"]) for row in normal
                ),
                "normal_event_count": len(normal),
                "pooled_positive_event_recall": (
                    float(np.mean([int(row["detected"]) for row in positive]))
                    if positive
                    else None
                ),
                "pooled_normal_event_alert_rate": (
                    float(np.mean([int(row["detected"]) for row in normal]))
                    if normal
                    else None
                ),
            }
        )

    full_label = "full_gat_one_class:ttc_persistent_relation"
    if full_label in summary:
        full_events = {
            (int(row["fold"]), row["split"], str(row["event_uid"])): int(
                row["detected"]
            )
            for row in events
            if row["backbone"] == "full_gat_one_class"
            and row["method"] == "ttc_persistent_relation"
            and int(row["positive"]) == 1
        }
        paired = {}
        for backbone in BACKBONES:
            if backbone == "full_gat_one_class":
                continue
            variant_events = {
                (int(row["fold"]), row["split"], str(row["event_uid"])): int(
                    row["detected"]
                )
                for row in events
                if row["backbone"] == backbone
                and row["method"] == "ttc_persistent_relation"
                and int(row["positive"]) == 1
            }
            if set(variant_events) != set(full_events):
                continue
            paired[backbone] = dict(
                Counter(
                    "both"
                    if full_events[key] and variant_events[key]
                    else "full_only"
                    if full_events[key]
                    else "variant_only"
                    if variant_events[key]
                    else "neither"
                    for key in full_events
                )
            )
        summary["paired_backbone_positive_counts"] = paired
    return summary


def run_complete_ablation(
    *,
    folds: Sequence[int],
    backbones: Sequence[str],
    index_root: Path = DEFAULT_INDEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    skipped = []
    for backbone in backbones:
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}")
        root = BACKBONES[backbone]
        for fold in folds:
            fold_metrics, fold_events = evaluate_backbone_fold(
                backbone=backbone,
                fold=fold,
                root=root,
                index_root=index_root,
                seed=seed,
                cache_root=cache_root,
            )
            if not fold_metrics:
                skipped.append(
                    {
                        "backbone": backbone,
                        "fold": fold,
                        "missing_checkpoint": str(
                            _checkpoint_path(root, fold).resolve()
                        ),
                    }
                )
                continue
            metrics.extend(fold_metrics)
            events.extend(fold_events)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "complete_ablation_metrics.csv"
    events_path = output_root / "complete_ablation_events.csv"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    result = {
        "protocol": "riskbench-complete-ablation-symmetric-pre-anchor-v1",
        "window": {
            "definition": f"0 < frames_to_anchor <= {WINDOW_FRAMES}",
            "applied_to": [
                "validation_normal",
                "test_normal",
                "test_positive",
            ],
            "reason": (
                "Prevent post-anchor hard-safe frames from being compared "
                "with pre-anchor-only positive frames."
            ),
        },
        "calibration": (
            "Each method/backbone is independently thresholded on inner "
            "hard-safe pre-anchor frames to match TTC frame FPR."
        ),
        "positive_frames_used_for_calibration": 0,
        "folds": list(folds),
        "backbones_requested": list(backbones),
        "skipped": skipped,
        "aggregate": _aggregate(metrics, events),
        "artifacts": {
            "metrics_csv": str(metrics_path.resolve()),
            "events_csv": str(events_path.resolve()),
        },
        "status": "complete_ablation_evaluation_complete",
    }
    summary_path = output_root / "complete_ablation_summary.json"
    result["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    backbones = [
        value.strip() for value in args.backbones.split(",") if value.strip()
    ]
    print(
        json.dumps(
            run_complete_ablation(
                folds=folds,
                backbones=backbones,
                index_root=args.index_root,
                output_root=args.output_root,
                cache_root=args.cache_root,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
