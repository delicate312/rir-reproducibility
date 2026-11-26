"""Family-grouped nested CV with hard-safe-only matched-FPR calibration."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest

from ..danger_signal.diagnostics import (
    apply_danger_calibration,
    robust_component_calibration,
    score_sequences,
)
from ..temporal_relation_train_eval import (
    _evaluate_dual_track,
    _group_dual_frames_by_event,
)
from .adapter import (
    DEFAULT_METADATA_ROOT,
    RISK_BENCH_GRAPH_INPUT_PROTOCOL,
    RISK_BENCH_TIMESTEP_SECONDS,
    discover_riskbench_records,
)
from .audited_gate_ablation import _attach_candidate_scores
from .development_ceiling_ablation import _attach_development_ceiling
from .early_window_sensitivity import _attach_window_scores
from .guarded_epoch_sweep import run_guarded_epoch_sweep
from .hard_safe_inhibition import FEATURES, _feature_vector
from .lazy_dataset import _hash_rank, _record_payload
from .onset_shock_ablation import _positive_reference
from .peak_or_sustain_gate import (
    PeakOrSustainConfig,
    attach_peak_or_sustain_response,
)
from .persistent_guarded_fusion import (
    ALERT_PERCENTILE,
    _attach_guarded_response,
    _attach_percentile_inputs,
)
from .stratified_eval import _load_model_and_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_riskbench_family_nested_cv_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_family_nested_cv_v1"
)
DEFAULT_FOLD_COUNT = 5
DEFAULT_SEED = 20260728
DEFAULT_EPOCHS = 3
DEFAULT_EVAL_EPOCHS = (1, 3)
WINDOW_FRAMES = 30
RANK_FEATURES = tuple(FEATURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--eval-epochs",
        default=",".join(str(value) for value in DEFAULT_EVAL_EPOCHS),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _payload_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _representative_records(
    records: Sequence[Any],
    *,
    seed: int,
) -> list[Any]:
    grouped: dict[str, list[Any]] = {}
    for record in records:
        if record.category not in {
            "non-interactive",
            "interactive",
            "collision",
        }:
            continue
        grouped.setdefault(record.scenario_group_id, []).append(record)
    return [
        min(
            variants,
            key=lambda record: _hash_rank(record.scenario_id, seed),
        )
        for variants in grouped.values()
    ]


def _assign_family_folds(
    representatives: Sequence[Any],
    *,
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    strata: dict[tuple[str, str], list[Any]] = {}
    for record in representatives:
        strata.setdefault(
            (record.category, record.target_type_name),
            [],
        ).append(record)
    for stratum, records in sorted(strata.items()):
        ordered = sorted(
            records,
            key=lambda record: _hash_rank(
                f"{stratum}:{record.scenario_group_id}",
                seed,
            ),
        )
        offset = int(_hash_rank(str(stratum), seed)[:8], 16) % fold_count
        for index, record in enumerate(ordered):
            assignments[record.scenario_group_id] = (
                index + offset
            ) % fold_count
    return assignments


def _assert_fold_isolation(
    splits: dict[str, list[Any]],
) -> dict[str, Any]:
    groups_by_split = {
        split: {record.scenario_group_id for record in records}
        for split, records in splits.items()
    }
    overlaps = {}
    names = list(groups_by_split)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                overlaps[f"{left}__{right}"] = sorted(overlap)
    if overlaps:
        raise ValueError(f"Family leakage detected: {overlaps}")
    return {
        "family_overlap_count": 0,
        "unique_family_count": len(
            set().union(*groups_by_split.values())
        ),
    }


def build_family_grouped_indexes(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    index_root: Path = DEFAULT_INDEX_ROOT,
    fold_count: int = DEFAULT_FOLD_COUNT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if fold_count < 3:
        raise ValueError("Nested CV requires at least three folds.")
    representatives = _representative_records(
        discover_riskbench_records(metadata_root),
        seed=seed,
    )
    assignments = _assign_family_folds(
        representatives,
        fold_count=fold_count,
        seed=seed,
    )
    index_root = index_root.expanduser().resolve()
    index_root.mkdir(parents=True, exist_ok=True)
    cache_root = index_root / "sequence_cache"
    fold_rows = []
    fold_paths = {}

    for outer_fold in range(fold_count):
        inner_fold = (outer_fold + 1) % fold_count
        splits = {
            "train_normal": [
                record
                for record in representatives
                if record.category == "non-interactive"
                and assignments[record.scenario_group_id]
                not in {outer_fold, inner_fold}
            ],
            "validation_normal": [
                record
                for record in representatives
                if record.category == "interactive"
                and assignments[record.scenario_group_id] == inner_fold
            ],
            "validation_positive": [
                record
                for record in representatives
                if record.category == "collision"
                and assignments[record.scenario_group_id] == inner_fold
            ],
            "test_normal": [
                record
                for record in representatives
                if record.category == "interactive"
                and assignments[record.scenario_group_id] == outer_fold
            ],
            "test_positive": [
                record
                for record in representatives
                if record.category == "collision"
                and assignments[record.scenario_group_id] == outer_fold
            ],
        }
        isolation = _assert_fold_isolation(splits)
        payload_splits = {
            split: [_record_payload(record) for record in records]
            for split, records in splits.items()
        }
        payload = {
            "format": "riskbench-metadata-lazy-v1",
            "protocol": "riskbench-family-grouped-nested-cv-v1",
            "graph_input_protocol": RISK_BENCH_GRAPH_INPUT_PROTOCOL,
            "metadata_root": str(metadata_root.expanduser().resolve()),
            "cache_root": str(cache_root.resolve()),
            "pre_critical_frames": 40,
            "post_critical_frames": 20,
            "max_normal_frames": 80,
            "seed": seed,
            "splits": payload_splits,
            "meta": {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "fold_count": fold_count,
                "sampling_unit": "scenario_group_id",
                "positive_outer_open_policy": (
                    "opened_once_after_checkpoint_and_matched_fpr_freeze"
                ),
                "gradient_allowed_splits": ["train_normal"],
                "threshold_allowed_splits": ["validation_normal"],
                "checkpoint_selection_allowed_splits": [
                    "validation_normal",
                    "validation_positive",
                ],
                "outer_evaluation_splits": [
                    "test_normal",
                    "test_positive",
                ],
                "legacy_mechanism_status": (
                    "retrospective fixed mechanism; legacy gate forms were "
                    "developed before this CV"
                ),
                "family_isolation": isolation,
                "manifest_sha256": _payload_hash(payload_splits),
            },
        }
        fold_root = index_root / f"fold_{outer_fold:02d}"
        fold_root.mkdir(parents=True, exist_ok=True)
        index_path = fold_root / "riskbench_nested_fold_index.json"
        index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        fold_paths[str(outer_fold)] = str(index_path)
        for split, records in splits.items():
            counts = Counter(record.target_type_name for record in records)
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "split": split,
                    "sequence_count": len(records),
                    "family_count": len(
                        {record.scenario_group_id for record in records}
                    ),
                    "target_type_counts": json.dumps(
                        dict(sorted(counts.items())),
                        sort_keys=True,
                    ),
                }
            )

    outer_collision_groups = Counter()
    outer_normal_groups = Counter()
    for fold_string, index_path_string in fold_paths.items():
        payload = json.loads(
            Path(index_path_string).read_text(encoding="utf-8")
        )
        for record in payload["splits"]["test_positive"]:
            outer_collision_groups[record["scenario_group_id"]] += 1
        for record in payload["splits"]["test_normal"]:
            outer_normal_groups[record["scenario_group_id"]] += 1
    if set(outer_collision_groups.values()) != {1}:
        raise ValueError("Positive families do not cover outer folds exactly once.")
    if set(outer_normal_groups.values()) != {1}:
        raise ValueError("Hard-safe families do not cover outer folds exactly once.")

    folds_csv = index_root / "nested_fold_composition.csv"
    _write_csv(folds_csv, fold_rows)
    summary = {
        "protocol": "riskbench-family-grouped-nested-cv-v1",
        "fold_count": fold_count,
        "seed": seed,
        "representative_family_count": len(representatives),
        "category_family_counts": dict(
            Counter(record.category for record in representatives)
        ),
        "outer_positive_family_count": len(outer_collision_groups),
        "outer_hard_safe_family_count": len(outer_normal_groups),
        "outer_family_coverage_exactly_once": True,
        "fold_indexes": fold_paths,
        "fold_composition_csv": str(folds_csv),
        "status": "family_grouped_nested_indexes_ready",
    }
    summary_path = index_root / "nested_cv_index_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _positive_percentile(reference: np.ndarray, value: float) -> float:
    return float(np.searchsorted(reference, value, side="right") / len(reference))


def _fit_normal_only_rank(
    frames: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> dict[str, Any]:
    validation = frames["validation_normal"]
    matrix = np.stack([_feature_vector(frame) for frame in validation])
    center = np.median(matrix, axis=0)
    scale = np.maximum(
        np.median(np.abs(matrix - center), axis=0) * 1.4826,
        0.05,
    )
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
    ).fit((matrix - center) / scale)
    validation_anomaly = -model.score_samples((matrix - center) / scale)
    anomaly_reference = np.sort(validation_anomaly)

    for split_frames in frames.values():
        split_matrix = np.stack(
            [_feature_vector(frame) for frame in split_frames]
        )
        anomaly = -model.score_samples((split_matrix - center) / scale)
        for frame, anomaly_value in zip(split_frames, anomaly):
            values = frame["aggregations"]
            anomaly_percentile = _positive_percentile(
                anomaly_reference,
                float(anomaly_value),
            )
            values["normal_only_anomaly_percentile"] = anomaly_percentile
            values["immune_rank_raw"] = max(
                float(values["physical_percentile"]),
                float(values["peak_or_sustain_guarded"]),
                anomaly_percentile,
            )
    return {
        "fit_split": "validation_normal",
        "positive_frames_used": 0,
        "fit_frame_count": len(validation),
        "feature_names": list(RANK_FEATURES),
        "n_estimators": 300,
        "seed": seed,
    }


def _fpr(scores: Sequence[float], threshold: float) -> float:
    return float(np.mean(np.asarray(scores, dtype=float) >= threshold))


def _matched_fpr_threshold(
    *,
    reference_scores: Sequence[float],
    target_fpr: float,
) -> tuple[float, float]:
    values = np.asarray(reference_scores, dtype=float)
    candidates = [
        float(np.nextafter(np.max(values), math.inf)),
        *sorted({float(value) for value in values}, reverse=True),
    ]
    eligible = [
        threshold
        for threshold in candidates
        if _fpr(values, threshold) <= target_fpr + 1e-12
    ]
    threshold = min(eligible)
    return threshold, _fpr(values, threshold)


def _event_rows(
    frames: dict[str, list[dict[str, Any]]],
    *,
    immune_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    methods = {
        "ttc": ("physical_percentile", ALERT_PERCENTILE),
        "immune_matched": ("immune_rank_raw", immune_threshold),
    }
    for split in ("test_positive", "test_normal"):
        positive = split.endswith("positive")
        for event_frames in _group_dual_frames_by_event(
            frames[split]
        ).values():
            usable = (
                [
                    frame
                    for frame in event_frames
                    if 0 < int(frame["frames_to_anchor"]) <= WINDOW_FRAMES
                ]
                if positive
                else list(event_frames)
            )
            row: dict[str, Any] = {
                "split": split,
                "event_uid": event_frames[0]["event_uid"],
                "positive": int(positive),
            }
            detections = {}
            for method, (score_key, threshold) in methods.items():
                hits = [
                    frame
                    for frame in usable
                    if float(frame["aggregations"][score_key]) >= threshold
                ]
                detections[method] = int(bool(hits))
                row[f"{method}_detected"] = detections[method]
                row[f"{method}_lead_seconds"] = (
                    ""
                    if not positive or not hits
                    else max(
                        int(frame["frames_to_anchor"])
                        for frame in hits
                    )
                    * RISK_BENCH_TIMESTEP_SECONDS
                )
            row["ttc_blind_rescued"] = int(
                detections["immune_matched"] and not detections["ttc"]
            )
            row["ttc_only"] = int(
                detections["ttc"] and not detections["immune_matched"]
            )
            rows.append(row)
    return rows


def _evaluate_fold_frames(
    frames: dict[str, list[dict[str, Any]]],
    *,
    fold: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calibration = robust_component_calibration(frames["validation_normal"])
    for split_frames in frames.values():
        apply_danger_calibration(split_frames, calibration)
    _attach_percentile_inputs(frames, frames["validation_normal"])
    for split_frames in frames.values():
        _attach_guarded_response(split_frames)
        attach_peak_or_sustain_response(
            split_frames,
            PeakOrSustainConfig(),
        )
    onset_reference = _positive_reference(frames["validation_normal"])
    _attach_window_scores(frames, onset_reference)
    _attach_candidate_scores(frames)
    _attach_development_ceiling(frames)
    rank_fit = _fit_normal_only_rank(frames, seed=seed)

    inner_ttc_fpr = _fpr(
        [
            frame["aggregations"]["physical_percentile"]
            for frame in frames["validation_normal"]
        ],
        ALERT_PERCENTILE,
    )
    immune_threshold, inner_immune_fpr = _matched_fpr_threshold(
        reference_scores=[
            frame["aggregations"]["immune_rank_raw"]
            for frame in frames["validation_normal"]
        ],
        target_fpr=inner_ttc_fpr,
    )
    metrics = []
    for method, score_key, threshold in (
        ("ttc", "physical_percentile", ALERT_PERCENTILE),
        ("immune_matched", "immune_rank_raw", immune_threshold),
    ):
        metrics.append(
            {
                "fold": fold,
                "method": method,
                "threshold": threshold,
                **_evaluate_dual_track(
                    normal_frames=frames["test_normal"],
                    positive_frames=frames["test_positive"],
                    aggregation=score_key,
                    label_window=f"pre_anchor_{WINDOW_FRAMES}",
                    threshold=threshold,
                    timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
                ),
            }
        )
    events = _event_rows(frames, immune_threshold=immune_threshold)
    positive_events = [row for row in events if row["positive"]]
    normal_events = [row for row in events if not row["positive"]]
    pairs = Counter(
        "both"
        if row["ttc_detected"] and row["immune_matched_detected"]
        else "ttc_only"
        if row["ttc_detected"]
        else "immune_only"
        if row["immune_matched_detected"]
        else "neither"
        for row in positive_events
    )
    ttc_blind = [
        row for row in positive_events if not row["ttc_detected"]
    ]
    summary = {
        "fold": fold,
        "inner_ttc_frame_fpr": inner_ttc_fpr,
        "inner_immune_frame_fpr": inner_immune_fpr,
        "matched_fpr_absolute_error": abs(
            inner_immune_fpr - inner_ttc_fpr
        ),
        "immune_threshold": immune_threshold,
        "positive_pair_counts": dict(pairs),
        "ttc_blind_event_count": len(ttc_blind),
        "ttc_blind_rescued_count": sum(
            int(row["immune_matched_detected"])
            for row in ttc_blind
        ),
        "outer_normal_event_alerts": {
            "ttc": sum(int(row["ttc_detected"]) for row in normal_events),
            "immune_matched": sum(
                int(row["immune_matched_detected"])
                for row in normal_events
            ),
            "event_count": len(normal_events),
        },
        "rank_fit": rank_fit,
    }
    return metrics, events, summary


def run_nested_fold(
    *,
    fold: int,
    index_path: Path,
    output_root: Path,
    epochs: int,
    eval_epochs: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    fold_root = output_root / f"fold_{fold:02d}"
    training_summary = run_guarded_epoch_sweep(
        dataset_path=index_path,
        output_root=fold_root / "training",
        epochs=epochs,
        eval_epochs=eval_epochs,
        seed=seed + fold,
        run_confirmation=False,
        evaluate_test_during_selection=False,
    )
    checkpoint = Path(training_summary["best_checkpoint_path"])
    model, normalized_splits, _, device = _load_model_and_sequences(
        index_path,
        checkpoint,
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
    metrics, events, evaluation = _evaluate_fold_frames(
        frames,
        fold=fold,
        seed=seed + fold,
    )
    fold_root.mkdir(parents=True, exist_ok=True)
    metrics_path = fold_root / "matched_fpr_metrics.csv"
    events_path = fold_root / "matched_fpr_events.csv"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    result = {
        "protocol": "riskbench-family-nested-matched-fpr-v1",
        "fold": fold,
        "index_path": str(index_path.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "outer_test_scored_during_selection": False,
        "matched_fpr_source": "validation_normal_only",
        "positive_frames_used_for_rank_fit_or_threshold": 0,
        "training_selection": training_summary["selection"],
        "evaluation": evaluation,
        "metrics": metrics,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "events_csv": str(events_path),
            "training_summary": training_summary["summary_path"],
        },
        "status": "nested_fold_complete",
    }
    summary_path = fold_root / "nested_fold_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _aggregate_fold_results(
    fold_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [
        row
        for result in fold_results
        for row in result["metrics"]
    ]
    metric_summary = {}
    for method in ("ttc", "immune_matched"):
        rows = [row for row in metrics if row["method"] == method]
        metric_summary[method] = {
            key: {
                "mean": float(np.mean([float(row[key]) for row in rows])),
                "std": float(
                    np.std([float(row[key]) for row in rows], ddof=0)
                ),
            }
            for key in (
                "frame_auroc",
                "normal_false_alarm_rate_at_threshold",
                "event_recall_at_threshold",
                "mean_lead_seconds",
            )
        }
    pairs = Counter()
    blind_count = 0
    blind_rescued = 0
    normal_event_count = 0
    normal_event_alerts = Counter()
    for result in fold_results:
        pairs.update(result["evaluation"]["positive_pair_counts"])
        blind_count += int(result["evaluation"]["ttc_blind_event_count"])
        blind_rescued += int(
            result["evaluation"]["ttc_blind_rescued_count"]
        )
        normal_alerts = result["evaluation"]["outer_normal_event_alerts"]
        normal_event_count += int(normal_alerts["event_count"])
        normal_event_alerts["ttc"] += int(normal_alerts["ttc"])
        normal_event_alerts["immune_matched"] += int(
            normal_alerts["immune_matched"]
        )
    positive_event_count = sum(pairs.values())
    ttc_event_hits = int(pairs["both"] + pairs["ttc_only"])
    immune_event_hits = int(pairs["both"] + pairs["immune_only"])
    paired_values = np.asarray(
        [
            *([(1, 1)] * int(pairs["both"])),
            *([(1, 0)] * int(pairs["ttc_only"])),
            *([(0, 1)] * int(pairs["immune_only"])),
            *([(0, 0)] * int(pairs["neither"])),
        ],
        dtype=float,
    )
    rng = np.random.default_rng(DEFAULT_SEED)
    bootstrap_indices = rng.integers(
        0,
        len(paired_values),
        size=(10_000, len(paired_values)),
    )
    bootstrap = paired_values[bootstrap_indices]
    recall_deltas = (
        bootstrap[:, :, 1].mean(axis=1)
        - bootstrap[:, :, 0].mean(axis=1)
    )
    discordant = int(pairs["ttc_only"] + pairs["immune_only"])
    observed_probability = (
        1.0
        if not discordant
        else math.comb(
            discordant,
            int(pairs["immune_only"]),
        )
        / (2.0**discordant)
    )
    mcnemar_p = (
        1.0
        if not discordant
        else min(
            1.0,
            sum(
                math.comb(discordant, value) / (2.0**discordant)
                for value in range(discordant + 1)
                if (
                    math.comb(discordant, value)
                    / (2.0**discordant)
                    <= observed_probability + 1e-15
                )
            ),
        )
    )
    direction_counts = {
        "recall_higher_fold_count": 0,
        "recall_equal_fold_count": 0,
        "recall_lower_fold_count": 0,
        "auroc_higher_fold_count": 0,
        "frame_fpr_not_higher_fold_count": 0,
    }
    max_inner_error = 0.0
    for result in fold_results:
        rows = {row["method"]: row for row in result["metrics"]}
        recall_delta = (
            float(rows["immune_matched"]["event_recall_at_threshold"])
            - float(rows["ttc"]["event_recall_at_threshold"])
        )
        if recall_delta > 1e-12:
            direction_counts["recall_higher_fold_count"] += 1
        elif recall_delta < -1e-12:
            direction_counts["recall_lower_fold_count"] += 1
        else:
            direction_counts["recall_equal_fold_count"] += 1
        direction_counts["auroc_higher_fold_count"] += int(
            float(rows["immune_matched"]["frame_auroc"])
            > float(rows["ttc"]["frame_auroc"])
        )
        direction_counts["frame_fpr_not_higher_fold_count"] += int(
            float(
                rows["immune_matched"][
                    "normal_false_alarm_rate_at_threshold"
                ]
            )
            <= float(
                rows["ttc"]["normal_false_alarm_rate_at_threshold"]
            )
        )
        max_inner_error = max(
            max_inner_error,
            float(result["evaluation"]["matched_fpr_absolute_error"]),
        )
    return {
        "completed_fold_count": len(fold_results),
        "metrics": metric_summary,
        "positive_pair_counts": dict(pairs),
        "pooled_positive_event_count": positive_event_count,
        "pooled_ttc_event_recall": (
            ttc_event_hits / positive_event_count
        ),
        "pooled_immune_event_recall": (
            immune_event_hits / positive_event_count
        ),
        "pooled_paired_recall_delta": (
            immune_event_hits - ttc_event_hits
        )
        / positive_event_count,
        "pooled_paired_recall_delta_bootstrap_ci95": [
            float(np.quantile(recall_deltas, 0.025)),
            float(np.quantile(recall_deltas, 0.975)),
        ],
        "mcnemar_exact_p_value": mcnemar_p,
        "ttc_blind_event_count": blind_count,
        "ttc_blind_rescued_count": blind_rescued,
        "ttc_blind_rescue_rate": (
            None if not blind_count else blind_rescued / blind_count
        ),
        "normal_event_count": normal_event_count,
        "normal_event_alerts": dict(normal_event_alerts),
        "normal_event_alert_rates": {
            method: count / normal_event_count
            for method, count in normal_event_alerts.items()
        },
        "direction_counts": direction_counts,
        "maximum_inner_matched_fpr_absolute_error": max_inner_error,
    }


def run_family_grouped_nested_cv(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    index_root: Path = DEFAULT_INDEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fold_count: int = DEFAULT_FOLD_COUNT,
    folds: Sequence[int] | None = None,
    epochs: int = DEFAULT_EPOCHS,
    eval_epochs: Sequence[int] = DEFAULT_EVAL_EPOCHS,
    seed: int = DEFAULT_SEED,
    build_only: bool = False,
) -> dict[str, Any]:
    indexes = build_family_grouped_indexes(
        metadata_root=metadata_root,
        index_root=index_root,
        fold_count=fold_count,
        seed=seed,
    )
    selected_folds = list(range(fold_count)) if folds is None else list(folds)
    if any(fold < 0 or fold >= fold_count for fold in selected_folds):
        raise ValueError(f"Fold selection out of range: {selected_folds}")
    if build_only:
        return indexes

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fold_results = [
        run_nested_fold(
            fold=fold,
            index_path=Path(indexes["fold_indexes"][str(fold)]),
            output_root=output_root,
            epochs=epochs,
            eval_epochs=eval_epochs,
            seed=seed,
        )
        for fold in selected_folds
    ]
    aggregate = _aggregate_fold_results(fold_results)
    result = {
        "protocol": "riskbench-family-nested-matched-fpr-v1",
        "fold_count": fold_count,
        "completed_folds": selected_folds,
        "epochs": epochs,
        "eval_epochs": list(eval_epochs),
        "seed": seed,
        "index_summary": indexes["summary_path"],
        "aggregate": aggregate,
        "fold_summaries": [
            result["summary_path"] for result in fold_results
        ],
        "claim_scope": (
            "Family-grouped retrospective cross-validation of a mechanism "
            "whose gate forms predate the CV. Outer fold scores are isolated; "
            "a future external dataset remains necessary."
        ),
        "status": (
            "nested_cv_complete"
            if len(selected_folds) == fold_count
            else "nested_cv_partial_smoke"
        ),
    }
    summary_path = output_root / "nested_cv_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_ints(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    return sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )


def main() -> None:
    args = parse_args()
    result = run_family_grouped_nested_cv(
        metadata_root=args.metadata_root,
        index_root=args.index_root,
        output_root=args.output_root,
        fold_count=args.fold_count,
        folds=_parse_ints(args.folds),
        epochs=args.epochs,
        eval_epochs=_parse_ints(args.eval_epochs) or [args.epochs],
        seed=args.seed,
        build_only=args.build_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
