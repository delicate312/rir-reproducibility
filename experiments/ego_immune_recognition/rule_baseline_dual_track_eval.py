"""Dual-track evaluation for TTC, inverse-distance, and closing-speed baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset_io import load_graph_dataset
from .schema import EDGE_FEATURE_NAMES
from .temporal_relation_train_eval import (
    DUAL_TRACK_DEFINITIONS,
    PROJECT_ROOT,
    _frame_auc_views,
    _label_window_mask_from_frames,
    _rate_above,
)


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_us101_counterfactual_dataset_v3"
    / "us101_ego_immune_counterfactual_dataset.pt"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "ego_immune_rule_baselines"
RULE_AGGREGATIONS = [
    "inverse_ttc_all_max",
    "inverse_ttc_ego_max",
    "inverse_distance_all_max",
    "inverse_distance_ego_max",
    "closing_speed_all_max",
    "closing_speed_ego_max",
]
EPSILON = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TTC / distance rule baselines under dual-track windows."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_rule_baseline_dual_track_eval(
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        threshold_quantile=args.threshold_quantile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_rule_baseline_dual_track_eval(
    *,
    dataset_path: Path,
    output_root: Path,
    threshold_quantile: float,
    track_definitions: list[dict[str, Any]] | None = None,
    timestep_seconds: float = 0.1,
) -> dict[str, Any]:
    if timestep_seconds <= 0.0:
        raise ValueError("timestep_seconds must be positive")
    track_definitions = track_definitions or DUAL_TRACK_DEFINITIONS
    package = load_graph_dataset(dataset_path)
    splits = package["splits"]

    validation_normal_frames = score_rule_frames(
        splits["validation_normal"],
        split="validation_normal",
    )
    test_normal_frames = score_rule_frames(
        splits["test_normal"],
        split="test_normal",
    )
    validation_positive_frames = score_rule_frames(
        splits["validation_positive"],
        split="validation_positive",
    )
    test_positive_frames = score_rule_frames(
        splits["test_positive"],
        split="test_positive",
    )
    thresholds = {
        aggregation: _aggregation_threshold(
            validation_normal_frames,
            aggregation=aggregation,
            quantile=threshold_quantile,
        )
        for aggregation in RULE_AGGREGATIONS
    }

    rows = []
    rows.extend(
        evaluate_rule_split(
            split="validation_positive",
            normal_frames=validation_normal_frames,
            positive_frames=validation_positive_frames,
            thresholds=thresholds,
            track_definitions=track_definitions,
            timestep_seconds=timestep_seconds,
        )
    )
    rows.extend(
        evaluate_rule_split(
            split="test_positive",
            normal_frames=test_normal_frames,
            positive_frames=test_positive_frames,
            thresholds=thresholds,
            track_definitions=track_definitions,
            timestep_seconds=timestep_seconds,
        )
    )

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "rule_dual_track_metrics.csv"
    summary_path = output_root / "rule_dual_track_summary.json"
    _write_csv(csv_path, rows)

    summary = {
        "dataset_path": str(dataset_path),
        "protocol": "rule-baseline-dual-track-causal-v1",
        "primary_frame_auroc": "alert_frames_vs_independent_normal_frames",
        "threshold_source": (
            f"validation_normal_{threshold_quantile:.2f}_quantile_per_rule"
        ),
        "thresholds": thresholds,
        "track_definitions": track_definitions,
        "timestep_seconds": timestep_seconds,
        "rule_definitions": _rule_definitions(),
        "best_rows_by_event_recall": _best_rows(rows, key="event_recall_at_threshold"),
        "best_rows_by_frame_auroc": _best_rows(rows, key="frame_auroc"),
        "strict_original_rows": [
            row
            for row in rows
            if row["track"] == "strict_original_window"
        ],
        "artifacts": {
            "metrics_csv": str(csv_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def score_rule_frames(
    sequences: list[list[Any]],
    *,
    split: str,
) -> list[dict[str, Any]]:
    frames = []
    for event_index, sequence in enumerate(sequences):
        for frame_position, graph in enumerate(sequence):
            frames.append(
                {
                    "split": split,
                    "event_index": event_index,
                    "scenario_id": str(graph.scenario_id),
                    "frame_position": frame_position,
                    "frames_to_anchor": int(graph.frames_to_physical_danger.item()),
                    "original_alert": int(graph.y_immune_alert.item()),
                    "physical": int(graph.y_physical_danger.item()),
                    "aggregations": rule_scores_for_graph(graph),
                }
            )
    return frames


def rule_scores_for_graph(graph: Any) -> dict[str, float]:
    edge_attr = graph.edge_attr.detach().cpu().float()
    if edge_attr.numel() == 0:
        return {aggregation: 0.0 for aggregation in RULE_AGGREGATIONS}

    distance = edge_attr[:, _edge_feature_index("distance_m")]
    closing_speed = edge_attr[:, _edge_feature_index("closing_speed_mps")]
    inverse_ttc = edge_attr[:, _edge_feature_index("inverse_ttc_per_s")]
    inverse_distance = 1.0 / torch.clamp(distance, min=EPSILON)
    ego_mask = _ego_incident_edge_mask(graph)

    return {
        "inverse_ttc_all_max": _max_or_zero(inverse_ttc),
        "inverse_ttc_ego_max": _max_or_zero(inverse_ttc[ego_mask]),
        "inverse_distance_all_max": _max_or_zero(inverse_distance),
        "inverse_distance_ego_max": _max_or_zero(inverse_distance[ego_mask]),
        "closing_speed_all_max": _max_or_zero(closing_speed),
        "closing_speed_ego_max": _max_or_zero(closing_speed[ego_mask]),
    }


def evaluate_rule_split(
    *,
    split: str,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    thresholds: dict[str, float],
    track_definitions: list[dict[str, Any]] | None = None,
    timestep_seconds: float = 0.1,
) -> list[dict[str, Any]]:
    track_definitions = track_definitions or DUAL_TRACK_DEFINITIONS
    rows = []
    for definition in track_definitions:
        label_window = str(definition["label_window"])
        track = str(definition["track"])
        for aggregation in RULE_AGGREGATIONS:
            rows.append(
                evaluate_rule_track(
                    split=split,
                    track=track,
                    aggregation=aggregation,
                    label_window=label_window,
                    normal_frames=normal_frames,
                    positive_frames=positive_frames,
                    threshold=thresholds[aggregation],
                    timestep_seconds=timestep_seconds,
                )
            )
    return rows


def evaluate_rule_track(
    *,
    split: str,
    track: str,
    aggregation: str,
    label_window: str,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    threshold: float,
    timestep_seconds: float = 0.1,
) -> dict[str, Any]:
    normal_scores = [
        float(frame["aggregations"][aggregation])
        for frame in normal_frames
    ]
    positive_scores = [
        float(frame["aggregations"][aggregation])
        for frame in positive_frames
    ]
    positive_labels = []
    event_leads = []
    detected_events = 0
    grouped = _group_by_event(positive_frames)
    for event_frames in grouped.values():
        mask = _label_window_mask_from_frames(event_frames, label_window)
        positive_labels.extend(int(value) for value in mask)
        detected_positions = [
            index
            for index, keep in enumerate(mask)
            if keep and float(event_frames[index]["aggregations"][aggregation]) >= threshold
        ]
        if detected_positions:
            detected_events += 1
            first = detected_positions[0]
            event_leads.append(max(int(event_frames[first]["frames_to_anchor"]), 0))

    alert_scores = [
        score
        for score, label in zip(positive_scores, positive_labels)
        if label == 1
    ]
    background_scores = [
        score
        for score, label in zip(positive_scores, positive_labels)
        if label == 0
    ]
    auc_views = _frame_auc_views(
        normal_scores=normal_scores,
        positive_scores=positive_scores,
        positive_labels=positive_labels,
    )
    return {
        "split": split,
        "track": track,
        "aggregation": aggregation,
        "label_window": label_window,
        "threshold": threshold,
        "frame_count": len(normal_scores) + len(alert_scores),
        "combined_frame_count": len(normal_scores) + len(positive_scores),
        "within_event_frame_count": len(positive_scores),
        "event_count": len(grouped),
        "positive_alert_frame_count": int(sum(positive_labels)),
        "positive_background_frame_count": len(background_scores),
        "frame_auroc": auc_views["independent_normal"],
        "frame_auroc_independent_normal": auc_views["independent_normal"],
        "frame_auroc_within_event": auc_views["within_event"],
        "frame_auroc_combined_legacy": auc_views["combined_legacy"],
        "normal_false_alarm_rate_at_threshold": _rate_above(
            normal_scores,
            threshold,
        ),
        "positive_alert_recall_at_threshold": _rate_above(
            alert_scores,
            threshold,
        ),
        "event_recall_at_threshold": (
            None if not grouped else float(detected_events / len(grouped))
        ),
        "mean_lead_seconds": (
            None
            if not event_leads
            else float(np.mean(event_leads) * timestep_seconds)
        ),
        "median_lead_seconds": (
            None
            if not event_leads
            else float(np.median(event_leads) * timestep_seconds)
        ),
        "mean_normal_score": (
            None if not normal_scores else float(np.mean(normal_scores))
        ),
        "mean_alert_score": (
            None if not alert_scores else float(np.mean(alert_scores))
        ),
    }


def _edge_feature_index(name: str) -> int:
    return EDGE_FEATURE_NAMES.index(name)


def _ego_incident_edge_mask(graph: Any) -> torch.Tensor:
    edge_index = graph.edge_index.detach().cpu()
    if edge_index.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool)
    ego_id = str(graph.ego_agent_id)
    keep = []
    for edge_position in range(int(edge_index.shape[1])):
        source = str(graph.node_ids[int(edge_index[0, edge_position])])
        target = str(graph.node_ids[int(edge_index[1, edge_position])])
        keep.append(source == ego_id or target == ego_id)
    return torch.tensor(keep, dtype=torch.bool)


def _max_or_zero(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.max(values).item())


def _aggregation_threshold(
    frames: list[dict[str, Any]],
    *,
    aggregation: str,
    quantile: float,
) -> float:
    values = [
        float(frame["aggregations"][aggregation])
        for frame in frames
    ]
    if not values:
        return float("inf")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _group_by_event(frames: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        grouped.setdefault(int(frame["event_index"]), []).append(frame)
    for event_frames in grouped.values():
        event_frames.sort(key=lambda frame: int(frame["frame_position"]))
    return grouped


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _best_rows(
    rows: list[dict[str, Any]],
    *,
    key: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    clean_rows = [row for row in rows if row.get(key) is not None]
    return sorted(clean_rows, key=lambda row: float(row[key]), reverse=True)[:limit]


def _rule_definitions() -> dict[str, str]:
    return {
        "inverse_ttc_all_max": "Maximum inverse TTC over all visible graph edges.",
        "inverse_ttc_ego_max": "Maximum inverse TTC over ego-incident visible edges.",
        "inverse_distance_all_max": "Maximum inverse distance over all visible graph edges.",
        "inverse_distance_ego_max": "Maximum inverse distance over ego-incident visible edges.",
        "closing_speed_all_max": "Maximum closing speed over all visible graph edges.",
        "closing_speed_ego_max": "Maximum closing speed over ego-incident visible edges.",
    }


if __name__ == "__main__":
    main()
