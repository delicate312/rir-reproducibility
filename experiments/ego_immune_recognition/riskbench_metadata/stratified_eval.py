"""Evaluate RiskBench model and rule scores by road-user type."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..dataset_io import load_graph_dataset
from ..temporal_relation_train_eval import (
    _NormalizedSequenceView,
    _aggregation_threshold,
    _dual_track_evaluation,
    _evaluate_dual_track,
    _load_training_module,
)
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .lazy_dataset import DEFAULT_OUTPUT_ROOT as DEFAULT_DATASET_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = DEFAULT_DATASET_ROOT / "riskbench_metadata_lazy_index.json"
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_metadata_lazy_300_temporal_v1"
    / "riskbench_temporal_immune_localizer.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_metadata_lazy_300_temporal_v1"
    / "stratified"
)
METHODS = {
    "temporal_causal_ewma": "ego_incident_causal_ewma",
    "same_frame_top3": "top3_mean",
    "inverse_ttc": "inverse_ttc_ego_max",
    "closing_speed": "closing_speed_ego_max",
    "fusion_causal_ttc": "fusion_causal_ttc_percentile_max",
    "fusion_top3_ttc": "fusion_top3_ttc_percentile_max",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--group-count", type=int, default=4)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _event_metadata(package: dict[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for records in package.get("sequence_index", {}).values():
        for record in records:
            category = str(record.get("category", ""))
            scenario_id = str(record.get("scenario_id", ""))
            if not category or not scenario_id:
                continue
            result[f"riskbench:{category}:{scenario_id}"] = {
                "target_object_type": str(record.get("target_type_name", "unknown")),
                "scenario_group_id": str(record.get("scenario_group_id", "")),
            }
    return result


def _load_model_and_sequences(
    dataset_path: Path,
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], torch.device]:
    package = load_graph_dataset(dataset_path)
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    training = _load_training_module()
    config = training.ModelConfig(**checkpoint["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training.build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalization = checkpoint["normalization"]
    normalized_splits = {
        split: _NormalizedSequenceView(
            sequences,
            normalization["x_mean"],
            normalization["x_std"],
            normalization["edge_mean"],
            normalization["edge_std"],
        )
        for split, sequences in package["splits"].items()
        if split != "train_positive"
    }
    return model, normalized_splits, package, device


def _attach_rule_aggregations(frames: list[dict[str, Any]]) -> None:
    for frame in frames:
        for key in ("inverse_ttc_ego_max", "closing_speed_ego_max"):
            frame["aggregations"][key] = float(frame["rule_scores"][key])


def _attach_percentile_fusions(
    frames_by_split: dict[str, list[dict[str, Any]]],
) -> None:
    calibration = frames_by_split["validation_normal"]
    reference = {
        key: np.sort(
            np.asarray(
                [float(frame["aggregations"][key]) for frame in calibration],
                dtype=float,
            )
        )
        for key in (
            "ego_incident_causal_ewma",
            "top3_mean",
            "inverse_ttc_ego_max",
        )
    }

    def percentile(key: str, value: float) -> float:
        values = reference[key]
        return float(np.searchsorted(values, value, side="right") / len(values))

    for frames in frames_by_split.values():
        for frame in frames:
            aggregations = frame["aggregations"]
            ttc = percentile(
                "inverse_ttc_ego_max",
                float(aggregations["inverse_ttc_ego_max"]),
            )
            causal = percentile(
                "ego_incident_causal_ewma",
                float(aggregations["ego_incident_causal_ewma"]),
            )
            top3 = percentile(
                "top3_mean",
                float(aggregations["top3_mean"]),
            )
            aggregations["fusion_causal_ttc_percentile_max"] = max(causal, ttc)
            aggregations["fusion_top3_ttc_percentile_max"] = max(top3, ttc)


def _type_frames(
    frames: list[dict[str, Any]],
    event_metadata: dict[str, dict[str, str]],
    target_type: str,
) -> list[dict[str, Any]]:
    return [
        frame
        for frame in frames
        if event_metadata.get(str(frame["event_uid"]), {}).get(
            "target_object_type"
        )
        == target_type
    ]


def _event_rows(
    frames: list[dict[str, Any]],
    event_metadata: dict[str, dict[str, str]],
    thresholds: dict[str, float],
    *,
    split: str,
    methods: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    methods = methods or METHODS
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        grouped[str(frame["event_uid"])].append(frame)
    rows = []
    for event_uid, event_frames in grouped.items():
        event_frames.sort(key=lambda frame: int(frame["frame_position"]))
        alert_frames = [frame for frame in event_frames if int(frame["original_alert"])]
        row: dict[str, Any] = {
            "split": split,
            "event_uid": event_uid,
            "target_object_type": event_metadata.get(event_uid, {}).get(
                "target_object_type", "unknown"
            ),
            "alert_frame_count": len(alert_frames),
        }
        detections = {}
        for method, aggregation in methods.items():
            detected = [
                frame
                for frame in alert_frames
                if float(frame["aggregations"][aggregation]) >= thresholds[aggregation]
            ]
            detections[method] = bool(detected)
            row[f"{method}_detected"] = int(bool(detected))
            row[f"{method}_peak_alert_score"] = (
                None
                if not alert_frames
                else max(
                    float(frame["aggregations"][aggregation])
                    for frame in alert_frames
                )
            )
            row[f"{method}_lead_seconds"] = (
                None
                if not detected
                else max(int(detected[0]["frames_to_anchor"]), 0)
                * RISK_BENCH_TIMESTEP_SECONDS
            )
        model_hit = detections["temporal_causal_ewma"]
        ttc_hit = detections["inverse_ttc"]
        row["model_vs_ttc"] = (
            "both"
            if model_hit and ttc_hit
            else "model_only"
            if model_hit
            else "ttc_only"
            if ttc_hit
            else "neither"
        )
        rows.append(row)
    return rows


def run_stratified_evaluation(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    group_count: int = 4,
) -> dict[str, Any]:
    model, splits, package, device = _load_model_and_sequences(
        dataset_path,
        checkpoint_path,
    )
    result = _dual_track_evaluation(
        model,
        validation_normal=splits["validation_normal"],
        validation_positive=splits["validation_positive"],
        test_normal=splits["test_normal"],
        test_positive=splits["test_positive"],
        device=device,
        group_count=group_count,
        threshold_quantile=0.95,
        track_definitions=[
            {
                "track": "official_early_alert_1s_causal",
                "aggregation": "ego_incident_causal_ewma",
                "label_window": "original",
            },
            {
                "track": "official_early_alert_1s_top3",
                "aggregation": "top3_mean",
                "label_window": "original",
            },
        ],
        timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
        return_frames=True,
    )
    frames = result["frames"]
    for split_frames in frames.values():
        _attach_rule_aggregations(split_frames)
    _attach_percentile_fusions(frames)
    thresholds = {
        aggregation: _aggregation_threshold(
            frames["validation_normal"],
            aggregation=aggregation,
            quantile=0.95,
        )
        for aggregation in METHODS.values()
    }
    metadata = _event_metadata(package)
    target_types = sorted(
        {
            values["target_object_type"]
            for values in metadata.values()
            if values["target_object_type"] != "none"
        }
    )
    metric_rows = []
    for split in ("validation_positive", "test_positive"):
        normal_split = (
            "validation_normal" if split == "validation_positive" else "test_normal"
        )
        for method, aggregation in METHODS.items():
            metrics = _evaluate_dual_track(
                normal_frames=frames[normal_split],
                positive_frames=frames[split],
                aggregation=aggregation,
                label_window="original",
                threshold=thresholds[aggregation],
                timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
            )
            metric_rows.append(
                {
                    "split": split,
                    "target_object_type": "ALL",
                    "method": method,
                    "matched_normal_event_count": len(
                        {str(frame["event_uid"]) for frame in frames[normal_split]}
                    ),
                    **metrics,
                }
            )
        for target_type in target_types:
            normal = _type_frames(frames[normal_split], metadata, target_type)
            positive = _type_frames(frames[split], metadata, target_type)
            if not normal or not positive:
                continue
            for method, aggregation in METHODS.items():
                metrics = _evaluate_dual_track(
                    normal_frames=normal,
                    positive_frames=positive,
                    aggregation=aggregation,
                    label_window="original",
                    threshold=thresholds[aggregation],
                    timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
                )
                metric_rows.append(
                    {
                        "split": split,
                        "target_object_type": target_type,
                        "method": method,
                        "matched_normal_event_count": len(
                            {str(frame["event_uid"]) for frame in normal}
                        ),
                        **metrics,
                    }
                )

    event_rows = []
    for split in ("validation_positive", "test_positive"):
        event_rows.extend(
            _event_rows(
                frames[split],
                metadata,
                thresholds,
                split=split,
            )
        )
    detection_counts = {
        split: {
            target_type: dict(
                Counter(
                    row["model_vs_ttc"]
                    for row in event_rows
                    if row["split"] == split
                    and row["target_object_type"] == target_type
                )
            )
            for target_type in target_types
        }
        for split in ("validation_positive", "test_positive")
    }
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "target_type_metrics.csv"
    events_path = output_root / "event_detection_pairs.csv"
    summary_path = output_root / "stratified_summary.json"
    _write_csv(metrics_path, metric_rows)
    _write_csv(events_path, event_rows)
    summary = {
        "protocol": "riskbench-target-type-same-input-stratified-v1",
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "checkpoint_path": str(checkpoint_path.expanduser().resolve()),
        "threshold_source": "global_validation_hard_safe_95th_percentile",
        "thresholds": thresholds,
        "target_types": target_types,
        "detection_pair_counts": detection_counts,
        "metric_rows": metric_rows,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "events_csv": str(events_path),
        },
        "status": "stratified_diagnostic_not_final_claim",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    result = run_stratified_evaluation(
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        output_root=args.output_root,
        group_count=args.group_count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
