"""Build a small metadata-only RiskBench three-pool PyG dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..label_protocol import FrameDangerLabel, SequenceDangerLabels
from ..pyg_packager import package_pyg_sequence, package_summary
from ..schema import EDGE_FEATURE_NAMES, RoadUserGraph
from .adapter import (
    DEFAULT_METADATA_ROOT,
    RiskBenchScenarioRecord,
    build_riskbench_graph_sequence,
)
from .manifest import DEFAULT_OUTPUT_ROOT, build_manifest


INVERSE_TTC_INDEX = EDGE_FEATURE_NAMES.index("inverse_ttc_per_s")
DISTANCE_INDEX = EDGE_FEATURE_NAMES.index("distance_m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-normal", type=int, default=12)
    parser.add_argument("--validation-normal", type=int, default=6)
    parser.add_argument("--test-normal", type=int, default=6)
    parser.add_argument("--validation-positive", type=int, default=6)
    parser.add_argument("--test-positive", type=int, default=6)
    parser.add_argument("--pre-critical-frames", type=int, default=40)
    parser.add_argument("--post-critical-frames", type=int, default=20)
    return parser.parse_args()


def _balanced_records(
    records: Sequence[RiskBenchScenarioRecord],
    split: str,
    count: int,
) -> list[RiskBenchScenarioRecord]:
    candidates = [record for record in records if record.split == split]
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} records for {split}, requested {count}")
    candidates.sort(
        key=lambda record: (
            -record.actor_count,
            -record.frame_count,
            record.scenario_id,
        )
    )
    by_type: dict[str, list[RiskBenchScenarioRecord]] = {}
    seen_groups = set()
    for record in candidates:
        if record.scenario_group_id in seen_groups:
            continue
        seen_groups.add(record.scenario_group_id)
        by_type.setdefault(record.target_type_name, []).append(record)
    type_names = sorted(by_type)
    selected: list[RiskBenchScenarioRecord] = []
    while len(selected) < count:
        made_progress = False
        for type_name in type_names:
            if by_type[type_name] and len(selected) < count:
                selected.append(by_type[type_name].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def _incident_scores(graph: RoadUserGraph) -> tuple[float, float | None]:
    if graph.edge_count == 0:
        return 0.0, None
    ego_position = graph.node_ids.index(graph.ego_agent_id)
    mask = (
        (graph.edge_index[0] == ego_position)
        | (graph.edge_index[1] == ego_position)
    )
    if not np.any(mask):
        return 0.0, None
    features = graph.edge_features[mask]
    return (
        float(features[:, INVERSE_TTC_INDEX].max(initial=0.0)),
        float(features[:, DISTANCE_INDEX].min(initial=np.inf)),
    )


def _target_presented_edges(
    graph: RoadUserGraph,
    risk_actor_ids: set[str],
) -> tuple[tuple[str, str, float], ...]:
    candidates = []
    for edge_position in range(graph.edge_count):
        source = graph.node_ids[int(graph.edge_index[0, edge_position])]
        target = graph.node_ids[int(graph.edge_index[1, edge_position])]
        if graph.ego_agent_id not in {source, target}:
            continue
        other = target if source == graph.ego_agent_id else source
        if other not in risk_actor_ids:
            continue
        edge = graph.edge_features[edge_position]
        score = float(edge[INVERSE_TTC_INDEX] + 0.05 / max(edge[DISTANCE_INDEX], 1e-3))
        candidates.append((source, target, score))
    candidates.sort(key=lambda value: -value[2])
    return tuple(candidates[:3])


def _official_labels(
    graphs: Sequence[RoadUserGraph],
    record: RiskBenchScenarioRecord,
    *,
    alert_lead_frames: int = 20,
) -> SequenceDangerLabels:
    anchor_position = None
    if record.category == "collision" and record.critical_frame is not None:
        for position, graph in enumerate(graphs):
            if graph.frame_index == record.critical_frame:
                anchor_position = position
                break
    risk_ids = set(record.risk_actor_ids)
    labels = []
    for position, graph in enumerate(graphs):
        inverse_ttc, min_distance = _incident_scores(graph)
        frames_to_anchor = None if anchor_position is None else anchor_position - position
        is_alert = bool(
            anchor_position is not None
            and max(0, anchor_position - alert_lead_frames) <= position < anchor_position
        )
        is_physical = bool(anchor_position is not None and position >= anchor_position)
        labels.append(
            FrameDangerLabel(
                frame_index=graph.frame_index,
                is_physical_danger=is_physical,
                is_immune_alert=is_alert,
                frames_to_physical_danger=frames_to_anchor,
                max_inverse_ttc_per_s=inverse_ttc,
                min_distance_m=min_distance,
                top_presented_edges=_target_presented_edges(graph, risk_ids),
            )
        )
    return SequenceDangerLabels(
        scenario_id=graphs[0].scenario_id,
        anchor_frame_position=anchor_position,
        frame_labels=tuple(labels),
        source_metadata={
            "protocol": "riskbench-official-critical-point-early-alert-v0",
            "official_category": record.category,
            "official_critical_frame": record.critical_frame,
            "alert_lead_frames": alert_lead_frames,
            "hard_safe_interactive_is_normal": record.category == "interactive",
        },
    )


def _attach_metadata(
    packaged: Any,
    record: RiskBenchScenarioRecord,
) -> None:
    for graph in packaged.graphs:
        graph.event_uid = f"riskbench:{record.category}:{record.scenario_id}"
        graph.pool_role = record.pool_role
        graph.riskbench_category = record.category
        graph.riskbench_source_scenario_id = record.scenario_id
        graph.riskbench_scenario_group_id = record.scenario_group_id
        graph.risk_actor_ids = list(record.risk_actor_ids)
        graph.target_object_type = record.target_type_name
        graph.critical_frame = record.critical_frame
        graph.behavior_window = record.behavior_window
        graph.metadata_only = True
        if record.risk_actor_ids:
            graph.correct_relation_pair = [
                str(graph.ego_agent_id),
                str(record.risk_actor_ids[0]),
            ]


def _package_record(
    record: RiskBenchScenarioRecord,
    *,
    pre_critical_frames: int,
    post_critical_frames: int,
) -> Any:
    adapted = build_riskbench_graph_sequence(
        record,
        pre_critical_frames=(
            pre_critical_frames if record.critical_frame is not None else None
        ),
        post_critical_frames=(
            post_critical_frames if record.critical_frame is not None else None
        ),
    )
    labels = _official_labels(adapted.graphs, record)
    packaged = package_pyg_sequence(adapted.graphs, labels)
    _attach_metadata(packaged, record)
    return packaged


def build_smoke_dataset(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    split_counts: dict[str, int] | None = None,
    pre_critical_frames: int = 40,
    post_critical_frames: int = 20,
) -> dict[str, Any]:
    split_counts = split_counts or {
        "train_normal": 12,
        "validation_normal": 6,
        "test_normal": 6,
        "validation_positive": 6,
        "test_positive": 6,
    }
    records, manifest_summary = build_manifest(metadata_root, output_root)
    selected_records = {
        split: _balanced_records(records, split, count)
        for split, count in split_counts.items()
    }
    packaged = {
        split: [
            _package_record(
                record,
                pre_critical_frames=pre_critical_frames,
                post_critical_frames=post_critical_frames,
            )
            for record in split_records
        ]
        for split, split_records in selected_records.items()
    }
    packaged["train_positive"] = []
    sequence_index = {
        split: [
            {
                "scenario_id": record.scenario_id,
                "scenario_group_id": record.scenario_group_id,
                "category": record.category,
                "pool_role": record.pool_role,
                "target_type_name": record.target_type_name,
                "risk_actor_ids": list(record.risk_actor_ids),
                "critical_frame": record.critical_frame,
                "source_frame_count": record.frame_count,
                "packaged_frame_count": len(sequence.graphs),
            }
            for record, sequence in zip(selected_records.get(split, []), sequences)
        ]
        for split, sequences in packaged.items()
    }
    output_root = output_root.expanduser().resolve()
    dataset_path = output_root / "riskbench_metadata_smoke_dataset.pt"
    payload = {
        "splits": {
            split: [sequence.graphs for sequence in sequences]
            for split, sequences in packaged.items()
        },
        "sequence_index": sequence_index,
        "meta": {
            "dataset_name": "RiskBench metadata-only ego immune recognition",
            "protocol": "riskbench-metadata-three-pool-smoke-v0",
            "schema_version": "ego-road-user-graph-v0",
            "metadata_only": True,
            "full_sensor_archives_required": False,
            "gradient_allowed_splits": ["train_normal"],
            "positive_gradient_sequence_count": 0,
            "pre_critical_frames": pre_critical_frames,
            "post_critical_frames": post_critical_frames,
        },
    }
    torch.save(payload, dataset_path)
    split_summaries = {
        split: package_summary(sequences)
        for split, sequences in packaged.items()
    }
    summary = {
        "protocol": "riskbench-metadata-three-pool-smoke-v0",
        "dataset_path": str(dataset_path),
        "manifest_summary_path": manifest_summary["summary_path"],
        "manifest_scenario_count": manifest_summary["scenario_count"],
        "requested_split_counts": split_counts,
        "split_summaries": split_summaries,
        "sample_category_counts": {
            split: dict(Counter(record.category for record in split_records))
            for split, split_records in selected_records.items()
        },
        "sample_target_type_counts": {
            split: dict(Counter(record.target_type_name for record in split_records))
            for split, split_records in selected_records.items()
        },
        "full_sensor_archives_required": False,
        "status": "metadata_adapter_smoke_ready_not_final_training",
    }
    summary_path = output_root / "smoke_dataset_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    split_counts = {
        "train_normal": args.train_normal,
        "validation_normal": args.validation_normal,
        "test_normal": args.test_normal,
        "validation_positive": args.validation_positive,
        "test_positive": args.test_positive,
    }
    summary = build_smoke_dataset(
        metadata_root=args.metadata_root,
        output_root=args.output_root,
        split_counts=split_counts,
        pre_critical_frames=args.pre_critical_frames,
        post_critical_frames=args.post_critical_frames,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
