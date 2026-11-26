"""PyTorch Geometric packaging for ego immune-recognition sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

    class Data:  # type: ignore[no-redef]
        """Small torch-saveable fallback when torch_geometric is unavailable."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

from .label_protocol import SequenceDangerLabels
from .schema import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, RoadUserGraph


@dataclass(frozen=True)
class PyGSequencePackage:
    """One packaged graph sequence with its label metadata."""

    scenario_id: str
    graphs: tuple[Data, ...]
    has_physical_danger: bool
    immune_alert_count: int


def graph_to_pyg_data(
    graph: RoadUserGraph,
    labels: SequenceDangerLabels,
    frame_position: int,
) -> Data:
    """Convert one canonical road-user graph into a PyG Data object."""

    frame_label = labels.frame_labels[frame_position]
    edge_presentation_mask = torch.zeros(
        graph.edge_count,
        dtype=torch.bool,
    )
    for source_id, target_id, _score in frame_label.top_presented_edges:
        for edge_position in range(graph.edge_count):
            source_node = int(graph.edge_index[0, edge_position])
            target_node = int(graph.edge_index[1, edge_position])
            if (
                graph.node_ids[source_node] == source_id
                and graph.node_ids[target_node] == target_id
            ):
                edge_presentation_mask[edge_position] = True
                break

    frames_to_anchor = (
        -1
        if frame_label.frames_to_physical_danger is None
        else int(frame_label.frames_to_physical_danger)
    )

    vehicle_ids = torch.tensor(
        [_stable_numeric_id(node_id) for node_id in graph.node_ids],
        dtype=torch.long,
    )

    data = Data(
        x=torch.from_numpy(graph.node_features).float(),
        edge_index=torch.from_numpy(graph.edge_index).long(),
        edge_attr=torch.from_numpy(graph.edge_features).float(),
        node_agent_types=torch.from_numpy(graph.node_agent_types).long(),
        vehicle_ids=vehicle_ids,
        observed_mask=torch.from_numpy(graph.observed_mask),
        valid_mask=torch.from_numpy(graph.valid_mask),
        edge_confidence=torch.from_numpy(graph.edge_confidence).float(),
        edge_presentation_mask=edge_presentation_mask,
        y_immune_alert=torch.tensor(
            [int(frame_label.is_immune_alert)],
            dtype=torch.long,
        ),
        y_physical_danger=torch.tensor(
            [int(frame_label.is_physical_danger)],
            dtype=torch.long,
        ),
        frames_to_physical_danger=torch.tensor(
            [frames_to_anchor],
            dtype=torch.long,
        ),
        max_inverse_ttc_per_s=torch.tensor(
            [frame_label.max_inverse_ttc_per_s],
            dtype=torch.float32,
        ),
        min_distance_m=torch.tensor(
            [
                -1.0
                if frame_label.min_distance_m is None
                else frame_label.min_distance_m
            ],
            dtype=torch.float32,
        ),
        frame_position=torch.tensor([frame_position], dtype=torch.long),
        frame_index=torch.tensor([graph.frame_index], dtype=torch.long),
        timestamp_s=torch.tensor([graph.timestamp_s], dtype=torch.float32),
        num_nodes=graph.node_count,
    )
    data.scenario_id = graph.scenario_id
    data.ego_agent_id = graph.ego_agent_id
    data.node_ids = list(graph.node_ids)
    data.node_feature_names = list(NODE_FEATURE_NAMES)
    data.edge_feature_names = list(EDGE_FEATURE_NAMES)
    return data


def _stable_numeric_id(value: str) -> int:
    """Convert dataset-specific road-user ids to stable int64 ids."""

    try:
        return int(value)
    except ValueError:
        # FNV-1a 63-bit hash keeps non-numeric Waymo-style ids usable for
        # temporal memory without relying on Python's randomized hash().
        result = 1469598103934665603
        for byte in value.encode("utf-8"):
            result ^= byte
            result *= 1099511628211
            result &= (1 << 63) - 1
        return int(result)


def package_pyg_sequence(
    graphs: Sequence[RoadUserGraph],
    labels: SequenceDangerLabels,
) -> PyGSequencePackage:
    """Package one canonical graph sequence and labels into PyG frames."""

    if len(graphs) != len(labels.frame_labels):
        raise ValueError("graphs and labels must have the same length.")
    if not graphs:
        raise ValueError("graphs must not be empty.")

    pyg_graphs = tuple(
        graph_to_pyg_data(graph, labels, frame_position)
        for frame_position, graph in enumerate(graphs)
    )
    return PyGSequencePackage(
        scenario_id=graphs[0].scenario_id,
        graphs=pyg_graphs,
        has_physical_danger=labels.has_physical_danger,
        immune_alert_count=labels.immune_alert_count,
    )


def package_summary(
    sequences: Sequence[PyGSequencePackage],
) -> dict[str, int | float]:
    """Return compact stats for a list of packaged sequences."""

    if not sequences:
        return {
            "sequence_count": 0,
            "frame_count": 0,
            "physical_danger_sequence_count": 0,
            "immune_alert_frame_count": 0,
        }

    frame_count = sum(len(sequence.graphs) for sequence in sequences)
    return {
        "sequence_count": len(sequences),
        "frame_count": frame_count,
        "physical_danger_sequence_count": sum(
            int(sequence.has_physical_danger)
            for sequence in sequences
        ),
        "immune_alert_frame_count": sum(
            sequence.immune_alert_count
            for sequence in sequences
        ),
        "mean_nodes_per_frame": sum(
            graph.num_nodes
            for sequence in sequences
            for graph in sequence.graphs
        )
        / max(frame_count, 1),
        "mean_edges_per_frame": sum(
            graph.edge_index.shape[1]
            for sequence in sequences
            for graph in sequence.graphs
        )
        / max(frame_count, 1),
    }
