"""Early-danger and immune-alert labeling protocol.

The protocol separates two ideas:

1. Physical danger anchor:
   a conservative, rule-based moment where an ego-related physical risk becomes
   explicit.  TTC-style rules are useful here as an event anchor.
2. Immune alert window:
   frames before the physical anchor where a model should recognize that the
   local interaction grammar is becoming suspicious.

This module does not define danger types.  It produces class-agnostic labels
for early danger recognition and optional relation-presentation candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .schema import EDGE_FEATURE_NAMES, RoadUserGraph


INVERSE_TTC_INDEX = EDGE_FEATURE_NAMES.index("inverse_ttc_per_s")
DISTANCE_INDEX = EDGE_FEATURE_NAMES.index("distance_m")


@dataclass(frozen=True)
class EarlyDangerLabelConfig:
    """Configuration for class-agnostic immune-alert labels."""

    physical_inverse_ttc_threshold_per_s: float = 0.5
    critical_distance_m: float = 1.5
    alert_lead_frames: int = 20
    alert_horizon_frames: int = 1
    top_presented_edges: int = 3
    ego_only: bool = True

    def __post_init__(self) -> None:
        if self.physical_inverse_ttc_threshold_per_s < 0.0:
            raise ValueError("physical inverse TTC threshold must be >= 0.")
        if self.critical_distance_m <= 0.0:
            raise ValueError("critical_distance_m must be positive.")
        if self.alert_lead_frames < 0:
            raise ValueError("alert_lead_frames must be non-negative.")
        if self.alert_horizon_frames < 0:
            raise ValueError("alert_horizon_frames must be non-negative.")
        if self.top_presented_edges <= 0:
            raise ValueError("top_presented_edges must be positive.")


@dataclass(frozen=True)
class FrameDangerLabel:
    """Frame-level label produced by the immune-alert protocol."""

    frame_index: int
    is_physical_danger: bool
    is_immune_alert: bool
    frames_to_physical_danger: int | None
    max_inverse_ttc_per_s: float
    min_distance_m: float | None
    top_presented_edges: tuple[tuple[str, str, float], ...]


@dataclass(frozen=True)
class SequenceDangerLabels:
    """Labels for a graph sequence."""

    scenario_id: str
    anchor_frame_position: int | None
    frame_labels: tuple[FrameDangerLabel, ...]
    source_metadata: dict[str, Any]

    @property
    def has_physical_danger(self) -> bool:
        return self.anchor_frame_position is not None

    @property
    def immune_alert_count(self) -> int:
        return sum(int(label.is_immune_alert) for label in self.frame_labels)

    @property
    def physical_danger_count(self) -> int:
        return sum(int(label.is_physical_danger) for label in self.frame_labels)

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "has_physical_danger": self.has_physical_danger,
            "anchor_frame_position": self.anchor_frame_position,
            "frame_count": len(self.frame_labels),
            "immune_alert_count": self.immune_alert_count,
            "physical_danger_count": self.physical_danger_count,
            "max_inverse_ttc_per_s": max(
                label.max_inverse_ttc_per_s for label in self.frame_labels
            )
            if self.frame_labels
            else 0.0,
            "min_distance_m": min(
                label.min_distance_m
                for label in self.frame_labels
                if label.min_distance_m is not None
            )
            if any(label.min_distance_m is not None for label in self.frame_labels)
            else None,
        }


def label_early_danger_sequence(
    graphs: Sequence[RoadUserGraph],
    config: EarlyDangerLabelConfig | None = None,
) -> SequenceDangerLabels:
    """Label one ego-centric graph sequence with early immune-alert targets."""

    if not graphs:
        raise ValueError("graphs must contain at least one RoadUserGraph.")

    config = config or EarlyDangerLabelConfig()
    scenario_id = graphs[0].scenario_id
    if any(graph.scenario_id != scenario_id for graph in graphs):
        raise ValueError("All graphs must share scenario_id.")

    frame_scores = [
        _frame_physical_scores(graph, config)
        for graph in graphs
    ]
    anchor_frame_position = _first_physical_anchor(frame_scores, config)

    labels: list[FrameDangerLabel] = []
    for position, (graph, scores) in enumerate(zip(graphs, frame_scores)):
        is_physical_danger = (
            anchor_frame_position is not None
            and position >= anchor_frame_position
            and _is_physical_danger(scores, config)
        )
        is_immune_alert = False
        frames_to_anchor: int | None = None
        if anchor_frame_position is not None:
            frames_to_anchor = anchor_frame_position - position
            alert_start = max(0, anchor_frame_position - config.alert_lead_frames)
            alert_end = max(0, anchor_frame_position - config.alert_horizon_frames)
            is_immune_alert = alert_start <= position <= alert_end

        labels.append(
            FrameDangerLabel(
                frame_index=graph.frame_index,
                is_physical_danger=is_physical_danger,
                is_immune_alert=is_immune_alert,
                frames_to_physical_danger=frames_to_anchor,
                max_inverse_ttc_per_s=scores["max_inverse_ttc_per_s"],
                min_distance_m=scores["min_distance_m"],
                top_presented_edges=_top_presented_edges(
                    graph,
                    config,
                ),
            )
        )

    return SequenceDangerLabels(
        scenario_id=scenario_id,
        anchor_frame_position=anchor_frame_position,
        frame_labels=tuple(labels),
        source_metadata={
            "protocol": "early-danger-immune-alert-v0",
            "physical_inverse_ttc_threshold_per_s": (
                config.physical_inverse_ttc_threshold_per_s
            ),
            "critical_distance_m": config.critical_distance_m,
            "alert_lead_frames": config.alert_lead_frames,
            "alert_horizon_frames": config.alert_horizon_frames,
            "ego_only": config.ego_only,
        },
    )


def _first_physical_anchor(
    frame_scores: Sequence[dict[str, float | None]],
    config: EarlyDangerLabelConfig,
) -> int | None:
    for position, scores in enumerate(frame_scores):
        if _is_physical_danger(scores, config):
            return position
    return None


def _is_physical_danger(
    scores: dict[str, float | None],
    config: EarlyDangerLabelConfig,
) -> bool:
    min_distance = scores["min_distance_m"]
    return (
        scores["max_inverse_ttc_per_s"]
        >= config.physical_inverse_ttc_threshold_per_s
        or (
            min_distance is not None
            and min_distance <= config.critical_distance_m
        )
    )


def _frame_physical_scores(
    graph: RoadUserGraph,
    config: EarlyDangerLabelConfig,
) -> dict[str, float | None]:
    edge_mask = _scored_edge_mask(graph, config)
    if not np.any(edge_mask):
        return {
            "max_inverse_ttc_per_s": 0.0,
            "min_distance_m": None,
        }

    edge_features = graph.edge_features[edge_mask]
    return {
        "max_inverse_ttc_per_s": float(
            edge_features[:, INVERSE_TTC_INDEX].max(initial=0.0)
        ),
        "min_distance_m": float(
            edge_features[:, DISTANCE_INDEX].min(initial=np.inf)
        ),
    }


def _top_presented_edges(
    graph: RoadUserGraph,
    config: EarlyDangerLabelConfig,
) -> tuple[tuple[str, str, float], ...]:
    edge_mask = _scored_edge_mask(graph, config)
    if not np.any(edge_mask):
        return tuple()

    scored_indices = np.where(edge_mask)[0]
    edge_features = graph.edge_features[scored_indices]
    distance = np.maximum(edge_features[:, DISTANCE_INDEX], 1e-3)
    inverse_ttc = edge_features[:, INVERSE_TTC_INDEX]

    # A simple presentation score: explicit TTC risk first, proximity second.
    presentation_score = inverse_ttc + 0.05 / distance
    order = np.argsort(-presentation_score)

    presented: list[tuple[str, str, float]] = []
    for edge_position in scored_indices[order[: config.top_presented_edges]]:
        source_node = int(graph.edge_index[0, edge_position])
        target_node = int(graph.edge_index[1, edge_position])
        presented.append(
            (
                graph.node_ids[source_node],
                graph.node_ids[target_node],
                float(presentation_score[
                    np.where(scored_indices == edge_position)[0][0]
                ]),
            )
        )
    return tuple(presented)


def _scored_edge_mask(
    graph: RoadUserGraph,
    config: EarlyDangerLabelConfig,
) -> np.ndarray:
    if graph.edge_count == 0:
        return np.zeros((0,), dtype=bool)
    if not config.ego_only:
        return np.ones((graph.edge_count,), dtype=bool)

    try:
        ego_node = graph.node_ids.index(graph.ego_agent_id)
    except ValueError as error:
        raise ValueError("Graph ego_agent_id is not present in node_ids.") from error

    return (
        (graph.edge_index[0] == ego_node)
        | (graph.edge_index[1] == ego_node)
    )
