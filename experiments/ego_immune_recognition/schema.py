"""Dataset-agnostic ego-centric schema for immune danger recognition.

The schema is intentionally independent from US-101, Waymo, InterHub, PyTorch,
and PyG.  Dataset adapters should convert raw data into:

    Scenario -> EgoFrame -> RoadUserGraph

US-101 can fill every road user as ``vehicle``.  Waymo-style datasets can fill
vehicles, pedestrians, cyclists, and unknown agents without changing the model
interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Sequence

import numpy as np


class AgentType(IntEnum):
    """Canonical road-user classes shared by all adapters."""

    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    MOTORCYCLIST = 4


AGENT_TYPE_NAMES = [
    "unknown",
    "vehicle",
    "pedestrian",
    "cyclist",
    "motorcyclist",
]

NODE_FEATURE_NAMES = [
    "ego_relative_x_m",
    "ego_relative_y_m",
    "velocity_x_mps",
    "velocity_y_mps",
    "speed_mps",
    "heading_sin",
    "heading_cos",
    "length_m",
    "width_m",
    "height_m",
    "is_unknown",
    "is_vehicle",
    "is_pedestrian",
    "is_cyclist",
    "is_motorcyclist",
    "observed_indicator",
    "observation_confidence",
]

EDGE_FEATURE_NAMES = [
    "relative_x_m",
    "relative_y_m",
    "relative_vx_mps",
    "relative_vy_mps",
    "distance_m",
    "closing_speed_mps",
    "inverse_ttc_per_s",
    "source_observation_confidence",
    "target_observation_confidence",
]

MIN_DISTANCE_M = 1e-3
MIN_CLOSING_SPEED_MPS = 1e-3
INVERSE_TTC_MAX_PER_S = 5.0


def _array2(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def _optional_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _one_hot_agent_type(agent_type: AgentType) -> np.ndarray:
    one_hot = np.zeros(len(AGENT_TYPE_NAMES), dtype=np.float32)
    one_hot[int(agent_type)] = 1.0
    return one_hot


@dataclass(frozen=True)
class RoadUserState:
    """State of one road user from the ego vehicle's observation frame."""

    agent_id: str
    agent_type: AgentType
    position_xy_m: Sequence[float]
    velocity_xy_mps: Sequence[float] = (0.0, 0.0)
    heading_rad: float | None = None
    length_m: float = 0.0
    width_m: float = 0.0
    height_m: float = 0.0
    valid: bool = True
    observed: bool = True
    observation_confidence: float = 1.0
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            str(self.agent_id),
        )
        object.__setattr__(
            self,
            "agent_type",
            AgentType(self.agent_type),
        )
        object.__setattr__(
            self,
            "position_xy_m",
            _array2(self.position_xy_m, "position_xy_m"),
        )
        object.__setattr__(
            self,
            "velocity_xy_mps",
            _array2(self.velocity_xy_mps, "velocity_xy_mps"),
        )
        object.__setattr__(
            self,
            "heading_rad",
            _optional_float(self.heading_rad, "heading_rad"),
        )

        for field_name in ["length_m", "width_m", "height_m"]:
            value = float(getattr(self, field_name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{field_name} must be a finite value >= 0.")
            object.__setattr__(self, field_name, value)

        confidence = float(self.observation_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("observation_confidence must be in [0, 1].")
        object.__setattr__(self, "observation_confidence", confidence)

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity_xy_mps))


@dataclass(frozen=True)
class EgoFrame:
    """One ego-centric frame containing all known road users."""

    scenario_id: str
    frame_index: int
    timestamp_s: float
    ego_agent_id: str
    road_users: Sequence[RoadUserState]
    map_features: Mapping[str, Any] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", str(self.scenario_id))
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "ego_agent_id", str(self.ego_agent_id))
        object.__setattr__(self, "road_users", tuple(self.road_users))

        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite.")
        if not self.road_users:
            raise ValueError("EgoFrame requires at least one road user.")
        if self.ego_agent_id not in {user.agent_id for user in self.road_users}:
            raise ValueError("ego_agent_id must refer to one road user.")

    @property
    def ego_state(self) -> RoadUserState:
        for user in self.road_users:
            if user.agent_id == self.ego_agent_id:
                return user
        raise RuntimeError("Validated EgoFrame lost its ego road user.")

    @property
    def visible_road_users(self) -> tuple[RoadUserState, ...]:
        return tuple(
            user
            for user in self.road_users
            if user.valid and user.observed
        )


@dataclass(frozen=True)
class RoadUserGraph:
    """Graph-ready tensor payload without binding to a deep-learning library."""

    scenario_id: str
    frame_index: int
    timestamp_s: float
    ego_agent_id: str
    node_ids: tuple[str, ...]
    node_agent_types: np.ndarray
    node_features: np.ndarray
    observed_mask: np.ndarray
    valid_mask: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_confidence: np.ndarray
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_count = len(self.node_ids)
        edge_count = int(self.edge_index.shape[1])

        if self.node_features.shape != (node_count, len(NODE_FEATURE_NAMES)):
            raise ValueError("node_features has an invalid shape.")
        if self.node_agent_types.shape != (node_count,):
            raise ValueError("node_agent_types has an invalid shape.")
        if self.observed_mask.shape != (node_count,):
            raise ValueError("observed_mask has an invalid shape.")
        if self.valid_mask.shape != (node_count,):
            raise ValueError("valid_mask has an invalid shape.")
        if self.edge_index.shape != (2, edge_count):
            raise ValueError("edge_index must have shape (2, edge_count).")
        if self.edge_features.shape != (edge_count, len(EDGE_FEATURE_NAMES)):
            raise ValueError("edge_features has an invalid shape.")
        if self.edge_confidence.shape != (edge_count,):
            raise ValueError("edge_confidence has an invalid shape.")
        if node_count and edge_count:
            if self.edge_index.min() < 0 or self.edge_index.max() >= node_count:
                raise ValueError("edge_index contains an invalid node index.")

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class Scenario:
    """Dataset-independent sequence container."""

    scenario_id: str
    dataset_name: str
    frames: Sequence[EgoFrame]
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", str(self.scenario_id))
        object.__setattr__(self, "dataset_name", str(self.dataset_name))
        object.__setattr__(self, "frames", tuple(self.frames))
        if not self.frames:
            raise ValueError("Scenario requires at least one frame.")
        for frame in self.frames:
            if frame.scenario_id != self.scenario_id:
                raise ValueError("All frames must share scenario_id.")

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def road_user_types(self) -> set[AgentType]:
        return {
            user.agent_type
            for frame in self.frames
            for user in frame.road_users
        }


def build_road_user_graph(
    frame: EgoFrame,
    *,
    max_observation_range_m: float = 80.0,
    edge_radius_m: float = 45.0,
    max_neighbors_per_node: int = 8,
    include_unobserved: bool = False,
) -> RoadUserGraph:
    """Convert an ego frame into a directed road-user graph.

    The function applies only dataset-neutral graph logic.  Visibility,
    occlusion, and distance-dependent uncertainty should be produced by an
    adapter or observation model before this function is called.
    """

    if max_observation_range_m <= 0.0:
        raise ValueError("max_observation_range_m must be positive.")
    if edge_radius_m <= 0.0:
        raise ValueError("edge_radius_m must be positive.")
    if max_neighbors_per_node <= 0:
        raise ValueError("max_neighbors_per_node must be positive.")

    ego = frame.ego_state
    selected: list[RoadUserState] = []
    for user in frame.road_users:
        if not user.valid:
            continue
        if not include_unobserved and not user.observed:
            continue
        ego_distance = float(
            np.linalg.norm(user.position_xy_m - ego.position_xy_m)
        )
        if ego_distance <= max_observation_range_m:
            selected.append(user)

    if ego.agent_id not in {user.agent_id for user in selected}:
        selected.insert(0, ego)

    selected = sorted(
        selected,
        key=lambda user: (
            user.agent_id != ego.agent_id,
            float(np.linalg.norm(user.position_xy_m - ego.position_xy_m)),
            user.agent_id,
        ),
    )

    node_ids = tuple(user.agent_id for user in selected)
    positions = np.stack(
        [user.position_xy_m for user in selected],
        axis=0,
    ).astype(np.float32)
    velocities = np.stack(
        [user.velocity_xy_mps for user in selected],
        axis=0,
    ).astype(np.float32)
    observed_mask = np.asarray(
        [user.observed for user in selected],
        dtype=bool,
    )
    valid_mask = np.asarray(
        [user.valid for user in selected],
        dtype=bool,
    )
    confidences = np.asarray(
        [user.observation_confidence for user in selected],
        dtype=np.float32,
    )

    ego_relative = positions - ego.position_xy_m[None, :]
    speeds = np.linalg.norm(velocities, axis=1)
    heading = np.asarray(
        [
            0.0 if user.heading_rad is None else user.heading_rad
            for user in selected
        ],
        dtype=np.float32,
    )
    sizes = np.asarray(
        [
            [user.length_m, user.width_m, user.height_m]
            for user in selected
        ],
        dtype=np.float32,
    )
    agent_types = np.asarray(
        [int(user.agent_type) for user in selected],
        dtype=np.int64,
    )
    type_one_hot = np.stack(
        [_one_hot_agent_type(user.agent_type) for user in selected],
        axis=0,
    )

    node_features = np.column_stack(
        [
            ego_relative,
            velocities,
            speeds,
            np.sin(heading),
            np.cos(heading),
            sizes,
            type_one_hot,
            observed_mask.astype(np.float32),
            confidences,
        ]
    ).astype(np.float32)

    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=-1,
    )

    edges: list[tuple[int, int]] = []
    for source in range(len(selected)):
        candidates = np.where(
            (distances[source] > 0.0)
            & (distances[source] <= edge_radius_m)
        )[0]
        candidates = candidates[np.argsort(distances[source, candidates])]
        for target in candidates[:max_neighbors_per_node]:
            edges.append((source, int(target)))

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_features = _compute_edge_features(
            edge_index=edge_index,
            positions=positions,
            velocities=velocities,
            confidences=confidences,
        )
        edge_confidence = (
            confidences[edge_index[0]]
            * confidences[edge_index[1]]
        ).astype(np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_features = np.empty(
            (0, len(EDGE_FEATURE_NAMES)),
            dtype=np.float32,
        )
        edge_confidence = np.empty((0,), dtype=np.float32)

    return RoadUserGraph(
        scenario_id=frame.scenario_id,
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        ego_agent_id=frame.ego_agent_id,
        node_ids=node_ids,
        node_agent_types=agent_types,
        node_features=node_features,
        observed_mask=observed_mask,
        valid_mask=valid_mask,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_confidence=edge_confidence,
        source_metadata={
            "schema_version": "ego-road-user-graph-v0",
            "node_feature_names": NODE_FEATURE_NAMES,
            "edge_feature_names": EDGE_FEATURE_NAMES,
            "max_observation_range_m": float(max_observation_range_m),
            "edge_radius_m": float(edge_radius_m),
            "include_unobserved": bool(include_unobserved),
        },
    )


def _compute_edge_features(
    *,
    edge_index: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    source = edge_index[0]
    target = edge_index[1]
    relative_position = positions[target] - positions[source]
    relative_velocity = velocities[target] - velocities[source]

    distance = np.linalg.norm(relative_position, axis=1).clip(
        MIN_DISTANCE_M,
    )
    unit_direction = relative_position / distance[:, None]
    closing_speed = np.maximum(
        -np.sum(relative_velocity * unit_direction, axis=1),
        0.0,
    )

    inverse_ttc = np.zeros_like(distance, dtype=np.float32)
    closing_mask = closing_speed > MIN_CLOSING_SPEED_MPS
    inverse_ttc[closing_mask] = np.minimum(
        closing_speed[closing_mask] / distance[closing_mask],
        INVERSE_TTC_MAX_PER_S,
    )

    return np.column_stack(
        [
            relative_position,
            relative_velocity,
            distance,
            closing_speed,
            inverse_ttc,
            confidences[source],
            confidences[target],
        ]
    ).astype(np.float32)
