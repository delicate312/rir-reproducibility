"""Adapt RiskBench metadata trajectories to the canonical ego graph schema.

The metadata bundle contains per-frame actor positions in an ego-relative
coordinate system. It is sufficient for a trajectory-only pilot without the
full camera, LiDAR, or CARLA archives. Velocities are estimated causally from
the current and previous observed positions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..schema import (
    AgentType,
    EgoFrame,
    RoadUserGraph,
    RoadUserState,
    Scenario,
    build_road_user_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_ROOT = PROJECT_ROOT / "data" / "riskbench" / "raw" / "metadata"
RISK_BENCH_TIMESTEP_SECONDS = 0.05
RISK_BENCH_GRAPH_INPUT_PROTOCOL = "riskbench-metadata-no-gt-target-features-v1"
SUPPORTED_CATEGORIES = (
    "non-interactive",
    "interactive",
    "collision",
    "obstacle",
)

_TYPE_CODE_TO_AGENT_TYPE = {
    "c": AgentType.VEHICLE,
    "t": AgentType.VEHICLE,
    "p": AgentType.PEDESTRIAN,
    "b": AgentType.CYCLIST,
    "m": AgentType.MOTORCYCLIST,
}
_TYPE_CODE_TO_NAME = {
    "c": "car",
    "t": "truck",
    "p": "pedestrian",
    "b": "bicyclist",
    "m": "motorcyclist",
    "0": "none",
}


@dataclass(frozen=True)
class RiskBenchScenarioRecord:
    """One RiskBench scenario and its official metadata labels."""

    scenario_id: str
    category: str
    state_path: Path
    scenario_group_id: str
    base_scenario_id: str
    weather: str
    traffic_density: str
    risk_actor_ids: tuple[str, ...]
    critical_frame: int | None
    behavior_window: tuple[int, int] | None
    target_type_code: str
    target_type_name: str
    target_agent_type: AgentType
    pool_role: str
    split: str
    frame_count: int
    actor_count: int


@dataclass(frozen=True)
class RiskBenchGraphSequence:
    """Canonical scenario plus graph-ready frames."""

    record: RiskBenchScenarioRecord
    scenario: Scenario
    graphs: tuple[RoadUserGraph, ...]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def split_scenario_variant(scenario_id: str) -> tuple[str, str, str]:
    """Separate a scenario script from weather and density augmentation."""

    parts = scenario_id.rstrip("_").rsplit("_", 2)
    if len(parts) != 3:
        return scenario_id.rstrip("_"), "unknown", "unknown"
    return parts[0], parts[1], parts[2]


def target_type_from_scenario_id(scenario_id: str) -> tuple[str, str, AgentType]:
    """Decode the interacting road-user type from the scenario taxonomy id."""

    base_scenario_id, _, _ = split_scenario_variant(scenario_id)
    tokens = base_scenario_id.split("_")
    type_code = tokens[3] if len(tokens) > 3 else "0"
    return (
        type_code,
        _TYPE_CODE_TO_NAME.get(type_code, "unknown"),
        _TYPE_CODE_TO_AGENT_TYPE.get(type_code, AgentType.UNKNOWN),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [item for item in value if item is not None]
        return None if not values else int(values[0])
    return int(value)


def _pool_role(category: str) -> str:
    return {
        "non-interactive": "safe_core",
        "interactive": "hard_safe",
        "collision": "positive_ood",
        "obstacle": "secondary_excluded",
    }[category]


def _group_fraction(group_id: str) -> float:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _split_for_group(category: str, group_id: str) -> str:
    """Assign whole augmentation families to one split deterministically."""

    fraction = _group_fraction(group_id)
    if category == "non-interactive":
        if fraction < 0.80:
            return "train_normal"
        if fraction < 0.90:
            return "validation_safe_reference"
        return "test_safe_reference"
    if category == "interactive":
        return "validation_normal" if fraction < 0.50 else "test_normal"
    if category == "collision":
        return "validation_positive" if fraction < 0.50 else "test_positive"
    return "secondary_excluded"


def _behavior_window(
    behavior_payload: Mapping[str, Any],
    base_scenario_id: str,
    weather: str,
    density: str,
) -> tuple[int, int] | None:
    variants = behavior_payload.get(base_scenario_id, {})
    value = variants.get(f"{weather}_{density}_")
    if not isinstance(value, Sequence) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def discover_riskbench_records(
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    *,
    categories: Sequence[str] = SUPPORTED_CATEGORIES,
) -> list[RiskBenchScenarioRecord]:
    """Inventory all metadata scenarios and create leakage-safe split records."""

    metadata_root = metadata_root.expanduser().resolve()
    if not metadata_root.exists():
        raise FileNotFoundError(metadata_root)

    unknown = set(categories) - set(SUPPORTED_CATEGORIES)
    if unknown:
        raise ValueError(f"Unsupported RiskBench categories: {sorted(unknown)}")

    records: list[RiskBenchScenarioRecord] = []
    for category in categories:
        state_root = metadata_root / "state" / category
        if not state_root.exists():
            raise FileNotFoundError(state_root)
        risk_path = metadata_root / "GT_risk" / f"{category}.json"
        critical_path = metadata_root / "GT_critical_point" / f"{category}.json"
        risk_payload = _load_json(risk_path) if risk_path.exists() else {}
        critical_payload = _load_json(critical_path) if critical_path.exists() else {}
        behavior_path = metadata_root / "behavior" / f"{category}.json"
        behavior_payload = _load_json(behavior_path) if behavior_path.exists() else {}

        for state_path in sorted(state_root.glob("*.json")):
            scenario_id = state_path.stem
            base_scenario_id, weather, density = split_scenario_variant(scenario_id)
            group_id = f"{category}:{base_scenario_id}"
            raw_risk_ids = risk_payload.get(scenario_id, [])
            risk_actor_ids = tuple(
                str(actor_id)
                for actor_id in raw_risk_ids
                if actor_id is not None
            )
            state_payload = _load_json(state_path)
            actor_ids = {
                str(actor_id)
                for frame_payload in state_payload.values()
                for actor_id in frame_payload
            }
            type_code, type_name, agent_type = target_type_from_scenario_id(scenario_id)
            records.append(
                RiskBenchScenarioRecord(
                    scenario_id=scenario_id,
                    category=category,
                    state_path=state_path,
                    scenario_group_id=group_id,
                    base_scenario_id=base_scenario_id,
                    weather=weather,
                    traffic_density=density,
                    risk_actor_ids=risk_actor_ids,
                    critical_frame=_optional_int(critical_payload.get(scenario_id)),
                    behavior_window=_behavior_window(
                        behavior_payload,
                        base_scenario_id,
                        weather,
                        density,
                    ),
                    target_type_code=type_code,
                    target_type_name=type_name,
                    target_agent_type=agent_type,
                    pool_role=_pool_role(category),
                    split=_split_for_group(category, group_id),
                    frame_count=len(state_payload),
                    actor_count=len(actor_ids),
                )
            )
    return records


def _frame_positions(
    state_payload: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[int, dict[str, np.ndarray]]:
    frames: dict[int, dict[str, np.ndarray]] = {}
    for raw_frame, raw_states in state_payload.items():
        frame_index = int(raw_frame)
        frames[frame_index] = {
            str(actor_id): np.asarray(position, dtype=np.float32)
            for actor_id, position in raw_states.items()
        }
    return frames


def _infer_ego_actor_id(
    positions: Mapping[int, Mapping[str, np.ndarray]],
) -> str | None:
    zero_counts: dict[str, int] = {}
    observation_counts: dict[str, int] = {}
    for states in positions.values():
        for actor_id, position in states.items():
            observation_counts[actor_id] = observation_counts.get(actor_id, 0) + 1
            if float(np.linalg.norm(position)) <= 1e-6:
                zero_counts[actor_id] = zero_counts.get(actor_id, 0) + 1
    candidates = [
        actor_id
        for actor_id, count in zero_counts.items()
        if count >= 3
        and count / max(observation_counts.get(actor_id, 1), 1) >= 0.95
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda actor_id: zero_counts[actor_id])


def _causal_relative_velocities(
    positions: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[int, dict[str, np.ndarray]]:
    velocities: dict[int, dict[str, np.ndarray]] = {}
    for frame_index in sorted(positions):
        current = positions[frame_index]
        previous = positions.get(frame_index - 1, {})
        frame_velocities = {}
        for actor_id, position in current.items():
            if actor_id in previous:
                frame_velocities[actor_id] = (
                    position - previous[actor_id]
                ) / RISK_BENCH_TIMESTEP_SECONDS
            else:
                frame_velocities[actor_id] = np.zeros(2, dtype=np.float32)
        velocities[frame_index] = frame_velocities
    return velocities


def _default_size(agent_type: AgentType, type_name: str) -> tuple[float, float, float]:
    if type_name == "truck":
        return 10.0, 2.5, 3.2
    return {
        AgentType.VEHICLE: (4.5, 1.8, 1.6),
        AgentType.PEDESTRIAN: (0.6, 0.6, 1.7),
        AgentType.CYCLIST: (1.8, 0.7, 1.7),
        AgentType.MOTORCYCLIST: (2.2, 0.8, 1.6),
    }.get(agent_type, (0.0, 0.0, 0.0))


def _selected_frame_indices(
    frame_indices: Sequence[int],
    critical_frame: int | None,
    pre_critical_frames: int | None,
    post_critical_frames: int | None,
) -> list[int]:
    if critical_frame is None or (
        pre_critical_frames is None and post_critical_frames is None
    ):
        return list(frame_indices)
    pre = 0 if pre_critical_frames is None else max(pre_critical_frames, 0)
    post = 0 if post_critical_frames is None else max(post_critical_frames, 0)
    start = critical_frame - pre
    end = critical_frame + post
    return [frame for frame in frame_indices if start <= frame <= end]


def load_riskbench_scenario(
    record: RiskBenchScenarioRecord,
    *,
    pre_critical_frames: int | None = None,
    post_critical_frames: int | None = None,
    max_range_m: float = 80.0,
) -> Scenario:
    """Load one metadata state file as an ego-relative canonical scenario."""

    if max_range_m <= 0.0:
        raise ValueError("max_range_m must be positive")
    state_payload = _load_json(record.state_path)
    positions = _frame_positions(state_payload)
    velocities = _causal_relative_velocities(positions)
    inferred_ego_id = _infer_ego_actor_id(positions)
    ego_agent_id = inferred_ego_id or "__riskbench_ego__"
    frame_indices = _selected_frame_indices(
        sorted(positions),
        record.critical_frame,
        pre_critical_frames,
        post_critical_frames,
    )
    if not frame_indices:
        raise ValueError(f"No selected frames for {record.scenario_id}")

    frames = []
    risk_ids = set(record.risk_actor_ids)
    for frame_position, frame_index in enumerate(frame_indices):
        states = positions[frame_index]
        frame_velocities = velocities[frame_index]
        road_users = [
            RoadUserState(
                agent_id=ego_agent_id,
                agent_type=AgentType.VEHICLE,
                position_xy_m=(0.0, 0.0),
                velocity_xy_mps=(0.0, 0.0),
                heading_rad=0.0,
                length_m=4.5,
                width_m=1.8,
                height_m=1.6,
                source_metadata={
                    "synthetic_ego_origin": inferred_ego_id is None,
                    "metadata_actor_id": inferred_ego_id,
                },
            )
        ]
        for actor_id, position in states.items():
            if actor_id == inferred_ego_id:
                continue
            if float(np.linalg.norm(position)) > max_range_m:
                continue
            # RiskBench exposes the risk actor/type as ground truth metadata.
            # Keep it for audits and localization labels, never as model input.
            agent_type = AgentType.UNKNOWN
            type_name = "unknown"
            velocity = frame_velocities[actor_id]
            speed = float(np.linalg.norm(velocity))
            heading = math.atan2(float(velocity[1]), float(velocity[0])) if speed >= 0.1 else None
            length, width, height = _default_size(agent_type, type_name)
            road_users.append(
                RoadUserState(
                    agent_id=actor_id,
                    agent_type=agent_type,
                    position_xy_m=position,
                    velocity_xy_mps=velocity,
                    heading_rad=heading,
                    length_m=length,
                    width_m=width,
                    height_m=height,
                    observed=True,
                    observation_confidence=1.0,
                    source_metadata={
                        "riskbench_risk_actor": actor_id in risk_ids,
                        "audit_target_type": (
                            record.target_type_name if actor_id in risk_ids else None
                        ),
                        "velocity_semantics": "causal_ego_relative_difference",
                    },
                )
            )
        frames.append(
            EgoFrame(
                scenario_id=record.scenario_id,
                frame_index=frame_index,
                timestamp_s=(frame_index - frame_indices[0]) * RISK_BENCH_TIMESTEP_SECONDS,
                ego_agent_id=ego_agent_id,
                road_users=road_users,
                source_metadata={
                    "source_frame_index": frame_index,
                    "frame_position": frame_position,
                    "risk_actor_ids": list(record.risk_actor_ids),
                    "critical_frame": record.critical_frame,
                    "is_critical_frame": frame_index == record.critical_frame,
                    "native_partial_observation": True,
                },
            )
        )

    return Scenario(
        scenario_id=record.scenario_id,
        dataset_name="RiskBench metadata",
        frames=frames,
        source_metadata={
            "protocol": "riskbench-metadata-ego-relative-v0",
            "category": record.category,
            "pool_role": record.pool_role,
            "split": record.split,
            "scenario_group_id": record.scenario_group_id,
            "base_scenario_id": record.base_scenario_id,
            "weather": record.weather,
            "traffic_density": record.traffic_density,
            "risk_actor_ids": list(record.risk_actor_ids),
            "critical_frame": record.critical_frame,
            "behavior_window": record.behavior_window,
            "target_type_name": record.target_type_name,
            "metadata_only": True,
            "graph_input_protocol": RISK_BENCH_GRAPH_INPUT_PROTOCOL,
            "gt_target_type_used_as_input": False,
            "position_semantics": "ego_relative_xy_m",
            "velocity_semantics": "causal_ego_relative_difference_mps",
            "timestep_seconds": RISK_BENCH_TIMESTEP_SECONDS,
            "state_path": str(record.state_path),
        },
    )


def build_riskbench_graph_sequence(
    record: RiskBenchScenarioRecord,
    *,
    pre_critical_frames: int | None = None,
    post_critical_frames: int | None = None,
    max_range_m: float = 80.0,
    edge_radius_m: float = 45.0,
    max_neighbors_per_node: int = 8,
) -> RiskBenchGraphSequence:
    """Build a canonical RiskBench sequence and directed relation graphs."""

    scenario = load_riskbench_scenario(
        record,
        pre_critical_frames=pre_critical_frames,
        post_critical_frames=post_critical_frames,
        max_range_m=max_range_m,
    )
    graphs = tuple(
        build_road_user_graph(
            frame,
            max_observation_range_m=max_range_m,
            edge_radius_m=edge_radius_m,
            max_neighbors_per_node=max_neighbors_per_node,
        )
        for frame in scenario.frames
    )
    return RiskBenchGraphSequence(record=record, scenario=scenario, graphs=graphs)
