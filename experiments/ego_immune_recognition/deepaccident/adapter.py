"""Adapt DeepAccident text labels to the canonical RIR observation schema.

The adapter intentionally uses the feature intersection shared with the
RiskBench experiment: focal-relative planar positions and velocities computed
from current and previous observations.  DeepAccident class labels, object
dimensions, raw velocity annotations, RGB, LiDAR, maps, future motion, and
collision identities are retained only as audit metadata or omitted; they do
not enter the graph features produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ..schema import (
    AgentType,
    EgoFrame,
    RoadUserGraph,
    RoadUserState,
    Scenario,
    build_road_user_graph,
)
from .mini_audit import (
    DEEPACCIDENT_TIMESTEP_SECONDS,
    ROLE_TO_FOLDER,
    parse_frame_label,
)


DEEPACCIDENT_SHARED_INPUT_PROTOCOL = (
    "deepaccident-ego-relative-shared-riskbench-features-v0"
)
SYNTHETIC_FOCAL_ID = "__deepaccident_focal__"
RISK_BENCH_MODEL_TIMESTEP_SECONDS = 0.05


@dataclass(frozen=True)
class DeepAccidentViewRecord:
    """One collision-involved focal view and its matched normal run."""

    scenario_id: str
    pair_key: str
    focal_role: str
    focal_actor_id: int
    target_actor_id: int
    collision_frame: int
    accident_label_root: Path
    normal_label_root: Path
    road_type: str
    weather: str
    localization_eligible: bool = True


@dataclass(frozen=True)
class DeepAccidentPairedGraphSequence:
    """Canonical accident/normal graph sequences for the same scenario setup."""

    record: DeepAccidentViewRecord
    accident: Scenario
    normal: Scenario
    accident_graphs: tuple[RoadUserGraph, ...]
    normal_graphs: tuple[RoadUserGraph, ...]


def causally_resample_scenario(
    scenario: Scenario,
    *,
    target_timestep_seconds: float = RISK_BENCH_MODEL_TIMESTEP_SECONDS,
) -> Scenario:
    """Convert native 10 Hz observations to the frozen model's 20 Hz clock.

    An observed frame is kept unchanged.  Between two successive native
    observations, one intermediate state is extrapolated from the current
    position and the velocity already available at that time.  The next
    native frame is never used to construct the intermediate state.  No frame
    is appended after the final recorded observation.
    """

    native_step = float(scenario.source_metadata.get("timestep_seconds", 0.0))
    if not math.isclose(native_step, DEEPACCIDENT_TIMESTEP_SECONDS):
        raise ValueError(
            "DeepAccident causal resampling expects native 0.1 s observations."
        )
    if not math.isclose(target_timestep_seconds * 2.0, native_step):
        raise ValueError("Only the prespecified causal 10 Hz to 20 Hz adapter is supported.")

    output_frames: list[EgoFrame] = []
    for index, frame in enumerate(scenario.frames):
        model_frame_index = int(frame.frame_index) * 2
        output_frames.append(
            EgoFrame(
                scenario_id=scenario.scenario_id,
                frame_index=model_frame_index,
                timestamp_s=float(frame.timestamp_s),
                ego_agent_id=frame.ego_agent_id,
                road_users=frame.road_users,
                map_features=frame.map_features,
                source_metadata={
                    **frame.source_metadata,
                    "native_frame_index": int(frame.frame_index),
                    "resampled_kind": "observed",
                    "model_timestep_seconds": target_timestep_seconds,
                },
            )
        )
        if index == len(scenario.frames) - 1:
            continue
        next_frame = scenario.frames[index + 1]
        if int(next_frame.frame_index) != int(frame.frame_index) + 1:
            continue
        extrapolated_users = []
        for user in frame.road_users:
            position = (
                np.asarray(user.position_xy_m, dtype=np.float32)
                + np.asarray(user.velocity_xy_mps, dtype=np.float32)
                * target_timestep_seconds
            )
            extrapolated_users.append(
                RoadUserState(
                    agent_id=user.agent_id,
                    agent_type=user.agent_type,
                    position_xy_m=position,
                    velocity_xy_mps=user.velocity_xy_mps,
                    heading_rad=user.heading_rad,
                    length_m=user.length_m,
                    width_m=user.width_m,
                    height_m=user.height_m,
                    valid=user.valid,
                    observed=user.observed,
                    observation_confidence=user.observation_confidence,
                    source_metadata={
                        **user.source_metadata,
                        "resampled_kind": "causal_half_step_extrapolation",
                    },
                )
            )
        output_frames.append(
            EgoFrame(
                scenario_id=scenario.scenario_id,
                frame_index=model_frame_index + 1,
                timestamp_s=float(frame.timestamp_s) + target_timestep_seconds,
                ego_agent_id=frame.ego_agent_id,
                road_users=extrapolated_users,
                map_features=frame.map_features,
                source_metadata={
                    **frame.source_metadata,
                    "native_frame_index": int(frame.frame_index),
                    "resampled_kind": "causal_half_step_extrapolation",
                    "model_timestep_seconds": target_timestep_seconds,
                },
            )
        )

    return Scenario(
        scenario_id=scenario.scenario_id,
        dataset_name=scenario.dataset_name,
        frames=output_frames,
        source_metadata={
            **scenario.source_metadata,
            "native_timestep_seconds": native_step,
            "timestep_seconds": target_timestep_seconds,
            "resampling": "causal_half_step_extrapolation_without_future_frames",
            "no_extension_after_last_observation": True,
        },
    )


def build_frozen_model_graphs(
    scenario: Scenario,
    *,
    max_input_frames: int = 40,
    max_range_m: float = 80.0,
    edge_radius_m: float = 45.0,
    max_neighbors_per_node: int = 8,
) -> tuple[RoadUserGraph, ...]:
    """Build the final causal 20 Hz input window for a frozen RIR model."""

    if max_input_frames <= 0:
        raise ValueError("max_input_frames must be positive")
    resampled = causally_resample_scenario(scenario)
    frames = resampled.frames[-max_input_frames:]
    return tuple(
        build_road_user_graph(
            frame,
            max_observation_range_m=max_range_m,
            edge_radius_m=edge_radius_m,
            max_neighbors_per_node=max_neighbors_per_node,
        )
        for frame in frames
    )


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_view_records(manifest_path: Path) -> list[DeepAccidentViewRecord]:
    """Load usable focal views from the audit manifest."""

    import csv

    records: list[DeepAccidentViewRecord] = []
    with manifest_path.expanduser().resolve().open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if not _as_bool(row["usable"]):
                continue
            records.append(
                DeepAccidentViewRecord(
                    scenario_id=row["scenario_id"],
                    pair_key=row["pair_key"],
                    focal_role=row["focal_role"],
                    focal_actor_id=int(row["focal_actor_id"]),
                    target_actor_id=int(row["target_actor_id"]),
                    collision_frame=int(row["collision_frame"]),
                    accident_label_root=Path(row["accident_label_root"]),
                    normal_label_root=Path(row["normal_label_root"]),
                    road_type=row["road_type"],
                    weather=row["weather"],
                    localization_eligible=_as_bool(
                        row.get("localization_eligible", "true")
                    ),
                )
            )
    return records


def _frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _read_positions(
    label_root: Path,
    *,
    end_frame: int | None,
) -> tuple[
    list[int],
    dict[int, dict[str, tuple[np.ndarray, bool]]],
]:
    label_paths = sorted(label_root.glob("*.txt"), key=_frame_index)
    if end_frame is not None:
        label_paths = [path for path in label_paths if _frame_index(path) <= end_frame]
    if not label_paths:
        raise ValueError(f"No label frames in {label_root}")

    frame_indices: list[int] = []
    positions: dict[int, dict[str, tuple[np.ndarray, bool]]] = {}
    for label_path in label_paths:
        frame_index = _frame_index(label_path)
        frame = parse_frame_label(label_path)
        states: dict[str, tuple[np.ndarray, bool]] = {}
        for obj in frame.objects:
            if obj.actor_id == -100:
                continue
            states[str(obj.actor_id)] = (
                np.asarray((obj.x_m, obj.y_m), dtype=np.float32),
                obj.camera_visible,
            )
        frame_indices.append(frame_index)
        positions[frame_index] = states
    return frame_indices, positions


def _causal_velocities(
    frame_indices: Iterable[int],
    positions: dict[int, dict[str, tuple[np.ndarray, bool]]],
) -> dict[int, dict[str, np.ndarray]]:
    velocities: dict[int, dict[str, np.ndarray]] = {}
    previous_states: dict[str, tuple[np.ndarray, bool]] = {}
    previous_frame: int | None = None
    for frame_index in frame_indices:
        current_states = positions[frame_index]
        current_velocities: dict[str, np.ndarray] = {}
        consecutive = previous_frame is not None and frame_index == previous_frame + 1
        for actor_id, (position, _) in current_states.items():
            if consecutive and actor_id in previous_states:
                current_velocities[actor_id] = (
                    position - previous_states[actor_id][0]
                ) / DEEPACCIDENT_TIMESTEP_SECONDS
            else:
                current_velocities[actor_id] = np.zeros(2, dtype=np.float32)
        velocities[frame_index] = current_velocities
        previous_states = current_states
        previous_frame = frame_index
    return velocities


def load_deepaccident_view(
    record: DeepAccidentViewRecord,
    *,
    outcome: str,
    match_accident_horizon: bool = True,
    max_range_m: float = 80.0,
) -> Scenario:
    """Load one focal view using only causal shared-input metadata.

    The matched normal run is truncated at the accident run's last observed
    frame by default.  This prevents the normal member from receiving a longer
    observation history than its paired accident member.
    """

    if outcome not in {"accident", "normal"}:
        raise ValueError("outcome must be 'accident' or 'normal'")
    if max_range_m <= 0:
        raise ValueError("max_range_m must be positive")

    accident_paths = sorted(record.accident_label_root.glob("*.txt"), key=_frame_index)
    if not accident_paths:
        raise ValueError(f"No accident frames in {record.accident_label_root}")
    accident_end_frame = _frame_index(accident_paths[-1])
    label_root = (
        record.accident_label_root if outcome == "accident" else record.normal_label_root
    )
    end_frame = accident_end_frame if match_accident_horizon else None
    frame_indices, positions = _read_positions(label_root, end_frame=end_frame)
    velocities = _causal_velocities(frame_indices, positions)

    unique_scenario_id = f"deepaccident:{record.pair_key}:{record.focal_role}:{outcome}"
    frames: list[EgoFrame] = []
    first_frame = frame_indices[0]
    for frame_position, frame_index in enumerate(frame_indices):
        road_users: list[RoadUserState] = [
            RoadUserState(
                agent_id=SYNTHETIC_FOCAL_ID,
                agent_type=AgentType.VEHICLE,
                position_xy_m=(0.0, 0.0),
                velocity_xy_mps=(0.0, 0.0),
                heading_rad=0.0,
                length_m=4.5,
                width_m=1.8,
                height_m=1.6,
                observed=True,
                observation_confidence=1.0,
                source_metadata={"synthetic_focal_origin": True},
            )
        ]
        for actor_id, (position, camera_visible) in positions[frame_index].items():
            if float(np.linalg.norm(position)) > max_range_m:
                continue
            velocity = velocities[frame_index][actor_id]
            speed = float(np.linalg.norm(velocity))
            road_users.append(
                RoadUserState(
                    agent_id=actor_id,
                    # Use only the feature intersection shared with RiskBench.
                    agent_type=AgentType.UNKNOWN,
                    position_xy_m=position,
                    velocity_xy_mps=velocity,
                    heading_rad=(
                        math.atan2(float(velocity[1]), float(velocity[0]))
                        if speed >= 0.1
                        else None
                    ),
                    length_m=0.0,
                    width_m=0.0,
                    height_m=0.0,
                    observed=camera_visible,
                    observation_confidence=1.0 if camera_visible else 0.0,
                    source_metadata={
                        "camera_visible": camera_visible,
                        "velocity_semantics": "causal_focal_relative_difference",
                    },
                )
            )
        frames.append(
            EgoFrame(
                scenario_id=unique_scenario_id,
                frame_index=frame_index,
                timestamp_s=(
                    frame_index - first_frame
                )
                * DEEPACCIDENT_TIMESTEP_SECONDS,
                ego_agent_id=SYNTHETIC_FOCAL_ID,
                road_users=road_users,
                source_metadata={
                    "source_frame_index": frame_index,
                    "frame_position": frame_position,
                    "outcome": outcome,
                    "native_partial_observation": True,
                    "collision_anchor_frame": record.collision_frame,
                    "time_to_collision_anchor_s": (
                        record.collision_frame - frame_index
                    )
                    * DEEPACCIDENT_TIMESTEP_SECONDS,
                },
            )
        )

    return Scenario(
        scenario_id=unique_scenario_id,
        dataset_name="DeepAccident mini",
        frames=frames,
        source_metadata={
            "protocol": DEEPACCIDENT_SHARED_INPUT_PROTOCOL,
            "pair_key": record.pair_key,
            "focal_role": record.focal_role,
            "outcome": outcome,
            "road_type": record.road_type,
            "weather": record.weather,
            "collision_anchor_frame": record.collision_frame,
            # Evaluation metadata only; never copied into node/edge features.
            "evaluation_target_actor_id": record.target_actor_id,
            "gt_target_identity_used_as_input": False,
            "gt_class_and_size_used_as_input": False,
            "provided_velocity_used_as_input": False,
            "rgb_lidar_or_map_used_as_input": False,
            "future_motion_used_as_input": False,
            "position_semantics": "focal_vehicle_relative_xy_m",
            "velocity_semantics": "causal_focal_relative_difference_mps",
            "timestep_seconds": DEEPACCIDENT_TIMESTEP_SECONDS,
            "matched_horizon": match_accident_horizon,
            "label_root": str(label_root),
        },
    )


def build_paired_graph_sequence(
    record: DeepAccidentViewRecord,
    *,
    max_range_m: float = 80.0,
    edge_radius_m: float = 45.0,
    max_neighbors_per_node: int = 8,
) -> DeepAccidentPairedGraphSequence:
    """Build matched accident/normal canonical graphs for one focal view."""

    accident = load_deepaccident_view(
        record,
        outcome="accident",
        max_range_m=max_range_m,
    )
    normal = load_deepaccident_view(
        record,
        outcome="normal",
        max_range_m=max_range_m,
    )

    def graphs(scenario: Scenario) -> tuple[RoadUserGraph, ...]:
        return tuple(
            build_road_user_graph(
                frame,
                max_observation_range_m=max_range_m,
                edge_radius_m=edge_radius_m,
                max_neighbors_per_node=max_neighbors_per_node,
            )
            for frame in scenario.frames
        )

    return DeepAccidentPairedGraphSequence(
        record=record,
        accident=accident,
        normal=normal,
        accident_graphs=graphs(accident),
        normal_graphs=graphs(normal),
    )
