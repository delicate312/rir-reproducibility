"""Ego-centric partial observation model.

This module degrades full annotation frames into a deployable approximation:
the ego vehicle observes nearby road users with distance-dependent confidence,
limited field of view, and simple line-of-sight occlusion.

The model is deliberately dataset-neutral.  US-101, Waymo, nuScenes, or InterHub
adapters should first build a complete ``EgoFrame`` and then call
``apply_ego_observation_model`` before graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import EgoFrame, RoadUserState


@dataclass(frozen=True)
class ObservationModelConfig:
    """Parameters for trajectory-level ego observation degradation."""

    max_range_m: float = 80.0
    full_confidence_range_m: float = 20.0
    min_visible_confidence: float = 0.15
    horizontal_fov_deg: float = 360.0
    occlusion_enabled: bool = True
    occlusion_angle_padding_deg: float = 1.0
    occlusion_depth_gap_m: float = 1.0
    position_noise_std_at_max_range_m: float = 0.0
    velocity_noise_std_at_max_range_mps: float = 0.0
    keep_unobserved: bool = True

    def __post_init__(self) -> None:
        if self.max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive.")
        if not 0.0 <= self.min_visible_confidence <= 1.0:
            raise ValueError("min_visible_confidence must be in [0, 1].")
        if self.full_confidence_range_m < 0.0:
            raise ValueError("full_confidence_range_m must be non-negative.")
        if self.full_confidence_range_m > self.max_range_m:
            raise ValueError(
                "full_confidence_range_m cannot exceed max_range_m."
            )
        if not 0.0 < self.horizontal_fov_deg <= 360.0:
            raise ValueError("horizontal_fov_deg must be in (0, 360].")
        if self.occlusion_depth_gap_m < 0.0:
            raise ValueError("occlusion_depth_gap_m must be non-negative.")
        if self.position_noise_std_at_max_range_m < 0.0:
            raise ValueError("position noise std must be non-negative.")
        if self.velocity_noise_std_at_max_range_mps < 0.0:
            raise ValueError("velocity noise std must be non-negative.")


def apply_ego_observation_model(
    frame: EgoFrame,
    config: ObservationModelConfig | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> EgoFrame:
    """Return a partial-observation version of ``frame``.

    The ego agent is always retained as observed with confidence 1.  Other road
    users outside range/FOV or behind an occluder are marked unobserved.  If
    ``keep_unobserved`` is false, unobserved non-ego users are removed.
    """

    config = config or ObservationModelConfig()
    rng = rng or np.random.default_rng(0)

    ego = frame.ego_state
    observations = _compute_visibility(frame, config)

    degraded_users: list[RoadUserState] = []
    for user in frame.road_users:
        if user.agent_id == frame.ego_agent_id:
            degraded_users.append(
                _replace_user_observation(
                    user,
                    observed=True,
                    confidence=1.0,
                    position_xy_m=user.position_xy_m,
                    velocity_xy_mps=user.velocity_xy_mps,
                )
            )
            continue

        observed, confidence, distance = observations[user.agent_id]
        if not observed and not config.keep_unobserved:
            continue

        position = np.asarray(user.position_xy_m, dtype=np.float32)
        velocity = np.asarray(user.velocity_xy_mps, dtype=np.float32)
        if observed:
            position, velocity = _apply_distance_noise(
                position=position,
                velocity=velocity,
                distance_m=distance,
                config=config,
                rng=rng,
            )

        degraded_users.append(
            _replace_user_observation(
                user,
                observed=observed,
                confidence=confidence,
                position_xy_m=position,
                velocity_xy_mps=velocity,
            )
        )

    return EgoFrame(
        scenario_id=frame.scenario_id,
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        ego_agent_id=frame.ego_agent_id,
        road_users=degraded_users,
        map_features=frame.map_features,
        source_metadata={
            **dict(frame.source_metadata),
            "observation_model": {
                "name": "ego_partial_observation_v0",
                "max_range_m": config.max_range_m,
                "horizontal_fov_deg": config.horizontal_fov_deg,
                "occlusion_enabled": config.occlusion_enabled,
                "keep_unobserved": config.keep_unobserved,
            },
        },
    )


def _compute_visibility(
    frame: EgoFrame,
    config: ObservationModelConfig,
) -> dict[str, tuple[bool, float, float]]:
    ego = frame.ego_state
    ego_position = np.asarray(ego.position_xy_m, dtype=np.float32)
    ego_heading = 0.0 if ego.heading_rad is None else float(ego.heading_rad)

    relative: dict[str, np.ndarray] = {}
    distances: dict[str, float] = {}
    bearings: dict[str, float] = {}
    in_view: dict[str, bool] = {}

    for user in frame.road_users:
        delta = np.asarray(user.position_xy_m, dtype=np.float32) - ego_position
        distance = float(np.linalg.norm(delta))
        bearing = _wrap_angle(float(np.arctan2(delta[1], delta[0])) - ego_heading)
        relative[user.agent_id] = delta
        distances[user.agent_id] = distance
        bearings[user.agent_id] = bearing
        in_view[user.agent_id] = (
            user.valid
            and distance <= config.max_range_m
            and _inside_fov(bearing, config.horizontal_fov_deg)
        )

    visible: dict[str, tuple[bool, float, float]] = {}
    for user in frame.road_users:
        if user.agent_id == frame.ego_agent_id:
            visible[user.agent_id] = (True, 1.0, 0.0)
            continue

        distance = distances[user.agent_id]
        observed = bool(in_view[user.agent_id])

        if observed and config.occlusion_enabled:
            observed = not _is_occluded(
                target=user,
                frame=frame,
                in_view=in_view,
                distances=distances,
                bearings=bearings,
                config=config,
            )

        confidence = (
            _distance_confidence(distance, config)
            if observed
            else 0.0
        )
        visible[user.agent_id] = (observed, confidence, distance)

    return visible


def _is_occluded(
    *,
    target: RoadUserState,
    frame: EgoFrame,
    in_view: dict[str, bool],
    distances: dict[str, float],
    bearings: dict[str, float],
    config: ObservationModelConfig,
) -> bool:
    target_distance = distances[target.agent_id]
    target_bearing = bearings[target.agent_id]

    for blocker in frame.road_users:
        if blocker.agent_id in {frame.ego_agent_id, target.agent_id}:
            continue
        if not in_view.get(blocker.agent_id, False):
            continue

        blocker_distance = distances[blocker.agent_id]
        if blocker_distance + config.occlusion_depth_gap_m >= target_distance:
            continue

        blocker_half_angle = _object_half_angle(
            width_m=max(blocker.width_m, 0.5),
            distance_m=max(blocker_distance, 0.5),
            padding_deg=config.occlusion_angle_padding_deg,
        )
        target_half_angle = _object_half_angle(
            width_m=max(target.width_m, 0.5),
            distance_m=max(target_distance, 0.5),
            padding_deg=0.0,
        )
        bearing_gap = abs(
            _wrap_angle(target_bearing - bearings[blocker.agent_id])
        )
        if bearing_gap <= blocker_half_angle + target_half_angle:
            return True

    return False


def _distance_confidence(
    distance_m: float,
    config: ObservationModelConfig,
) -> float:
    if distance_m <= config.full_confidence_range_m:
        return 1.0
    denominator = max(
        config.max_range_m - config.full_confidence_range_m,
        1e-6,
    )
    ratio = min(
        max((distance_m - config.full_confidence_range_m) / denominator, 0.0),
        1.0,
    )
    return float(
        1.0
        - ratio * (1.0 - config.min_visible_confidence)
    )


def _apply_distance_noise(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    distance_m: float,
    config: ObservationModelConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ratio = min(max(distance_m / config.max_range_m, 0.0), 1.0)
    position_std = config.position_noise_std_at_max_range_m * ratio * ratio
    velocity_std = config.velocity_noise_std_at_max_range_mps * ratio * ratio

    if position_std > 0.0:
        position = position + rng.normal(0.0, position_std, size=2)
    if velocity_std > 0.0:
        velocity = velocity + rng.normal(0.0, velocity_std, size=2)

    return (
        np.asarray(position, dtype=np.float32),
        np.asarray(velocity, dtype=np.float32),
    )


def _replace_user_observation(
    user: RoadUserState,
    *,
    observed: bool,
    confidence: float,
    position_xy_m: np.ndarray,
    velocity_xy_mps: np.ndarray,
) -> RoadUserState:
    return RoadUserState(
        agent_id=user.agent_id,
        agent_type=user.agent_type,
        position_xy_m=position_xy_m,
        velocity_xy_mps=velocity_xy_mps,
        heading_rad=user.heading_rad,
        length_m=user.length_m,
        width_m=user.width_m,
        height_m=user.height_m,
        valid=user.valid,
        observed=observed,
        observation_confidence=confidence,
        source_metadata=user.source_metadata,
    )


def _inside_fov(
    bearing_rad: float,
    horizontal_fov_deg: float,
) -> bool:
    if horizontal_fov_deg >= 360.0:
        return True
    half_fov = np.deg2rad(horizontal_fov_deg / 2.0)
    return abs(bearing_rad) <= half_fov


def _object_half_angle(
    *,
    width_m: float,
    distance_m: float,
    padding_deg: float,
) -> float:
    return float(
        np.arctan2(width_m / 2.0, distance_m)
        + np.deg2rad(padding_deg)
    )


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)
