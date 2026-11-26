"""
Shared directed-relation schema for TRAIL.

The project deliberately does not define a hand-written scalar
``interaction_strength``.  Graph edges contain only observable pair state.
TRAIL learns a latent, directed interaction state from the temporal evolution
of these observations.
"""

from __future__ import annotations

import numpy as np


RELATION_SCHEMA_VERSION = "trail-directed-response-v2"

# Observable, directed relation state.  For an edge i -> j, source quantities
# belong to i and target quantities belong to j.
EDGE_ATTRIBUTE_NAMES = [
    "relative_lateral_offset_ft",
    "relative_longitudinal_offset_ft",
    "relative_longitudinal_speed_fps",
    "source_acceleration_fps2",
    "target_acceleration_fps2",
    "inverse_ttc_per_s",
    "same_lane_indicator",
    "adjacent_lane_indicator",
]

EDGE_RAW_NAMES = list(EDGE_ATTRIBUTE_NAMES)

INVERSE_TTC_MAX_PER_S = 2.0
MIN_LONGITUDINAL_GAP_FT = 1e-3
MIN_CLOSING_SPEED_FPS = 1e-3


def compute_directed_relation_observations(
    edge_index: np.ndarray,
    positions: np.ndarray,
    speeds: np.ndarray,
    accelerations: np.ndarray,
    lane_ids: np.ndarray,
) -> np.ndarray:
    """Return raw directed pair observations for every graph edge.

    No channel is presented as an interaction label.  In particular,
    ``inverse_ttc_per_s`` is a conventional physical observable, while the
    remaining channels retain direction, motion, response, and lane context.
    """

    edge_count = int(edge_index.shape[1])

    if edge_count == 0:
        return np.empty(
            (0, len(EDGE_ATTRIBUTE_NAMES)),
            dtype=np.float32,
        )

    source = edge_index[0]
    target = edge_index[1]

    relative_position = (
        positions[target] - positions[source]
    ).astype(np.float32)

    relative_lateral = relative_position[:, 0]
    relative_longitudinal = relative_position[:, 1]
    relative_speed = (
        speeds[target] - speeds[source]
    ).astype(np.float32)

    # Closing speed is symmetric for the physical pair but is computed from
    # the directed signed longitudinal offset.  It is positive only when the
    # two vehicles are moving toward one another along the road axis.
    direction = np.sign(relative_longitudinal)
    closing_speed = np.maximum(
        -direction * relative_speed,
        0.0,
    ).astype(np.float32)

    longitudinal_gap = np.maximum(
        np.abs(relative_longitudinal),
        MIN_LONGITUDINAL_GAP_FT,
    )

    inverse_ttc = np.zeros(
        edge_count,
        dtype=np.float32,
    )

    closing_mask = (
        closing_speed > MIN_CLOSING_SPEED_FPS
    )

    inverse_ttc[closing_mask] = np.minimum(
        closing_speed[closing_mask]
        / longitudinal_gap[closing_mask],
        INVERSE_TTC_MAX_PER_S,
    )

    valid_lanes = (
        (lane_ids[source] > 0)
        & (lane_ids[target] > 0)
    )
    lane_delta = np.abs(
        lane_ids[source] - lane_ids[target]
    )

    same_lane = (
        valid_lanes & (lane_delta == 0)
    ).astype(np.float32)
    adjacent_lane = (
        valid_lanes & (lane_delta == 1)
    ).astype(np.float32)

    observations = np.stack(
        [
            relative_lateral,
            relative_longitudinal,
            relative_speed,
            accelerations[source],
            accelerations[target],
            inverse_ttc,
            same_lane,
            adjacent_lane,
        ],
        axis=1,
    ).astype(np.float32)

    if not np.isfinite(observations).all():
        raise ValueError(
            "Directed relation observations contain NaN or Inf."
        )

    return observations

