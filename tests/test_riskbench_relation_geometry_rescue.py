from __future__ import annotations

import numpy as np

from experiments.ego_immune_recognition.riskbench_metadata.relation_geometry_rescue import (
    cpa_risk,
    closest_approach,
)


def test_closest_approach_detects_crossing_course() -> None:
    tcpa, dcpa, future = closest_approach(
        np.asarray([10.0, 10.0]),
        np.asarray([-5.0, -5.0]),
        horizon_seconds=3.0,
    )

    assert future
    assert abs(tcpa - 2.0) < 1e-6
    assert dcpa < 1e-6


def test_closest_approach_rejects_diverging_relation() -> None:
    _, _, future = closest_approach(
        np.asarray([5.0, 0.0]),
        np.asarray([2.0, 0.0]),
        horizon_seconds=3.0,
    )

    assert not future


def test_cpa_risk_prefers_future_low_clearance_conflict() -> None:
    conflict = cpa_risk(
        np.asarray([10.0, 5.0]),
        np.asarray([-5.0, -2.5]),
        horizon_seconds=3.0,
        clearance_m=2.5,
    )
    safe = cpa_risk(
        np.asarray([10.0, 5.0]),
        np.asarray([-5.0, 2.5]),
        horizon_seconds=3.0,
        clearance_m=2.5,
    )

    assert conflict["dcpa_m"] < safe["dcpa_m"]
    assert conflict["raw"] > safe["raw"]
