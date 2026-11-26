from __future__ import annotations

from experiments.ego_immune_recognition.riskbench_metadata.relation_geometry_rescue import (
    _relation_alert_sequences,
)
from experiments.ego_immune_recognition.riskbench_metadata.safe_response_rescue import (
    SafeResponseConfig,
)


def _frame(position: int, score: float) -> dict:
    return {
        "event_index": 0,
        "event_uid": "event",
        "frame_position": position,
        "geometry_relations": {"actor": {"cpa_only": score}},
    }


def test_recovery_drop_inhibits_falling_relation() -> None:
    config = SafeResponseConfig(
        horizon_seconds=2.0,
        clearance_m=2.5,
        signal="cpa_only",
        threshold_quantile=0.8,
        score_threshold=0.7,
        support_frames=1,
        window_frames=3,
        minimum_rise=0.0,
        max_recovery_drop=0.05,
        minimum_current_delta=-0.05,
    )
    sequence = _relation_alert_sequences(
        [_frame(0, 0.9), _frame(1, 0.8)],
        config=config,
    )["0"]

    assert sequence[0][1]
    assert not sequence[1][1]


def test_persistent_rising_relation_remains_active() -> None:
    config = SafeResponseConfig(
        horizon_seconds=2.0,
        clearance_m=2.5,
        signal="cpa_only",
        threshold_quantile=0.8,
        score_threshold=0.7,
        support_frames=2,
        window_frames=3,
        minimum_rise=0.0,
        max_recovery_drop=0.05,
        minimum_current_delta=-0.01,
    )
    sequence = _relation_alert_sequences(
        [_frame(0, 0.75), _frame(1, 0.82)],
        config=config,
    )["0"]

    assert not sequence[0][1]
    assert sequence[1][1]
