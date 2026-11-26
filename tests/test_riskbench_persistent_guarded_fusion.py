from experiments.ego_immune_recognition.riskbench_metadata.persistent_guarded_fusion import (
    ALERT_PERCENTILE,
    _attach_guarded_response,
)
from experiments.ego_immune_recognition.riskbench_metadata.peak_or_sustain_gate import (
    PeakOrSustainConfig,
    attach_peak_or_sustain_response,
)


def _frame(
    position: int,
    *,
    physical: float,
    relation: float,
    persistence: float,
    recovery: float,
    inverse_ttc: float,
) -> dict:
    return {
        "event_index": 0,
        "frame_position": position,
        "aggregations": {
            "physical_percentile": physical,
            "relation_percentile": relation,
            "persistence_percentile": persistence,
            "recovery_percentile": recovery,
            "inverse_ttc_ego_max": inverse_ttc,
        },
    }


def test_persistent_relation_can_rescue_ttc_negative_frame() -> None:
    frames = [
        _frame(
            index,
            physical=0.60,
            relation=0.97,
            persistence=0.93,
            recovery=0.20,
            inverse_ttc=0.20,
        )
        for index in range(3)
    ]

    _attach_guarded_response(frames)

    assert frames[-1]["aggregations"]["persistent_relation_gate"] == 1.0
    assert frames[-1]["aggregations"]["persistent_guarded"] >= ALERT_PERCENTILE


def test_recovery_inhibits_only_marginal_physical_alert() -> None:
    frames = [
        _frame(
            index,
            physical=0.97,
            relation=0.60,
            persistence=0.60,
            recovery=0.90,
            inverse_ttc=0.80 - 0.05 * index,
        )
        for index in range(4)
    ]

    _attach_guarded_response(frames)

    assert frames[-1]["aggregations"]["physical_inhibition_gate"] == 1.0
    assert frames[-1]["aggregations"]["persistent_guarded"] < ALERT_PERCENTILE


def test_severe_physical_guard_cannot_be_inhibited() -> None:
    frames = [
        _frame(
            index,
            physical=0.995,
            relation=0.60,
            persistence=0.60,
            recovery=0.99,
            inverse_ttc=1.00 - 0.05 * index,
        )
        for index in range(4)
    ]

    _attach_guarded_response(frames)

    assert frames[-1]["aggregations"]["severe_physical_guard"] == 1.0
    assert frames[-1]["aggregations"]["physical_inhibition_gate"] == 0.0
    assert frames[-1]["aggregations"]["persistent_guarded"] >= ALERT_PERCENTILE


def test_supported_peak_can_rescue_two_frame_peak() -> None:
    relations = [0.76, 0.81, 0.84, 0.88, 0.91, 0.95, 0.97]
    frames = [
        _frame(
            index,
            physical=0.60,
            relation=relation,
            persistence=0.86,
            recovery=0.20,
            inverse_ttc=0.20,
        )
        for index, relation in enumerate(relations)
    ]
    _attach_guarded_response(frames)
    attach_peak_or_sustain_response(
        frames,
        PeakOrSustainConfig(
            support_window_frames=7,
            minimum_support_frames=4,
            minimum_relation_rise=0.15,
        ),
    )

    assert frames[-1]["aggregations"]["supported_peak_gate"] == 1.0
    assert (
        frames[-1]["aggregations"]["peak_or_sustain_guarded"]
        >= ALERT_PERCENTILE
    )


def test_isolated_peak_is_not_supported() -> None:
    relations = [0.40, 0.45, 0.50, 0.98]
    frames = [
        _frame(
            index,
            physical=0.60,
            relation=relation,
            persistence=0.90,
            recovery=0.20,
            inverse_ttc=0.20,
        )
        for index, relation in enumerate(relations)
    ]
    _attach_guarded_response(frames)
    attach_peak_or_sustain_response(
        frames,
        PeakOrSustainConfig(
            support_window_frames=4,
            minimum_support_frames=3,
        ),
    )

    assert frames[-1]["aggregations"]["supported_peak_gate"] == 0.0
    assert (
        frames[-1]["aggregations"]["peak_or_sustain_guarded"]
        < ALERT_PERCENTILE
    )
