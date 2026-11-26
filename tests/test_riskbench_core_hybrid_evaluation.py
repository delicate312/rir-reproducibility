from __future__ import annotations

from experiments.ego_immune_recognition.riskbench_metadata.core_hybrid_evaluation import (
    event_alert,
    event_score,
    threshold_at_event_fpr,
)


def test_threshold_at_event_fpr_respects_budget() -> None:
    threshold, achieved = threshold_at_event_fpr(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        0.20,
    )
    assert threshold == 0.5
    assert achieved == 0.20


def test_event_score_and_earliest_lead() -> None:
    trace = [
        ({"frames_to_anchor": 10}, 0.4),
        ({"frames_to_anchor": 8}, 0.8),
        ({"frames_to_anchor": 6}, 0.9),
    ]
    assert event_score(trace) == 0.9
    detected, lead = event_alert(trace, 0.75)
    assert detected
    assert lead == 0.4
