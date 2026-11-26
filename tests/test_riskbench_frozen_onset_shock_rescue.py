from __future__ import annotations

from experiments.ego_immune_recognition.riskbench_metadata.frozen_onset_shock_rescue import (
    _augment_trace,
)


def _frame(position: int, onset: float, *, aligned: bool = True) -> dict:
    return {
        "event_uid": "event",
        "frame_position": position,
        "frames_to_anchor": 10 - position,
        "relation_components": {
            "ego->target": {
                "target_id": "target",
                "ego_outgoing": 1,
                "onset_shock_percentile": onset,
            }
        },
        "aligned_geometry_relations": (
            {
                "target": {
                    "cpa_only": 0.5,
                    "pair_aligned_geomean": 0.4,
                }
            }
            if aligned
            else {}
        ),
    }


def test_exact_aligned_onset_shock_activates_positive_branch() -> None:
    trace = _augment_trace([_frame(0, 1.0)])
    assert trace[0][1] == 1.0


def test_unaligned_onset_shock_does_not_activate_branch() -> None:
    trace = _augment_trace([_frame(0, 1.0, aligned=False)])
    assert trace[0][1] == 0.0


def test_subthreshold_onset_shock_does_not_activate_branch() -> None:
    trace = _augment_trace([_frame(0, 0.998)])
    assert trace[0][1] == 0.0


def test_late_onset_shock_does_not_count_as_early_alert() -> None:
    frame = _frame(0, 1.0)
    frame["frames_to_anchor"] = 4
    trace = _augment_trace([frame])
    assert trace[0][1] == 0.0
