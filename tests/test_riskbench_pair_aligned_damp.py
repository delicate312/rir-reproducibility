from __future__ import annotations

from experiments.ego_immune_recognition.riskbench_metadata.pair_aligned_damp_evaluation import (
    _aligned_trace,
)


def _frame(position: int, cpa: float, aligned: float) -> dict:
    return {
        "frame_position": position,
        "frames_to_anchor": 10 - position,
        "aligned_geometry_relations": {
            "target": {
                "cpa_only": cpa,
                "pair_aligned_geomean": aligned,
            }
        },
    }


def test_severe_cpa_bypasses_pair_support() -> None:
    trace = _aligned_trace([_frame(0, 0.9995, 0.2)])
    assert trace[0][1] == 0.9995


def test_non_severe_cpa_does_not_bypass_pair_support() -> None:
    trace = _aligned_trace([_frame(0, 0.995, 0.9)])
    assert trace[0][1] == 0.0


def test_moderate_cpa_requires_three_pair_aligned_frames() -> None:
    trace = _aligned_trace(
        [
            _frame(0, 0.90, 0.60),
            _frame(1, 0.91, 0.75),
            _frame(2, 0.92, 0.90),
        ]
    )
    assert [score for _, score in trace[:2]] == [0.0, 0.0]
    assert abs(trace[2][1] - 0.75) < 1e-9
