import numpy as np

from experiments.ego_immune_recognition.riskbench_metadata.mainstream_baseline_evaluation import (
    CenterKalman,
    _center_risk,
    _right_padded_window,
)


def test_center_risk_prefers_future_conflict_over_receding_motion():
    approaching, approaching_distance, _ = _center_risk(
        np.asarray([10.0, 0.0]), np.asarray([-5.0, 0.0])
    )
    receding, receding_distance, _ = _center_risk(
        np.asarray([10.0, 0.0]), np.asarray([5.0, 0.0])
    )
    assert approaching_distance < receding_distance
    assert approaching > receding


def test_center_kalman_estimates_finite_motion():
    kalman = CenterKalman(np.asarray([10.0, 0.0]), 0.05)
    for x in (9.8, 9.6, 9.4, 9.2):
        kalman.update(np.asarray([x, 0.0]))
    assert np.all(np.isfinite(kalman.position))
    assert np.all(np.isfinite(kalman.velocity))
    assert kalman.velocity[0] < 0.0


def test_right_padded_window_is_causal_and_reports_length():
    trace = {
        frame: np.full(9, frame, dtype=np.float32)
        for frame in range(1, 8)
    }
    values, length = _right_padded_window(trace, end_frame=4, window_frames=3)
    assert length == 3
    assert values[:, 0].tolist() == [2.0, 3.0, 4.0]
