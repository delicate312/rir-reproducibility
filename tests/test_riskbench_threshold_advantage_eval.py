from __future__ import annotations

import pytest

from experiments.ego_immune_recognition.riskbench_metadata.threshold_advantage_eval import (
    _lift_summary,
    _pooling_sensitivity,
    maximum_rolling_mean,
)


def test_maximum_rolling_mean_uses_contiguous_peak_neighborhood() -> None:
    values = [0.1, 0.2, 0.9, 0.8, 0.7, 0.1]

    assert maximum_rolling_mean(values, 3) == pytest.approx(0.8)


def test_lift_summary_reports_model_win_rate() -> None:
    rows = [
        {"peak_neighborhood_lift": 0.2},
        {"peak_neighborhood_lift": 0.1},
        {"peak_neighborhood_lift": -0.1},
    ]

    result = _lift_summary(rows, bootstrap_samples=100, seed=7)

    assert result["event_count"] == 3
    assert result["mean_lift"] == pytest.approx(0.2 / 3.0)
    assert result["model_win_rate"] == pytest.approx(2.0 / 3.0)


def test_pooling_sensitivity_uses_matching_model_and_ttc_widths() -> None:
    rows = [
        {
            "positive": 1,
            "ttc_detected": 0,
            "model_pool_1": 0.9,
            "ttc_pool_1": 0.4,
            "lift_pool_1": 0.5,
            "model_pool_5": 0.8,
            "ttc_pool_5": 0.5,
            "lift_pool_5": 0.3,
        },
        {
            "positive": 0,
            "ttc_detected": 0,
            "model_pool_1": 0.2,
            "ttc_pool_1": 0.3,
            "lift_pool_1": -0.1,
            "model_pool_5": 0.1,
            "ttc_pool_5": 0.4,
            "lift_pool_5": -0.3,
        },
    ]

    result = _pooling_sensitivity(
        rows,
        pooling_frames=[1, 5],
        bootstrap_samples=100,
        seed=7,
    )

    assert [row["pooling_frames"] for row in result] == [1, 5]
    assert result[0]["ttc_negative_model_auroc"] == pytest.approx(1.0)
    assert result[0]["ttc_negative_ttc_auroc"] == pytest.approx(1.0)
    assert result[1]["ttc_blind_mean_lift"] == pytest.approx(0.3)
