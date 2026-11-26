from __future__ import annotations

import pytest

from experiments.ego_immune_recognition.riskbench_metadata.pair_damp_v2_evaluation import (
    summarize_continuous_metrics,
    summarize_events,
)


def _row(
    *,
    positive: bool,
    ttc: float,
    baseline: float,
    onset: float,
    ttc_detected: bool = False,
    baseline_detected: bool = False,
    onset_detected: bool = False,
) -> dict:
    row = {
        "fpr_budget": 0.05,
        "positive": int(positive),
        "ttc_detected": int(ttc_detected),
        "baseline_detected": int(baseline_detected),
        "onset_detected": int(onset_detected),
        "baseline_lead_seconds": 0.7 if baseline_detected else "",
        "onset_lead_seconds": 0.8 if onset_detected else "",
    }
    for width in (1, 5, 10):
        row[f"ttc_pool_{width}"] = ttc
        row[f"baseline_pool_{width}"] = baseline
        row[f"onset_pool_{width}"] = onset
        row[f"baseline_vs_ttc_lift_{width}"] = baseline - ttc
        row[f"onset_vs_ttc_lift_{width}"] = onset - ttc
    return row


def test_continuous_summary_keeps_lift_and_discrimination_separate() -> None:
    positive = _row(positive=True, ttc=0.2, baseline=0.7, onset=0.9)
    normal = _row(positive=False, ttc=0.2, baseline=0.4, onset=0.3)
    summary = summarize_continuous_metrics([positive, normal])[0]
    assert summary["ttc_negative_metrics"]["onset"]["auroc"] == 1.0
    assert summary["onset_vs_ttc_positive"]["mean_lift"] == pytest.approx(0.7)
    assert summary["onset_vs_baseline_positive"]["mean_lift"] == pytest.approx(0.2)
    assert summary["net_onset_vs_ttc_lift"]["mean_difference"] == pytest.approx(0.6)


def test_event_summary_reports_incremental_rescue_and_false_rescue() -> None:
    rows = [
        _row(
            positive=True,
            ttc=0.2,
            baseline=0.8,
            onset=1.0,
            onset_detected=True,
        ),
        _row(
            positive=False,
            ttc=0.2,
            baseline=0.7,
            onset=1.0,
            onset_detected=True,
        ),
    ]
    summaries = summarize_events(rows)
    onset = next(row for row in summaries if row["method"] == "onset")
    assert onset["ttc_blind_rescues"] == 1
    assert onset["false_rescues"] == 1
    assert onset["mean_rescue_lead_seconds"] == pytest.approx(0.8)
