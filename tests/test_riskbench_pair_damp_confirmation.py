from __future__ import annotations

from experiments.ego_immune_recognition.riskbench_metadata.pair_damp_confirmation import (
    _cpa_trace,
    _paired_comparison,
    _summarize_events,
)


def test_cpa_trace_uses_maximum_exact_relation() -> None:
    frame = {
        "frame_position": 1,
        "frames_to_anchor": 4,
        "aligned_geometry_relations": {
            "a": {"cpa_only": 0.4},
            "b": {"cpa_only": 0.8},
        },
    }
    assert _cpa_trace([frame])[0][1] == 0.8


def test_event_summary_counts_only_incremental_rescues() -> None:
    rows = [
        {"positive": 1, "ttc_detected": 0, "blind_rescue": 1, "final_detected": 1, "method_lead_seconds": 0.5},
        {"positive": 1, "ttc_detected": 1, "blind_rescue": 0, "final_detected": 1, "method_lead_seconds": ""},
        {"positive": 0, "ttc_detected": 0, "blind_rescue": 1, "final_detected": 1, "method_lead_seconds": ""},
        {"positive": 0, "ttc_detected": 1, "blind_rescue": 0, "final_detected": 1, "method_lead_seconds": ""},
    ]
    summary = _summarize_events(rows, fold="all", budget=0.1, method="method")
    assert summary["positive_hits"] == 2
    assert summary["ttc_blind_rescues"] == 1
    assert summary["false_rescues"] == 1
    assert summary["ttc_negative_hard_safe_count"] == 1


def test_paired_comparison_reports_damp_only_rescue() -> None:
    rows = []
    for method, rescue in (("cpa_peak", 0), ("frozen_pair_damp", 1)):
        rows.append(
            {
                "fold": 0,
                "event_uid": "positive",
                "positive": 1,
                "method": method,
                "fpr_budget": 0.1,
                "blind_rescue": rescue,
            }
        )
        rows.append(
            {
                "fold": 0,
                "event_uid": "normal",
                "positive": 0,
                "method": method,
                "fpr_budget": 0.1,
                "blind_rescue": 0,
            }
        )
    comparison = _paired_comparison(rows, budget=0.1)
    assert comparison["ttc_blind_positive_rescue"]["counts"]["damp_only"] == 1
