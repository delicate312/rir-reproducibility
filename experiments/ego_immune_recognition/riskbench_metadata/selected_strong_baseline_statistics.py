"""Align new baselines with the manuscript's nested-selected RIR decision head."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS = PROJECT_ROOT / "results"
OUTPUT = RESULTS / "riskbench_selected_strong_baselines_v1"
RIR_FILES = {
    0.05: RESULTS / "ego_immune_riskbench_damp_necessity_v2_nested/outer_event_predictions.csv",
    0.10: RESULTS / "ego_immune_riskbench_damp_necessity_review10_v1/outer_event_predictions.csv",
}
COMPARATOR_FILES = (
    RESULTS / "riskbench_actor_query_transformer_v1/all_seed_events.csv",
    RESULTS / "riskbench_probabilistic_rollout_v2/all_seed_events.csv",
    RESULTS / "riskbench_social_gan_clean_evaluation_v1/all_seed_events.csv",
)
METHODS = (
    "supervised_actor_query_transformer",
    "rollout_adaptive_cv_ca_shared_risk",
    "rollout_ca_shared_risk",
    "social_gan_safe_only_retrained_shared_risk",
)
REFERENCE = "rir_nested_selected"


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _load_budget(budget: float) -> pd.DataFrame:
    rir = pd.read_csv(RIR_FILES[budget])
    rir = rir[rir["method"].eq(REFERENCE)].copy()
    rir = rir.rename(columns={
        "method_score": "score", "method_detected": "detected",
        "method_lead_seconds": "lead_seconds",
    })
    pieces = [rir]
    for path in COMPARATOR_FILES:
        data = pd.read_csv(path)
        pieces.append(data[data["fpr_budget"].eq(budget) & data["method"].isin(METHODS)])
    return pd.concat(pieces, ignore_index=True)


def _population(data: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "overall_positive":
        return data[data["positive"].eq(1)]
    if name == "hidden_positive":
        return data[data["positive"].eq(1) & data["ttc_detected"].eq(0)]
    return data[data["positive"].eq(0) & data["ttc_detected"].eq(0)]


def run(replicates: int = 5000) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for budget in (0.05, 0.10):
        data = _load_budget(budget)
        for method in (REFERENCE,) + METHODS:
            current = data[data["method"].eq(method)]
            positive = _population(current, "overall_positive")
            hidden = _population(current, "hidden_positive")
            hard_safe = _population(current, "hidden_hard_safe")
            summaries.append({
                "fpr_budget": budget, "method": method,
                "positive_family_count": positive["scenario_group_id"].nunique(),
                "hidden_positive_family_count": hidden["scenario_group_id"].nunique(),
                "hidden_hard_safe_family_count": hard_safe["scenario_group_id"].nunique(),
                "overall_recall": positive["final_detected"].mean(),
                "hidden_rescue_recall": hidden["blind_rescue"].mean(),
                "additional_candidate_rate": hard_safe["blind_rescue"].mean(),
            })
        for comparator in METHODS:
            for population, field, metric in (
                ("overall_positive", "final_detected", "overall_recall_delta"),
                ("hidden_positive", "blind_rescue", "hidden_rescue_delta"),
                ("hidden_hard_safe", "blind_rescue", "additional_candidate_delta"),
            ):
                current = _population(data, population)
                averages = current.groupby(
                    ["method", "scenario_group_id", "event_uid"]
                )[field].mean()
                values = averages.to_dict()
                left = {(family, uid) for method, family, uid in values if method == REFERENCE}
                right = {(family, uid) for method, family, uid in values if method == comparator}
                by_family: dict[str, list[float]] = defaultdict(list)
                for family, uid in left & right:
                    by_family[family].append(
                        values[(REFERENCE, family, uid)] - values[(comparator, family, uid)]
                    )
                families = sorted(by_family)
                observed = float(np.mean([value for family in families for value in by_family[family]]))
                rng = np.random.default_rng(20260913)
                draws = np.asarray([
                    np.mean([
                        value for family in rng.choice(families, len(families), replace=True)
                        for value in by_family[family]
                    ]) for _ in range(replicates)
                ])
                lower = (float(np.sum(draws <= 0)) + 1) / (replicates + 1)
                upper = (float(np.sum(draws >= 0)) + 1) / (replicates + 1)
                comparisons.append({
                    "fpr_budget": budget, "reference": REFERENCE,
                    "comparator": comparator, "metric": metric,
                    "paired_family_count": len(families),
                    "estimate_reference_minus_comparator": observed,
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "two_sided_bootstrap_p": min(1.0, 2 * min(lower, upper)),
                    "bootstrap_replicates": replicates,
                })
    _write(OUTPUT / "aligned_summary.csv", summaries)
    _write(OUTPUT / "paired_family_bootstrap.csv", comparisons)
    return summaries, comparisons


if __name__ == "__main__":
    summary, comparison = run()
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(comparison).to_string(index=False))
