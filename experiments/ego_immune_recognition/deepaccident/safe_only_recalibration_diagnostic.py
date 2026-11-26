"""Post-hoc diagnosis of normal-only threshold transfer on DeepAccident.

The model and score are frozen.  For each held-out scenario family, a relation
threshold is selected using only physical-negative matched-normal views from
the other families.  No accident score or held-out-family normal score enters
that threshold.  Because this analysis was motivated after inspecting the
fixed-threshold result, it is exploratory rather than confirmatory evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..riskbench_metadata.core_hybrid_evaluation import threshold_at_event_fpr
from .summarize_external_validation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _cluster_rate,
    _format_rate,
    _write_csv,
)


BUDGETS = (0.05, 0.10)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric(
    rows: list[dict[str, Any]],
    *,
    budget: float,
    name: str,
    outcome: str,
    denominator_key: str | None,
    numerator_key: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    current = [
        {key: str(value) for key, value in row.items()}
        for row in rows
        if float(row["target_added_candidate_budget"]) == budget
        and row["outcome"] == outcome
    ]
    estimate, low, high, numerator, denominator = _cluster_rate(
        current,
        denominator=(
            (lambda row: True)
            if denominator_key is None
            else (lambda row: not int(row[denominator_key]))
        ),
        numerator=lambda row: bool(int(row[numerator_key])),
        rng=rng,
    )
    return {
        "budget": budget,
        "metric": name,
        "estimate": estimate,
        "family_bootstrap_ci_low": low,
        "family_bootstrap_ci_high": high,
        "numerator_seed_views": numerator,
        "denominator_seed_views": denominator,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-results", type=Path, required=True)
    parser.add_argument("--guarded-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"Refuse to overwrite diagnostic output: {output_root}")

    relation_rows = _read(args.relation_results.expanduser().resolve() / "events.csv")
    guarded_rows = [
        row
        for row in _read(
            args.guarded_results.expanduser().resolve() / "union_events.csv"
        )
        if float(row["budget"]) == 0.10
    ]
    relation = {
        (
            float(row["budget"]),
            int(row["seed"]),
            row["view_uid"],
            row["outcome"],
        ): row
        for row in relation_rows
        if row["method"] == "rir_all_safe_selected"
    }

    output_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for row in guarded_rows:
            seed = int(row["seed"])
            family = row["family"]
            calibration = [
                other
                for other in guarded_rows
                if int(other["seed"]) == seed
                and other["outcome"] == "normal"
                and other["family"] != family
                and not int(other["physical_hit"])
            ]
            calibration_scores = [
                float(
                    relation[(budget, seed, other["view_uid"], "normal")][
                        "score"
                    ]
                )
                for other in calibration
            ]
            threshold, calibration_rate = threshold_at_event_fpr(
                calibration_scores, budget
            )
            score = float(
                relation[
                    (budget, seed, row["view_uid"], row["outcome"])
                ]["score"]
            )
            relation_hit = score >= threshold
            physical_hit = bool(int(row["physical_hit"]))
            output_rows.append(
                {
                    "target_added_candidate_budget": budget,
                    "seed": seed,
                    "fold": int(row["fold"]),
                    "family": family,
                    "view_uid": row["view_uid"],
                    "outcome": row["outcome"],
                    "calibration_family_count": len(
                        {other["family"] for other in calibration}
                    ),
                    "calibration_physical_negative_normal_views": len(calibration),
                    "threshold": threshold,
                    "calibration_candidate_rate": calibration_rate,
                    "relation_score": score,
                    "physical_hit": int(physical_hit),
                    "relation_hit": int(relation_hit),
                    "relation_only_rescue": int(relation_hit and not physical_hit),
                    "union_hit": int(relation_hit or physical_hit),
                }
            )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = []
    for budget in BUDGETS:
        metrics.extend(
            [
                _metric(
                    output_rows,
                    budget=budget,
                    name="physical_accident_recall",
                    outcome="accident",
                    denominator_key=None,
                    numerator_key="physical_hit",
                    rng=rng,
                ),
                _metric(
                    output_rows,
                    budget=budget,
                    name="rir_relation_accident_recall",
                    outcome="accident",
                    denominator_key=None,
                    numerator_key="relation_hit",
                    rng=rng,
                ),
                _metric(
                    output_rows,
                    budget=budget,
                    name="union_accident_recall",
                    outcome="accident",
                    denominator_key=None,
                    numerator_key="union_hit",
                    rng=rng,
                ),
                _metric(
                    output_rows,
                    budget=budget,
                    name="rir_rescue_among_physical_negative_accidents",
                    outcome="accident",
                    denominator_key="physical_hit",
                    numerator_key="relation_hit",
                    rng=rng,
                ),
                _metric(
                    output_rows,
                    budget=budget,
                    name="added_rir_candidates_among_physical_negative_normals",
                    outcome="normal",
                    denominator_key="physical_hit",
                    numerator_key="relation_hit",
                    rng=rng,
                ),
                _metric(
                    output_rows,
                    budget=budget,
                    name="union_matched_normal_candidate_rate",
                    outcome="normal",
                    denominator_key=None,
                    numerator_key="union_hit",
                    rng=rng,
                ),
            ]
        )

    output_root.mkdir(parents=True)
    _write_csv(output_root / "cross_fitted_events.csv", output_rows)
    _write_csv(output_root / "family_bootstrap_metrics.csv", metrics)
    lookup = {(row["budget"], row["metric"]): row for row in metrics}
    thresholds = {
        str(budget): {
            "median": float(
                np.median(
                    [
                        row["threshold"]
                        for row in output_rows
                        if row["target_added_candidate_budget"] == budget
                    ]
                )
            ),
            "min": min(
                row["threshold"]
                for row in output_rows
                if row["target_added_candidate_budget"] == budget
            ),
            "max": max(
                row["threshold"]
                for row in output_rows
                if row["target_added_candidate_budget"] == budget
            ),
        }
        for budget in BUDGETS
    }
    payload = {
        "status": "post_hoc_safe_only_cross_fitted_threshold_diagnostic",
        "selection_timing": (
            "Motivated after inspecting the predeclared fixed-threshold external result; "
            "must not be described as confirmatory or preregistered."
        ),
        "calibration_rule": (
            "For each held-out family and frozen seed, choose the least strict observed "
            "threshold below the target candidate budget using only physical-negative "
            "matched-normal views from every other family."
        ),
        "accident_labels_used_for_threshold": False,
        "held_out_family_normal_used_for_threshold": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "thresholds": thresholds,
        "metrics": metrics,
    }
    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Post-hoc safe-only external threshold diagnostic",
        "",
        "This is an exploratory diagnosis, not the primary independent result. Each "
        "family is held out from threshold calibration, and only matched-normal views "
        "from the other families are used.",
        "",
        "| Added-relation budget | Physical recall | RIR relation recall | Union recall | Hidden rescue | Realized added candidates | Union normal candidates |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        lines.append(
            f"| {budget:.0%} | "
            f"{_format_rate(lookup[(budget, 'physical_accident_recall')])} | "
            f"{_format_rate(lookup[(budget, 'rir_relation_accident_recall')])} | "
            f"{_format_rate(lookup[(budget, 'union_accident_recall')])} | "
            f"{_format_rate(lookup[(budget, 'rir_rescue_among_physical_negative_accidents')])} | "
            f"{_format_rate(lookup[(budget, 'added_rir_candidates_among_physical_negative_normals')])} | "
            f"{_format_rate(lookup[(budget, 'union_matched_normal_candidate_rate')])} |"
        )
    lines.extend(
        [
            "",
            "The fixed RiskBench 10% threshold yielded only 1.4% added candidates on "
            "DeepAccident. The cross-fitted diagnostic tests whether the missing rescue "
            "is attributable to threshold transfer rather than lack of useful relation scores.",
            "",
        ]
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print((output_root / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
