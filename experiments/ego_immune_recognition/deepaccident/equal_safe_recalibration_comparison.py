"""Compare DeepAccident scores under equal safe-only cross-fitted calibration.

For every method, seed, budget, and held-out family, the operating threshold is
estimated only from physical-reference-negative matched-normal views belonging
to the other families.  The underlying model and event scores remain frozen.
This analysis was designed after inspecting the zero-adaptation result and is
therefore an exploratory mechanism/calibration diagnostic.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..riskbench_metadata.core_hybrid_evaluation import threshold_at_event_fpr
from .summarize_external_validation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _cluster_rate,
    _format_rate,
    _write_csv,
)


BUDGETS = (0.05, 0.10)
METHODS = (
    "rir_all_safe_selected",
    "cpa_peak",
    "cpa_temporal",
    "cpa_ema_warmup",
    "cpa_cusum_warmup",
    "raw_residual_hybrid",
    "constant_damp_control",
)
PRIMARY_COMPARATORS = ("cpa_peak", "raw_residual_hybrid")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_strings(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{key: str(value) for key, value in row.items()} for row in rows]


def _metric(
    rows: list[dict[str, Any]],
    *,
    budget: float,
    method: str,
    name: str,
    outcome: str,
    denominator: Callable[[dict[str, str]], bool],
    numerator_key: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    current = _as_strings(
        [
            row
            for row in rows
            if float(row["budget"]) == budget
            and row["method"] == method
            and row["outcome"] == outcome
        ]
    )
    estimate, low, high, numerator, denominator_count = _cluster_rate(
        current,
        denominator=denominator,
        numerator=lambda row: bool(int(row[numerator_key])),
        rng=rng,
    )
    return {
        "budget": budget,
        "method": method,
        "metric": name,
        "estimate": estimate,
        "family_bootstrap_ci_low": low,
        "family_bootstrap_ci_high": high,
        "numerator_seed_views": numerator,
        "denominator_seed_views": denominator_count,
    }


def _family_counts(
    rows: list[dict[str, Any]],
    *,
    denominator: Callable[[dict[str, Any]], bool],
    numerator_key: str,
) -> tuple[list[str], np.ndarray]:
    families = sorted({str(row["family"]) for row in rows})
    counts = []
    for family in families:
        eligible = [
            row
            for row in rows
            if row["family"] == family and denominator(row)
        ]
        counts.append(
            (sum(bool(int(row[numerator_key])) for row in eligible), len(eligible))
        )
    return families, np.asarray(counts, dtype=float)


def _paired_difference(
    rows: list[dict[str, Any]],
    *,
    budget: float,
    method_a: str,
    method_b: str,
    metric: str,
    outcome: str,
    denominator: Callable[[dict[str, Any]], bool],
    numerator_key: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    method_rows = {}
    for method in (method_a, method_b):
        method_rows[method] = [
            row
            for row in rows
            if float(row["budget"]) == budget
            and row["method"] == method
            and row["outcome"] == outcome
        ]
    families_a, counts_a = _family_counts(
        method_rows[method_a],
        denominator=denominator,
        numerator_key=numerator_key,
    )
    families_b, counts_b = _family_counts(
        method_rows[method_b],
        denominator=denominator,
        numerator_key=numerator_key,
    )
    if families_a != families_b:
        raise ValueError("Methods do not contain identical family units")

    def difference(indices: np.ndarray) -> float | None:
        denominator_a = float(counts_a[indices, 1].sum())
        denominator_b = float(counts_b[indices, 1].sum())
        if denominator_a == 0 or denominator_b == 0:
            return None
        return float(
            counts_a[indices, 0].sum() / denominator_a
            - counts_b[indices, 0].sum() / denominator_b
        )

    all_indices = np.arange(len(families_a))
    point = difference(all_indices)
    boot = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(families_a), size=len(families_a))
        value = difference(sample)
        if value is not None:
            boot.append(value)
    low, high = np.quantile(np.asarray(boot), (0.025, 0.975))
    return {
        "budget": budget,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "difference": point,
        "family_bootstrap_ci_low": float(low),
        "family_bootstrap_ci_high": float(high),
        "bootstrap_replicates_retained": len(boot),
    }


def _ranking_metrics(
    source_rows: list[dict[str, str]], budget: float, method: str
) -> dict[str, Any]:
    seed_metrics = []
    for seed in sorted({int(row["seed"]) for row in source_rows}):
        current = [
            row
            for row in source_rows
            if float(row["budget"]) == budget
            and row["method"] == method
            and int(row["seed"]) == seed
        ]
        labels = np.asarray([row["outcome"] == "accident" for row in current])
        scores = np.asarray([float(row["score"]) for row in current])
        paired = defaultdict(dict)
        for row in current:
            paired[row["view_uid"]][row["outcome"]] = float(row["score"])
        ordering = np.mean(
            [
                values["accident"] > values["normal"]
                for values in paired.values()
                if {"accident", "normal"}.issubset(values)
            ]
        )
        seed_metrics.append(
            {
                "seed": seed,
                "auroc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "standardized_pauroc_5": float(
                    roc_auc_score(labels, scores, max_fpr=0.05)
                ),
                "standardized_pauroc_10": float(
                    roc_auc_score(labels, scores, max_fpr=0.10)
                ),
                "paired_accident_score_higher": float(ordering),
            }
        )
    return {
        "budget": budget,
        "method": method,
        **{
            f"{key}_mean_over_seeds": float(
                np.mean([row[key] for row in seed_metrics])
            )
            for key in (
                "auroc",
                "average_precision",
                "standardized_pauroc_5",
                "standardized_pauroc_10",
                "paired_accident_score_higher",
            )
        },
    }


def _ranking_triplet(rows: list[dict[str, str]]) -> dict[str, float]:
    labels = np.asarray([row["outcome"] == "accident" for row in rows])
    scores = np.asarray([float(row["score"]) for row in rows])
    paired: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        paired[row["view_uid"]][row["outcome"]].append(float(row["score"]))
    comparisons = [
        accident > normal
        for values in paired.values()
        for accident, normal in zip(values["accident"], values["normal"])
    ]
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "paired_accident_score_higher": float(
            np.mean(
                [
                    value for value in comparisons
                ]
            )
        ),
    }


def _ranking_differences(
    source_rows: list[dict[str, str]],
    *,
    budget: float,
    comparator: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    methods = ("rir_all_safe_selected", comparator)
    seeds = sorted({int(row["seed"]) for row in source_rows})
    families = sorted({row["family"] for row in source_rows})
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if float(row["budget"]) == budget and row["method"] in methods:
            grouped[(row["method"], int(row["seed"]), row["family"])].append(row)

    def differences(sample: np.ndarray) -> dict[str, float]:
        by_method: dict[str, dict[str, list[float]]] = {}
        for method in methods:
            seed_values = []
            for seed in seeds:
                rows = [
                    row
                    for family_index in sample
                    for row in grouped[(method, seed, families[int(family_index)])]
                ]
                seed_values.append(_ranking_triplet(rows))
            by_method[method] = {
                key: float(np.mean([values[key] for values in seed_values]))
                for key in seed_values[0]
            }
        return {
            key: by_method[methods[0]][key] - by_method[methods[1]][key]
            for key in by_method[methods[0]]
        }

    point = differences(np.arange(len(families)))
    bootstrap = {key: [] for key in point}
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(families), size=len(families))
        values = differences(sample)
        for key, value in values.items():
            bootstrap[key].append(value)
    output = []
    for metric, value in point.items():
        low, high = np.quantile(np.asarray(bootstrap[metric]), (0.025, 0.975))
        output.append(
            {
                "budget": budget,
                "method_a": methods[0],
                "method_b": methods[1],
                "metric": metric,
                "difference": value,
                "family_bootstrap_ci_low": float(low),
                "family_bootstrap_ci_high": float(high),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-results", type=Path, required=True)
    parser.add_argument("--guarded-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"Refuse to overwrite comparison output: {output_root}")

    source_rows = _read(args.relation_results.expanduser().resolve() / "events.csv")
    guarded_rows = [
        row
        for row in _read(
            args.guarded_results.expanduser().resolve() / "union_events.csv"
        )
        if float(row["budget"]) == 0.10
    ]
    physical = {
        (int(row["seed"]), row["view_uid"], row["outcome"]): bool(
            int(row["physical_hit"])
        )
        for row in guarded_rows
    }

    calibrated_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for method in METHODS:
            method_rows = [
                row
                for row in source_rows
                if float(row["budget"]) == budget and row["method"] == method
            ]
            for row in method_rows:
                seed = int(row["seed"])
                family = row["family"]
                calibration = [
                    other
                    for other in method_rows
                    if int(other["seed"]) == seed
                    and other["outcome"] == "normal"
                    and other["family"] != family
                    and not physical[(seed, other["view_uid"], "normal")]
                ]
                threshold, calibration_rate = threshold_at_event_fpr(
                    [float(other["score"]) for other in calibration], budget
                )
                score = float(row["score"])
                relation_hit = score >= threshold
                physical_hit = physical[(seed, row["view_uid"], row["outcome"])]
                calibrated_rows.append(
                    {
                        "budget": budget,
                        "method": method,
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
                        "score": score,
                        "physical_hit": int(physical_hit),
                        "relation_hit": int(relation_hit),
                        "relation_only_rescue": int(
                            relation_hit and not physical_hit
                        ),
                        "union_hit": int(relation_hit or physical_hit),
                    }
                )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = []
    for budget in BUDGETS:
        for method in METHODS:
            metrics.extend(
                [
                    _metric(
                        calibrated_rows,
                        budget=budget,
                        method=method,
                        name="relation_accident_recall",
                        outcome="accident",
                        denominator=lambda row: True,
                        numerator_key="relation_hit",
                        rng=rng,
                    ),
                    _metric(
                        calibrated_rows,
                        budget=budget,
                        method=method,
                        name="union_accident_recall",
                        outcome="accident",
                        denominator=lambda row: True,
                        numerator_key="union_hit",
                        rng=rng,
                    ),
                    _metric(
                        calibrated_rows,
                        budget=budget,
                        method=method,
                        name="hidden_rescue",
                        outcome="accident",
                        denominator=lambda row: not int(row["physical_hit"]),
                        numerator_key="relation_hit",
                        rng=rng,
                    ),
                    _metric(
                        calibrated_rows,
                        budget=budget,
                        method=method,
                        name="added_candidates",
                        outcome="normal",
                        denominator=lambda row: not int(row["physical_hit"]),
                        numerator_key="relation_hit",
                        rng=rng,
                    ),
                    _metric(
                        calibrated_rows,
                        budget=budget,
                        method=method,
                        name="union_normal_candidates",
                        outcome="normal",
                        denominator=lambda row: True,
                        numerator_key="union_hit",
                        rng=rng,
                    ),
                ]
            )

    comparisons = []
    for budget in BUDGETS:
        for comparator in PRIMARY_COMPARATORS:
            for metric, outcome, denominator, numerator_key in (
                (
                    "relation_accident_recall",
                    "accident",
                    lambda row: True,
                    "relation_hit",
                ),
                (
                    "union_accident_recall",
                    "accident",
                    lambda row: True,
                    "union_hit",
                ),
                (
                    "hidden_rescue",
                    "accident",
                    lambda row: not int(row["physical_hit"]),
                    "relation_hit",
                ),
                (
                    "added_candidates",
                    "normal",
                    lambda row: not int(row["physical_hit"]),
                    "relation_hit",
                ),
            ):
                comparisons.append(
                    _paired_difference(
                        calibrated_rows,
                        budget=budget,
                        method_a="rir_all_safe_selected",
                        method_b=comparator,
                        metric=metric,
                        outcome=outcome,
                        denominator=denominator,
                        numerator_key=numerator_key,
                        rng=rng,
                    )
                )

    ranking = [
        _ranking_metrics(source_rows, budget, method)
        for budget in BUDGETS
        for method in METHODS
    ]
    ranking_differences = [
        row
        for budget in BUDGETS
        for comparator in PRIMARY_COMPARATORS
        for row in _ranking_differences(
            source_rows,
            budget=budget,
            comparator=comparator,
            rng=rng,
        )
    ]
    output_root.mkdir(parents=True)
    _write_csv(output_root / "cross_fitted_events.csv", calibrated_rows)
    _write_csv(output_root / "family_bootstrap_metrics.csv", metrics)
    _write_csv(output_root / "paired_method_differences.csv", comparisons)
    _write_csv(output_root / "ranking_metrics.csv", ranking)
    _write_csv(output_root / "ranking_differences.csv", ranking_differences)

    lookup = {
        (row["budget"], row["method"], row["metric"]): row for row in metrics
    }
    lines = [
        "# Equal safe-only recalibration comparison",
        "",
        "Exploratory post-hoc diagnostic. Every method uses the same leave-one-family-out "
        "physical-negative normal calibration pool and target added-candidate budget. "
        "No accident score or held-out-family normal score selects a threshold.",
        "",
    ]
    for budget in BUDGETS:
        lines.extend(
            [
                f"## {budget:.0%} added-candidate budget",
                "",
                "| Method | Relation recall | Union recall | Hidden rescue | Added candidates | Union normal candidates |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            lines.append(
                f"| {method} | "
                f"{_format_rate(lookup[(budget, method, 'relation_accident_recall')])} | "
                f"{_format_rate(lookup[(budget, method, 'union_accident_recall')])} | "
                f"{_format_rate(lookup[(budget, method, 'hidden_rescue')])} | "
                f"{_format_rate(lookup[(budget, method, 'added_candidates')])} | "
                f"{_format_rate(lookup[(budget, method, 'union_normal_candidates')])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "This diagnostic was specified after the fixed RiskBench-threshold transfer was "
            "inspected. Bootstrap intervals resample the already cross-fitted family decisions; "
            "they do not repeat threshold estimation inside every bootstrap replicate. Results "
            "must therefore remain supplementary/exploratory and cannot replace the frozen "
            "39-family external result.",
            "",
        ]
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    (output_root / "analysis.json").write_text(
        json.dumps(
            {
                "status": "post_hoc_equal_safe_only_recalibration_diagnostic",
                "budgets": list(BUDGETS),
                "methods": list(METHODS),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "threshold_uses_accident_scores": False,
                "threshold_uses_held_out_family_normal": False,
                "metrics": metrics,
                "comparisons": comparisons,
                "ranking_metrics": ranking,
                "ranking_differences": ranking_differences,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print((output_root / "SUMMARY.md").read_text(encoding="utf-8"))
    print("\nPrimary paired differences (RIR minus comparator):")
    for row in comparisons:
        print(
            f"{row['budget']:.0%} {row['metric']} vs {row['method_b']}: "
            f"{row['difference']:+.3f} "
            f"[{row['family_bootstrap_ci_low']:+.3f}, "
            f"{row['family_bootstrap_ci_high']:+.3f}]"
        )


if __name__ == "__main__":
    main()
