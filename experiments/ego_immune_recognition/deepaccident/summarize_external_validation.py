"""Summarize one or more DeepAccident external-validation batches by family."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import numpy as np


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260910
BUDGETS = (0.10, 0.05)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cluster_rate(
    rows: list[dict[str, str]],
    *,
    denominator: Callable[[dict[str, str]], bool],
    numerator: Callable[[dict[str, str]], bool],
    rng: np.random.Generator,
) -> tuple[float | None, float | None, float | None, int, int]:
    families = sorted({row["family"] for row in rows})
    counts = []
    for family in families:
        current = [row for row in rows if row["family"] == family]
        eligible = [row for row in current if denominator(row)]
        counts.append((sum(numerator(row) for row in eligible), len(eligible)))
    total_num = sum(item[0] for item in counts)
    total_den = sum(item[1] for item in counts)
    if total_den == 0:
        return None, None, None, total_num, total_den
    point = total_num / total_den
    array = np.asarray(counts, dtype=float)
    boot = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(families), size=len(families))
        denominator_sum = float(array[sample, 1].sum())
        if denominator_sum:
            boot.append(float(array[sample, 0].sum() / denominator_sum))
    low, high = np.quantile(np.asarray(boot), (0.025, 0.975))
    return point, float(low), float(high), total_num, total_den


def _metric(
    rows: list[dict[str, str]],
    *,
    budget: float,
    name: str,
    outcome: str,
    denominator_extra: Callable[[dict[str, str]], bool],
    numerator_key: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    current = [
        row
        for row in rows
        if float(row["budget"]) == budget and row["outcome"] == outcome
    ]
    point, low, high, numerator, denominator = _cluster_rate(
        current,
        denominator=denominator_extra,
        numerator=lambda row: bool(int(row[numerator_key])),
        rng=rng,
    )
    return {
        "budget": budget,
        "metric": name,
        "estimate": point,
        "family_bootstrap_ci_low": low,
        "family_bootstrap_ci_high": high,
        "numerator_seed_views": numerator,
        "denominator_seed_views": denominator,
        "family_count": len({row["family"] for row in current}),
    }


def _format_rate(row: dict[str, Any]) -> str:
    if row["estimate"] is None:
        return "not estimable"
    return (
        f"{float(row['estimate']):.1%} "
        f"[{float(row['family_bootstrap_ci_low']):.1%}, "
        f"{float(row['family_bootstrap_ci_high']):.1%}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-results", type=Path, action="append", required=True)
    parser.add_argument("--guarded-results", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--interpretation", required=True)
    args = parser.parse_args()
    if len(args.relation_results) != len(args.guarded_results):
        raise ValueError("Relation and guarded result batches must correspond one-to-one")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"Refuse to overwrite summary output: {output_root}")

    relation_rows = [
        row
        for root in args.relation_results
        for row in _read(root.expanduser().resolve() / "events.csv")
    ]
    guarded_rows = [
        row
        for root in args.guarded_results
        for row in _read(root.expanduser().resolve() / "union_events.csv")
    ]
    localization_rows = [
        row
        for root in args.relation_results
        for row in _read(root.expanduser().resolve() / "localization.csv")
    ]
    families = sorted({row["family"] for row in guarded_rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = []
    for budget in BUDGETS:
        metrics.extend(
            [
                _metric(
                    guarded_rows,
                    budget=budget,
                    name="physical_accident_recall",
                    outcome="accident",
                    denominator_extra=lambda row: True,
                    numerator_key="physical_hit",
                    rng=rng,
                ),
                _metric(
                    guarded_rows,
                    budget=budget,
                    name="rir_relation_accident_recall",
                    outcome="accident",
                    denominator_extra=lambda row: True,
                    numerator_key="relation_hit",
                    rng=rng,
                ),
                _metric(
                    guarded_rows,
                    budget=budget,
                    name="union_accident_recall",
                    outcome="accident",
                    denominator_extra=lambda row: True,
                    numerator_key="union_hit",
                    rng=rng,
                ),
                _metric(
                    guarded_rows,
                    budget=budget,
                    name="rir_rescue_among_physical_negative_accidents",
                    outcome="accident",
                    denominator_extra=lambda row: not int(row["physical_hit"]),
                    numerator_key="relation_hit",
                    rng=rng,
                ),
                _metric(
                    guarded_rows,
                    budget=budget,
                    name="physical_matched_normal_candidate_rate",
                    outcome="normal",
                    denominator_extra=lambda row: True,
                    numerator_key="physical_hit",
                    rng=rng,
                ),
                _metric(
                    guarded_rows,
                    budget=budget,
                    name="added_rir_candidates_among_physical_negative_normals",
                    outcome="normal",
                    denominator_extra=lambda row: not int(row["physical_hit"]),
                    numerator_key="relation_hit",
                    rng=rng,
                ),
                _metric(
                    guarded_rows,
                    budget=budget,
                    name="union_matched_normal_candidate_rate",
                    outcome="normal",
                    denominator_extra=lambda row: True,
                    numerator_key="union_hit",
                    rng=rng,
                ),
            ]
        )

    primary_relation = [
        row
        for row in relation_rows
        if float(row["budget"]) == 0.10
        and row["method"] == "rir_all_safe_selected"
        and row["outcome"] == "accident"
    ]
    primary_localization = [
        row
        for row in localization_rows
        if float(row["budget"]) == 0.10 and int(row["hit"])
    ]
    lead_times = [
        float(row["first_crossing_lead_seconds"])
        for row in primary_relation
        if row["first_crossing_lead_seconds"] not in ("", "None")
    ]
    localization_rate = (
        mean(int(row["target_top1"]) for row in primary_localization)
        if primary_localization
        else None
    )

    output_root.mkdir(parents=True)
    _write_csv(output_root / "family_bootstrap_metrics.csv", metrics)
    summary = {
        "title": args.title,
        "family_count": len(families),
        "accident_view_count": len(
            {
                row["view_uid"]
                for row in guarded_rows
                if row["outcome"] == "accident"
            }
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_rir_detected_seed_views": len(primary_localization),
        "primary_localization_top1": localization_rate,
        "primary_rir_lead_time_mean_seconds": mean(lead_times) if lead_times else None,
        "primary_rir_lead_time_median_seconds": (
            float(np.median(lead_times)) if lead_times else None
        ),
        "interpretation": args.interpretation,
        "metrics": metrics,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lookup = {(row["budget"], row["metric"]): row for row in metrics}
    lines = [
        f"# {args.title}",
        "",
        f"Independent units: {len(families)} scenario families; 95% intervals use "
        f"{BOOTSTRAP_REPLICATES:,} family-cluster bootstrap samples.",
        "",
        "| Budget | Physical recall | RIR relation recall | Physical OR RIR recall | RIR rescue among physical-negative accidents | Physical normal candidates | Added RIR candidates | Union normal candidates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        lines.append(
            f"| {budget:.0%} | "
            f"{_format_rate(lookup[(budget, 'physical_accident_recall')])} | "
            f"{_format_rate(lookup[(budget, 'rir_relation_accident_recall')])} | "
            f"{_format_rate(lookup[(budget, 'union_accident_recall')])} | "
            f"{_format_rate(lookup[(budget, 'rir_rescue_among_physical_negative_accidents')])} | "
            f"{_format_rate(lookup[(budget, 'physical_matched_normal_candidate_rate')])} | "
            f"{_format_rate(lookup[(budget, 'added_rir_candidates_among_physical_negative_normals')])} | "
            f"{_format_rate(lookup[(budget, 'union_matched_normal_candidate_rate')])} |"
        )
    lines.extend(
        [
            "",
            f"At the primary 10% relation operating point, collision-partner top-1 "
            f"localization is {localization_rate:.1%} among detected seed-view cases. "
            f"Mean/median first-crossing lead time within the observable interval is "
            f"{mean(lead_times):.2f}/{float(np.median(lead_times)):.2f} s.",
            "",
            args.interpretation,
            "",
        ]
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print((output_root / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
