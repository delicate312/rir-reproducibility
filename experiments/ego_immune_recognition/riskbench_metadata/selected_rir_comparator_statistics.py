"""Paired family-bootstrap comparisons for nested-selected RIR.

This analysis joins the final hard-safe-only nested-selected RIR predictions
with the already frozen controlled-comparator predictions.  Repeated-seed
outputs are averaged within each event before scenario families are resampled.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .pair_damp_freeze import PROJECT_ROOT


RIR_EVENTS = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_damp_necessity_v2_nested"
    / "outer_event_predictions.csv"
)
CONTROLLED_EVENTS = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_controlled_baselines_v2"
    / "all_seed_events.csv"
)
OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_damp_necessity_v2_nested"
    / "selected_rir_vs_comparators_bootstrap.csv"
)
REFERENCE = "rir_nested_selected"
COMPARATORS = (
    "ttc",
    "center_cv",
    "center_kalman",
    "supervised_gru_event",
    "supervised_transformer_event",
)
REPLICATES = 10_000


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir-events", type=Path, default=RIR_EVENTS)
    parser.add_argument("--controlled-events", type=Path, default=CONTROLLED_EVENTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fpr-budget", type=float, default=0.05)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    return parser.parse_args()


def _load(
    rir_events: Path,
    controlled_events: Path,
    fpr_budget: float,
) -> pd.DataFrame:
    rir = pd.read_csv(rir_events)
    rir = rir[
        rir["fpr_budget"].eq(fpr_budget) & rir["method"].eq(REFERENCE)
    ].copy()
    controlled = pd.read_csv(controlled_events)
    controlled = controlled[
        controlled["fpr_budget"].eq(fpr_budget)
        & controlled["method"].isin(COMPARATORS)
    ].copy()
    return pd.concat(
        [
            rir[
                [
                    "method",
                    "scenario_group_id",
                    "event_uid",
                    "positive",
                    "ttc_detected",
                    "final_detected",
                    "blind_rescue",
                ]
            ],
            controlled[
                [
                    "method",
                    "scenario_group_id",
                    "event_uid",
                    "positive",
                    "ttc_detected",
                    "final_detected",
                    "blind_rescue",
                ]
            ],
        ],
        ignore_index=True,
    )


def run(
    replicates: int = REPLICATES,
    *,
    rir_events: Path = RIR_EVENTS,
    controlled_events: Path = CONTROLLED_EVENTS,
    output_path: Path = OUTPUT,
    fpr_budget: float = 0.05,
) -> list[dict[str, Any]]:
    rows = _load(rir_events, controlled_events, fpr_budget).to_dict("records")
    rng = np.random.default_rng(20260824)
    output: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        for metric, population, field in (
            ("overall_recall_delta", "positive", "final_detected"),
            ("blind_rescue_delta", "blind_positive", "blind_rescue"),
            ("incremental_fpr_delta", "blind_normal", "blind_rescue"),
        ):
            values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            for row in rows:
                positive = bool(int(row["positive"]))
                blind = not bool(int(row["ttc_detected"]))
                include = (
                    (population == "positive" and positive)
                    or (population == "blind_positive" and positive and blind)
                    or (population == "blind_normal" and not positive and blind)
                )
                if include and row["method"] in (REFERENCE, comparator):
                    values[
                        (
                            str(row["method"]),
                            str(row["scenario_group_id"]),
                            str(row["event_uid"]),
                        )
                    ].append(float(row[field]))
            paired: dict[str, list[float]] = defaultdict(list)
            keys = {
                (family, uid)
                for method, family, uid in values
                if method == REFERENCE
            }
            for family, uid in keys:
                left = values.get((REFERENCE, family, uid))
                right = values.get((comparator, family, uid))
                if left and right:
                    paired[family].append(float(np.mean(left) - np.mean(right)))
            families = sorted(paired)
            observed = [value for family in families for value in paired[family]]
            draws = np.empty(replicates, dtype=float)
            for index in range(replicates):
                sampled = rng.choice(families, size=len(families), replace=True)
                draws[index] = float(
                    np.mean(
                        [value for family in sampled for value in paired[family]]
                    )
                )
            lower_tail = (float(np.sum(draws <= 0.0)) + 1.0) / (replicates + 1.0)
            upper_tail = (float(np.sum(draws >= 0.0)) + 1.0) / (replicates + 1.0)
            output.append(
                {
                    "reference": REFERENCE,
                    "comparator": comparator,
                    "metric": metric,
                    "paired_family_count": len(families),
                    "estimate_reference_minus_comparator": float(np.mean(observed)),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "two_sided_bootstrap_p": min(
                        1.0, 2.0 * min(lower_tail, upper_tail)
                    ),
                    "bootstrap_replicates": replicates,
                }
            )
    _write_csv(output_path, output)
    return output


if __name__ == "__main__":
    args = parse_args()
    for row in run(
        args.replicates,
        rir_events=args.rir_events,
        controlled_events=args.controlled_events,
        output_path=args.output,
        fpr_budget=args.fpr_budget,
    ):
        print(row)
