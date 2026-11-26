"""Quantify how a frozen relation score can prioritize manual scene review.

The input is an outer-test event sheet produced after hard-safe-only threshold
calibration.  No positive label is used to fit a score or threshold here.  The
binary operating point is reported exactly as frozen; ranking statistics are
descriptive outer-test retrieval metrics for the physics-reference-negative
pool and must not be described as threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_METHOD = "rir_nested_selected"
DEFAULT_REVIEW_FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.50)
DEFAULT_TARGET_RECALLS = (0.50, 0.75, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--fpr-budget", type=float)
    return parser.parse_args()


def _read_rows(
    path: Path, method: str, fpr_budget: float | None
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["method"] == method
            and (
                fpr_budget is None
                or math.isclose(float(row["fpr_budget"]), fpr_budget)
            )
        ]
    if not rows:
        raise ValueError(f"No rows for method={method!r} in {path}")
    return rows


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array))


def _event_level(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["event_uid"], []).append(row)

    output: list[dict[str, Any]] = []
    for uid, members in grouped.items():
        positive = {int(row["positive"]) for row in members}
        ttc_detected = {int(row["ttc_detected"]) for row in members}
        if len(positive) != 1 or len(ttc_detected) != 1:
            raise ValueError(f"Inconsistent labels for {uid}")
        output.append(
            {
                "event_uid": uid,
                "positive": positive.pop(),
                "physical_reference_detected": ttc_detected.pop(),
                "mean_score": _mean(
                    float(row.get("method_score", row.get("score", "nan")))
                    for row in members
                ),
                "mean_candidate": _mean(float(row["blind_rescue"]) for row in members),
                "seed_count": len(members),
                "target_type": members[0].get("target_type", ""),
            }
        )
    return output


def _ranking_rows(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blind = [row for row in events if not row["physical_reference_detected"]]
    blind.sort(key=lambda row: (-float(row["mean_score"]), row["event_uid"]))
    positives = sum(int(row["positive"]) for row in blind)
    total = len(blind)
    prevalence = positives / total

    rows: list[dict[str, Any]] = []
    for fraction in DEFAULT_REVIEW_FRACTIONS:
        reviewed = min(total, max(1, math.ceil(total * fraction)))
        hits = sum(int(row["positive"]) for row in blind[:reviewed])
        precision = hits / reviewed
        rows.append(
            {
                "review_fraction": fraction,
                "reviewed_events": reviewed,
                "hidden_danger_hits": hits,
                "hidden_danger_count": positives,
                "recall": hits / positives,
                "precision": precision,
                "enrichment_over_random": precision / prevalence,
            }
        )

    labels = np.asarray([int(row["positive"]) for row in blind], dtype=int)
    scores = np.asarray([float(row["mean_score"]) for row in blind], dtype=float)
    summary = {
        "physics_reference_negative_event_count": total,
        "hidden_danger_event_count": positives,
        "hidden_danger_prevalence": prevalence,
        "ranking_auroc": float(roc_auc_score(labels, scores)),
        "ranking_average_precision": float(average_precision_score(labels, scores)),
    }
    return rows, summary


def _target_recall_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blind = [row for row in events if not row["physical_reference_detected"]]
    blind.sort(key=lambda row: (-float(row["mean_score"]), row["event_uid"]))
    positives = sum(int(row["positive"]) for row in blind)
    output: list[dict[str, Any]] = []
    for target in DEFAULT_TARGET_RECALLS:
        required_hits = math.ceil(target * positives)
        hits = 0
        reviewed = 0
        for reviewed, row in enumerate(blind, start=1):
            hits += int(row["positive"])
            if hits >= required_hits:
                break
        fraction = reviewed / len(blind)
        output.append(
            {
                "target_recall": target,
                "achieved_recall": hits / positives,
                "reviewed_events": reviewed,
                "total_events": len(blind),
                "review_fraction": fraction,
                "workload_reduction_vs_full_review": 1.0 - fraction,
                "precision": hits / reviewed,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    events_path: Path,
    output_root: Path,
    method: str,
    fpr_budget: float | None = None,
) -> dict[str, Any]:
    rows = _read_rows(events_path, method, fpr_budget)
    events = _event_level(rows)
    ranking, summary = _ranking_rows(events)
    target_recall = _target_recall_rows(events)

    seed_metrics = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = [row for row in rows if int(row["seed"]) == seed]
        blind_positive = [
            row for row in selected
            if int(row["positive"]) and not int(row["ttc_detected"])
        ]
        blind_normal = [
            row for row in selected
            if not int(row["positive"]) and not int(row["ttc_detected"])
        ]
        seed_metrics.append(
            {
                "seed": seed,
                "hidden_danger_recall": _mean(float(row["blind_rescue"]) for row in blind_positive),
                "additional_candidate_rate": _mean(float(row["blind_rescue"]) for row in blind_normal),
            }
        )
    summary.update(
        {
            "method": method,
            "fpr_budget": float(rows[0]["fpr_budget"]),
            "binary_hidden_danger_recall_mean": _mean(row["hidden_danger_recall"] for row in seed_metrics),
            "binary_hidden_danger_recall_std": float(np.std([row["hidden_danger_recall"] for row in seed_metrics], ddof=1)),
            "binary_additional_candidate_rate_mean": _mean(row["additional_candidate_rate"] for row in seed_metrics),
            "binary_additional_candidate_rate_std": float(np.std([row["additional_candidate_rate"] for row in seed_metrics], ddof=1)),
            "ranking_scope": "outer-test physics-reference-negative events; descriptive, not threshold selection",
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "review_budget_curve.csv", ranking)
    _write_csv(output_root / "target_recall_workload.csv", target_recall)
    _write_csv(output_root / "binary_seed_metrics.csv", seed_metrics)
    (output_root / "review_burden_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {"summary": summary, "review_budget_curve": ranking, "target_recall_workload": target_recall}


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(args.events, args.output_root, args.method, args.fpr_budget),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
