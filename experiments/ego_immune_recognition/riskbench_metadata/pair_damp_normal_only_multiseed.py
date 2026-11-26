"""Formal multi-seed family-CV evaluation for the frozen Pair-DAMP method.

Each replicate trains five graph-free one-class models.  Checkpoints are
selected only by validation-normal objective loss.  The outer test splits are
not scored until every checkpoint in the replicate has been selected and a
hash receipt has been written.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .complete_structure_ablation import BACKBONES
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT
from .frozen_onset_shock_rescue import _fold_rows
from .guarded_epoch_sweep import run_guarded_epoch_sweep
from .normal_only_checkpoint_audit import _attach_event_metadata
from .pair_damp_freeze import FOLD_COUNT, PROJECT_ROOT, sha256_file
from .pair_damp_v2_evaluation import (
    PRIMARY_CONTINUOUS_FPR_BUDGET,
    summarize_continuous_metrics,
    summarize_events,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_normal_only_multiseed_v1"
)
DEFAULT_SEEDS = (20260812, 20260813, 20260814)
MODEL_NAME = "Temporal_Immune_Localizer_without_Graph_Message_Passing"
AGGREGATE_METRICS = (
    "positive_recall",
    "ttc_blind_rescue_recall",
    "incremental_rescue_fpr",
    "hard_safe_event_fpr",
    "mean_rescue_lead_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-epochs", default="1,3")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("Expected at least one integer.")
    return output


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _compatible_training_summary(
    path: Path, *, seed: int, epochs: int, eval_epochs: Sequence[int]
) -> bool:
    if not path.exists():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        int(summary.get("seed", -1)) == seed
        and int(summary.get("epochs", -1)) == epochs
        and list(summary.get("eval_epochs", [])) == list(eval_epochs)
        and summary.get("model_name") == MODEL_NAME
        and summary.get("checkpoint_selection_mode") == "normal_only_val_loss"
        and int(summary.get("positive_sequences_scored_during_selection", -1))
        == 0
        and not bool(summary.get("evaluate_test_during_selection", True))
    )


def _training_receipt(seed_root: Path, seed: int) -> list[dict[str, Any]]:
    rows = []
    for fold in range(FOLD_COUNT):
        fold_root = seed_root / f"fold_{fold:02d}"
        summary = json.loads(
            (fold_root / "guarded_epoch_sweep_summary.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint = fold_root / "best_guarded_checkpoint.pt"
        rows.append(
            {
                "replicate_seed": seed,
                "fold": fold,
                "training_seed": int(summary["seed"]),
                "selected_epoch": int(summary["selection"]["best_epoch"]),
                "validation_normal_total": float(
                    summary["selection"]["validation_normal"]["val_total"]
                ),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "positive_sequences_scored_during_selection": int(
                    summary["positive_sequences_scored_during_selection"]
                ),
                "outer_test_scored_during_selection": int(
                    bool(summary["evaluate_test_during_selection"])
                ),
            }
        )
    return rows


def run_seed(
    *,
    seed: int,
    index_root: Path = DEFAULT_INDEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    epochs: int = 3,
    eval_epochs: Sequence[int] = (1, 3),
    force: bool = False,
) -> dict[str, Any]:
    index_root = index_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    seed_root = output_root / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    eval_epochs = sorted({int(value) for value in eval_epochs if value <= epochs})
    if epochs not in eval_epochs:
        eval_epochs.append(epochs)

    for fold in range(FOLD_COUNT):
        fold_root = seed_root / f"fold_{fold:02d}"
        summary_path = fold_root / "guarded_epoch_sweep_summary.json"
        training_seed = seed + fold
        if not force and _compatible_training_summary(
            summary_path,
            seed=training_seed,
            epochs=epochs,
            eval_epochs=eval_epochs,
        ):
            print(
                json.dumps({"seed": seed, "fold": fold, "status": "reused"}),
                flush=True,
            )
            continue
        print(
            json.dumps({"seed": seed, "fold": fold, "status": "training"}),
            flush=True,
        )
        run_guarded_epoch_sweep(
            dataset_path=(
                index_root
                / f"fold_{fold:02d}"
                / "riskbench_nested_fold_index.json"
            ),
            output_root=fold_root,
            epochs=epochs,
            eval_epochs=eval_epochs,
            seed=training_seed,
            objective_mode="one_class_prototype",
            train_safety_policy="label_normal",
            run_confirmation=False,
            evaluate_test_during_selection=False,
            model_name=MODEL_NAME,
            checkpoint_selection_mode="normal_only_val_loss",
        )

    receipt_rows = _training_receipt(seed_root, seed)
    if any(
        row["positive_sequences_scored_during_selection"]
        or row["outer_test_scored_during_selection"]
        for row in receipt_rows
    ):
        raise RuntimeError("Selection firewall violation; outer evaluation aborted.")
    receipt = {
        "protocol": "pair-damp-normal-only-family-cv-replicate-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replicate_seed": seed,
        "selection_rule": "minimum validation-normal objective loss",
        "positive_sequences_scored_during_selection": 0,
        "outer_test_scored_before_freeze": False,
        "checkpoints": receipt_rows,
        "status": "all_five_checkpoints_frozen_before_outer_test",
    }
    receipt_path = seed_root / "pre_outer_test_checkpoint_freeze.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt_sha256 = sha256_file(receipt_path)

    historical_root = BACKBONES["graph_free_one_class"]
    BACKBONES["graph_free_one_class"] = seed_root
    try:
        rows: list[dict[str, Any]] = []
        audits = []
        for fold in range(FOLD_COUNT):
            fold_rows, audit = _fold_rows(
                fold=fold,
                index_root=index_root,
                cache_root=seed_root / "pair_frame_cache",
            )
            rows.extend(fold_rows)
            audits.append(audit)
    finally:
        BACKBONES["graph_free_one_class"] = historical_root

    _attach_event_metadata(rows, index_root)
    event_summary = summarize_events(rows)
    continuous_summary = summarize_continuous_metrics(
        [
            row
            for row in rows
            if float(row["fpr_budget"]) == PRIMARY_CONTINUOUS_FPR_BUDGET
        ]
    )
    _write_csv(seed_root / "outer_test_events.csv", rows)
    _write_csv(seed_root / "outer_test_event_summary.csv", event_summary)
    _write_csv(seed_root / "training_receipt.csv", receipt_rows)
    result = {
        "protocol": "pair-damp-normal-only-family-cv-replicate-v1",
        "replicate_seed": seed,
        "family_grouped_outer_folds": FOLD_COUNT,
        "checkpoint_freeze_receipt": str(receipt_path),
        "checkpoint_freeze_receipt_sha256": receipt_sha256,
        "selection_positive_sequences_scored": 0,
        "selection_outer_test_sequences_scored": 0,
        "event_summary": event_summary,
        "continuous_summary": continuous_summary,
        "fold_audits": audits,
        "status": "replicate_complete",
    }
    summary_path = seed_root / "replicate_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    seed_summaries = []
    rows = []
    for seed in seeds:
        summary = json.loads(
            (output_root / f"seed_{seed}" / "replicate_summary.json").read_text(
                encoding="utf-8"
            )
        )
        seed_summaries.append(summary)
        for row in summary["event_summary"]:
            rows.append({"seed": int(seed), **row})

    aggregate_rows = []
    for budget in sorted({float(row["fpr_budget"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            selected = [
                row
                for row in rows
                if float(row["fpr_budget"]) == budget
                and str(row["method"]) == method
            ]
            output: dict[str, Any] = {
                "fpr_budget": budget,
                "method": method,
                "seed_count": len(selected),
            }
            for metric in AGGREGATE_METRICS:
                values = np.asarray(
                    [
                        float(row[metric])
                        for row in selected
                        if row.get(metric, "") != ""
                    ],
                    dtype=float,
                )
                output[f"{metric}_mean"] = (
                    float(values.mean()) if values.size else ""
                )
                output[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                )
            aggregate_rows.append(output)

    _write_csv(output_root / "all_seed_event_summary.csv", rows)
    _write_csv(output_root / "multi_seed_aggregate.csv", aggregate_rows)
    result = {
        "protocol": "pair-damp-normal-only-family-cv-multiseed-v1",
        "seeds": [int(seed) for seed in seeds],
        "replicate_count": len(seed_summaries),
        "training_run_count": len(seed_summaries) * FOLD_COUNT,
        "checkpoint_selection": "validation-normal objective loss only",
        "outer_test_opened_after_each_replicate_freeze": True,
        "aggregate": aggregate_rows,
        "status": "multiseed_aggregate_complete",
    }
    summary_path = output_root / "multi_seed_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    seeds = _parse_ints(args.seeds)
    if args.aggregate_only:
        result = aggregate(seeds=seeds, output_root=args.output_root)
    elif args.seed is not None:
        result = run_seed(
            seed=args.seed,
            index_root=args.index_root,
            output_root=args.output_root,
            epochs=args.epochs,
            eval_epochs=_parse_ints(args.eval_epochs),
            force=args.force,
        )
    else:
        for seed in seeds:
            run_seed(
                seed=seed,
                index_root=args.index_root,
                output_root=args.output_root,
                epochs=args.epochs,
                eval_epochs=_parse_ints(args.eval_epochs),
                force=args.force,
            )
        result = aggregate(seeds=seeds, output_root=args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
