"""Audit Pair-DAMP v2 after strictly normal-only checkpoint selection.

This is a post-hoc provenance and sensitivity audit.  It uses checkpoint
candidates that were already trained with safe-core gradients, selects among
them using validation-normal objective loss only, and evaluates the selected
models on the already-opened Pair-DAMP v2 exact-variant holdout.  Consequently
the output is not a new independent holdout claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

from .complete_structure_ablation import BACKBONES
from .frozen_onset_shock_rescue import _fold_rows
from .pair_damp_freeze import FOLD_COUNT, PROJECT_ROOT, sha256_file
from .pair_damp_v2_evaluation import (
    PRIMARY_CONTINUOUS_FPR_BUDGET,
    summarize_continuous_metrics,
    summarize_events,
)


DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_graph_free_ablation_v1"
)
DEFAULT_INDEX_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "ego_immune_riskbench_pair_damp_v2_holdout"
)
DEFAULT_BASELINE_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_pair_damp_v2_holdout"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_v2_normal_only_checkpoint_audit_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select_normal_only_checkpoint(source_root: Path, fold: int) -> dict[str, Any]:
    fold_root = source_root / f"fold_{fold:02d}"
    summary_path = fold_root / "guarded_epoch_sweep_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eval_epochs = {int(epoch) for epoch in summary["eval_epochs"]}
    candidates = [
        row for row in summary["history"] if int(row["epoch"]) in eval_epochs
    ]
    selected = min(
        candidates,
        key=lambda row: (
            float(row["val_total"]),
            float(row["val_next_edge"]),
            float(row["val_prototype_compactness"]),
            int(row["epoch"]),
        ),
    )
    epoch = int(selected["epoch"])
    checkpoint = fold_root / "checkpoints" / f"epoch_{epoch:03d}.pt"
    historical = fold_root / "best_guarded_checkpoint.pt"
    return {
        "fold": fold,
        "seed": int(summary["seed"]),
        "candidate_epochs": sorted(eval_epochs),
        "selected_epoch": epoch,
        "validation_normal_total": float(selected["val_total"]),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "historical_best_epoch": int(summary["selection"]["best_epoch"]),
        "historical_checkpoint_sha256": sha256_file(historical),
        "changed_from_historical": sha256_file(checkpoint) != sha256_file(historical),
        "positive_sequences_scored_for_selection": 0,
        "positive_frames_used_for_selection": 0,
    }


def _materialize_checkpoint_root(
    selections: Sequence[dict[str, Any]], output_root: Path
) -> Path:
    root = output_root / "normal_only_checkpoints"
    for row in selections:
        fold_root = root / f"fold_{int(row['fold']):02d}"
        fold_root.mkdir(parents=True, exist_ok=True)
        destination = fold_root / "best_guarded_checkpoint.pt"
        source = Path(str(row["checkpoint_path"]))
        if destination.exists():
            if sha256_file(destination) != str(row["checkpoint_sha256"]):
                raise ValueError(f"Existing audit checkpoint differs: {destination}")
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return root


def _reuse_unchanged_caches(
    selections: Sequence[dict[str, Any]], baseline_root: Path, cache_root: Path
) -> None:
    source_root = baseline_root / "pair_frame_cache" / "graph_free_one_class"
    destination_root = cache_root / "graph_free_one_class"
    destination_root.mkdir(parents=True, exist_ok=True)
    for row in selections:
        if bool(row["changed_from_historical"]):
            continue
        fold = int(row["fold"])
        source = source_root / f"fold_{fold:02d}.pkl"
        destination = destination_root / source.name
        if not source.exists() or destination.exists():
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _paired_delta(
    baseline_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {
        (float(row["fpr_budget"]), row["method"]): row for row in baseline_rows
    }
    output = []
    for row in audit_rows:
        key = (float(row["fpr_budget"]), row["method"])
        old = baseline[key]
        output.append(
            {
                "fpr_budget": key[0],
                "method": key[1],
                "positive_recall_delta": (
                    float(row["positive_recall"]) - float(old["positive_recall"])
                ),
                "ttc_blind_rescue_delta": (
                    int(row["ttc_blind_rescues"])
                    - int(old["ttc_blind_rescues"])
                ),
                "hard_safe_alert_delta": (
                    int(row["hard_safe_alerts"])
                    - int(old["hard_safe_alerts"])
                ),
                "incremental_rescue_fpr_delta": (
                    float(row["incremental_rescue_fpr"])
                    - float(old["incremental_rescue_fpr"])
                ),
            }
        )
    return output


def _event_metadata(index_root: Path) -> dict[tuple[int, str], dict[str, str]]:
    output: dict[tuple[int, str], dict[str, str]] = {}
    for fold in range(FOLD_COUNT):
        payload = json.loads(
            (
                index_root
                / f"fold_{fold:02d}"
                / "riskbench_nested_fold_index.json"
            ).read_text(encoding="utf-8")
        )
        for split in ("test_positive", "test_normal"):
            for record in payload["splits"][split]:
                uid = f"riskbench:{record['category']}:{record['scenario_id']}"
                output[(fold, uid)] = {
                    "category": str(record.get("category", "unknown")),
                    "scenario_id": str(record.get("scenario_id", "unknown")),
                    "target_type": str(record.get("target_type_name", "unknown")),
                    "weather": str(record.get("weather", "unknown")),
                    "traffic_density": str(
                        record.get("traffic_density", "unknown")
                    ),
                }
    return output


def _attach_event_metadata(
    rows: Sequence[dict[str, Any]], index_root: Path
) -> None:
    metadata = _event_metadata(index_root)
    for row in rows:
        key = (int(row["fold"]), str(row["event_uid"]))
        if key not in metadata:
            raise KeyError(f"No event metadata for {key}")
        row.update(metadata[key])


def _stratified_summary(
    rows: Sequence[dict[str, Any]], stratum_key: str
) -> list[dict[str, Any]]:
    output = []
    strata = sorted({str(row[stratum_key]) for row in rows})
    for budget in sorted({float(row["fpr_budget"]) for row in rows}):
        for stratum in strata:
            selected = [
                row
                for row in rows
                if float(row["fpr_budget"]) == budget
                and str(row[stratum_key]) == stratum
            ]
            positive = [row for row in selected if int(row["positive"])]
            normal = [row for row in selected if not int(row["positive"])]
            blind_positive = [
                row for row in positive if not int(row["ttc_detected"])
            ]
            blind_normal = [
                row for row in normal if not int(row["ttc_detected"])
            ]
            for method in ("ttc", "baseline", "onset"):
                detected_key = (
                    "ttc_detected" if method == "ttc" else f"{method}_detected"
                )
                rescue_key = (
                    None if method == "ttc" else f"{method}_blind_rescue"
                )
                output.append(
                    {
                        "stratum": stratum_key,
                        "value": stratum,
                        "fpr_budget": budget,
                        "method": method,
                        "positive_event_count": len(positive),
                        "positive_hits": sum(
                            int(row["ttc_detected"])
                            or int(row[detected_key])
                            for row in positive
                        ),
                        "ttc_blind_positive_count": len(blind_positive),
                        "ttc_blind_rescues": (
                            0
                            if rescue_key is None
                            else sum(int(row[rescue_key]) for row in blind_positive)
                        ),
                        "normal_event_count": len(normal),
                        "normal_alerts": sum(
                            int(row["ttc_detected"])
                            or int(row[detected_key])
                            for row in normal
                        ),
                        "ttc_negative_normal_count": len(blind_normal),
                        "false_rescues": (
                            0
                            if rescue_key is None
                            else sum(int(row[rescue_key]) for row in blind_normal)
                        ),
                    }
                )
    return output


def run_audit(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    index_root: Path = DEFAULT_INDEX_ROOT,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    index_root = index_root.expanduser().resolve()
    baseline_root = baseline_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selections = [
        _select_normal_only_checkpoint(source_root, fold)
        for fold in range(FOLD_COUNT)
    ]
    checkpoint_root = _materialize_checkpoint_root(selections, output_root)
    cache_root = output_root / "pair_frame_cache"
    _reuse_unchanged_caches(selections, baseline_root, cache_root)

    historical_root = BACKBONES["graph_free_one_class"]
    BACKBONES["graph_free_one_class"] = checkpoint_root
    try:
        rows = []
        audits = []
        for fold in range(FOLD_COUNT):
            fold_rows, audit = _fold_rows(
                fold=fold,
                index_root=index_root,
                cache_root=cache_root,
            )
            rows.extend(fold_rows)
            audits.append(audit)
    finally:
        BACKBONES["graph_free_one_class"] = historical_root

    _attach_event_metadata(rows, index_root)

    event_summary = summarize_events(rows)
    continuous_source = [
        row
        for row in rows
        if float(row["fpr_budget"]) == PRIMARY_CONTINUOUS_FPR_BUDGET
    ]
    continuous_summary = summarize_continuous_metrics(continuous_source)
    baseline_summary = json.loads(
        (baseline_root / "v2_holdout_summary.json").read_text(encoding="utf-8")
    )
    comparison = _paired_delta(baseline_summary["event_summary"], event_summary)
    stratified = []
    for key in ("target_type", "weather", "traffic_density"):
        stratified.extend(_stratified_summary(rows, key))

    _write_csv(output_root / "normal_only_selections.csv", selections)
    _write_csv(output_root / "normal_only_holdout_events.csv", rows)
    _write_csv(output_root / "normal_only_event_summary.csv", event_summary)
    _write_csv(output_root / "normal_only_comparison.csv", comparison)
    _write_csv(output_root / "normal_only_stratified_summary.csv", stratified)
    result = {
        "protocol": "pair-damp-v2-normal-only-checkpoint-sensitivity-v1",
        "evidence_tier": "posthoc_on_already_opened_exact_variant_holdout",
        "independent_holdout_claim_allowed": False,
        "selection_rule": (
            "minimum validation-normal total objective loss among saved "
            "candidate epochs; ties broken without positive data"
        ),
        "positive_sequences_scored_for_checkpoint_selection": 0,
        "positive_frames_used_for_checkpoint_selection": 0,
        "selections": selections,
        "changed_fold_count": sum(
            int(row["changed_from_historical"]) for row in selections
        ),
        "event_summary": event_summary,
        "continuous_summary": continuous_summary,
        "comparison_to_historical_v2": comparison,
        "stratified_summary": stratified,
        "fold_audits": audits,
        "status": "normal_only_checkpoint_sensitivity_complete",
    }
    summary_path = output_root / "normal_only_checkpoint_audit_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_audit(
                source_root=args.source_root,
                index_root=args.index_root,
                baseline_root=args.baseline_root,
                output_root=args.output_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
