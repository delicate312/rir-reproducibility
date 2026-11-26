"""Seal the final pair-specific DAMP method and an unseen-variant holdout.

The local RiskBench metadata has already exposed every collision family during
development.  This protocol therefore selects a new exact variant from each
available family and assigns it only to the fold whose model excluded that
family during training.  It is a confirmatory unseen-variant evaluation, not an
external unseen-family evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .adapter import (
    DEFAULT_METADATA_ROOT,
    RISK_BENCH_GRAPH_INPUT_PROTOCOL,
    discover_riskbench_records,
)
from .complete_structure_ablation import BACKBONES, _checkpoint_path
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT
from .lazy_dataset import _hash_rank, _record_payload


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_riskbench_pair_damp_confirmation_v1"
)
DEFAULT_FREEZE_PATH = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_frozen_v1"
    / "frozen_protocol_manifest.json"
)
SELECTION_SEED = 20260803
FOLD_COUNT = 5
FPR_BUDGETS = (0.05, 0.10)
FROZEN_METHOD = {
    "name": "CPA + graph-free exact-pair DAMP",
    "backbone": "graph_free_one_class",
    "training_objective": "one_class_prototype_safe_core_only",
    "cpa_horizon_seconds": 2.0,
    "cpa_clearance_m": 3.5,
    "severe_cpa_percentile": 0.999,
    "pair_support_frames": 3,
    "pair_score": "sqrt(CPA percentile * exact-pair DAMP percentile)",
    "damp_weights": {"stress": 0.25, "damage": 0.35, "persistence": 0.40},
    "recovery_beta": 0.10,
    "evaluation_window_frames": 30,
    "timestep_seconds": 0.05,
    "threshold_fit_split": "validation_normal_ttc_negative_events_only",
    "positive_events_used_for_threshold_or_calibration": 0,
    "fpr_budgets": list(FPR_BUDGETS),
}
FROZEN_CODE_PATHS = (
    "experiments/ego_immune_recognition/riskbench_metadata/pair_damp_freeze.py",
    "experiments/ego_immune_recognition/riskbench_metadata/pair_damp_confirmation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/pair_aligned_damp_evaluation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/relation_geometry_rescue.py",
    "experiments/ego_immune_recognition/riskbench_metadata/core_hybrid_evaluation.py",
    "experiments/ego_immune_recognition/danger_signal/diagnostics.py",
    "experiments/ego_immune_recognition/temporal_relation_train_eval.py",
    "experiments/ego_immune_recognition/schema.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--source-index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _walk_scenario_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        if "scenario_id" in value:
            yield str(value["scenario_id"])
        for nested in value.values():
            yield from _walk_scenario_ids(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_scenario_ids(nested)


def _prior_index_inventory(processed_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    scenario_ids: set[str] = set()
    inventory = []
    for path in sorted(processed_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        ids = set(_walk_scenario_ids(payload))
        if not ids:
            continue
        scenario_ids.update(ids)
        inventory.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "scenario_id_count": len(ids),
            }
        )
    return scenario_ids, inventory


def _source_fold_payloads(index_root: Path) -> dict[int, dict[str, Any]]:
    return {
        fold: json.loads(
            (
                index_root
                / f"fold_{fold:02d}"
                / "riskbench_nested_fold_index.json"
            ).read_text(encoding="utf-8")
        )
        for fold in range(FOLD_COUNT)
    }


def _outer_family_assignments(
    folds: dict[int, dict[str, Any]],
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for fold, payload in folds.items():
        for split in ("test_normal", "test_positive"):
            for record in payload["splits"][split]:
                family = str(record["scenario_group_id"])
                if family in assignments:
                    raise ValueError(f"Family appears in multiple outer folds: {family}")
                assignments[family] = fold
    return assignments


def _select_new_variants(
    records: Sequence[Any],
    *,
    prior_ids: set[str],
    assignments: dict[str, int],
    seed: int,
) -> dict[int, dict[str, list[Any]]]:
    candidates: dict[str, list[Any]] = {}
    for record in records:
        if record.category not in {"interactive", "collision"}:
            continue
        if record.scenario_id in prior_ids:
            continue
        if record.scenario_group_id not in assignments:
            continue
        candidates.setdefault(record.scenario_group_id, []).append(record)

    selected = {
        family: min(
            variants,
            key=lambda item: _hash_rank(item.scenario_id, seed),
        )
        for family, variants in candidates.items()
    }
    by_fold = {
        fold: {"test_normal": [], "test_positive": []}
        for fold in range(FOLD_COUNT)
    }
    for family, record in selected.items():
        split = "test_positive" if record.category == "collision" else "test_normal"
        by_fold[assignments[family]][split].append(record)
    for fold_splits in by_fold.values():
        for records_for_split in fold_splits.values():
            records_for_split.sort(key=lambda item: item.scenario_id)
    return by_fold


def _assert_family_excluded_from_model_training(
    source: dict[str, Any],
    selected: dict[str, list[Any]],
) -> dict[str, int]:
    fitted_families = {
        str(record["scenario_group_id"])
        for split in ("train_normal", "validation_normal", "validation_positive")
        for record in source["splits"][split]
    }
    test_families = {
        record.scenario_group_id
        for records in selected.values()
        for record in records
    }
    overlap = fitted_families & test_families
    if overlap:
        raise ValueError(f"Holdout family leaked into fold fitting data: {sorted(overlap)}")
    return {
        "fitted_family_overlap_count": 0,
        "test_family_count": len(test_families),
    }


def seal_pair_damp_confirmation(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    source_index_root: Path = DEFAULT_INDEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    freeze_path: Path = DEFAULT_FREEZE_PATH,
    seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    freeze_path = freeze_path.expanduser().resolve()
    if freeze_path.exists() or output_root.exists():
        raise FileExistsError(
            "Frozen confirmation artifacts already exist; refusing to reseal."
        )
    source_index_root = source_index_root.expanduser().resolve()
    source_folds = _source_fold_payloads(source_index_root)
    assignments = _outer_family_assignments(source_folds)
    prior_ids, prior_inventory = _prior_index_inventory(
        PROJECT_ROOT / "data" / "processed"
    )
    selected = _select_new_variants(
        discover_riskbench_records(metadata_root),
        prior_ids=prior_ids,
        assignments=assignments,
        seed=seed,
    )
    positive_count = sum(len(value["test_positive"]) for value in selected.values())
    normal_count = sum(len(value["test_normal"]) for value in selected.values())
    if positive_count < 100 or normal_count < 100:
        raise ValueError(
            f"Insufficient unused variants: positives={positive_count}, normals={normal_count}"
        )

    output_root.mkdir(parents=True)
    cache_root = output_root / "sequence_cache"
    index_hashes = {}
    selected_records = []
    fold_counts = []
    for fold in range(FOLD_COUNT):
        source = source_folds[fold]
        isolation = _assert_family_excluded_from_model_training(source, selected[fold])
        payload_splits = {
            **{
                split: source["splits"][split]
                for split in ("train_normal", "validation_normal", "validation_positive")
            },
            **{
                split: [_record_payload(record) for record in records]
                for split, records in selected[fold].items()
            },
        }
        for split in ("test_normal", "test_positive"):
            selected_records.extend(payload_splits[split])
        payload = {
            **{key: value for key, value in source.items() if key not in {"splits", "meta", "cache_root", "protocol"}},
            "protocol": "riskbench-frozen-pair-damp-unseen-variant-v1",
            "graph_input_protocol": RISK_BENCH_GRAPH_INPUT_PROTOCOL,
            "metadata_root": str(metadata_root.expanduser().resolve()),
            "cache_root": str(cache_root.resolve()),
            "splits": payload_splits,
            "meta": {
                **source.get("meta", {}),
                "confirmation_outer_fold": fold,
                "holdout_tier": "unseen_exact_variant_family_excluded_model",
                "development_family_overlap_expected": True,
                "exact_variant_used_in_prior_index_count": 0,
                "frozen_parameter_updates_allowed": False,
                "outer_test_labels_allowed_for_model_or_threshold_fit": False,
                "family_exclusion_audit": isolation,
                "manifest_sha256": sha256_payload(payload_splits),
            },
        }
        fold_root = output_root / f"fold_{fold:02d}"
        fold_root.mkdir(parents=True)
        index_path = fold_root / "riskbench_nested_fold_index.json"
        index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        index_hashes[str(fold)] = sha256_file(index_path)
        fold_counts.append(
            {
                "fold": fold,
                "test_positive": len(payload_splits["test_positive"]),
                "test_normal": len(payload_splits["test_normal"]),
            }
        )

    backbone = str(FROZEN_METHOD["backbone"])
    checkpoints = {
        str(fold): {
            "path": str(_checkpoint_path(BACKBONES[backbone], fold).resolve()),
            "sha256": sha256_file(_checkpoint_path(BACKBONES[backbone], fold)),
        }
        for fold in range(FOLD_COUNT)
    }
    code_hashes = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in FROZEN_CODE_PATHS
    }
    source_index_hashes = {
        str(fold): sha256_file(
            source_index_root
            / f"fold_{fold:02d}"
            / "riskbench_nested_fold_index.json"
        )
        for fold in range(FOLD_COUNT)
    }
    selected_state_hashes = {
        str(Path(record["state_path"]).resolve()): sha256_file(Path(record["state_path"]))
        for record in selected_records
    }
    frozen_payload = {
        "protocol": "riskbench-pair-damp-confirmation-freeze-v1",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "method": FROZEN_METHOD,
        "method_sha256": sha256_payload(FROZEN_METHOD),
        "holdout": {
            "tier": "unseen_exact_variant_family_excluded_model",
            "positive_count": positive_count,
            "normal_count": normal_count,
            "fold_counts": fold_counts,
            "index_root": str(output_root),
            "index_hashes": index_hashes,
            "selected_state_hashes": selected_state_hashes,
            "prior_index_scenario_id_count": len(prior_ids),
            "prior_index_inventory": prior_inventory,
            "development_family_overlap_expected": True,
            "unseen_family_claim_allowed": False,
        },
        "checkpoints": checkpoints,
        "source_index_hashes": source_index_hashes,
        "code_hashes": code_hashes,
        "restrictions": {
            "method_or_parameter_updates_after_seal": 0,
            "holdout_positive_labels_used_for_fitting": 0,
            "holdout_evaluation_allowed_once": True,
            "external_unseen_family_confirmation_still_required": True,
        },
    }
    frozen_payload["freeze_receipt_sha256"] = sha256_payload(frozen_payload)
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(json.dumps(frozen_payload, indent=2), encoding="utf-8")
    return frozen_payload


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            seal_pair_damp_confirmation(
                metadata_root=args.metadata_root,
                source_index_root=args.source_index_root,
                output_root=args.output_root,
                freeze_path=args.freeze_path,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
