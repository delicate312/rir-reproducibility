"""Seal Pair-DAMP v2 and a second unseen-exact-variant holdout."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .adapter import (
    DEFAULT_METADATA_ROOT,
    RISK_BENCH_GRAPH_INPUT_PROTOCOL,
    discover_riskbench_records,
)
from .complete_structure_ablation import BACKBONES, _checkpoint_path
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT
from .frozen_onset_shock_rescue import (
    MAX_ADDITIONAL_VALIDATION_FPR,
    MIN_ONSET_LEAD_SECONDS,
    ONSET_QUANTILES,
    POOLING_FRAMES,
)
from .lazy_dataset import _record_payload
from .pair_damp_freeze import (
    FOLD_COUNT,
    FROZEN_METHOD,
    PROJECT_ROOT,
    _assert_family_excluded_from_model_training,
    _outer_family_assignments,
    _prior_index_inventory,
    _select_new_variants,
    _source_fold_payloads,
    sha256_file,
    sha256_payload,
)


SELECTION_SEED = 20260804
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_riskbench_pair_damp_v2_holdout"
)
DEFAULT_FREEZE_PATH = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_v2_frozen"
    / "frozen_protocol_manifest.json"
)
V2_METHOD = {
    **FROZEN_METHOD,
    "name": "CPA + graph-free exact-pair DAMP + onset-shock rescue",
    "onset_shock": {
        "definition": (
            "recovery-shaped residual/damage drop within first three valid "
            "predictions of an exact ego-target CPA-aligned relation"
        ),
        "quantile_candidates": list(ONSET_QUANTILES),
        "selection": (
            "least strict candidate with no more than two percentage points "
            "additional event FPR on validation-normal TTC-negative events"
        ),
        "max_additional_validation_fpr": MAX_ADDITIONAL_VALIDATION_FPR,
        "minimum_lead_seconds": MIN_ONSET_LEAD_SECONDS,
    },
    "continuous_metrics": {
        "pooling_frames": list(POOLING_FRAMES),
        "pooling_seconds": [0.05, 0.25, 0.50],
        "metrics": ["paired_score_lift", "win_rate", "auroc", "average_precision"],
    },
}
V2_CODE_PATHS = (
    "experiments/ego_immune_recognition/riskbench_metadata/pair_damp_v2_freeze.py",
    "experiments/ego_immune_recognition/riskbench_metadata/pair_damp_v2_evaluation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/frozen_onset_shock_rescue.py",
    "experiments/ego_immune_recognition/riskbench_metadata/pair_aligned_damp_evaluation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/relation_geometry_rescue.py",
    "experiments/ego_immune_recognition/riskbench_metadata/core_hybrid_evaluation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/threshold_advantage_eval.py",
    "experiments/ego_immune_recognition/danger_signal/diagnostics.py",
    "experiments/ego_immune_recognition/temporal_relation_train_eval.py",
    "experiments/ego_immune_recognition/schema.py",
)


def verify_v2_freeze(freeze_path: Path, index_root: Path) -> dict[str, Any]:
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    receipt = frozen.pop("freeze_receipt_sha256")
    if sha256_payload(frozen) != receipt:
        raise ValueError("Pair-DAMP v2 freeze receipt mismatch.")
    frozen["freeze_receipt_sha256"] = receipt
    for relative, expected in frozen["code_hashes"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"Frozen v2 code changed: {relative}")
    for fold, expected in frozen["holdout"]["index_hashes"].items():
        path = index_root / f"fold_{int(fold):02d}" / "riskbench_nested_fold_index.json"
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen v2 index changed: {path}")
    for payload in frozen["checkpoints"].values():
        if sha256_file(Path(payload["path"])) != payload["sha256"]:
            raise ValueError(f"Frozen v2 checkpoint changed: {payload['path']}")
    for path, expected in frozen["holdout"]["selected_state_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"Frozen v2 state changed: {path}")
    return frozen


def seal_v2_holdout(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    source_index_root: Path = DEFAULT_INDEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    freeze_path: Path = DEFAULT_FREEZE_PATH,
    seed: int = SELECTION_SEED,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    freeze_path = freeze_path.expanduser().resolve()
    if output_root.exists() or freeze_path.exists():
        raise FileExistsError("Pair-DAMP v2 holdout is already sealed.")
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
            f"Insufficient unused v2 variants: positives={positive_count}, normals={normal_count}"
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
            **{
                key: value
                for key, value in source.items()
                if key not in {"splits", "meta", "cache_root", "protocol"}
            },
            "protocol": "riskbench-pair-damp-v2-second-unseen-variant",
            "graph_input_protocol": RISK_BENCH_GRAPH_INPUT_PROTOCOL,
            "metadata_root": str(metadata_root.expanduser().resolve()),
            "cache_root": str(cache_root.resolve()),
            "splits": payload_splits,
            "meta": {
                **source.get("meta", {}),
                "confirmation_outer_fold": fold,
                "holdout_tier": "second_unseen_exact_variant_family_excluded_model",
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

    backbone = str(V2_METHOD["backbone"])
    checkpoints = {
        str(fold): {
            "path": str(_checkpoint_path(BACKBONES[backbone], fold).resolve()),
            "sha256": sha256_file(_checkpoint_path(BACKBONES[backbone], fold)),
        }
        for fold in range(FOLD_COUNT)
    }
    selected_state_hashes = {
        str(Path(record["state_path"]).resolve()): sha256_file(Path(record["state_path"]))
        for record in selected_records
    }
    payload = {
        "protocol": "riskbench-pair-damp-v2-freeze",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "method": V2_METHOD,
        "method_sha256": sha256_payload(V2_METHOD),
        "holdout": {
            "tier": "second_unseen_exact_variant_family_excluded_model",
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
        "code_hashes": {
            relative: sha256_file(PROJECT_ROOT / relative)
            for relative in V2_CODE_PATHS
        },
        "restrictions": {
            "method_or_parameter_updates_after_seal": 0,
            "holdout_positive_labels_used_for_fitting": 0,
            "holdout_evaluation_allowed_once": True,
            "external_unseen_family_confirmation_still_required": True,
        },
    }
    payload["freeze_receipt_sha256"] = sha256_payload(payload)
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(seal_v2_holdout(), indent=2))
