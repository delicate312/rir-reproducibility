"""Freeze and reload the deterministic v3 three-pool recognition baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import temporal_relation_train_eval as baseline_eval


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_us101_counterfactual_dataset_v3"
    / "us101_ego_immune_counterfactual_dataset.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_baselines" / "v3_smoke3"
)
BASELINE_ID = "ego-immune-v3-three-pool-one-class-epoch3"
BASELINE_CONFIG = {
    "epochs": 3,
    "hidden_channels": 32,
    "latent_channels": 16,
    "learning_rate": 1e-3,
    "group_count": 4,
    "threshold_quantile": 0.95,
    "seed": 20260713,
    "objective_mode": "one_class_prototype",
    "train_safety_policy": "kinematic_safe_segments",
    "safe_min_segment_frames": 8,
}
TRACKED_SOURCE_FILES = [
    PROJECT_ROOT / "02_alt_temporal_train.py",
    PROJECT_ROOT
    / "experiments"
    / "ego_immune_recognition"
    / "temporal_relation_train_eval.py",
    PROJECT_ROOT
    / "experiments"
    / "ego_immune_recognition"
    / "epoch_dual_track_sweep.py",
    PROJECT_ROOT
    / "experiments"
    / "ego_immune_recognition"
    / "schema.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_baseline_runtime(
    dataset_path: Path,
) -> tuple[
    Any,
    torch.nn.Module,
    torch.device,
    dict[str, list[list[Any]]],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    config = BASELINE_CONFIG
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    training = baseline_eval._load_training_module()
    package = torch.load(dataset_path.resolve(), weights_only=False)

    source_train = package["splits"]["train_normal"]
    train_sequences = baseline_eval._apply_training_safety_policy(
        source_train,
        policy=str(config["train_safety_policy"]),
        min_segment_frames=int(config["safe_min_segment_frames"]),
    )
    x_mean, x_std, edge_mean, edge_std = baseline_eval._fit_normalization(
        train_sequences
    )
    normalization = {
        "x_mean": x_mean,
        "x_std": x_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }

    split_names = [
        "train_normal",
        "validation_normal",
        "validation_positive",
        "test_normal",
        "test_positive",
    ]
    raw_splits = {name: package["splits"][name] for name in split_names}
    raw_splits["train_normal"] = train_sequences
    splits = {
        name: baseline_eval._normalize_sequences(
            sequences,
            x_mean,
            x_std,
            edge_mean,
            edge_std,
        )
        for name, sequences in raw_splits.items()
    }

    device = torch.device("cpu")
    model_config = training.ModelConfig(
        model_name=training.RELATION_MODEL_NAME,
        in_channels=int(splits["train_normal"][0][0].x.shape[1]),
        edge_attr_dim=int(splits["train_normal"][0][0].edge_attr.shape[1]),
        hidden_channels=int(config["hidden_channels"]),
        latent_channels=int(config["latent_channels"]),
        dropout=0.10,
    )
    model = training.build_model(model_config).to(device)
    return training, model, device, splits, normalization, package


def train_frozen_baseline(
    *,
    dataset_path: Path,
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "baseline_checkpoint.pt"
    manifest_path = output_root / "baseline_manifest.json"
    if checkpoint_path.exists() and not force:
        raise FileExistsError(
            f"Frozen baseline already exists: {checkpoint_path}. Use --force to replace it."
        )

    training, model, device, splits, normalization, package = (
        prepare_baseline_runtime(dataset_path)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(BASELINE_CONFIG["learning_rate"]),
        weight_decay=1e-5,
    )
    history = []
    for epoch in range(1, int(BASELINE_CONFIG["epochs"]) + 1):
        train_metrics = training.train_relation_localizer_one_epoch(
            model=model,
            sequences=splits["train_normal"],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=int(BASELINE_CONFIG["epochs"]),
            objective_mode=str(BASELINE_CONFIG["objective_mode"]),
        )
        val_metrics = training.evaluate_relation_localizer_one_epoch(
            model=model,
            sequences=splits["validation_normal"],
            device=device,
            epoch=epoch,
            total_epochs=int(BASELINE_CONFIG["epochs"]),
            objective_mode=str(BASELINE_CONFIG["objective_mode"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_next_edge": float(train_metrics["next_edge"]),
                "val_next_edge": float(val_metrics["next_edge"]),
                "train_total": float(train_metrics["total"]),
                "val_total": float(val_metrics["total"]),
            }
        )

    event_metadata = baseline_eval._load_counterfactual_event_metadata(package)
    evaluation = baseline_eval._dual_track_evaluation(
        model,
        validation_normal=splits["validation_normal"],
        validation_positive=splits["validation_positive"],
        test_normal=splits["test_normal"],
        test_positive=splits["test_positive"],
        device=device,
        group_count=int(BASELINE_CONFIG["group_count"]),
        threshold_quantile=float(BASELINE_CONFIG["threshold_quantile"]),
        event_metadata_by_uid=event_metadata,
    )
    torch.save(
        {
            "baseline_id": BASELINE_ID,
            "config": BASELINE_CONFIG,
            "model_state_dict": model.state_dict(),
            "normalization": normalization,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "history": history,
        },
        checkpoint_path,
    )
    tracked_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in TRACKED_SOURCE_FILES
    }
    manifest = {
        "baseline_id": BASELINE_ID,
        "status": "frozen_do_not_overwrite",
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": BASELINE_CONFIG,
        "source_sha256": tracked_hashes,
        "history": history,
        "reference_metrics": evaluation["splits"],
        "reload_command": (
            "python -m experiments.ego_immune_recognition.danger_signal.diagnostics"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_frozen_baseline(
    *, checkpoint_path: Path, dataset_path: Path | None = None
) -> tuple[
    Any,
    torch.nn.Module,
    torch.device,
    dict[str, list[list[Any]]],
    dict[str, Any],
]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    manifest_path = checkpoint_path.parent / "baseline_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen baseline manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash no longer matches the frozen manifest.")
    for relative_path, expected_hash in manifest["source_sha256"].items():
        source_path = PROJECT_ROOT / relative_path
        if sha256_file(source_path) != expected_hash:
            raise ValueError(
                f"Frozen baseline source changed: {relative_path}. "
                "Keep the baseline source or freeze a new baseline id."
            )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("baseline_id") != BASELINE_ID:
        raise ValueError(f"Unexpected baseline id: {checkpoint.get('baseline_id')!r}")
    resolved_dataset = (
        Path(checkpoint["dataset_path"])
        if dataset_path is None
        else dataset_path.expanduser().resolve()
    )
    if sha256_file(resolved_dataset) != checkpoint["dataset_sha256"]:
        raise ValueError("Dataset hash no longer matches the frozen baseline.")
    training, model, device, splits, _, package = prepare_baseline_runtime(
        resolved_dataset
    )
    del training
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, splits, package, checkpoint


def main() -> None:
    args = parse_args()
    result = train_frozen_baseline(
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
