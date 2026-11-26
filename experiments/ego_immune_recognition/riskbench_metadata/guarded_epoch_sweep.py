"""Train RiskBench checkpoints and select epochs by guarded alert metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..dataset_io import load_graph_dataset
from ..danger_signal.diagnostics import (
    apply_danger_calibration,
    false_alarm_morphology,
    robust_component_calibration,
    score_sequences,
)
from ..schema import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from ..temporal_relation_train_eval import (
    ONE_CLASS_SCORING_CONTRACT,
    _apply_training_safety_policy,
    _fit_normalization,
    _load_training_module,
    _normalize_sequences,
)
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .persistent_guarded_fusion import (
    ALERT_PERCENTILE,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIRMATION_INDEX,
    DEFAULT_DEVELOPMENT_INDEX,
    METHODS,
    _attach_guarded_response,
    _attach_percentile_inputs,
    _method_metrics,
    run_persistent_guarded_fusion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_guarded_epoch_sweep_v1"
)
DEFAULT_EVAL_EPOCHS = [1, 3, 5, 10, 20]
MAX_VALIDATION_FPR_INCREASE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DEVELOPMENT_INDEX,
    )
    parser.add_argument(
        "--confirmation-index",
        type=Path,
        default=DEFAULT_CONFIRMATION_INDEX,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--eval-epochs",
        default=",".join(str(epoch) for epoch in DEFAULT_EVAL_EPOCHS),
    )
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--group-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Relation-localizer model variant; defaults to the full model.",
    )
    parser.add_argument(
        "--objective-mode",
        default="one_class_prototype",
        choices=[
            "one_class_prototype",
            "causal_future",
            "hybrid_reconstruction",
        ],
    )
    parser.add_argument(
        "--train-safety-policy",
        default="label_normal",
        choices=["label_normal", "kinematic_safe_segments"],
    )
    parser.add_argument("--safe-min-segment-frames", type=int, default=8)
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument(
        "--skip-test-during-selection",
        action="store_true",
        help="Do not score outer/test splits while selecting a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-selection-mode",
        default="guarded_positive",
        choices=["guarded_positive", "normal_only_val_loss"],
        help=(
            "Select checkpoints using the historical validation danger metrics "
            "or strictly from validation-normal objective loss."
        ),
    )
    return parser.parse_args()


def parse_eval_epochs(value: str, *, max_epoch: int) -> list[int]:
    epochs = {
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    }
    if any(epoch <= 0 for epoch in epochs):
        raise ValueError("Evaluation epochs must be positive.")
    epochs = {epoch for epoch in epochs if epoch <= max_epoch}
    epochs.add(max_epoch)
    return sorted(epochs)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_payload(
    *,
    dataset_path: Path,
    epoch: int,
    seed: int,
    objective_mode: str,
    train_safety_policy: str,
    config: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    normalization: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "format": "temporal-immune-localizer-checkpoint-v1",
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "epoch": epoch,
        "seed": seed,
        "objective_mode": objective_mode,
        "train_safety_policy": train_safety_policy,
        "model_config": {
            "model_name": config.model_name,
            "in_channels": config.in_channels,
            "edge_attr_dim": config.edge_attr_dim,
            "hidden_channels": config.hidden_channels,
            "latent_channels": config.latent_channels,
            "dropout": config.dropout,
        },
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "normalization": {
            key: value.detach().cpu()
            for key, value in normalization.items()
        },
        "timestep_seconds": RISK_BENCH_TIMESTEP_SECONDS,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "one_class_scoring_contract": ONE_CLASS_SCORING_CONTRACT,
    }


def _evaluate_checkpoint(
    *,
    model: torch.nn.Module,
    device: torch.device,
    group_count: int,
    splits: dict[str, Any],
    epoch: int,
    include_test: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored_split_names = {
        "validation_normal",
        "validation_positive",
        *(
            {"test_normal", "test_positive"}
            if include_test
            else set()
        ),
    }
    frames = {
        split: score_sequences(
            model,
            sequences,
            device,
            split=split,
            group_count=group_count,
        )
        for split, sequences in splits.items()
        if split in scored_split_names
    }
    calibration = robust_component_calibration(frames["validation_normal"])
    for split_frames in frames.values():
        apply_danger_calibration(split_frames, calibration)
    _attach_percentile_inputs(frames, frames["validation_normal"])
    for split_frames in frames.values():
        _attach_guarded_response(split_frames)

    metric_rows = []
    false_alarm_rows = []
    evaluations = [
        ("validation", "validation_normal", "validation_positive"),
    ]
    if include_test:
        evaluations.append(("test", "test_normal", "test_positive"))
    for split, normal_split, positive_split in evaluations:
        for row in _method_metrics(
            normal_frames=frames[normal_split],
            positive_frames=frames[positive_split],
            split=split,
        ):
            metric_rows.append({"epoch": epoch, **row})
        for method, score_key in METHODS.items():
            false_alarm_rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "method": method,
                    **false_alarm_morphology(
                        frames[normal_split],
                        score_key=score_key,
                        threshold=ALERT_PERCENTILE,
                    ),
                }
            )
    return metric_rows, false_alarm_rows


def _metric(
    rows: Sequence[dict[str, Any]],
    *,
    epoch: int,
    split: str,
    method: str,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if int(row["epoch"]) == epoch
        and row["split"] == split
        and row["method"] == method
    )


def checkpoint_selection_key(
    metric_rows: Sequence[dict[str, Any]],
    epoch: int,
) -> tuple[int, float, float, float, int]:
    ttc = _metric(
        metric_rows,
        epoch=epoch,
        split="validation",
        method="ttc_percentile",
    )
    guarded = _metric(
        metric_rows,
        epoch=epoch,
        split="validation",
        method="persistent_guarded",
    )
    guarded_fpr = float(guarded["normal_false_alarm_rate_at_threshold"])
    ttc_fpr = float(ttc["normal_false_alarm_rate_at_threshold"])
    eligible = int(
        guarded_fpr <= ttc_fpr + MAX_VALIDATION_FPR_INCREASE
    )
    return (
        eligible,
        float(guarded["event_recall_at_threshold"]),
        float(guarded["frame_auroc"]),
        -guarded_fpr,
        -epoch,
    )


def _selection_summary(
    metric_rows: Sequence[dict[str, Any]],
    eval_epochs: Sequence[int],
) -> dict[str, Any]:
    best_epoch = max(
        eval_epochs,
        key=lambda epoch: checkpoint_selection_key(metric_rows, epoch),
    )
    return {
        "best_epoch": best_epoch,
        "criterion": (
            "eligible if guarded validation frame FPR <= TTC FPR + 0.01; "
            "then maximize guarded validation event recall, AUROC, and lower FPR; "
            "prefer earlier epoch on exact ties"
        ),
        "selection_key": list(
            checkpoint_selection_key(metric_rows, best_epoch)
        ),
        "validation_ttc": _metric(
            metric_rows,
            epoch=best_epoch,
            split="validation",
            method="ttc_percentile",
        ),
        "validation_guarded": _metric(
            metric_rows,
            epoch=best_epoch,
            split="validation",
            method="persistent_guarded",
        ),
        "test_guarded_diagnostic": next(
            (
                row
                for row in metric_rows
                if int(row["epoch"]) == best_epoch
                and row["split"] == "test"
                and row["method"] == "persistent_guarded"
            ),
            None,
        ),
    }


def _normal_only_selection_summary(
    history_rows: Sequence[dict[str, Any]],
    eval_epochs: Sequence[int],
) -> dict[str, Any]:
    """Select a checkpoint without scoring any positive sequence."""
    candidates = [
        row for row in history_rows if int(row["epoch"]) in set(eval_epochs)
    ]
    if not candidates:
        raise ValueError("No normal-only checkpoint candidates were evaluated.")
    best = min(
        candidates,
        key=lambda row: (
            float(row["val_total"]),
            float(row["val_next_edge"]),
            float(row["val_prototype_compactness"]),
            int(row["epoch"]),
        ),
    )
    return {
        "best_epoch": int(best["epoch"]),
        "criterion": (
            "minimize validation-normal total objective loss; break ties by "
            "validation-normal next-edge loss, prototype compactness, and "
            "earlier epoch"
        ),
        "selection_key": [
            float(best["val_total"]),
            float(best["val_next_edge"]),
            float(best["val_prototype_compactness"]),
            int(best["epoch"]),
        ],
        "validation_normal": dict(best),
        "positive_sequences_scored_during_selection": 0,
        "positive_frames_used_for_selection": 0,
    }


def run_guarded_epoch_sweep(
    *,
    dataset_path: Path = DEFAULT_DEVELOPMENT_INDEX,
    confirmation_index: Path = DEFAULT_CONFIRMATION_INDEX,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    epochs: int = 20,
    eval_epochs: Sequence[int] = DEFAULT_EVAL_EPOCHS,
    hidden_channels: int = 32,
    latent_channels: int = 16,
    learning_rate: float = 1e-3,
    group_count: int = 4,
    seed: int = 20260721,
    objective_mode: str = "one_class_prototype",
    train_safety_policy: str = "label_normal",
    safe_min_segment_frames: int = 8,
    run_confirmation: bool = True,
    evaluate_test_during_selection: bool = True,
    model_name: str | None = None,
    checkpoint_selection_mode: str = "guarded_positive",
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if checkpoint_selection_mode not in {
        "guarded_positive",
        "normal_only_val_loss",
    }:
        raise ValueError(
            f"Unknown checkpoint selection mode: {checkpoint_selection_mode}"
        )
    eval_epochs = sorted(
        {int(epoch) for epoch in eval_epochs if 0 < int(epoch) <= epochs}
    )
    if epochs not in eval_epochs:
        eval_epochs.append(epochs)

    torch.manual_seed(seed)
    np.random.seed(seed)
    package = load_graph_dataset(dataset_path)
    training = _load_training_module()
    raw_train = package["splits"]["train_normal"]
    train_sequences = _apply_training_safety_policy(
        raw_train,
        policy=train_safety_policy,
        min_segment_frames=safe_min_segment_frames,
    )
    x_mean, x_std, edge_mean, edge_std = _fit_normalization(train_sequences)
    normalization = {
        "x_mean": x_mean,
        "x_std": x_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }
    normalized_splits = {
        split: _normalize_sequences(
            sequences,
            x_mean,
            x_std,
            edge_mean,
            edge_std,
        )
        for split, sequences in package["splits"].items()
        if split != "train_positive"
    }
    normalized_splits["train_normal"] = _normalize_sequences(
        train_sequences,
        x_mean,
        x_std,
        edge_mean,
        edge_std,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = training.ModelConfig(
        model_name=model_name or training.RELATION_MODEL_NAME,
        in_channels=int(normalized_splits["train_normal"][0][0].x.shape[1]),
        edge_attr_dim=int(
            normalized_splits["train_normal"][0][0].edge_attr.shape[1]
        ),
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        dropout=0.10,
    )
    model = training.build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    output_root = output_root.expanduser().resolve()
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    history_rows = []
    metric_rows = []
    false_alarm_rows = []
    checkpoints = {}

    for epoch in range(1, epochs + 1):
        train_metrics = training.train_relation_localizer_one_epoch(
            model=model,
            sequences=normalized_splits["train_normal"],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            objective_mode=objective_mode,
        )
        val_metrics = training.evaluate_relation_localizer_one_epoch(
            model=model,
            sequences=normalized_splits["validation_normal"],
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            objective_mode=objective_mode,
        )
        history_row = {
            "epoch": epoch,
            "train_total": float(train_metrics["total"]),
            "train_masked_edge": float(train_metrics["masked_edge"]),
            "train_next_edge": float(train_metrics["next_edge"]),
            "train_prototype_compactness": float(
                train_metrics["prototype_compactness"]
            ),
            "val_total": float(val_metrics["total"]),
            "val_masked_edge": float(val_metrics["masked_edge"]),
            "val_next_edge": float(val_metrics["next_edge"]),
            "val_prototype_compactness": float(
                val_metrics["prototype_compactness"]
            ),
            "mask_fraction": float(train_metrics["mask_fraction"]),
        }
        history_rows.append(history_row)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_total": history_row["train_total"],
                    "val_total": history_row["val_total"],
                    "guarded_evaluation": epoch in eval_epochs,
                }
            ),
            flush=True,
        )
        if epoch not in eval_epochs:
            continue

        checkpoint_path = checkpoint_root / f"epoch_{epoch:03d}.pt"
        torch.save(
            _checkpoint_payload(
                dataset_path=dataset_path,
                epoch=epoch,
                seed=seed,
                objective_mode=objective_mode,
                train_safety_policy=train_safety_policy,
                config=config,
                model=model,
                optimizer=optimizer,
                normalization=normalization,
            ),
            checkpoint_path,
        )
        checkpoints[str(epoch)] = str(checkpoint_path)
        if checkpoint_selection_mode == "guarded_positive":
            epoch_metrics, epoch_false_alarms = _evaluate_checkpoint(
                model=model,
                device=device,
                group_count=group_count,
                splits=normalized_splits,
                epoch=epoch,
                include_test=evaluate_test_during_selection,
            )
            metric_rows.extend(epoch_metrics)
            false_alarm_rows.extend(epoch_false_alarms)

    selection = (
        _selection_summary(metric_rows, eval_epochs)
        if checkpoint_selection_mode == "guarded_positive"
        else _normal_only_selection_summary(history_rows, eval_epochs)
    )
    best_epoch = int(selection["best_epoch"])
    best_checkpoint = Path(checkpoints[str(best_epoch)])
    best_checkpoint_path = output_root / "best_guarded_checkpoint.pt"
    best_checkpoint_path.write_bytes(best_checkpoint.read_bytes())

    confirmation_summary = None
    if run_confirmation:
        confirmation_summary = run_persistent_guarded_fusion(
            development_index=dataset_path,
            confirmation_index=confirmation_index,
            checkpoint_path=best_checkpoint_path,
            output_root=output_root / "best_epoch_full_evaluation",
            group_count=group_count,
        )

    history_path = output_root / "guarded_epoch_history.csv"
    metrics_path = output_root / "guarded_epoch_metrics.csv"
    false_alarm_path = output_root / "guarded_epoch_false_alarms.csv"
    summary_path = output_root / "guarded_epoch_sweep_summary.json"
    _write_csv(history_path, history_rows)
    _write_csv(metrics_path, metric_rows)
    _write_csv(false_alarm_path, false_alarm_rows)
    summary = {
        "protocol": "riskbench-guarded-checkpoint-sweep-v1",
        "dataset_path": str(dataset_path.expanduser().resolve()),
        "confirmation_index": str(
            confirmation_index.expanduser().resolve()
        ),
        "epochs": epochs,
        "eval_epochs": list(eval_epochs),
        "seed": seed,
        "device": str(device),
        "objective_mode": objective_mode,
        "train_safety_policy": train_safety_policy,
        "hidden_channels": hidden_channels,
        "latent_channels": latent_channels,
        "learning_rate": learning_rate,
        "group_count": group_count,
        "model_name": config.model_name,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "positive_sequences_scored_during_selection": int(
            checkpoint_selection_mode == "guarded_positive"
        ),
        "evaluate_test_during_selection": evaluate_test_during_selection,
        "selection": selection,
        "best_checkpoint_path": str(best_checkpoint_path),
        "checkpoints": checkpoints,
        "history": history_rows,
        "metrics": metric_rows,
        "confirmation": confirmation_summary,
        "artifacts": {
            "history_csv": str(history_path),
            "metrics_csv": str(metrics_path),
            "false_alarms_csv": str(false_alarm_path),
        },
        "status": "guarded_epoch_sweep_complete",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    result = run_guarded_epoch_sweep(
        dataset_path=args.dataset_path,
        confirmation_index=args.confirmation_index,
        output_root=args.output_root,
        epochs=args.epochs,
        eval_epochs=parse_eval_epochs(
            args.eval_epochs,
            max_epoch=args.epochs,
        ),
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        learning_rate=args.learning_rate,
        group_count=args.group_count,
        seed=args.seed,
        objective_mode=args.objective_mode,
        train_safety_policy=args.train_safety_policy,
        safe_min_segment_frames=args.safe_min_segment_frames,
        run_confirmation=not args.skip_confirmation,
        evaluate_test_during_selection=(
            not args.skip_test_during_selection
        ),
        model_name=args.model_name,
        checkpoint_selection_mode=args.checkpoint_selection_mode,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
