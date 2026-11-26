"""Smoke training with the existing temporal relation / Immune-GAE model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from .dataset_io import load_graph_dataset
from .schema import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = PROJECT_ROOT / "02_alt_temporal_train.py"
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_us101_counterfactual_dataset_v3"
    / "us101_ego_immune_counterfactual_dataset.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_us101_temporal_relation_smoke"
)
DUAL_TRACK_AGGREGATIONS = [
    "top3_mean",
    "ego_incident_causal_ewma",
]
DEFAULT_OBJECTIVE_MODE = "one_class_prototype"
ONE_CLASS_SCORING_CONTRACT = {
    "primary_online_score": "ego_incident_causal_ewma",
    "primary_signal": "causal_future_relation_prediction_residual",
    "prototype_role": (
        "safe_core_compactness_regularizer_and_auxiliary_diagnostic"
    ),
    "prototype_fused_score_is_primary": False,
}
EGO_CAUSAL_EWMA_ALPHA = 0.35
PAIRED_RULE_SCORE_KEYS = {
    "inverse_ttc": "inverse_ttc_ego_max",
    "closing_speed": "closing_speed_ego_max",
    "inverse_distance": "inverse_distance_ego_max",
}
OBSERVABLE_DIVERGENCE_CONFIG = {
    "position_floor_m": 0.5,
    "position_low_confidence_extra_m": 1.5,
    "velocity_floor_mps": 0.3,
    "velocity_low_confidence_extra_mps": 1.0,
    "relation_distance_floor_m": 0.75,
    "relation_closing_speed_floor_mps": 0.5,
    "relation_inverse_ttc_floor_per_s": 0.05,
}
SEVERITY_DEFINITIONS = {
    "mild": "severity_score <= train composite q33",
    "medium": "train composite q33 < severity_score <= q67",
    "strong": "severity_score > train composite q67",
    "fallback": "legacy packages use extra_delay_frames bins",
}
TRAIN_SAFETY_POLICIES = ["label_normal", "kinematic_safe_segments"]
DUAL_TRACK_DEFINITIONS = [
    {
        "track": "strict_original_window",
        "aggregation": "top3_mean",
        "label_window": "original",
        "description": "Conservative legacy protocol: original metadata alert window + top-3 all-edge surprise.",
    },
    {
        "track": "early_immune_pre_anchor_10",
        "aggregation": "ego_incident_causal_ewma",
        "label_window": "pre_anchor_10",
        "description": "Immune-style early alert: sustained causal ego-relation forecast surprise in the 1s pre-anchor window.",
    },
    {
        "track": "early_immune_original_dilate_5",
        "aggregation": "ego_incident_causal_ewma",
        "label_window": "original_dilate_5",
        "description": "Immune-style timing-tolerant alert: sustained causal ego-relation forecast surprise near the original window.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-train existing temporal relation localizer on ego data."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--group-count", type=int, default=4)
    parser.add_argument(
        "--objective-mode",
        choices=[
            "one_class_prototype",
            "causal_future",
            "hybrid_reconstruction",
        ],
        default=DEFAULT_OBJECTIVE_MODE,
    )
    parser.add_argument(
        "--train-safety-policy",
        choices=TRAIN_SAFETY_POLICIES,
        default="kinematic_safe_segments",
    )
    parser.add_argument("--safe-min-segment-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_temporal_relation_smoke(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        latent_channels=args.latent_channels,
        learning_rate=args.learning_rate,
        group_count=args.group_count,
        seed=args.seed,
        objective_mode=args.objective_mode,
        train_safety_policy=args.train_safety_policy,
        safe_min_segment_frames=args.safe_min_segment_frames,
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "temporal_relation_smoke_summary.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_path": str(output_path), **result}, indent=2))


def run_temporal_relation_smoke(
    *,
    dataset_path: Path,
    epochs: int,
    hidden_channels: int,
    latent_channels: int,
    learning_rate: float,
    group_count: int,
    seed: int,
    objective_mode: str = DEFAULT_OBJECTIVE_MODE,
    train_safety_policy: str = "kinematic_safe_segments",
    safe_min_segment_frames: int = 8,
    timestep_seconds: float = 0.1,
    dual_track_definitions: list[dict[str, Any]] | None = None,
    checkpoint_path: Path | None = None,
    run_status: str = "temporal_relation_dual_track_smoke_not_final_run",
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if timestep_seconds <= 0.0:
        raise ValueError("timestep_seconds must be positive.")
    track_definitions = dual_track_definitions or DUAL_TRACK_DEFINITIONS

    torch.manual_seed(seed)
    np.random.seed(seed)
    training = _load_training_module()
    package = load_graph_dataset(dataset_path)
    event_metadata_by_uid = _load_counterfactual_event_metadata(package)

    train_sequences = package["splits"]["train_normal"]
    validation_normal = package["splits"]["validation_normal"]
    validation_positive = package["splits"]["validation_positive"]
    test_normal = package["splits"]["test_normal"]
    test_positive = package["splits"]["test_positive"]
    source_training_safety_audit = _training_safety_audit(train_sequences)
    train_sequences = _apply_training_safety_policy(
        train_sequences,
        policy=train_safety_policy,
        min_segment_frames=safe_min_segment_frames,
    )
    training_safety_audit = {
        "policy": train_safety_policy,
        "safe_min_segment_frames": safe_min_segment_frames,
        "source": source_training_safety_audit,
        "effective": _training_safety_audit(train_sequences),
    }
    three_pool_protocol = _three_pool_protocol_audit(
        safe_core=train_sequences,
        hard_normal=validation_normal,
        positive_ood=_ConcatenatedSequenceView(
            validation_positive,
            test_positive,
        ),
    )

    x_mean, x_std, edge_mean, edge_std = _fit_normalization(train_sequences)
    train_sequences = _normalize_sequences(train_sequences, x_mean, x_std, edge_mean, edge_std)
    validation_normal = _normalize_sequences(validation_normal, x_mean, x_std, edge_mean, edge_std)
    validation_positive = _normalize_sequences(validation_positive, x_mean, x_std, edge_mean, edge_std)
    test_normal = _normalize_sequences(test_normal, x_mean, x_std, edge_mean, edge_std)
    test_positive = _normalize_sequences(test_positive, x_mean, x_std, edge_mean, edge_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = training.ModelConfig(
        model_name=training.RELATION_MODEL_NAME,
        in_channels=int(train_sequences[0][0].x.shape[1]),
        edge_attr_dim=int(train_sequences[0][0].edge_attr.shape[1]),
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

    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = training.train_relation_localizer_one_epoch(
            model=model,
            sequences=train_sequences,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            objective_mode=objective_mode,
        )
        val_metrics = training.evaluate_relation_localizer_one_epoch(
            model=model,
            sequences=validation_normal,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            objective_mode=objective_mode,
        )
        history.append(
            {
                "epoch": epoch,
                "train_total": train_metrics["total"],
                "train_masked_edge": train_metrics["masked_edge"],
                "train_next_edge": train_metrics["next_edge"],
                "train_prototype_compactness": train_metrics[
                    "prototype_compactness"
                ],
                "val_total": val_metrics["total"],
                "val_masked_edge": val_metrics["masked_edge"],
                "val_next_edge": val_metrics["next_edge"],
                "val_prototype_compactness": val_metrics[
                    "prototype_compactness"
                ],
                "mask_fraction": train_metrics["mask_fraction"],
            }
        )

    threshold = _normal_threshold(
        model,
        validation_normal,
        device,
        group_count=group_count,
        quantile=0.95,
    )
    validation_eval = _evaluate_split_pair(
        model,
        normal_sequences=validation_normal,
        positive_sequences=validation_positive,
        device=device,
        group_count=group_count,
        threshold=threshold,
        timestep_seconds=timestep_seconds,
    )
    test_eval = _evaluate_split_pair(
        model,
        normal_sequences=test_normal,
        positive_sequences=test_positive,
        device=device,
        group_count=group_count,
        threshold=threshold,
        timestep_seconds=timestep_seconds,
    )
    dual_track_evaluation = _dual_track_evaluation(
        model,
        validation_normal=validation_normal,
        validation_positive=validation_positive,
        test_normal=test_normal,
        test_positive=test_positive,
        device=device,
        group_count=group_count,
        threshold_quantile=0.95,
        track_definitions=track_definitions,
        timestep_seconds=timestep_seconds,
        event_metadata_by_uid=event_metadata_by_uid,
    )

    resolved_checkpoint = None
    if checkpoint_path is not None:
        resolved_checkpoint = checkpoint_path.expanduser().resolve()
        resolved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "temporal-immune-localizer-checkpoint-v1",
                "dataset_path": str(dataset_path.expanduser().resolve()),
                "epoch": epochs,
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
                    "x_mean": x_mean.detach().cpu(),
                    "x_std": x_std.detach().cpu(),
                    "edge_mean": edge_mean.detach().cpu(),
                    "edge_std": edge_std.detach().cpu(),
                },
                "normal_threshold_95_quantile": threshold,
                "timestep_seconds": timestep_seconds,
                "node_feature_names": NODE_FEATURE_NAMES,
                "edge_feature_names": EDGE_FEATURE_NAMES,
                "one_class_scoring_contract": ONE_CLASS_SCORING_CONTRACT,
            },
            resolved_checkpoint,
        )

    return {
        "dataset_path": str(dataset_path),
        "model": "Temporal_Immune_Localizer",
        "objective_mode": objective_mode,
        "train_safety_policy": train_safety_policy,
        "training_safety_audit": training_safety_audit,
        "three_pool_protocol": three_pool_protocol,
        "prototype_state": _prototype_state_summary(model),
        "one_class_scoring_contract": ONE_CLASS_SCORING_CONTRACT,
        "epochs": epochs,
        "timestep_seconds": timestep_seconds,
        "device": str(device),
        "train_sequence_count": len(train_sequences),
        "validation_positive_sequence_count": len(validation_positive),
        "test_positive_sequence_count": len(test_positive),
        "in_channels": config.in_channels,
        "edge_attr_dim": config.edge_attr_dim,
        "hidden_channels": hidden_channels,
        "latent_channels": latent_channels,
        "checkpoint_path": (
            None if resolved_checkpoint is None else str(resolved_checkpoint)
        ),
        "validation_threshold_95fpr_normal": threshold,
        "validation": validation_eval,
        "test": test_eval,
        "dual_track_evaluation": dual_track_evaluation,
        "history": history,
        "status": run_status,
    }


def _load_training_module():
    module_name = "ego_immune_temporal_training_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fit_normalization(
    sequences: Sequence[Sequence[Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_sum = x_square_sum = edge_sum = edge_square_sum = None
    x_count = edge_count = 0
    for sequence in sequences:
        for graph in sequence:
            x = graph.x.detach().cpu().double()
            if x_sum is None:
                x_sum = torch.zeros((1, x.shape[1]), dtype=torch.float64)
                x_square_sum = torch.zeros_like(x_sum)
            x_sum += x.sum(dim=0, keepdim=True)
            x_square_sum += x.square().sum(dim=0, keepdim=True)
            x_count += int(x.shape[0])
            if graph.edge_attr.numel() == 0:
                continue
            edge = graph.edge_attr.detach().cpu().double()
            if edge_sum is None:
                edge_sum = torch.zeros((1, edge.shape[1]), dtype=torch.float64)
                edge_square_sum = torch.zeros_like(edge_sum)
            edge_sum += edge.sum(dim=0, keepdim=True)
            edge_square_sum += edge.square().sum(dim=0, keepdim=True)
            edge_count += int(edge.shape[0])
    if x_count == 0 or edge_count == 0:
        raise ValueError("Cannot normalize an empty graph dataset")
    x_mean = x_sum / x_count
    edge_mean = edge_sum / edge_count
    x_variance = (x_square_sum / x_count - x_mean.square()).clamp_min(1e-12)
    edge_variance = (
        edge_square_sum / edge_count - edge_mean.square()
    ).clamp_min(1e-12)
    return (
        x_mean.float(),
        x_variance.sqrt().float().clamp_min(1e-6),
        edge_mean.float(),
        edge_variance.sqrt().float().clamp_min(1e-6),
    )


class _NormalizedSequenceView(Sequence[list[Any]]):
    def __init__(
        self,
        sequences: Sequence[Sequence[Any]],
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        edge_mean: torch.Tensor,
        edge_std: torch.Tensor,
    ) -> None:
        self.sequences = sequences
        self.x_mean = x_mean
        self.x_std = x_std
        self.edge_mean = edge_mean
        self.edge_std = edge_std

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> list[Any]:
        normalized = []
        for graph in self.sequences[index]:
            clone = graph.clone()
            clone.raw_x = clone.x.float().clone()
            clone.raw_edge_attr = clone.edge_attr.float().clone()
            clone.x = (clone.x.float() - self.x_mean) / self.x_std
            if clone.edge_attr.numel() > 0:
                clone.edge_attr = (
                    clone.edge_attr.float() - self.edge_mean
                ) / self.edge_std
            normalized.append(clone)
        return normalized

    def __iter__(self) -> Iterator[list[Any]]:
        for index in range(len(self)):
            yield self[index]


class _ConcatenatedSequenceView(Sequence[Sequence[Any]]):
    def __init__(self, *sources: Sequence[Sequence[Any]]) -> None:
        self.sources = sources

    def __len__(self) -> int:
        return sum(len(source) for source in self.sources)

    def __getitem__(self, index: int) -> Sequence[Any]:
        if index < 0:
            index += len(self)
        for source in self.sources:
            if index < len(source):
                return source[index]
            index -= len(source)
        raise IndexError(index)


def _normalize_sequences(
    sequences: Sequence[Sequence[Any]],
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    edge_mean: torch.Tensor,
    edge_std: torch.Tensor,
) -> Sequence[list[Any]]:
    return _NormalizedSequenceView(
        sequences,
        x_mean,
        x_std,
        edge_mean,
        edge_std,
    )


def _load_counterfactual_event_metadata(
    package: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    source_path = package.get("meta", {}).get("source_path")
    if not source_path:
        return {}
    path = Path(str(source_path)).expanduser()
    if not path.exists():
        return {}
    source = torch.load(path, weights_only=False)
    fields = [
        "event_uid",
        "variant_type",
        "extra_delay_frames",
        "extra_delay_seconds",
        "severity",
        "severity_protocol",
        "severity_score",
        "severity_risk_percentile_train",
        "severity_lead_percentile_train",
        "peak_directional_risk_increment",
        "mean_directional_risk_increment",
        "effective_lead_window_frames",
        "effective_lead_window_seconds",
        "early_window_start_index",
        "early_window_end_index",
        "first_directional_hazard_index",
        "directional_hazard_window_frames",
        "physical_danger_index",
        "anomaly_peak_index",
        "max_speed_difference_fps",
        "max_displacement_difference_ft",
    ]
    return {
        str(event["event_uid"]): {
            field: event.get(field)
            for field in fields
        }
        for event in source.get("paired_events", [])
    }


def _training_safety_audit(
    sequences: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    frame_count = 0
    alert_count = 0
    physical_count = 0
    inverse_ttc_count = 0
    closing_3_count = 0
    closing_5_count = 0
    risky_sequences = 0
    for sequence in sequences:
        sequence_has_inverse_ttc_risk = False
        for graph in sequence:
            frame_count += 1
            alert_count += int(graph.y_immune_alert.item())
            physical_count += int(graph.y_physical_danger.item())
            scores = _ego_rule_scores_for_graph(graph)
            inverse_ttc_risk = scores["inverse_ttc_ego_max"] >= 0.2
            inverse_ttc_count += int(inverse_ttc_risk)
            closing_3_count += int(scores["closing_speed_ego_max"] >= 3.0)
            closing_5_count += int(scores["closing_speed_ego_max"] >= 5.0)
            sequence_has_inverse_ttc_risk |= inverse_ttc_risk
        risky_sequences += int(sequence_has_inverse_ttc_risk)
    return {
        "sequence_count": len(sequences),
        "frame_count": frame_count,
        "labeled_immune_alert_frame_count": alert_count,
        "labeled_physical_danger_frame_count": physical_count,
        "ego_inverse_ttc_ge_0_2_frame_count": inverse_ttc_count,
        "ego_inverse_ttc_ge_0_2_frame_rate": _fraction(
            inverse_ttc_count, frame_count
        ),
        "ego_inverse_ttc_ge_0_2_sequence_count": int(risky_sequences),
        "ego_closing_speed_ge_3_frame_count": closing_3_count,
        "ego_closing_speed_ge_3_frame_rate": _fraction(
            closing_3_count, frame_count
        ),
        "ego_closing_speed_ge_5_frame_count": closing_5_count,
        "ego_closing_speed_ge_5_frame_rate": _fraction(
            closing_5_count, frame_count
        ),
        "audit_warning": (
            "label-normal data crosses generic kinematic thresholds; apply "
            "dataset-specific semantics before filtering"
            if inverse_ttc_count
            else None
        ),
    }


def _apply_training_safety_policy(
    sequences: list[list[Any]],
    *,
    policy: str,
    min_segment_frames: int,
) -> list[list[Any]]:
    if policy not in TRAIN_SAFETY_POLICIES:
        raise ValueError(f"Unknown training safety policy: {policy}")
    if min_segment_frames <= 1:
        raise ValueError("min_segment_frames must be greater than 1")
    if policy == "label_normal":
        return sequences

    safe_segments = []
    for sequence in sequences:
        current = []
        for graph in sequence:
            scores = _ego_rule_scores_for_graph(graph)
            safe = (
                scores["inverse_ttc_ego_max"] < 0.2
                and scores["closing_speed_ego_max"] < 3.0
            )
            if safe:
                current.append(graph)
                continue
            if len(current) >= min_segment_frames:
                safe_segments.append(current)
            current = []
        if len(current) >= min_segment_frames:
            safe_segments.append(current)
    if not safe_segments:
        raise ValueError(
            "kinematic_safe_segments produced no trainable sequences"
        )
    return safe_segments


def _three_pool_protocol_audit(
    *,
    safe_core: Sequence[Sequence[Any]],
    hard_normal: Sequence[Sequence[Any]],
    positive_ood: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    def stats(sequences: Sequence[Sequence[Any]]) -> dict[str, int]:
        frame_count = alert_count = physical_count = 0
        for sequence in sequences:
            for graph in sequence:
                frame_count += 1
                alert_count += int(graph.y_immune_alert.item())
                physical_count += int(graph.y_physical_danger.item())
        return {
            "frame_count": frame_count,
            "alert_count": alert_count,
            "physical_count": physical_count,
        }

    safe_stats = stats(safe_core)
    hard_stats = stats(hard_normal)
    ood_stats = stats(positive_ood)
    safe_alert_labels = safe_stats["alert_count"]
    safe_physical_labels = safe_stats["physical_count"]
    if safe_alert_labels or safe_physical_labels:
        raise ValueError("safe-core contains positive danger labels")
    return {
        "protocol": "three-pool-one-class-v1",
        "optimization_pool": "safe_core_only",
        "threshold_calibration_pool": "hard_but_safe_normal_only",
        "evaluation_pool": "positive_ood_only",
        "safe_core_sequence_count": len(safe_core),
        "safe_core_frame_count": safe_stats["frame_count"],
        "hard_normal_sequence_count": len(hard_normal),
        "hard_normal_frame_count": hard_stats["frame_count"],
        "positive_ood_sequence_count": len(positive_ood),
        "positive_ood_frame_count": ood_stats["frame_count"],
        "safe_core_positive_label_count": (
            safe_alert_labels + safe_physical_labels
        ),
        "hard_normal_positive_label_count": (
            hard_stats["alert_count"] + hard_stats["physical_count"]
        ),
        "positive_ood_gradient_steps": 0,
        "hard_normal_gradient_steps": 0,
    }


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _prototype_state_summary(model: torch.nn.Module) -> dict[str, Any]:
    initialized = bool(
        getattr(
            model,
            "safe_relation_prototype_initialized",
            torch.tensor(False),
        ).item()
    )
    if not initialized:
        return {"initialized": False}
    center = model.safe_relation_prototype_center.detach().cpu()
    scale = model.safe_relation_prototype_scale.detach().cpu()
    return {
        "initialized": True,
        "center_l2_norm": float(torch.linalg.vector_norm(center).item()),
        "mean_scale": float(scale.mean().item()),
        "min_scale": float(scale.min().item()),
        "max_scale": float(scale.max().item()),
    }


def _normal_threshold(
    model: torch.nn.Module,
    sequences: list[list[Any]],
    device: torch.device,
    *,
    group_count: int,
    quantile: float,
) -> float:
    scores, _labels = _score_sequences(
        model,
        sequences,
        device,
        group_count=group_count,
        force_label=0,
    )
    if not scores:
        return float("inf")
    return float(np.quantile(np.asarray(scores, dtype=float), quantile))


def _evaluate_split_pair(
    model: torch.nn.Module,
    *,
    normal_sequences: list[list[Any]],
    positive_sequences: list[list[Any]],
    device: torch.device,
    group_count: int,
    threshold: float,
    timestep_seconds: float = 0.1,
) -> dict[str, float | int | None]:
    normal_scores, normal_labels = _score_sequences(
        model,
        normal_sequences,
        device,
        group_count=group_count,
        force_label=0,
    )
    positive_scores, positive_labels = _score_sequences(
        model,
        positive_sequences,
        device,
        group_count=group_count,
    )
    del normal_labels
    alert_scores = [
        score
        for score, label in zip(positive_scores, positive_labels)
        if label == 1
    ]
    background_scores = [
        score
        for score, label in zip(positive_scores, positive_labels)
        if label == 0
    ]
    auc_views = _frame_auc_views(
        normal_scores=normal_scores,
        positive_scores=positive_scores,
        positive_labels=positive_labels,
    )
    event_metrics = _event_lead_time_metrics(
        model,
        positive_sequences,
        device,
        group_count=group_count,
        threshold=threshold,
        timestep_seconds=timestep_seconds,
    )
    return {
        "frame_count": len(normal_scores) + len(alert_scores),
        "combined_frame_count": len(normal_scores) + len(positive_scores),
        "within_event_frame_count": len(positive_scores),
        "positive_alert_frame_count": int(sum(positive_labels)),
        "positive_background_frame_count": len(background_scores),
        "frame_auroc": auc_views["independent_normal"],
        "frame_auroc_independent_normal": auc_views["independent_normal"],
        "frame_auroc_within_event": auc_views["within_event"],
        "frame_auroc_combined_legacy": auc_views["combined_legacy"],
        "normal_false_alarm_rate_at_threshold": _rate_above(
            normal_scores,
            threshold,
        ),
        "positive_alert_recall_at_threshold": _rate_above(
            alert_scores,
            threshold,
        ),
        "mean_normal_score": float(np.mean(normal_scores)) if normal_scores else None,
        "mean_alert_score": float(np.mean(alert_scores)) if alert_scores else None,
        **event_metrics,
    }


@torch.no_grad()
def _score_sequences(
    model: torch.nn.Module,
    sequences: list[list[Any]],
    device: torch.device,
    *,
    group_count: int,
    force_label: int | None = None,
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    model.eval()
    for sequence in sequences:
        outputs = model.score_sequence_masked_groups(
            sequence,
            device,
            group_count=group_count,
            surprise_mode="absolute",
        )
        for graph, output in zip(sequence, outputs):
            edge_scores = output["edge_scores"].detach().cpu()
            if edge_scores.numel() == 0:
                score = 0.0
            else:
                k = min(3, edge_scores.numel())
                score = float(torch.topk(edge_scores, k=k).values.mean().item())
            label = (
                int(force_label)
                if force_label is not None
                else int(graph.y_immune_alert.item())
            )
            scores.append(score)
            labels.append(label)
    return scores, labels


@torch.no_grad()
def _event_lead_time_metrics(
    model: torch.nn.Module,
    positive_sequences: list[list[Any]],
    device: torch.device,
    *,
    group_count: int,
    threshold: float,
    timestep_seconds: float = 0.1,
) -> dict[str, float | int | None]:
    event_count = len(positive_sequences)
    if event_count == 0:
        return {
            "event_count": 0,
            "event_recall_at_threshold": None,
            "mean_lead_frames": None,
            "median_lead_frames": None,
            "mean_lead_seconds": None,
            "median_lead_seconds": None,
        }

    detected_leads: list[int] = []
    for sequence in positive_sequences:
        outputs = model.score_sequence_masked_groups(
            sequence,
            device,
            group_count=group_count,
            surprise_mode="absolute",
        )
        frame_scores = []
        labels = []
        frames_to_anchor = []
        for graph, output in zip(sequence, outputs):
            edge_scores = output["edge_scores"].detach().cpu()
            if edge_scores.numel() == 0:
                score = 0.0
            else:
                k = min(3, edge_scores.numel())
                score = float(torch.topk(edge_scores, k=k).values.mean().item())
            frame_scores.append(score)
            labels.append(int(graph.y_immune_alert.item()))
            frames_to_anchor.append(int(graph.frames_to_physical_danger.item()))

        alert_positions = [
            index
            for index, label in enumerate(labels)
            if label == 1
        ]
        if not alert_positions:
            continue
        detected_positions = [
            index
            for index in alert_positions
            if frame_scores[index] >= threshold
        ]
        if not detected_positions:
            continue
        first_detection = min(detected_positions)
        detected_leads.append(max(frames_to_anchor[first_detection], 0))

    if not detected_leads:
        return {
            "event_count": event_count,
            "event_recall_at_threshold": 0.0,
            "mean_lead_frames": None,
            "median_lead_frames": None,
            "mean_lead_seconds": None,
            "median_lead_seconds": None,
        }

    lead_array = np.asarray(detected_leads, dtype=float)
    return {
        "event_count": event_count,
        "event_recall_at_threshold": float(len(detected_leads) / event_count),
        "mean_lead_frames": float(np.mean(lead_array)),
        "median_lead_frames": float(np.median(lead_array)),
        "mean_lead_seconds": float(np.mean(lead_array) * timestep_seconds),
        "median_lead_seconds": float(np.median(lead_array) * timestep_seconds),
    }


@torch.no_grad()
def _dual_track_evaluation(
    model: torch.nn.Module,
    *,
    validation_normal: list[list[Any]],
    validation_positive: list[list[Any]],
    test_normal: list[list[Any]],
    test_positive: list[list[Any]],
    device: torch.device,
    group_count: int,
    threshold_quantile: float,
    track_definitions: list[dict[str, Any]] | None = None,
    timestep_seconds: float = 0.1,
    return_frames: bool = False,
    event_metadata_by_uid: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_metadata_by_uid = event_metadata_by_uid or {}
    track_definitions = track_definitions or DUAL_TRACK_DEFINITIONS
    normal_frames = _dual_track_score_frames(
        model,
        validation_normal,
        device,
        group_count=group_count,
        split="validation_normal",
    )
    validation_frames = _dual_track_score_frames(
        model,
        validation_positive,
        device,
        group_count=group_count,
        split="validation_positive",
    )
    test_normal_frames = _dual_track_score_frames(
        model,
        test_normal,
        device,
        group_count=group_count,
        split="test_normal",
    )
    test_frames = _dual_track_score_frames(
        model,
        test_positive,
        device,
        group_count=group_count,
        split="test_positive",
    )
    one_class_calibration = _fit_one_class_calibration(normal_frames)
    for frames in [
        normal_frames,
        validation_frames,
        test_normal_frames,
        test_frames,
    ]:
        _apply_one_class_calibration(frames, one_class_calibration)
    aggregations = sorted(
        {str(definition["aggregation"]) for definition in track_definitions}
    )
    thresholds = {
        aggregation: _aggregation_threshold(
            normal_frames,
            aggregation=aggregation,
            quantile=threshold_quantile,
        )
        for aggregation in aggregations
    }

    track_results = {
        "validation_positive": _evaluate_dual_tracks_for_split(
            normal_frames=normal_frames,
            positive_frames=validation_frames,
            thresholds=thresholds,
            track_definitions=track_definitions,
            timestep_seconds=timestep_seconds,
        ),
        "test_positive": _evaluate_dual_tracks_for_split(
            normal_frames=test_normal_frames,
            positive_frames=test_frames,
            thresholds=thresholds,
            track_definitions=track_definitions,
            timestep_seconds=timestep_seconds,
        ),
    }
    relation_localization = {
        "validation_positive": _dual_track_relation_summary(validation_frames),
        "test_positive": _dual_track_relation_summary(test_frames),
    }
    validation_paired = _paired_counterfactual_diagnostics(
        normal_frames=normal_frames,
        positive_frames=validation_frames,
        split="validation_positive",
        event_metadata_by_uid=event_metadata_by_uid,
    )
    test_paired = _paired_counterfactual_diagnostics(
        normal_frames=test_normal_frames,
        positive_frames=test_frames,
        split="test_positive",
        event_metadata_by_uid=event_metadata_by_uid,
    )
    result = {
        "protocol": "dual-track-causal-paired-observable-v3",
        "primary_frame_auroc": "alert_frames_vs_independent_normal_frames",
        "one_class_scoring_contract": ONE_CLASS_SCORING_CONTRACT,
        "threshold_source": (
            f"validation_normal_{threshold_quantile:.2f}_quantile_per_aggregation"
        ),
        "threshold_calibration_pool": "hard_but_safe_validation_normal",
        "one_class_component_calibration": one_class_calibration,
        "thresholds": thresholds,
        "track_definitions": track_definitions,
        "timestep_seconds": timestep_seconds,
        "splits": track_results,
        "relation_localization": relation_localization,
        "paired_counterfactual": {
            "validation_positive": validation_paired["summaries"],
            "test_positive": test_paired["summaries"],
        },
        "paired_counterfactual_severity": {
            "validation_positive": validation_paired["severity_summaries"],
            "test_positive": test_paired["severity_summaries"],
        },
        "paired_observable_events": {
            "validation_positive": validation_paired["event_summaries"],
            "test_positive": test_paired["event_summaries"],
        },
        "observable_divergence_config": OBSERVABLE_DIVERGENCE_CONFIG,
        "severity_definitions": SEVERITY_DEFINITIONS,
    }
    if return_frames:
        result["frames"] = {
            "validation_normal": normal_frames,
            "validation_positive": validation_frames,
            "test_normal": test_normal_frames,
            "test_positive": test_frames,
        }
        result["paired_frames"] = {
            "validation_positive": validation_paired["frames"],
            "test_positive": test_paired["frames"],
        }
    return result


def _fit_one_class_calibration(
    hard_normal_frames: list[dict[str, Any]],
) -> dict[str, float]:
    def robust_stats(key: str) -> tuple[float, float]:
        values = np.asarray(
            [
                float(frame["aggregations"][key])
                for frame in hard_normal_frames
            ],
            dtype=float,
        )
        if values.size == 0:
            return 0.0, 1.0
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        return center, max(1.4826 * mad, 1e-3)

    residual_center, residual_scale = robust_stats(
        "ego_incident_causal_ewma"
    )
    prototype_center, prototype_scale = robust_stats(
        "ego_incident_prototype_ewma"
    )
    return {
        "source": "hard_but_safe_validation_normal",
        "residual_center": residual_center,
        "residual_scale": residual_scale,
        "prototype_center": prototype_center,
        "prototype_scale": prototype_scale,
        "combination": "residual_robust_z_plus_prototype_robust_z",
    }


def _apply_one_class_calibration(
    frames: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> None:
    for frame in frames:
        aggregations = frame["aggregations"]
        residual_z = (
            float(aggregations["ego_incident_causal_ewma"])
            - float(calibration["residual_center"])
        ) / float(calibration["residual_scale"])
        prototype_z = (
            float(aggregations["ego_incident_prototype_ewma"])
            - float(calibration["prototype_center"])
        ) / float(calibration["prototype_scale"])
        aggregations["ego_incident_one_class_calibrated"] = (
            residual_z + prototype_z
        )


def _evaluate_dual_tracks_for_split(
    *,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    thresholds: dict[str, float],
    track_definitions: list[dict[str, Any]] | None = None,
    timestep_seconds: float = 0.1,
) -> dict[str, dict[str, float | int | None]]:
    track_definitions = track_definitions or DUAL_TRACK_DEFINITIONS
    return {
        definition["track"]: _evaluate_dual_track(
            normal_frames=normal_frames,
            positive_frames=positive_frames,
            aggregation=str(definition["aggregation"]),
            label_window=str(definition["label_window"]),
            threshold=thresholds[str(definition["aggregation"])],
            timestep_seconds=timestep_seconds,
        )
        for definition in track_definitions
    }


def _evaluate_dual_track(
    *,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    aggregation: str,
    label_window: str,
    threshold: float,
    timestep_seconds: float = 0.1,
) -> dict[str, float | int | None]:
    normal_scores = [
        float(frame["aggregations"][aggregation])
        for frame in normal_frames
    ]
    positive_scores = [
        float(frame["aggregations"][aggregation])
        for frame in positive_frames
    ]
    positive_labels = []
    event_leads = []
    detected_event_count = 0
    grouped = _group_dual_frames_by_event(positive_frames)
    for event_frames in grouped.values():
        mask = _label_window_mask_from_frames(event_frames, label_window)
        positive_labels.extend(int(value) for value in mask)
        detected_positions = [
            index
            for index, keep in enumerate(mask)
            if keep and float(event_frames[index]["aggregations"][aggregation]) >= threshold
        ]
        if detected_positions:
            detected_event_count += 1
            first_detection = detected_positions[0]
            event_leads.append(
                max(int(event_frames[first_detection]["frames_to_anchor"]), 0)
            )

    alert_scores = [
        score
        for score, label in zip(positive_scores, positive_labels)
        if label == 1
    ]
    background_scores = [
        score
        for score, label in zip(positive_scores, positive_labels)
        if label == 0
    ]
    auc_views = _frame_auc_views(
        normal_scores=normal_scores,
        positive_scores=positive_scores,
        positive_labels=positive_labels,
    )
    return {
        "aggregation": aggregation,
        "label_window": label_window,
        "threshold": threshold,
        "frame_count": len(normal_scores) + len(alert_scores),
        "combined_frame_count": len(normal_scores) + len(positive_scores),
        "within_event_frame_count": len(positive_scores),
        "event_count": len(grouped),
        "positive_alert_frame_count": int(sum(positive_labels)),
        "positive_background_frame_count": len(background_scores),
        # Primary paper-facing AUROC: alert frames versus independent normal
        # sequences only.  The two diagnostic views remain explicit below.
        "frame_auroc": auc_views["independent_normal"],
        "frame_auroc_independent_normal": auc_views["independent_normal"],
        "frame_auroc_within_event": auc_views["within_event"],
        "frame_auroc_combined_legacy": auc_views["combined_legacy"],
        "normal_false_alarm_rate_at_threshold": _rate_above(
            normal_scores,
            threshold,
        ),
        "positive_alert_recall_at_threshold": _rate_above(
            alert_scores,
            threshold,
        ),
        "event_recall_at_threshold": (
            None
            if not grouped
            else float(detected_event_count / len(grouped))
        ),
        "mean_lead_frames": (
            None if not event_leads else float(np.mean(event_leads))
        ),
        "median_lead_frames": (
            None if not event_leads else float(np.median(event_leads))
        ),
        "mean_lead_seconds": (
            None
            if not event_leads
            else float(np.mean(event_leads) * timestep_seconds)
        ),
        "median_lead_seconds": (
            None
            if not event_leads
            else float(np.median(event_leads) * timestep_seconds)
        ),
        "mean_normal_score": (
            None if not normal_scores else float(np.mean(normal_scores))
        ),
        "mean_alert_score": (
            None if not alert_scores else float(np.mean(alert_scores))
        ),
    }


@torch.no_grad()
def _dual_track_score_frames(
    model: torch.nn.Module,
    sequences: list[list[Any]],
    device: torch.device,
    *,
    group_count: int,
    split: str,
) -> list[dict[str, Any]]:
    model.eval()
    scored_frames = []
    for event_index, sequence in enumerate(sequences):
        outputs = model.score_sequence_masked_groups(
            sequence,
            device,
            group_count=group_count,
            surprise_mode="absolute",
        )
        sequence_frames = []
        previous_ego = None
        causal_ewma = 0.0
        one_class_ewma = 0.0
        prototype_ewma = 0.0
        for frame_position, (graph, output) in enumerate(zip(sequence, outputs)):
            edge_scores = output["edge_scores"].detach().cpu()
            top3 = _topk_mean(edge_scores, 3)
            ego_top3 = _ego_incident_score(
                graph,
                edge_scores,
                mode="top3_mean",
            )
            causal_scores = output["causal_next_edge_scores"].detach().cpu()
            prototype_scores = output["causal_prototype_scores"].detach().cpu()
            one_class_scores = output["one_class_edge_scores"].detach().cpu()
            causal_coverage = output["causal_next_coverage"].detach().cpu()
            ego_causal_top3 = _ego_incident_score(
                graph,
                causal_scores,
                mode="top3_mean",
                valid_mask=causal_coverage,
            )
            causal_ewma = _ewma_update(
                causal_ewma,
                ego_causal_top3,
                alpha=EGO_CAUSAL_EWMA_ALPHA,
            )
            ego_prototype_top3 = _ego_incident_score(
                graph,
                prototype_scores,
                mode="top3_mean",
                valid_mask=causal_coverage,
            )
            prototype_ewma = _ewma_update(
                prototype_ewma,
                ego_prototype_top3,
                alpha=EGO_CAUSAL_EWMA_ALPHA,
            )
            ego_one_class_top3 = _ego_incident_score(
                graph,
                one_class_scores,
                mode="top3_mean",
                valid_mask=causal_coverage,
            )
            one_class_ewma = _ewma_update(
                one_class_ewma,
                ego_one_class_top3,
                alpha=EGO_CAUSAL_EWMA_ALPHA,
            )
            aggregations = {
                "top3_mean": top3,
                "ego_incident_causal_raw": ego_causal_top3,
                "ego_incident_causal_ewma": causal_ewma,
                "ego_incident_prototype_raw": ego_prototype_top3,
                "ego_incident_prototype_ewma": prototype_ewma,
                "ego_incident_one_class_raw": ego_one_class_top3,
                "ego_incident_one_class_ewma": one_class_ewma,
                # Retained for backward-compatible diagnostics only.
                "ego_incident_positive_delta": (
                    0.0
                    if previous_ego is None
                    else max(ego_top3 - previous_ego, 0.0)
                ),
            }
            top_edge, top_edge_score = _top_edge_from_scores(
                graph,
                causal_scores,
                valid_mask=causal_coverage,
            )
            target_source, target_target = _target_pair_from_graph(graph)
            target_rank = _target_pair_rank(
                graph=graph,
                edge_scores=causal_scores,
                target_source=target_source,
                target_target=target_target,
                valid_mask=causal_coverage,
            )
            rule_scores = _ego_rule_scores_for_graph(graph)
            observable_state = _observable_state_for_graph(graph)
            sequence_frames.append(
                {
                    "split": split,
                    "event_index": event_index,
                    "event_uid": str(
                        getattr(graph, "event_uid", graph.scenario_id)
                    ),
                    "event_variant": str(
                        getattr(graph, "event_variant", "")
                    ),
                    "scenario_id": str(graph.scenario_id),
                    "frame_position": frame_position,
                    "frames_to_anchor": int(
                        graph.frames_to_physical_danger.item()
                    ),
                    "original_alert": int(graph.y_immune_alert.item()),
                    "physical": int(graph.y_physical_danger.item()),
                    "aggregations": aggregations,
                    "top_edge": top_edge,
                    "top_edge_score": top_edge_score,
                    "target_source": target_source,
                    "target_target": target_target,
                    "rule_scores": rule_scores,
                    "_observable_state": observable_state,
                    **target_rank,
                }
            )
            previous_ego = ego_top3
        scored_frames.extend(sequence_frames)
    return scored_frames


def _paired_counterfactual_diagnostics(
    *,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    split: str,
    event_metadata_by_uid: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_metadata_by_uid = event_metadata_by_uid or {}
    normal_by_key = {
        _paired_frame_key(frame): frame
        for frame in normal_frames
    }
    grouped_positive = _group_dual_frames_by_event(positive_frames)
    paired_rows: list[dict[str, Any]] = []
    missing_normal_frame_count = 0

    for event_index, event_frames in grouped_positive.items():
        event_uid = str(event_frames[0].get("event_uid", event_index))
        event_metadata = event_metadata_by_uid.get(event_uid, {})
        extra_delay_frames = _optional_int(
            event_metadata.get("extra_delay_frames")
        )
        severity = str(
            event_metadata.get("severity")
            or _counterfactual_severity(extra_delay_frames)
        )
        label_masks = {
            str(definition["label_window"]): _label_window_mask_from_frames(
                event_frames,
                str(definition["label_window"]),
            )
            for definition in DUAL_TRACK_DEFINITIONS
        }
        event_rows = []
        for frame_position, positive_frame in enumerate(event_frames):
            normal_frame = normal_by_key.get(_paired_frame_key(positive_frame))
            if normal_frame is None:
                missing_normal_frame_count += 1
                continue

            normal_raw = float(
                normal_frame["aggregations"]["ego_incident_causal_raw"]
            )
            positive_raw = float(
                positive_frame["aggregations"]["ego_incident_causal_raw"]
            )
            normal_ewma = float(
                normal_frame["aggregations"]["ego_incident_causal_ewma"]
            )
            positive_ewma = float(
                positive_frame["aggregations"]["ego_incident_causal_ewma"]
            )
            normal_one_class_raw = float(
                normal_frame["aggregations"].get(
                    "ego_incident_one_class_raw",
                    normal_raw,
                )
            )
            positive_one_class_raw = float(
                positive_frame["aggregations"].get(
                    "ego_incident_one_class_raw",
                    positive_raw,
                )
            )
            normal_one_class_ewma = float(
                normal_frame["aggregations"].get(
                    "ego_incident_one_class_ewma",
                    normal_ewma,
                )
            )
            positive_one_class_ewma = float(
                positive_frame["aggregations"].get(
                    "ego_incident_one_class_ewma",
                    positive_ewma,
                )
            )
            normal_prototype_ewma = float(
                normal_frame["aggregations"].get(
                    "ego_incident_prototype_ewma",
                    normal_one_class_ewma - normal_ewma,
                )
            )
            positive_prototype_ewma = float(
                positive_frame["aggregations"].get(
                    "ego_incident_prototype_ewma",
                    positive_one_class_ewma - positive_ewma,
                )
            )
            normal_one_class_calibrated = float(
                normal_frame["aggregations"].get(
                    "ego_incident_one_class_calibrated",
                    normal_one_class_ewma,
                )
            )
            positive_one_class_calibrated = float(
                positive_frame["aggregations"].get(
                    "ego_incident_one_class_calibrated",
                    positive_one_class_ewma,
                )
            )
            normal_target = normal_frame.get("target_pair_best_score")
            positive_target = positive_frame.get("target_pair_best_score")
            target_delta = (
                None
                if normal_target is None or positive_target is None
                else float(positive_target) - float(normal_target)
            )
            divergence = _observable_divergence(
                normal_frame.get("_observable_state"),
                positive_frame.get("_observable_state"),
            )
            row = {
                "split": split,
                "event_index": event_index,
                "event_uid": str(positive_frame.get("event_uid", event_index)),
                "normal_scenario_id": str(normal_frame["scenario_id"]),
                "counterfactual_scenario_id": str(positive_frame["scenario_id"]),
                "frame_position": int(positive_frame["frame_position"]),
                "frames_to_anchor": int(positive_frame["frames_to_anchor"]),
                "original_alert": int(positive_frame["original_alert"]),
                "physical": int(positive_frame["physical"]),
                "normal_causal_raw": normal_raw,
                "counterfactual_causal_raw": positive_raw,
                "paired_causal_raw_delta": positive_raw - normal_raw,
                "normal_causal_ewma": normal_ewma,
                "counterfactual_causal_ewma": positive_ewma,
                "paired_causal_ewma_delta": positive_ewma - normal_ewma,
                "normal_one_class_raw": normal_one_class_raw,
                "counterfactual_one_class_raw": positive_one_class_raw,
                "paired_one_class_raw_delta": (
                    positive_one_class_raw - normal_one_class_raw
                ),
                "normal_one_class_ewma": normal_one_class_ewma,
                "counterfactual_one_class_ewma": positive_one_class_ewma,
                "paired_one_class_ewma_delta": (
                    positive_one_class_ewma - normal_one_class_ewma
                ),
                "normal_prototype_ewma": normal_prototype_ewma,
                "counterfactual_prototype_ewma": positive_prototype_ewma,
                "paired_prototype_ewma_delta": (
                    positive_prototype_ewma - normal_prototype_ewma
                ),
                "normal_one_class_calibrated": normal_one_class_calibrated,
                "counterfactual_one_class_calibrated": (
                    positive_one_class_calibrated
                ),
                "paired_one_class_calibrated_delta": (
                    positive_one_class_calibrated
                    - normal_one_class_calibrated
                ),
                "normal_target_pair_score": normal_target,
                "counterfactual_target_pair_score": positive_target,
                "paired_target_pair_delta": target_delta,
                "extra_delay_frames": extra_delay_frames,
                "extra_delay_seconds": _optional_float(
                    event_metadata.get("extra_delay_seconds")
                ),
                "severity_protocol": event_metadata.get(
                    "severity_protocol"
                ),
                "severity_score": _optional_float(
                    event_metadata.get("severity_score")
                ),
                "severity_risk_percentile_train": _optional_float(
                    event_metadata.get("severity_risk_percentile_train")
                ),
                "severity_lead_percentile_train": _optional_float(
                    event_metadata.get("severity_lead_percentile_train")
                ),
                "peak_directional_risk_increment": _optional_float(
                    event_metadata.get("peak_directional_risk_increment")
                ),
                "effective_lead_window_frames": _optional_int(
                    event_metadata.get("effective_lead_window_frames")
                ),
                "early_window_start_index": _optional_int(
                    event_metadata.get("early_window_start_index")
                ),
                "early_window_end_index": _optional_int(
                    event_metadata.get("early_window_end_index")
                ),
                "source_first_directional_hazard_index": _optional_int(
                    event_metadata.get("first_directional_hazard_index")
                ),
                "source_directional_hazard_window_frames": _optional_int(
                    event_metadata.get("directional_hazard_window_frames")
                ),
                "source_physical_danger_index": _optional_int(
                    event_metadata.get("physical_danger_index")
                ),
                "anomaly_peak_index": _optional_int(
                    event_metadata.get("anomaly_peak_index")
                ),
                "max_speed_difference_fps": _optional_float(
                    event_metadata.get("max_speed_difference_fps")
                ),
                "max_displacement_difference_ft": _optional_float(
                    event_metadata.get("max_displacement_difference_ft")
                ),
                "severity": severity,
                **divergence,
            }
            for short_name, score_key in PAIRED_RULE_SCORE_KEYS.items():
                normal_rule = normal_frame.get("rule_scores", {}).get(score_key)
                positive_rule = positive_frame.get("rule_scores", {}).get(score_key)
                row[f"normal_{short_name}_ego"] = normal_rule
                row[f"counterfactual_{short_name}_ego"] = positive_rule
                row[f"paired_{short_name}_ego_delta"] = (
                    None
                    if normal_rule is None or positive_rule is None
                    else float(positive_rule) - float(normal_rule)
                )
            for label_window, mask in label_masks.items():
                row[f"label_{label_window}"] = int(mask[frame_position])
            event_rows.append(row)

        observable_rows = [
            row for row in event_rows if row["observable_divergence"]
        ]
        first_observable = (
            None
            if not observable_rows
            else min(int(row["frame_position"]) for row in observable_rows)
        )
        anchor_positions = [
            int(row["frame_position"]) + int(row["frames_to_anchor"])
            for row in event_rows
            if int(row["frames_to_anchor"]) >= 0
        ]
        anchor_position = None if not anchor_positions else anchor_positions[0]
        first_types = ""
        if first_observable is not None:
            first_types = next(
                str(row["observable_divergence_types"])
                for row in event_rows
                if int(row["frame_position"]) == first_observable
            )
        for row in event_rows:
            position = int(row["frame_position"])
            row["first_observable_divergence_position"] = first_observable
            row["first_observable_divergence_types"] = first_types
            row["observable_lead_to_anchor_frames"] = (
                None
                if first_observable is None or anchor_position is None
                else anchor_position - first_observable
            )
            row["label_observable_to_anchor"] = int(
                first_observable is not None
                and anchor_position is not None
                and first_observable <= position <= anchor_position
            )
        paired_rows.extend(event_rows)

    summaries = {}
    paired_definitions = [
        definition
        for definition in DUAL_TRACK_DEFINITIONS
        if str(definition["aggregation"]).startswith("ego_incident_")
    ] + [
        {
            "track": "paired_observable_to_anchor",
            "aggregation": "ego_incident_causal_ewma",
            "label_window": "observable_to_anchor",
        }
    ]
    severity_summaries = {}
    for definition in paired_definitions:
        aggregation = str(definition["aggregation"])
        track = str(definition["track"])
        label_window = str(definition["label_window"])
        summaries[track] = _summarize_paired_counterfactual_rows(
            paired_rows,
            aggregation=aggregation,
            label_window=label_window,
            expected_positive_frame_count=len(positive_frames),
            missing_normal_frame_count=missing_normal_frame_count,
        )
        severity_summaries[track] = {}
        for severity in ["mild", "medium", "strong", "unknown"]:
            severity_rows = [
                row for row in paired_rows if row["severity"] == severity
            ]
            if not severity_rows:
                continue
            severity_summaries[track][severity] = (
                _summarize_paired_counterfactual_rows(
                    severity_rows,
                    aggregation=aggregation,
                    label_window=label_window,
                    expected_positive_frame_count=len(severity_rows),
                    missing_normal_frame_count=0,
                )
            )
        for metric in [
            "paired_delta_auroc",
            "paired_target_delta_auroc",
            "paired_prototype_delta_auroc",
            "paired_inverse_ttc_delta_auroc",
            "paired_closing_speed_delta_auroc",
            "paired_inverse_distance_delta_auroc",
        ]:
            tier_values = [
                metrics.get(metric)
                for severity_name, metrics in severity_summaries[track].items()
                if severity_name in SEVERITY_DEFINITIONS
                and metrics.get(metric) is not None
            ]
            summaries[track][f"severity_macro_{metric}"] = _mean_or_none(
                [float(value) for value in tier_values]
            )

    return {
        "summaries": summaries,
        "severity_summaries": severity_summaries,
        "event_summaries": _paired_observable_event_summaries(paired_rows),
        "frames": paired_rows,
    }


def _summarize_paired_counterfactual_rows(
    rows: list[dict[str, Any]],
    *,
    aggregation: str,
    label_window: str,
    expected_positive_frame_count: int,
    missing_normal_frame_count: int,
) -> dict[str, Any]:
    label_key = f"label_{label_window}"
    score_key = {
        "ego_incident_causal_ewma": "paired_causal_ewma_delta",
        "ego_incident_causal_raw": "paired_causal_raw_delta",
        "ego_incident_one_class_ewma": "paired_one_class_ewma_delta",
        "ego_incident_one_class_raw": "paired_one_class_raw_delta",
        "ego_incident_one_class_calibrated": (
            "paired_one_class_calibrated_delta"
        ),
    }.get(aggregation, "paired_causal_raw_delta")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["event_index"]), []).append(row)
    regions: dict[tuple[int, int], str] = {}
    event_peak_in_alert = []
    event_alert_mean_gt_background = []
    event_alert_mean_gt_pre_window = []
    for event_rows in grouped.values():
        event_rows.sort(key=lambda row: int(row["frame_position"]))
        event_labels = [int(row[label_key]) for row in event_rows]
        event_deltas = [float(row[score_key]) for row in event_rows]
        alert_indices = [
            index for index, label in enumerate(event_labels) if label == 1
        ]
        if alert_indices:
            first_alert = min(alert_indices)
            last_alert = max(alert_indices)
        else:
            first_alert = len(event_rows)
            last_alert = -1
        for index, row in enumerate(event_rows):
            if event_labels[index] == 1:
                region = "alert"
            elif index < first_alert:
                region = "pre_window"
            elif index > last_alert:
                region = "post_window"
            else:
                region = "between_alert_segments"
            regions[
                (int(row["event_index"]), int(row["frame_position"]))
            ] = region
        if event_deltas and any(event_labels):
            peak_index = int(np.argmax(np.asarray(event_deltas, dtype=float)))
            event_peak_in_alert.append(event_labels[peak_index] == 1)
        event_alert = [
            delta
            for delta, label in zip(event_deltas, event_labels)
            if label == 1
        ]
        event_background = [
            delta
            for delta, label in zip(event_deltas, event_labels)
            if label == 0
        ]
        if event_alert and event_background:
            event_alert_mean_gt_background.append(
                float(np.mean(event_alert)) > float(np.mean(event_background))
            )
        event_pre_window = event_deltas[:first_alert]
        if event_alert and event_pre_window:
            event_alert_mean_gt_pre_window.append(
                float(np.mean(event_alert)) > float(np.mean(event_pre_window))
            )

    labels = [int(row[label_key]) for row in rows]
    paired_deltas = [float(row[score_key]) for row in rows]
    raw_key = (
        "paired_one_class_raw_delta"
        if "one_class" in aggregation
        else "paired_causal_raw_delta"
    )
    raw_deltas = [float(row[raw_key]) for row in rows]
    row_regions = [
        regions[(int(row["event_index"]), int(row["frame_position"]))]
        for row in rows
    ]
    alert_deltas = [
        delta
        for delta, region in zip(paired_deltas, row_regions)
        if region == "alert"
    ]
    pre_window_deltas = [
        delta
        for delta, region in zip(paired_deltas, row_regions)
        if region == "pre_window"
    ]
    post_window_deltas = [
        delta
        for delta, region in zip(paired_deltas, row_regions)
        if region == "post_window"
    ]
    background_deltas = [
        delta
        for delta, label in zip(paired_deltas, labels)
        if label == 0
    ]
    raw_alert_deltas = [
        delta
        for delta, region in zip(raw_deltas, row_regions)
        if region == "alert"
    ]
    raw_pre_window_deltas = [
        delta
        for delta, region in zip(raw_deltas, row_regions)
        if region == "pre_window"
    ]
    target_labels = []
    target_deltas = []
    target_alert_deltas = []
    target_pre_window_deltas = []
    for row, label, region in zip(rows, labels, row_regions):
        target_delta = row.get("paired_target_pair_delta")
        if target_delta is None:
            continue
        target_delta = float(target_delta)
        target_labels.append(label)
        target_deltas.append(target_delta)
        if region == "alert":
            target_alert_deltas.append(target_delta)
        elif region == "pre_window":
            target_pre_window_deltas.append(target_delta)

    prototype_deltas = [
        float(row["paired_prototype_ewma_delta"]) for row in rows
    ]
    prototype_alert_deltas = [
        value
        for value, region in zip(prototype_deltas, row_regions)
        if region == "alert"
    ]
    prototype_pre_window_deltas = [
        value
        for value, region in zip(prototype_deltas, row_regions)
        if region == "pre_window"
    ]

    rule_metrics = {}
    for short_name in PAIRED_RULE_SCORE_KEYS:
        key = f"paired_{short_name}_ego_delta"
        scored = [
            (float(row[key]), label, region)
            for row, label, region in zip(rows, labels, row_regions)
            if row.get(key) is not None
        ]
        rule_alert = [value for value, _label, region in scored if region == "alert"]
        rule_pre = [
            value for value, _label, region in scored if region == "pre_window"
        ]
        rule_labels = [label for _value, label, _region in scored]
        rule_values = [value for value, _label, _region in scored]
        prefix = f"paired_{short_name}_delta"
        rule_metrics.update(
            {
                f"{prefix}_auroc": _paired_binary_auc(rule_pre, rule_alert),
                f"{prefix}_auroc_all_background": _roc_auc(
                    rule_labels,
                    rule_values,
                ),
                f"mean_alert_{prefix}": _mean_or_none(rule_alert),
                f"mean_pre_window_{prefix}": _mean_or_none(rule_pre),
                f"mean_alert_minus_pre_window_{prefix}": (
                    None
                    if not rule_alert or not rule_pre
                    else float(np.mean(rule_alert) - np.mean(rule_pre))
                ),
                f"{prefix}_event_alert_gt_pre_rate": _event_signal_rate(
                    grouped,
                    label_key=label_key,
                    score_key=key,
                ),
            }
        )

    return {
        "aggregation": aggregation,
        "label_window": label_window,
        "event_count": len(grouped),
        "expected_positive_frame_count": expected_positive_frame_count,
        "matched_frame_count": len(rows),
        "missing_normal_frame_count": missing_normal_frame_count,
        "alert_frame_count": int(sum(labels)),
        "background_frame_count": int(len(labels) - sum(labels)),
        "pre_window_frame_count": len(pre_window_deltas),
        "post_window_frame_count": len(post_window_deltas),
        # Primary paired view: alert versus the same event before the window.
        # Post-window danger is not treated as a negative.
        "paired_delta_auroc": _paired_binary_auc(
            pre_window_deltas,
            alert_deltas,
        ),
        "paired_delta_auroc_pre_window": _paired_binary_auc(
            pre_window_deltas,
            alert_deltas,
        ),
        "paired_delta_auroc_all_background": _roc_auc(labels, paired_deltas),
        "paired_raw_delta_auroc": _paired_binary_auc(
            raw_pre_window_deltas,
            raw_alert_deltas,
        ),
        "paired_raw_delta_auroc_all_background": _roc_auc(labels, raw_deltas),
        "paired_target_delta_auroc": _paired_binary_auc(
            target_pre_window_deltas,
            target_alert_deltas,
        ),
        "paired_target_delta_auroc_all_background": _roc_auc(
            target_labels,
            target_deltas,
        ),
        "paired_prototype_delta_auroc": _paired_binary_auc(
            prototype_pre_window_deltas,
            prototype_alert_deltas,
        ),
        "paired_prototype_delta_auroc_all_background": _roc_auc(
            labels,
            prototype_deltas,
        ),
        "mean_alert_paired_prototype_delta": _mean_or_none(
            prototype_alert_deltas
        ),
        "mean_pre_window_paired_prototype_delta": _mean_or_none(
            prototype_pre_window_deltas
        ),
        "mean_alert_paired_delta": _mean_or_none(alert_deltas),
        "mean_pre_window_paired_delta": _mean_or_none(pre_window_deltas),
        "mean_post_window_paired_delta": _mean_or_none(post_window_deltas),
        "mean_background_paired_delta": _mean_or_none(background_deltas),
        "mean_alert_minus_pre_window_delta": (
            None
            if not alert_deltas or not pre_window_deltas
            else float(np.mean(alert_deltas) - np.mean(pre_window_deltas))
        ),
        "mean_alert_minus_background_delta": (
            None
            if not alert_deltas or not background_deltas
            else float(np.mean(alert_deltas) - np.mean(background_deltas))
        ),
        "alert_counterfactual_win_rate": _positive_rate(alert_deltas),
        "pre_window_counterfactual_win_rate": _positive_rate(
            pre_window_deltas
        ),
        "post_window_counterfactual_win_rate": _positive_rate(
            post_window_deltas
        ),
        "background_counterfactual_win_rate": _positive_rate(background_deltas),
        "target_alert_counterfactual_win_rate": _positive_rate(
            target_alert_deltas
        ),
        "target_scored_frame_count": len(target_deltas),
        "event_peak_in_alert_rate": _bool_rate(event_peak_in_alert),
        "event_alert_mean_gt_background_rate": _bool_rate(
            event_alert_mean_gt_background
        ),
        "event_alert_mean_gt_pre_window_rate": _bool_rate(
            event_alert_mean_gt_pre_window
        ),
        **rule_metrics,
    }


def _ego_rule_scores_for_graph(graph: Any) -> dict[str, float]:
    edge_attr = getattr(graph, "raw_edge_attr", graph.edge_attr).detach().cpu()
    if edge_attr.numel() == 0:
        return {
            "inverse_ttc_ego_max": 0.0,
            "inverse_distance_ego_max": 0.0,
            "closing_speed_ego_max": 0.0,
        }
    edge_index = graph.edge_index.detach().cpu()
    node_ids = [str(node_id) for node_id in graph.node_ids]
    ego_id = str(graph.ego_agent_id)
    ego_mask = torch.tensor(
        [
            node_ids[int(source)] == ego_id or node_ids[int(target)] == ego_id
            for source, target in edge_index.t().tolist()
        ],
        dtype=torch.bool,
    )
    distance = edge_attr[:, EDGE_FEATURE_NAMES.index("distance_m")]
    closing_speed = edge_attr[:, EDGE_FEATURE_NAMES.index("closing_speed_mps")]
    inverse_ttc = edge_attr[:, EDGE_FEATURE_NAMES.index("inverse_ttc_per_s")]
    inverse_distance = 1.0 / torch.clamp(distance, min=1e-3)

    def ego_max(values: torch.Tensor) -> float:
        selected = values[ego_mask]
        return 0.0 if selected.numel() == 0 else float(selected.max().item())

    return {
        "inverse_ttc_ego_max": ego_max(inverse_ttc),
        "inverse_distance_ego_max": ego_max(inverse_distance),
        "closing_speed_ego_max": ego_max(closing_speed),
    }


def _observable_state_for_graph(graph: Any) -> dict[str, Any]:
    raw_x = getattr(graph, "raw_x", graph.x).detach().cpu()
    raw_edge_attr = getattr(
        graph,
        "raw_edge_attr",
        graph.edge_attr,
    ).detach().cpu()
    edge_index = graph.edge_index.detach().cpu()
    node_ids = [str(node_id) for node_id in graph.node_ids]
    position_slice = [
        NODE_FEATURE_NAMES.index("ego_relative_x_m"),
        NODE_FEATURE_NAMES.index("ego_relative_y_m"),
    ]
    velocity_slice = [
        NODE_FEATURE_NAMES.index("velocity_x_mps"),
        NODE_FEATURE_NAMES.index("velocity_y_mps"),
    ]
    confidence_index = NODE_FEATURE_NAMES.index("observation_confidence")
    nodes = {
        node_id: {
            "position": raw_x[index, position_slice].numpy(),
            "velocity": raw_x[index, velocity_slice].numpy(),
            "confidence": float(raw_x[index, confidence_index].item()),
        }
        for index, node_id in enumerate(node_ids)
    }
    edge_confidence = getattr(graph, "edge_confidence", None)
    edge_confidence = (
        None if edge_confidence is None else edge_confidence.detach().cpu()
    )
    edges = {}
    for edge_position, (source, target) in enumerate(edge_index.t().tolist()):
        key = (node_ids[int(source)], node_ids[int(target)])
        edges[key] = {
            "distance": float(
                raw_edge_attr[
                    edge_position,
                    EDGE_FEATURE_NAMES.index("distance_m"),
                ].item()
            ),
            "closing_speed": float(
                raw_edge_attr[
                    edge_position,
                    EDGE_FEATURE_NAMES.index("closing_speed_mps"),
                ].item()
            ),
            "inverse_ttc": float(
                raw_edge_attr[
                    edge_position,
                    EDGE_FEATURE_NAMES.index("inverse_ttc_per_s"),
                ].item()
            ),
            "confidence": (
                1.0
                if edge_confidence is None
                else float(edge_confidence[edge_position].item())
            ),
        }
    return {
        "ego_id": str(graph.ego_agent_id),
        "nodes": nodes,
        "edges": edges,
    }


def _observable_divergence(
    normal_state: dict[str, Any] | None,
    counterfactual_state: dict[str, Any] | None,
) -> dict[str, Any]:
    empty = {
        "max_position_divergence_m": 0.0,
        "position_uncertainty_at_max_m": None,
        "max_velocity_divergence_mps": 0.0,
        "velocity_uncertainty_at_max_mps": None,
        "max_relation_distance_divergence_m": 0.0,
        "max_relation_closing_speed_divergence_mps": 0.0,
        "max_relation_inverse_ttc_divergence_per_s": 0.0,
        "position_observable": 0,
        "velocity_observable": 0,
        "relation_observable": 0,
        "relation_topology_changed": 0,
        "observable_divergence": 0,
        "observable_divergence_types": "",
    }
    if not normal_state or not counterfactual_state:
        return empty

    position_max = 0.0
    position_threshold_at_max = None
    velocity_max = 0.0
    velocity_threshold_at_max = None
    position_observable = False
    velocity_observable = False
    common_nodes = set(normal_state["nodes"]) & set(counterfactual_state["nodes"])
    for node_id in common_nodes:
        normal_node = normal_state["nodes"][node_id]
        positive_node = counterfactual_state["nodes"][node_id]
        confidence = min(
            float(normal_node["confidence"]),
            float(positive_node["confidence"]),
        )
        position_threshold = (
            OBSERVABLE_DIVERGENCE_CONFIG["position_floor_m"]
            + (1.0 - confidence)
            * OBSERVABLE_DIVERGENCE_CONFIG["position_low_confidence_extra_m"]
        )
        velocity_threshold = (
            OBSERVABLE_DIVERGENCE_CONFIG["velocity_floor_mps"]
            + (1.0 - confidence)
            * OBSERVABLE_DIVERGENCE_CONFIG[
                "velocity_low_confidence_extra_mps"
            ]
        )
        position_delta = float(
            np.linalg.norm(
                positive_node["position"] - normal_node["position"]
            )
        )
        velocity_delta = float(
            np.linalg.norm(
                positive_node["velocity"] - normal_node["velocity"]
            )
        )
        if position_delta >= position_max:
            position_max = position_delta
            position_threshold_at_max = position_threshold
        if velocity_delta >= velocity_max:
            velocity_max = velocity_delta
            velocity_threshold_at_max = velocity_threshold
        position_observable = position_observable or position_delta > position_threshold
        velocity_observable = velocity_observable or velocity_delta > velocity_threshold

    normal_edges = normal_state["edges"]
    positive_edges = counterfactual_state["edges"]
    ego_id = str(counterfactual_state["ego_id"])
    normal_ego_edges = {
        key for key in normal_edges if ego_id in key
    }
    positive_ego_edges = {
        key for key in positive_edges if ego_id in key
    }
    topology_changed = normal_ego_edges != positive_ego_edges
    relation_distance_max = 0.0
    relation_closing_max = 0.0
    relation_ttc_max = 0.0
    relation_observable = topology_changed
    for key in normal_ego_edges & positive_ego_edges:
        normal_edge = normal_edges[key]
        positive_edge = positive_edges[key]
        confidence = min(
            float(normal_edge["confidence"]),
            float(positive_edge["confidence"]),
        )
        distance_delta = abs(
            float(positive_edge["distance"]) - float(normal_edge["distance"])
        )
        closing_delta = abs(
            float(positive_edge["closing_speed"])
            - float(normal_edge["closing_speed"])
        )
        ttc_delta = abs(
            float(positive_edge["inverse_ttc"])
            - float(normal_edge["inverse_ttc"])
        )
        relation_distance_max = max(relation_distance_max, distance_delta)
        relation_closing_max = max(relation_closing_max, closing_delta)
        relation_ttc_max = max(relation_ttc_max, ttc_delta)
        relation_observable = relation_observable or (
            distance_delta
            > OBSERVABLE_DIVERGENCE_CONFIG["relation_distance_floor_m"]
            + (1.0 - confidence)
            * OBSERVABLE_DIVERGENCE_CONFIG["position_low_confidence_extra_m"]
            or closing_delta
            > OBSERVABLE_DIVERGENCE_CONFIG[
                "relation_closing_speed_floor_mps"
            ]
            + (1.0 - confidence)
            * OBSERVABLE_DIVERGENCE_CONFIG[
                "velocity_low_confidence_extra_mps"
            ]
            or ttc_delta
            > OBSERVABLE_DIVERGENCE_CONFIG[
                "relation_inverse_ttc_floor_per_s"
            ]
        )

    types = []
    if position_observable:
        types.append("position")
    if velocity_observable:
        types.append("velocity")
    if relation_observable:
        types.append("relation")
    return {
        "max_position_divergence_m": position_max,
        "position_uncertainty_at_max_m": position_threshold_at_max,
        "max_velocity_divergence_mps": velocity_max,
        "velocity_uncertainty_at_max_mps": velocity_threshold_at_max,
        "max_relation_distance_divergence_m": relation_distance_max,
        "max_relation_closing_speed_divergence_mps": relation_closing_max,
        "max_relation_inverse_ttc_divergence_per_s": relation_ttc_max,
        "position_observable": int(position_observable),
        "velocity_observable": int(velocity_observable),
        "relation_observable": int(relation_observable),
        "relation_topology_changed": int(topology_changed),
        "observable_divergence": int(bool(types)),
        "observable_divergence_types": "|".join(types),
    }


def _counterfactual_severity(extra_delay_frames: int | None) -> str:
    if extra_delay_frames is None:
        return "unknown"
    if extra_delay_frames <= 7:
        return "mild"
    if extra_delay_frames <= 13:
        return "medium"
    return "strong"


def _paired_observable_event_summaries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["event_index"]), []).append(row)
    summaries = []
    for event_index, event_rows in grouped.items():
        event_rows.sort(key=lambda row: int(row["frame_position"]))
        first = event_rows[0]
        observable_rows = [
            row for row in event_rows if int(row["observable_divergence"])
        ]
        summaries.append(
            {
                "split": first["split"],
                "event_index": event_index,
                "event_uid": first["event_uid"],
                "severity": first["severity"],
                "extra_delay_frames": first["extra_delay_frames"],
                "extra_delay_seconds": first["extra_delay_seconds"],
                "severity_protocol": first.get("severity_protocol"),
                "severity_score": first.get("severity_score"),
                "severity_risk_percentile_train": first.get(
                    "severity_risk_percentile_train"
                ),
                "severity_lead_percentile_train": first.get(
                    "severity_lead_percentile_train"
                ),
                "peak_directional_risk_increment": first.get(
                    "peak_directional_risk_increment"
                ),
                "effective_lead_window_frames": first.get(
                    "effective_lead_window_frames"
                ),
                "early_window_start_index": first["early_window_start_index"],
                "early_window_end_index": first["early_window_end_index"],
                "source_physical_danger_index": first[
                    "source_physical_danger_index"
                ],
                "anomaly_peak_index": first["anomaly_peak_index"],
                "first_observable_divergence_position": first[
                    "first_observable_divergence_position"
                ],
                "first_observable_divergence_types": first[
                    "first_observable_divergence_types"
                ],
                "observable_lead_to_anchor_frames": first[
                    "observable_lead_to_anchor_frames"
                ],
                "observable_frame_count": len(observable_rows),
                "max_position_divergence_m": max(
                    float(row["max_position_divergence_m"])
                    for row in event_rows
                ),
                "max_velocity_divergence_mps": max(
                    float(row["max_velocity_divergence_mps"])
                    for row in event_rows
                ),
                "max_relation_inverse_ttc_divergence_per_s": max(
                    float(row["max_relation_inverse_ttc_divergence_per_s"])
                    for row in event_rows
                ),
                "max_speed_difference_fps": first[
                    "max_speed_difference_fps"
                ],
                "max_displacement_difference_ft": first[
                    "max_displacement_difference_ft"
                ],
            }
        )
    return summaries


def _event_signal_rate(
    grouped: dict[int, list[dict[str, Any]]],
    *,
    label_key: str,
    score_key: str,
) -> float | None:
    comparisons = []
    for event_rows in grouped.values():
        event_rows.sort(key=lambda row: int(row["frame_position"]))
        labels = [int(row[label_key]) for row in event_rows]
        alert_indices = [index for index, label in enumerate(labels) if label]
        if not alert_indices:
            continue
        first_alert = min(alert_indices)
        alert = [
            float(row[score_key])
            for row, label in zip(event_rows, labels)
            if label and row.get(score_key) is not None
        ]
        pre = [
            float(row[score_key])
            for row in event_rows[:first_alert]
            if row.get(score_key) is not None
        ]
        if alert and pre:
            comparisons.append(float(np.mean(alert)) > float(np.mean(pre)))
    return _bool_rate(comparisons)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _paired_frame_key(frame: dict[str, Any]) -> tuple[str, int]:
    event_uid = str(frame.get("event_uid", ""))
    if not event_uid:
        event_uid = f"event-index-{int(frame['event_index'])}"
    return event_uid, int(frame["frame_position"])


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _paired_binary_auc(
    negative_scores: list[float],
    positive_scores: list[float],
) -> float | None:
    return _roc_auc(
        [0] * len(negative_scores) + [1] * len(positive_scores),
        negative_scores + positive_scores,
    )


def _positive_rate(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values) > 0.0))


def _bool_rate(values: list[bool]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _frame_auc_views(
    *,
    normal_scores: list[float],
    positive_scores: list[float],
    positive_labels: list[int],
) -> dict[str, float | None]:
    alert_scores = [
        score
        for score, label in zip(positive_scores, positive_labels)
        if label == 1
    ]
    return {
        "independent_normal": _roc_auc(
            [0] * len(normal_scores) + [1] * len(alert_scores),
            normal_scores + alert_scores,
        ),
        "within_event": _roc_auc(
            positive_labels,
            positive_scores,
        ),
        "combined_legacy": _roc_auc(
            [0] * len(normal_scores) + positive_labels,
            normal_scores + positive_scores,
        ),
    }


def _ewma_update(previous: float, value: float, *, alpha: float) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    return float(alpha * value + (1.0 - alpha) * previous)


def _dual_track_relation_summary(
    frames: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    grouped = _group_dual_frames_by_event(frames)
    best_ranks = []
    top1_events = 0
    top3_events = 0
    top5_events = 0
    visible_rates = []
    edge_rates = []
    peak_hit_counts = {
        "exact_directed": 0,
        "exact_reverse": 0,
        "one_target_node": 0,
        "off_target": 0,
        "no_edge": 0,
    }
    for event_frames in grouped.values():
        alert_frames = [
            frame
            for frame in event_frames
            if int(frame["original_alert"]) == 1
        ]
        if alert_frames:
            visible_rates.append(
                np.mean([bool(frame["target_visible"]) for frame in alert_frames])
            )
            edge_rates.append(
                np.mean([bool(frame["target_edge_exists"]) for frame in alert_frames])
            )
        ranks = [
            int(frame["target_pair_best_rank"])
            for frame in alert_frames
            if frame["target_pair_best_rank"] is not None
        ]
        if ranks:
            best = min(ranks)
            best_ranks.append(best)
            top1_events += int(best <= 1)
            top3_events += int(best <= 3)
            top5_events += int(best <= 5)
        peak_frame = max(
            event_frames,
            key=lambda frame: float(frame["aggregations"]["top3_mean"]),
        )
        hit_type = _top_edge_hit_type(
            top_edge=str(peak_frame["top_edge"]),
            target_source=str(peak_frame["target_source"]),
            target_target=str(peak_frame["target_target"]),
        )
        peak_hit_counts[hit_type] = peak_hit_counts.get(hit_type, 0) + 1

    event_count = len(grouped)
    return {
        "event_count": event_count,
        "mean_target_visible_alert_rate": (
            None if not visible_rates else float(np.mean(visible_rates))
        ),
        "mean_target_edge_alert_rate": (
            None if not edge_rates else float(np.mean(edge_rates))
        ),
        "mean_best_target_rank_in_alert": (
            None if not best_ranks else float(np.mean(best_ranks))
        ),
        "median_best_target_rank_in_alert": (
            None if not best_ranks else float(np.median(best_ranks))
        ),
        "target_top1_event_rate": (
            None if not event_count else float(top1_events / event_count)
        ),
        "target_top3_event_rate": (
            None if not event_count else float(top3_events / event_count)
        ),
        "target_top5_event_rate": (
            None if not event_count else float(top5_events / event_count)
        ),
        "peak_top_edge_hit_counts": peak_hit_counts,
    }


def _aggregation_threshold(
    frames: list[dict[str, Any]],
    *,
    aggregation: str,
    quantile: float,
) -> float:
    values = [
        float(frame["aggregations"][aggregation])
        for frame in frames
    ]
    if not values:
        return float("inf")
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def _label_window_mask_from_frames(
    frames: list[dict[str, Any]],
    label_window: str,
) -> list[bool]:
    original = [int(frame["original_alert"]) == 1 for frame in frames]
    if label_window == "original":
        return original
    if label_window.startswith("original_dilate_"):
        radius = int(label_window.rsplit("_", 1)[-1])
        return _dilate_mask(original, radius)
    if label_window.startswith("pre_anchor_"):
        lead_frames = int(label_window.rsplit("_", 1)[-1])
        return [
            0 < int(frame["frames_to_anchor"]) <= lead_frames
            for frame in frames
        ]
    raise ValueError(f"Unknown label window: {label_window}")


def _dilate_mask(mask: list[bool], radius: int) -> list[bool]:
    result = []
    for index in range(len(mask)):
        start = max(index - radius, 0)
        end = min(index + radius + 1, len(mask))
        result.append(any(mask[start:end]))
    return result


def _group_dual_frames_by_event(
    frames: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        grouped.setdefault(int(frame["event_index"]), []).append(frame)
    for event_frames in grouped.values():
        event_frames.sort(key=lambda frame: int(frame["frame_position"]))
    return grouped


def _target_pair_from_graph(graph: Any) -> tuple[str, str]:
    pair = getattr(graph, "correct_relation_pair", None)
    if pair is not None and len(pair) >= 2:
        return str(pair[0]), str(pair[1])
    stimulus = getattr(graph, "stimulus_vehicle_id", None)
    response = getattr(graph, "response_vehicle_id", None)
    return _scalar_id_string(stimulus), _scalar_id_string(response)


def _target_pair_rank(
    *,
    graph: Any,
    edge_scores: torch.Tensor,
    target_source: str,
    target_target: str,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    node_ids = [str(node_id) for node_id in graph.node_ids]
    visible = bool(target_source and target_target) and (
        target_source in node_ids and target_target in node_ids
    )
    if not visible or edge_scores.numel() == 0:
        return {
            "target_visible": visible,
            "target_edge_exists": False,
            "target_pair_best_score": None,
            "target_pair_best_rank": None,
        }
    edge_index = graph.edge_index.detach().cpu()
    if valid_mask is None:
        valid_mask = torch.ones_like(edge_scores, dtype=torch.bool)
    else:
        valid_mask = valid_mask.detach().cpu().bool()
        if valid_mask.shape != edge_scores.shape:
            raise ValueError("valid_mask must match the edge score shape.")
    target_positions = []
    for edge_position in range(int(edge_index.shape[1])):
        source = node_ids[int(edge_index[0, edge_position])]
        target = node_ids[int(edge_index[1, edge_position])]
        if {source, target} == {target_source, target_target}:
            target_positions.append(edge_position)
    if not target_positions:
        return {
            "target_visible": visible,
            "target_edge_exists": False,
            "target_pair_best_score": None,
            "target_pair_best_rank": None,
        }
    scored_target_positions = [
        position
        for position in target_positions
        if bool(valid_mask[position])
    ]
    if not scored_target_positions:
        return {
            "target_visible": visible,
            "target_edge_exists": True,
            "target_pair_best_score": None,
            "target_pair_best_rank": None,
        }
    target_scores = edge_scores[scored_target_positions]
    best_score = float(torch.max(target_scores).item())
    rank = int(torch.sum(edge_scores[valid_mask] > best_score).item()) + 1
    return {
        "target_visible": visible,
        "target_edge_exists": True,
        "target_pair_best_score": best_score,
        "target_pair_best_rank": rank,
    }


def _top_edge_from_scores(
    graph: Any,
    edge_scores: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[str, float]:
    if edge_scores.numel() == 0:
        return "", 0.0
    if valid_mask is None:
        valid_mask = torch.ones_like(edge_scores, dtype=torch.bool)
    else:
        valid_mask = valid_mask.detach().cpu().bool()
        if valid_mask.shape != edge_scores.shape:
            raise ValueError("valid_mask must match the edge score shape.")
    valid_positions = torch.where(valid_mask)[0]
    if valid_positions.numel() == 0:
        return "", 0.0
    top_local_index = int(torch.argmax(edge_scores[valid_positions]).item())
    top_index = int(valid_positions[top_local_index].item())
    source_node = int(graph.edge_index[0, top_index])
    target_node = int(graph.edge_index[1, top_index])
    return (
        f"{graph.node_ids[source_node]}->{graph.node_ids[target_node]}",
        float(edge_scores[top_index].item()),
    )


def _top_edge_hit_type(
    *,
    top_edge: str,
    target_source: str,
    target_target: str,
) -> str:
    if "->" not in top_edge:
        return "no_edge"
    source, target = top_edge.split("->", 1)
    if source == target_source and target == target_target:
        return "exact_directed"
    if source == target_target and target == target_source:
        return "exact_reverse"
    if len({source, target} & {target_source, target_target}) == 1:
        return "one_target_node"
    return "off_target"


def _ego_incident_score(
    graph: Any,
    edge_scores: torch.Tensor,
    *,
    mode: str,
    valid_mask: torch.Tensor | None = None,
) -> float:
    if edge_scores.numel() == 0:
        return 0.0
    ego_id = str(graph.ego_agent_id)
    edge_index = graph.edge_index.detach().cpu()
    keep = []
    for edge_position in range(int(edge_index.shape[1])):
        source = str(graph.node_ids[int(edge_index[0, edge_position])])
        target = str(graph.node_ids[int(edge_index[1, edge_position])])
        keep.append(source == ego_id or target == ego_id)
    mask = torch.tensor(keep, dtype=torch.bool)
    if valid_mask is not None:
        valid_mask = valid_mask.detach().cpu().bool()
        if valid_mask.shape != mask.shape:
            raise ValueError("valid_mask must match the edge score shape.")
        mask = mask & valid_mask
    if not bool(mask.any()):
        return 0.0
    values = edge_scores[mask]
    if mode == "top3_mean":
        return _topk_mean(values, 3)
    if mode == "max":
        return float(torch.max(values).item())
    raise ValueError(mode)


def _topk_mean(values: torch.Tensor, k: int) -> float:
    if values.numel() == 0:
        return 0.0
    return float(
        torch.topk(values, k=min(k, int(values.numel()))).values.mean().item()
    )


def _scalar_id_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return ""
        return str(int(value.detach().cpu().flatten()[0].item()))
    return str(value)


def _rate_above(scores: Iterable[float], threshold: float) -> float | None:
    scores = list(scores)
    if not scores:
        return None
    return float(np.mean(np.asarray(scores) >= threshold))


def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    positives = scores_array[labels_array == 1]
    negatives = scores_array[labels_array == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return float(wins / (len(positives) * len(negatives)))


if __name__ == "__main__":
    main()
