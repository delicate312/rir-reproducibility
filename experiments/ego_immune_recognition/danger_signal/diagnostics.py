"""Evaluate explicit danger signals without changing the frozen baseline model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .. import temporal_relation_train_eval as baseline_eval
from ..schema import EDGE_FEATURE_NAMES
from .baseline import DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_ROOT, load_frozen_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = DEFAULT_OUTPUT_ROOT / "baseline_checkpoint.pt"
DEFAULT_DIAGNOSTIC_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_danger_signal_v1_1"
)
PERSISTENCE_DECAY = 0.80
RECOVERY_BETA = 0.10
SELF_CONTEXT_WARMUP_FRAMES = 10
ONSET_SHOCK_PREDICTIONS = 3
ACTIVATION_WEIGHTS = {
    "danger_stress": 0.25,
    "danger_damage": 0.35,
    "danger_persistence": 0.40,
}
RAW_COMPONENTS = [
    "danger_stress_raw",
    "danger_damage_raw",
    "danger_persistence_raw",
    "danger_recovery_raw",
]
SCORE_KEYS = [
    "baseline_top3_mean",
    "baseline_causal_ewma",
    "danger_stress",
    "danger_damage",
    "danger_persistence",
    "danger_activation",
    "danger_activation_recovery_only",
    "danger_activation_inhibited",
    "inverse_ttc_ego_max",
    "closing_speed_ego_max",
]
TRACKS = [
    ("strict_original_window", "original"),
    ("early_immune_pre_anchor_10", "pre_anchor_10"),
    ("early_immune_original_dilate_5", "original_dilate_5"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DIAGNOSTIC_ROOT)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def directed_edge_keys(graph: Any) -> list[str]:
    edge_index = graph.edge_index.detach().cpu()
    return [
        f"{graph.node_ids[int(edge_index[0, index])]}->{graph.node_ids[int(edge_index[1, index])]}"
        for index in range(int(edge_index.shape[1]))
    ]


def directional_damage(
    graph: Any,
    expected_edge_attr: torch.Tensor,
    coverage: torch.Tensor,
) -> torch.Tensor:
    """Unexpected movement in the dangerous direction, in normalized units."""
    actual = graph.edge_attr.detach().cpu().float()
    expected = expected_edge_attr.detach().cpu().float()
    coverage = coverage.detach().cpu().bool()
    if actual.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    residual = actual - expected
    distance_index = EDGE_FEATURE_NAMES.index("distance_m")
    closing_index = EDGE_FEATURE_NAMES.index("closing_speed_mps")
    inverse_ttc_index = EDGE_FEATURE_NAMES.index("inverse_ttc_per_s")
    damage = (
        torch.relu(-residual[:, distance_index])
        + torch.relu(residual[:, closing_index])
        + torch.relu(residual[:, inverse_ttc_index])
    ) / 3.0
    return torch.where(coverage, damage, torch.zeros_like(damage))


def edge_danger_components(
    *,
    graph: Any,
    causal_scores: torch.Tensor,
    damage_scores: torch.Tensor,
    coverage: torch.Tensor,
    previous: dict[str, tuple[float, float, float]],
    persistence_decay: float = PERSISTENCE_DECAY,
    covered_observation_counts: dict[str, int] | None = None,
    recovery_initialization_predictions: int = 0,
    onset_shock_predictions: int = ONSET_SHOCK_PREDICTIONS,
) -> dict[str, torch.Tensor]:
    """Compute non-learned DAMP-like components for every directed relation."""
    if recovery_initialization_predictions < 0:
        raise ValueError(
            "recovery_initialization_predictions must be non-negative"
        )
    if onset_shock_predictions < 0:
        raise ValueError("onset_shock_predictions must be non-negative")
    causal_scores = causal_scores.detach().cpu().float()
    damage_scores = damage_scores.detach().cpu().float()
    coverage = coverage.detach().cpu().bool()
    stress = torch.zeros_like(causal_scores)
    persistence = torch.zeros_like(causal_scores)
    recovery = torch.zeros_like(causal_scores)
    onset_shock = torch.zeros_like(causal_scores)
    for index, edge_key in enumerate(directed_edge_keys(graph)):
        if not bool(coverage[index]):
            continue
        prior_covered_observations = (
            0
            if covered_observation_counts is None
            else int(covered_observation_counts.get(edge_key, 0))
        )
        current_residual = float(causal_scores[index])
        current_damage = float(damage_scores[index])
        prior = previous.get(edge_key)
        if prior is None:
            current_stress = 0.0
            current_persistence = (1.0 - persistence_decay) * current_damage
            current_recovery = 0.0
        else:
            previous_residual, previous_damage, previous_persistence = prior
            current_stress = max(current_residual - previous_residual, 0.0)
            current_recovery = 0.5 * (
                max(previous_residual - current_residual, 0.0)
                + max(previous_damage - current_damage, 0.0)
            )
            if prior_covered_observations < onset_shock_predictions:
                onset_shock[index] = current_recovery
            if (
                prior_covered_observations
                < recovery_initialization_predictions
            ):
                current_recovery = 0.0
            current_persistence = (
                persistence_decay * previous_persistence
                + (1.0 - persistence_decay) * (current_stress + current_damage)
            )
        stress[index] = current_stress
        persistence[index] = current_persistence
        recovery[index] = current_recovery
        previous[edge_key] = (
            current_residual,
            current_damage,
            current_persistence,
        )
        if covered_observation_counts is not None:
            covered_observation_counts[edge_key] = (
                prior_covered_observations + 1
            )
    return {
        "stress": stress,
        "damage": damage_scores,
        "persistence": persistence,
        "recovery": recovery,
        "onset_shock": onset_shock,
    }


@torch.no_grad()
def score_sequences(
    model: torch.nn.Module,
    sequences: list[list[Any]],
    device: torch.device,
    *,
    split: str,
    group_count: int,
    recovery_initialization_predictions: int = 0,
    onset_shock_predictions: int = ONSET_SHOCK_PREDICTIONS,
    include_relation_components: bool = False,
) -> list[dict[str, Any]]:
    model.eval()
    frames: list[dict[str, Any]] = []
    for event_index, sequence in enumerate(sequences):
        outputs = model.score_sequence_masked_groups(
            sequence,
            device,
            group_count=group_count,
            surprise_mode="absolute",
        )
        previous_edges: dict[str, tuple[float, float, float]] = {}
        covered_observation_counts: dict[str, int] = {}
        causal_ewma = 0.0
        for frame_position, (graph, output) in enumerate(zip(sequence, outputs)):
            causal_scores = output["causal_next_edge_scores"].detach().cpu()
            coverage = output["causal_next_coverage"].detach().cpu().bool()
            damage_scores = directional_damage(
                graph,
                output["expected_edge_attr"],
                coverage,
            )
            components = edge_danger_components(
                graph=graph,
                causal_scores=causal_scores,
                damage_scores=damage_scores,
                coverage=coverage,
                previous=previous_edges,
                covered_observation_counts=covered_observation_counts,
                recovery_initialization_predictions=(
                    recovery_initialization_predictions
                ),
                onset_shock_predictions=onset_shock_predictions,
            )
            causal_raw = baseline_eval._ego_incident_score(
                graph,
                causal_scores,
                mode="top3_mean",
                valid_mask=coverage,
            )
            causal_ewma = baseline_eval._ewma_update(
                causal_ewma,
                causal_raw,
                alpha=baseline_eval.EGO_CAUSAL_EWMA_ALPHA,
            )
            edge_scores = output["edge_scores"].detach().cpu()
            rules = baseline_eval._ego_rule_scores_for_graph(graph)
            aggregations = {
                "baseline_top3_mean": baseline_eval._topk_mean(edge_scores, 3),
                "baseline_causal_raw": causal_raw,
                "baseline_causal_ewma": causal_ewma,
                "danger_stress_raw": baseline_eval._ego_incident_score(
                    graph, components["stress"], mode="top3_mean", valid_mask=coverage
                ),
                "danger_damage_raw": baseline_eval._ego_incident_score(
                    graph, components["damage"], mode="top3_mean", valid_mask=coverage
                ),
                "danger_persistence_raw": baseline_eval._ego_incident_score(
                    graph,
                    components["persistence"],
                    mode="top3_mean",
                    valid_mask=coverage,
                ),
                "danger_recovery_raw": baseline_eval._ego_incident_score(
                    graph,
                    components["recovery"],
                    mode="top3_mean",
                    valid_mask=coverage,
                ),
                "danger_onset_shock_raw": baseline_eval._ego_incident_score(
                    graph,
                    components["onset_shock"],
                    mode="top3_mean",
                    valid_mask=coverage,
                ),
                "inverse_ttc_ego_max": float(rules["inverse_ttc_ego_max"]),
                "closing_speed_ego_max": float(rules["closing_speed_ego_max"]),
            }
            relation_components = None
            if include_relation_components:
                relation_components = {}
                edge_index = graph.edge_index.detach().cpu()
                node_ids = [str(value) for value in graph.node_ids]
                ego_id = str(graph.ego_agent_id)
                keys = directed_edge_keys(graph)
                for edge_position, edge_key in enumerate(keys):
                    if not bool(coverage[edge_position]):
                        continue
                    source_index = int(edge_index[0, edge_position])
                    target_index = int(edge_index[1, edge_position])
                    source_id = node_ids[source_index]
                    target_id = node_ids[target_index]
                    relation_components[edge_key] = {
                        "source_id": source_id,
                        "target_id": target_id,
                        "ego_incident": int(
                            source_id == ego_id or target_id == ego_id
                        ),
                        "ego_outgoing": int(source_id == ego_id),
                        "causal_raw": float(causal_scores[edge_position]),
                        "stress_raw": float(components["stress"][edge_position]),
                        "damage_raw": float(components["damage"][edge_position]),
                        "persistence_raw": float(
                            components["persistence"][edge_position]
                        ),
                        "recovery_raw": float(components["recovery"][edge_position]),
                        "onset_shock_raw": float(
                            components["onset_shock"][edge_position]
                        ),
                    }
            source_frame_index = int(
                torch.as_tensor(
                    getattr(graph, "frame_index", frame_position)
                )
                .view(-1)[0]
                .item()
            )
            critical_frame = int(
                torch.as_tensor(
                    getattr(graph, "critical_frame", source_frame_index)
                )
                .view(-1)[0]
                .item()
            )
            frame_row = {
                    "split": split,
                    "event_index": event_index,
                    "event_uid": str(getattr(graph, "event_uid", graph.scenario_id)),
                    "event_variant": str(getattr(graph, "event_variant", "")),
                    "frame_position": frame_position,
                    "source_frame_index": source_frame_index,
                    "critical_frame": critical_frame,
                    "eval_frames_to_anchor": (
                        critical_frame - source_frame_index
                    ),
                    "frames_to_anchor": int(graph.frames_to_physical_danger.item()),
                    "original_alert": int(graph.y_immune_alert.item()),
                    "physical": int(graph.y_physical_danger.item()),
                    "aggregations": aggregations,
                }
            if relation_components is not None:
                frame_row["relation_components"] = relation_components
            frames.append(frame_row)
    return frames


def robust_component_calibration(
    hard_normal_frames: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    calibration = {}
    for key in RAW_COMPONENTS:
        values = np.asarray(
            [float(frame["aggregations"][key]) for frame in hard_normal_frames],
            dtype=float,
        )
        center = float(np.median(values)) if values.size else 0.0
        mad = float(np.median(np.abs(values - center))) if values.size else 0.0
        upper_scale = (
            float(np.quantile(values, 0.90) - center) if values.size else 0.0
        )
        calibration[key] = {
            "center": center,
            "scale": max(1.4826 * mad, upper_scale, 1e-3),
        }
    return calibration


def apply_danger_calibration(
    frames: list[dict[str, Any]],
    calibration: dict[str, dict[str, float]],
    *,
    recovery_beta: float = RECOVERY_BETA,
) -> None:
    normalized_names = {
        "danger_stress_raw": "danger_stress",
        "danger_damage_raw": "danger_damage",
        "danger_persistence_raw": "danger_persistence",
        "danger_recovery_raw": "danger_recovery",
    }
    for frame in frames:
        values = frame["aggregations"]
        for raw_key, output_key in normalized_names.items():
            stats = calibration[raw_key]
            values[output_key] = max(
                (float(values[raw_key]) - stats["center"]) / stats["scale"],
                0.0,
            )
        activation = sum(
            ACTIVATION_WEIGHTS[key] * float(values[key])
            for key in ACTIVATION_WEIGHTS
        )
        values["danger_activation"] = activation
        values["danger_activation_recovery_only"] = max(
            activation - recovery_beta * float(values["danger_recovery"]),
            0.0,
        )
    apply_self_context_inhibition(frames)


def apply_self_context_inhibition(
    frames: list[dict[str, Any]],
    *,
    warmup_frames: int = SELF_CONTEXT_WARMUP_FRAMES,
) -> None:
    """Subtract only the online-safe sequence prefix, never future frames."""
    if warmup_frames <= 0:
        raise ValueError("warmup_frames must be positive")
    for event_frames in baseline_eval._group_dual_frames_by_event(frames).values():
        warmup = event_frames[:warmup_frames]
        if not warmup:
            continue
        context_center = float(
            np.median(
                [
                    float(
                        frame["aggregations"][
                            "danger_activation_recovery_only"
                        ]
                    )
                    for frame in warmup
                ]
            )
        )
        for frame in event_frames:
            values = frame["aggregations"]
            values["danger_self_context"] = context_center
            values["danger_activation_inhibited"] = max(
                float(values["danger_activation_recovery_only"])
                - context_center,
                0.0,
            )


def evaluate_score(
    *,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    score_key: str,
    label_window: str,
    threshold: float,
) -> dict[str, Any]:
    return baseline_eval._evaluate_dual_track(
        normal_frames=normal_frames,
        positive_frames=positive_frames,
        aggregation=score_key,
        label_window=label_window,
        threshold=threshold,
    )


def longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def false_alarm_morphology(
    frames: list[dict[str, Any]], *, score_key: str, threshold: float
) -> dict[str, Any]:
    grouped = baseline_eval._group_dual_frames_by_event(frames)
    longest_runs = []
    for event_frames in grouped.values():
        longest_runs.append(
            longest_true_run(
                float(frame["aggregations"][score_key]) >= threshold
                for frame in event_frames
            )
        )
    scores = [float(frame["aggregations"][score_key]) for frame in frames]
    return {
        "score": score_key,
        "frame_count": len(scores),
        "frame_false_alarm_rate": float(np.mean(np.asarray(scores) >= threshold)),
        "event_count": len(longest_runs),
        "event_false_alarm_rate": float(np.mean(np.asarray(longest_runs) >= 1)),
        "sustained_3_frame_event_false_alarm_rate": float(
            np.mean(np.asarray(longest_runs) >= 3)
        ),
        "mean_longest_false_alarm_run": float(np.mean(longest_runs)),
        "max_false_alarm_run": int(max(longest_runs, default=0)),
    }


def paired_delta_auc(
    *,
    normal_frames: list[dict[str, Any]],
    positive_frames: list[dict[str, Any]],
    score_key: str,
    label_window: str,
) -> float | None:
    normal_by_key = {
        (frame["event_uid"], int(frame["frame_position"])): frame
        for frame in normal_frames
    }
    negative_deltas: list[float] = []
    positive_deltas: list[float] = []
    for event_frames in baseline_eval._group_dual_frames_by_event(
        positive_frames
    ).values():
        mask = baseline_eval._label_window_mask_from_frames(event_frames, label_window)
        alert_positions = [index for index, value in enumerate(mask) if value]
        if not alert_positions:
            continue
        first_alert = min(alert_positions)
        for index, frame in enumerate(event_frames):
            normal = normal_by_key.get(
                (frame["event_uid"], int(frame["frame_position"]))
            )
            if normal is None:
                continue
            delta = float(frame["aggregations"][score_key]) - float(
                normal["aggregations"][score_key]
            )
            if mask[index]:
                positive_deltas.append(delta)
            elif index < first_alert:
                negative_deltas.append(delta)
    return baseline_eval._roc_auc(
        [0] * len(negative_deltas) + [1] * len(positive_deltas),
        negative_deltas + positive_deltas,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def frame_rows(frames_by_split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    curve_keys = [
        "baseline_causal_ewma",
        "danger_stress",
        "danger_damage",
        "danger_persistence",
        "danger_recovery",
        "danger_activation",
        "danger_activation_recovery_only",
        "danger_activation_inhibited",
        "inverse_ttc_ego_max",
        "closing_speed_ego_max",
    ]
    for split, frames in frames_by_split.items():
        for frame in frames:
            rows.append(
                {
                    "split": split,
                    "event_index": frame["event_index"],
                    "event_uid": frame["event_uid"],
                    "event_variant": frame["event_variant"],
                    "frame_position": frame["frame_position"],
                    "frames_to_anchor": frame["frames_to_anchor"],
                    "original_alert": frame["original_alert"],
                    "physical": frame["physical"],
                    **{
                        key: frame["aggregations"][key]
                        for key in curve_keys
                    },
                }
            )
    return rows


def render_event_curves(
    frames: list[dict[str, Any]], output_root: Path
) -> list[str]:
    import matplotlib.pyplot as plt

    curve_root = output_root / "event_curves"
    curve_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for event_frames in baseline_eval._group_dual_frames_by_event(frames).values():
        uid = str(event_frames[0]["event_uid"])
        x = [int(frame["frame_position"]) for frame in event_frames]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for key, label, color in [
            ("danger_stress", "Stress", "#d55e00"),
            ("danger_damage", "Damage", "#cc79a7"),
            ("danger_persistence", "Persistence", "#0072b2"),
        ]:
            axes[0].plot(
                x,
                [float(frame["aggregations"][key]) for frame in event_frames],
                label=label,
                color=color,
                linewidth=1.8,
            )
        axes[0].legend(loc="upper left", ncol=3, frameon=False)
        axes[0].set_ylabel("Calibrated signal")
        axes[0].grid(alpha=0.2)
        axes[1].plot(
            x,
            [float(frame["aggregations"]["danger_activation"]) for frame in event_frames],
            label="Activation",
            color="#e69f00",
            linewidth=1.5,
        )
        axes[1].plot(
            x,
            [
                float(frame["aggregations"]["danger_activation_inhibited"])
                for frame in event_frames
            ],
            label="Activation - recovery - self context",
            color="#009e73",
            linewidth=2.2,
        )
        recovery_axis = axes[1].twinx()
        recovery_axis.plot(
            x,
            [float(frame["aggregations"]["danger_recovery"]) for frame in event_frames],
            label="Recovery evidence",
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )
        recovery_axis.set_ylabel("Recovery evidence", color="#666666")
        recovery_axis.tick_params(axis="y", colors="#666666")
        alert = [bool(frame["original_alert"]) for frame in event_frames]
        alert_x = [value for value, keep in zip(x, alert) if keep]
        if alert_x:
            for axis in axes:
                axis.axvspan(min(alert_x), max(alert_x), color="#f0e442", alpha=0.16)
        axes[1].legend(loc="upper left", ncol=2, frameon=False)
        axes[1].set_xlabel("Frame position")
        axes[1].set_ylabel("Combined response")
        axes[1].grid(alpha=0.2)
        fig.suptitle(uid)
        fig.tight_layout()
        path = curve_root / f"{uid}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def find_metric(
    rows: list[dict[str, Any]], *, split: str, track: str, score: str
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["split"] == split and row["track"] == track and row["score"] == score
    )


def decision_gate(
    metrics: list[dict[str, Any]], false_alarm_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gains = []
    for split in ["validation_positive", "test_positive"]:
        for track in [
            "early_immune_pre_anchor_10",
            "early_immune_original_dilate_5",
        ]:
            baseline = find_metric(
                metrics, split=split, track=track, score="baseline_causal_ewma"
            )
            candidate = find_metric(
                metrics,
                split=split,
                track=track,
                score="danger_activation_inhibited",
            )
            gains.append(
                {
                    "split": split,
                    "track": track,
                    "baseline_auroc": baseline["frame_auroc"],
                    "candidate_auroc": candidate["frame_auroc"],
                    "gain": candidate["frame_auroc"] - baseline["frame_auroc"],
                }
            )
    hard_normal = {row["score"]: row for row in false_alarm_rows}
    false_alarm_delta = (
        hard_normal["danger_activation_inhibited"]["frame_false_alarm_rate"]
        - hard_normal["baseline_causal_ewma"]["frame_false_alarm_rate"]
    )
    stable = (
        all(row["gain"] >= 0.01 for row in gains)
        and false_alarm_delta <= 0.01
    )
    return {
        "criterion": (
            "candidate gains >= 0.01 on both early tracks in validation and test, "
            "with held-out hard-normal frame FPR increase <= 0.01"
        ),
        "gains": gains,
        "hard_normal_frame_fpr_delta": false_alarm_delta,
        "eligible_for_learnable_head": stable,
        "decision": (
            "promote_to_learnable_head"
            if stable
            else "retain_as_diagnostic_only"
        ),
    }


def run_diagnostics(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    output_root: Path,
    render_plots: bool = True,
) -> dict[str, Any]:
    model, device, splits, _, checkpoint = load_frozen_baseline(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
    )
    frames_by_split = {
        split: score_sequences(
            model,
            sequences,
            device,
            split=split,
            group_count=int(checkpoint["config"]["group_count"]),
        )
        for split, sequences in splits.items()
        if split != "train_normal"
    }
    calibration = robust_component_calibration(
        frames_by_split["validation_normal"]
    )
    for frames in frames_by_split.values():
        apply_danger_calibration(frames, calibration)

    warmup_positive_label_count = sum(
        int(frame["original_alert"]) + int(frame["physical"])
        for split in ["validation_positive", "test_positive"]
        for event_frames in baseline_eval._group_dual_frames_by_event(
            frames_by_split[split]
        ).values()
        for frame in event_frames[:SELF_CONTEXT_WARMUP_FRAMES]
    )
    if warmup_positive_label_count:
        raise ValueError("Self-context warmup overlaps a positive label.")

    thresholds = {
        score: baseline_eval._aggregation_threshold(
            frames_by_split["validation_normal"],
            aggregation=score,
            quantile=float(checkpoint["config"]["threshold_quantile"]),
        )
        for score in SCORE_KEYS
    }
    metrics = []
    paired_rows = []
    for split, normal_split in [
        ("validation_positive", "validation_normal"),
        ("test_positive", "test_normal"),
    ]:
        for track, label_window in TRACKS:
            for score in SCORE_KEYS:
                result = evaluate_score(
                    normal_frames=frames_by_split[normal_split],
                    positive_frames=frames_by_split[split],
                    score_key=score,
                    label_window=label_window,
                    threshold=thresholds[score],
                )
                metrics.append(
                    {"split": split, "track": track, "score": score, **result}
                )
                paired_rows.append(
                    {
                        "split": split,
                        "track": track,
                        "score": score,
                        "paired_delta_auroc": paired_delta_auc(
                            normal_frames=frames_by_split[normal_split],
                            positive_frames=frames_by_split[split],
                            score_key=score,
                            label_window=label_window,
                        ),
                    }
                )

    false_alarm_rows = [
        false_alarm_morphology(
            frames_by_split["test_normal"],
            score_key=score,
            threshold=thresholds[score],
        )
        for score in SCORE_KEYS
    ]
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "danger_dual_track_metrics.csv"
    paired_path = output_root / "danger_paired_metrics.csv"
    frames_path = output_root / "danger_signal_frames.csv"
    false_alarm_path = output_root / "hard_normal_false_alarm.csv"
    write_csv(metrics_path, metrics)
    write_csv(paired_path, paired_rows)
    write_csv(frames_path, frame_rows(frames_by_split))
    write_csv(false_alarm_path, false_alarm_rows)
    plot_paths = (
        render_event_curves(frames_by_split["test_positive"], output_root)
        if render_plots
        else []
    )
    gate = decision_gate(metrics, false_alarm_rows)
    summary = {
        "protocol": "danger-signal-diagnostics-v1.1",
        "baseline_id": checkpoint["baseline_id"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "danger_signal_definition": {
            "stress": "positive temporal derivative of per-edge causal residual",
            "damage": (
                "positive directional forecast error in distance, closing speed, "
                "and inverse TTC"
            ),
            "persistence": (
                f"edge-state EWMA of stress + damage with decay={PERSISTENCE_DECAY}"
            ),
            "recovery": (
                "joint decrease in causal residual and directional damage"
            ),
            "activation_weights": ACTIVATION_WEIGHTS,
            "recovery_beta": RECOVERY_BETA,
            "self_context": (
                f"median activation-minus-recovery in first "
                f"{SELF_CONTEXT_WARMUP_FRAMES} online frames"
            ),
            "self_context_warmup_positive_label_count": (
                warmup_positive_label_count
            ),
            "calibration": "robust validation hard-normal calibration only",
        },
        "component_calibration": calibration,
        "thresholds": thresholds,
        "decision_gate": gate,
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "paired_metrics_csv": str(paired_path),
            "frame_curves_csv": str(frames_path),
            "hard_normal_false_alarm_csv": str(false_alarm_path),
            "event_curve_pngs": plot_paths,
        },
    }
    summary_path = output_root / "danger_signal_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()
    result = run_diagnostics(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        render_plots=not args.no_plots,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
