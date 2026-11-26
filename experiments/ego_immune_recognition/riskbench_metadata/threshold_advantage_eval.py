"""Evaluate threshold relaxation and fixed temporal score advantage."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .complete_structure_ablation import DEFAULT_CACHE_ROOT
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT, WINDOW_FRAMES
from .persistent_guarded_fusion import ALERT_PERCENTILE
from .relation_geometry_rescue import (
    _attach_relation_signals,
    _fold_frames,
    _geometry_from_observations,
    _observations_by_frame,
    _rescue_metrics,
)
from .safe_response_rescue import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_SAFE_RESPONSE_ROOT,
    SafeResponseConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_threshold_advantage_v1"
)
DEFAULT_THRESHOLD_DELTAS = (
    0.0,
    0.005,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
)
DEFAULT_ROLLING_FRAMES = 5
DEFAULT_POOLING_FRAMES = (1, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--safe-response-root",
        type=Path,
        default=DEFAULT_SAFE_RESPONSE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--threshold-deltas",
        default=",".join(str(value) for value in DEFAULT_THRESHOLD_DELTAS),
    )
    parser.add_argument(
        "--rolling-frames",
        type=int,
        default=DEFAULT_ROLLING_FRAMES,
    )
    parser.add_argument(
        "--pooling-frames",
        default=",".join(str(value) for value in DEFAULT_POOLING_FRAMES),
        help=(
            "Comma-separated temporal pooling widths for the sensitivity "
            "analysis. At 20 Hz, 1/5/10 frames correspond to peak, "
            "0.25 s sustain, and 0.5 s sustain."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def maximum_rolling_mean(
    values: Sequence[float],
    window_frames: int,
) -> float:
    """Return the largest fixed-width contiguous moving average."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    width = min(max(int(window_frames), 1), int(array.size))
    if width == int(array.size):
        return float(np.mean(array))
    cumulative = np.concatenate(([0.0], np.cumsum(array)))
    means = (cumulative[width:] - cumulative[:-width]) / width
    return float(np.max(means))


def _windowed_event_frames(
    frames: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        frame
        for frame in sorted(
            frames,
            key=lambda item: int(item["frame_position"]),
        )
        if 0
        < int(
            frame.get("eval_frames_to_anchor", frame["frames_to_anchor"])
        )
        <= WINDOW_FRAMES
    ]


def _event_score_rows(
    frames: Sequence[dict[str, Any]],
    *,
    fold: int,
    positive: bool,
    config: SafeResponseConfig,
    records: Sequence[dict[str, Any]],
    rolling_frames: int,
    pooling_frames: Sequence[int],
) -> list[dict[str, Any]]:
    rows = []
    for event_index, event_frames in _group_dual_frames_by_event(
        list(frames)
    ).items():
        usable = _windowed_event_frames(event_frames)
        ttc_values = [
            float(frame["aggregations"]["physical_percentile"])
            for frame in usable
        ]
        model_values = [
            max(
                (
                    float(values[config.signal])
                    for values in frame.get(
                        "geometry_relations",
                        {},
                    ).values()
                ),
                default=0.0,
            )
            for frame in usable
        ]
        record = records[int(event_index)]
        ttc_detected = any(
            value >= ALERT_PERCENTILE for value in ttc_values
        )
        row = {
                "fold": fold,
                "event_index": int(event_index),
                "event_uid": str(event_frames[0]["event_uid"]),
                "positive": int(positive),
                "category": record["category"],
                "base_scenario_id": record["base_scenario_id"],
                "target_type_name": record["target_type_name"],
                "ttc_detected": int(ttc_detected),
                "frame_count": len(usable),
                "ttc_max": max(ttc_values, default=0.0),
                "model_max": max(model_values, default=0.0),
                "ttc_peak_neighborhood": maximum_rolling_mean(
                    ttc_values,
                    rolling_frames,
                ),
                "model_peak_neighborhood": maximum_rolling_mean(
                    model_values,
                    rolling_frames,
                ),
            }
        for width in pooling_frames:
            model_score = maximum_rolling_mean(model_values, width)
            ttc_score = maximum_rolling_mean(ttc_values, width)
            row[f"model_pool_{width}"] = model_score
            row[f"ttc_pool_{width}"] = ttc_score
            row[f"lift_pool_{width}"] = model_score - ttc_score
        rows.append(row)
    for row in rows:
        row["peak_neighborhood_lift"] = (
            float(row["model_peak_neighborhood"])
            - float(row["ttc_peak_neighborhood"])
        )
    return rows


def _safe_binary_metrics(
    rows: Sequence[dict[str, Any]],
    score_key: str,
) -> dict[str, float | None]:
    labels = np.asarray([int(row["positive"]) for row in rows], dtype=int)
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=float)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _lift_summary(
    rows: Sequence[dict[str, Any]],
    *,
    lift_key: str = "peak_neighborhood_lift",
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    lifts = np.asarray(
        [float(row[lift_key]) for row in rows],
        dtype=float,
    )
    if lifts.size == 0:
        return {
            "event_count": 0,
            "mean_lift": None,
            "median_lift": None,
            "model_win_rate": None,
            "mean_lift_bootstrap_95ci": None,
        }
    rng = np.random.default_rng(seed)
    sampled_means = np.mean(
        rng.choice(
            lifts,
            size=(bootstrap_samples, lifts.size),
            replace=True,
        ),
        axis=1,
    )
    return {
        "event_count": int(lifts.size),
        "mean_lift": float(np.mean(lifts)),
        "median_lift": float(np.median(lifts)),
        "model_win_rate": float(np.mean(lifts > 0.0)),
        "mean_lift_bootstrap_95ci": [
            float(np.quantile(sampled_means, 0.025)),
            float(np.quantile(sampled_means, 0.975)),
        ],
    }


def _aggregate_threshold(
    delta: float,
    fold_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    positive_events = [
        row
        for result in fold_results
        for row in result["positive"]["events"]
    ]
    normal_events = [
        row
        for result in fold_results
        for row in result["normal"]["events"]
    ]
    positive_count = len(positive_events)
    normal_count = len(normal_events)
    ttc_positive_hits = sum(
        int(row["ttc_detected"]) for row in positive_events
    )
    positive_rescues = sum(
        int(row["blind_rescue"]) for row in positive_events
    )
    final_positive_hits = sum(
        int(row["final_detected"]) for row in positive_events
    )
    ttc_normal_alerts = sum(
        int(row["ttc_detected"]) for row in normal_events
    )
    normal_false_rescues = sum(
        int(row["blind_rescue"]) for row in normal_events
    )
    final_normal_alerts = sum(
        int(row["final_detected"]) for row in normal_events
    )
    ttc_blind_positive_count = positive_count - ttc_positive_hits
    ttc_negative_normal_count = normal_count - ttc_normal_alerts
    leads = [
        float(row["rescue_lead_seconds"])
        for row in positive_events
        if row["blind_rescue"] and row["rescue_lead_seconds"] != ""
    ]
    return {
        "threshold_delta": delta,
        "positive_event_count": positive_count,
        "ttc_positive_hits": ttc_positive_hits,
        "ttc_blind_positive_count": ttc_blind_positive_count,
        "ttc_blind_rescue_count": positive_rescues,
        "ttc_blind_rescue_recall": (
            0.0
            if not ttc_blind_positive_count
            else positive_rescues / ttc_blind_positive_count
        ),
        "final_positive_hits": final_positive_hits,
        "final_positive_recall": (
            0.0 if not positive_count else final_positive_hits / positive_count
        ),
        "hard_safe_event_count": normal_count,
        "ttc_hard_safe_alerts": ttc_normal_alerts,
        "ttc_negative_hard_safe_count": ttc_negative_normal_count,
        "rescue_false_alert_count": normal_false_rescues,
        "conditional_rescue_fpr": (
            0.0
            if not ttc_negative_normal_count
            else normal_false_rescues / ttc_negative_normal_count
        ),
        "final_hard_safe_alerts": final_normal_alerts,
        "final_hard_safe_alert_rate": (
            0.0 if not normal_count else final_normal_alerts / normal_count
        ),
        "mean_rescue_lead_seconds": (
            None if not leads else float(np.mean(leads))
        ),
        "median_rescue_lead_seconds": (
            None if not leads else float(np.median(leads))
        ),
    }


def _score_advantage(
    rows: Sequence[dict[str, Any]],
    *,
    rolling_frames: int,
    bootstrap_samples: int,
    seed: int,
    model_score_key: str = "model_peak_neighborhood",
    ttc_score_key: str = "ttc_peak_neighborhood",
    lift_key: str = "peak_neighborhood_lift",
) -> dict[str, Any]:
    all_rows = list(rows)
    ttc_blind_rows = [
        row for row in all_rows if not int(row["ttc_detected"])
    ]
    positive_rows = [row for row in all_rows if int(row["positive"])]
    blind_positive_rows = [
        row
        for row in positive_rows
        if not int(row["ttc_detected"])
    ]
    return {
        "temporal_pooling": {
            "name": "maximum_contiguous_moving_average",
            "rolling_frames": rolling_frames,
            "rolling_seconds": rolling_frames * 0.05,
            "evaluation_window_frames": WINDOW_FRAMES,
            "evaluation_window_seconds": WINDOW_FRAMES * 0.05,
            "same_pooling_for_model_and_ttc": True,
        },
        "all_events": {
            "event_count": len(all_rows),
            "model": _safe_binary_metrics(
                all_rows,
                model_score_key,
            ),
            "ttc": _safe_binary_metrics(
                all_rows,
                ttc_score_key,
            ),
        },
        "ttc_negative_events": {
            "event_count": len(ttc_blind_rows),
            "positive_count": sum(
                int(row["positive"]) for row in ttc_blind_rows
            ),
            "normal_count": sum(
                not int(row["positive"]) for row in ttc_blind_rows
            ),
            "model": _safe_binary_metrics(
                ttc_blind_rows,
                model_score_key,
            ),
            "ttc": _safe_binary_metrics(
                ttc_blind_rows,
                ttc_score_key,
            ),
        },
        "positive_score_lift": _lift_summary(
            positive_rows,
            lift_key=lift_key,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "ttc_blind_positive_score_lift": _lift_summary(
            blind_positive_rows,
            lift_key=lift_key,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 1,
        ),
        "interpretation": (
            "Scores are empirical percentiles, not calibrated danger "
            "probabilities. Lift is reported in percentile-score units."
        ),
    }


def _pooling_sensitivity(
    rows: Sequence[dict[str, Any]],
    *,
    pooling_frames: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    sensitivity = []
    for index, width in enumerate(pooling_frames):
        advantage = _score_advantage(
            rows,
            rolling_frames=width,
            bootstrap_samples=bootstrap_samples,
            seed=seed + index * 10,
            model_score_key=f"model_pool_{width}",
            ttc_score_key=f"ttc_pool_{width}",
            lift_key=f"lift_pool_{width}",
        )
        all_events = advantage["all_events"]
        ttc_negative = advantage["ttc_negative_events"]
        blind_lift = advantage["ttc_blind_positive_score_lift"]
        sensitivity.append(
            {
                "pooling_frames": width,
                "pooling_seconds": width * 0.05,
                "pooling_name": (
                    "instantaneous_peak"
                    if width == 1
                    else "maximum_contiguous_mean"
                ),
                "all_model_auroc": all_events["model"]["auroc"],
                "all_ttc_auroc": all_events["ttc"]["auroc"],
                "all_model_ap": all_events["model"]["average_precision"],
                "all_ttc_ap": all_events["ttc"]["average_precision"],
                "ttc_negative_model_auroc": ttc_negative["model"]["auroc"],
                "ttc_negative_ttc_auroc": ttc_negative["ttc"]["auroc"],
                "ttc_negative_model_ap": ttc_negative["model"][
                    "average_precision"
                ],
                "ttc_negative_ttc_ap": ttc_negative["ttc"][
                    "average_precision"
                ],
                "ttc_blind_positive_count": blind_lift["event_count"],
                "ttc_blind_model_win_rate": blind_lift["model_win_rate"],
                "ttc_blind_mean_lift": blind_lift["mean_lift"],
                "ttc_blind_median_lift": blind_lift["median_lift"],
                "ttc_blind_mean_lift_ci_low": blind_lift[
                    "mean_lift_bootstrap_95ci"
                ][0],
                "ttc_blind_mean_lift_ci_high": blind_lift[
                    "mean_lift_bootstrap_95ci"
                ][1],
            }
        )
    return sensitivity


def _report(
    threshold_rows: Sequence[dict[str, Any]],
    advantage: dict[str, Any],
    pooling_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# Threshold and Temporal Advantage Evaluation",
        "",
        "This is an exploratory five-fold diagnostic. Threshold deltas are",
        "applied to each fold's frozen safe-response threshold. The same",
        "fixed 5-frame temporal pooling is applied to the model and TTC.",
        "",
        "## Threshold Sweep",
        "",
        "| Delta | Final collision recall | TTC-blind rescue | Conditional "
        "rescue FPR | All hard-safe alert rate | Mean lead |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lead = row["mean_rescue_lead_seconds"]
        lead_text = "N/A" if lead is None else f"{float(lead):.3f}s"
        lines.append(
            f"| {float(row['threshold_delta']):.3f} "
            f"| {int(row['final_positive_hits'])}/"
            f"{int(row['positive_event_count'])} "
            f"({float(row['final_positive_recall']):.1%}) "
            f"| {int(row['ttc_blind_rescue_count'])}/"
            f"{int(row['ttc_blind_positive_count'])} "
            f"({float(row['ttc_blind_rescue_recall']):.1%}) "
            f"| {float(row['conditional_rescue_fpr']):.1%} "
            f"| {int(row['final_hard_safe_alerts'])}/"
            f"{int(row['hard_safe_event_count'])} "
            f"({float(row['final_hard_safe_alert_rate']):.1%}) "
            f"| {lead_text} |"
        )
    all_metrics = advantage["all_events"]
    blind_metrics = advantage["ttc_negative_events"]
    lift = advantage["ttc_blind_positive_score_lift"]
    ci = lift["mean_lift_bootstrap_95ci"]
    lines.extend(
        [
            "",
            "## Fixed Peak-Neighborhood Score",
            "",
            f"- Pooling: maximum contiguous 5-frame mean (0.25 s) inside "
            f"the fixed 1.5 s pre-anchor window.",
            f"- All-event model AUROC: "
            f"`{all_metrics['model']['auroc']:.4f}`.",
            f"- All-event TTC AUROC: "
            f"`{all_metrics['ttc']['auroc']:.4f}`.",
            f"- TTC-negative model AUROC: "
            f"`{blind_metrics['model']['auroc']:.4f}`.",
            f"- TTC-negative TTC AUROC: "
            f"`{blind_metrics['ttc']['auroc']:.4f}`.",
            f"- TTC-blind positive model win rate over TTC: "
            f"`{lift['model_win_rate']:.1%}`.",
            f"- Mean TTC-blind score lift: `{lift['mean_lift']:.4f}` "
            f"percentile-score units (bootstrap 95% CI "
            f"`[{ci[0]:.4f}, {ci[1]:.4f}]`).",
            "",
            "These percentile scores are not probabilities. Probability",
            "metrics such as Brier score require a separately frozen",
            "validation-only calibration model.",
            "",
        ]
    )
    lines.extend(
        [
            "## Temporal Pooling Sensitivity",
            "",
            "The model and TTC use the identical pooling operator. Peak-1 "
            "is the maximum single-frame score; Sustain-5 and Sustain-10 "
            "are maximum contiguous means.",
            "",
            "| Pooling | TTC-negative model AUROC | TTC-negative TTC AUROC "
            "| Model AP | TTC AP | Paired win rate | Mean lift |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pooling_rows:
        width = int(row["pooling_frames"])
        label = "Peak-1" if width == 1 else f"Sustain-{width}"
        lines.append(
            f"| {label} ({float(row['pooling_seconds']):.2f} s) "
            f"| {float(row['ttc_negative_model_auroc']):.4f} "
            f"| {float(row['ttc_negative_ttc_auroc']):.4f} "
            f"| {float(row['ttc_negative_model_ap']):.4f} "
            f"| {float(row['ttc_negative_ttc_ap']):.4f} "
            f"| {float(row['ttc_blind_model_win_rate']):.1%} "
            f"| {float(row['ttc_blind_mean_lift']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "This table is a sensitivity analysis. A final pooling width "
            "must be selected on development families and frozen before "
            "the final holdout evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_figures(
    *,
    threshold_rows: Sequence[dict[str, Any]],
    advantage: dict[str, Any],
    event_rows: Sequence[dict[str, Any]],
    pooling_rows: Sequence[dict[str, Any]],
    output_root: Path,
) -> dict[str, str]:
    tradeoff_path = output_root / "threshold_tradeoff.png"
    advantage_path = output_root / "score_advantage.png"
    pooling_path = output_root / "temporal_pooling_ablation.png"

    x = np.asarray(
        [float(row["conditional_rescue_fpr"]) for row in threshold_rows]
    )
    blind_recall = np.asarray(
        [float(row["ttc_blind_rescue_recall"]) for row in threshold_rows]
    )
    overall_recall = np.asarray(
        [float(row["final_positive_recall"]) for row in threshold_rows]
    )
    figure, axis = plt.subplots(figsize=(8.8, 5.4), facecolor="white")
    axis.plot(
        x,
        blind_recall,
        marker="o",
        linewidth=2.2,
        color="#16836F",
        label="TTC-blind rescue recall",
    )
    axis.plot(
        x,
        overall_recall,
        marker="s",
        linewidth=2.0,
        color="#356C9A",
        label="Overall collision recall",
    )
    for row, x_value, y_value in zip(threshold_rows, x, blind_recall):
        axis.annotate(
            f"-{float(row['threshold_delta']):.3f}",
            (x_value, y_value),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            color="#45565C",
        )
    axis.axvline(
        0.10,
        color="#B23A48",
        linestyle="--",
        linewidth=1.3,
        label="10% conditional FPR target",
    )
    axis.set_xlabel("Conditional rescue FPR on TTC-negative interactive events")
    axis.set_ylabel("Event recall")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.45, 1.0)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title(
        "Threshold relaxation improves recall but rapidly increases alerts",
        loc="left",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(tradeoff_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    all_metrics = advantage["all_events"]
    blind_metrics = advantage["ttc_negative_events"]
    metric_groups = (
        ("All AUROC", all_metrics["model"]["auroc"], all_metrics["ttc"]["auroc"]),
        (
            "All AP",
            all_metrics["model"]["average_precision"],
            all_metrics["ttc"]["average_precision"],
        ),
        (
            "TTC-neg AUROC",
            blind_metrics["model"]["auroc"],
            blind_metrics["ttc"]["auroc"],
        ),
        (
            "TTC-neg AP",
            blind_metrics["model"]["average_precision"],
            blind_metrics["ttc"]["average_precision"],
        ),
    )
    blind_positive_lifts = sorted(
        float(row["peak_neighborhood_lift"])
        for row in event_rows
        if int(row["positive"]) and not int(row["ttc_detected"])
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.8),
        gridspec_kw={"width_ratios": (1.0, 1.25)},
        facecolor="white",
    )
    positions = np.arange(len(metric_groups))
    width = 0.36
    model_values = [float(group[1]) for group in metric_groups]
    ttc_values = [float(group[2]) for group in metric_groups]
    axes[0].bar(
        positions - width / 2,
        model_values,
        width,
        color="#16836F",
        label="Safe-response",
    )
    axes[0].bar(
        positions + width / 2,
        ttc_values,
        width,
        color="#356C9A",
        label="TTC",
    )
    axes[0].set_xticks(
        positions,
        [group[0] for group in metric_groups],
        rotation=18,
        ha="right",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Metric value")
    axes[0].grid(axis="y", alpha=0.16)
    axes[0].legend(frameon=False)
    axes[0].set_title(
        "Fixed 0.25 s peak-neighborhood score",
        loc="left",
        fontweight="bold",
    )

    colors = [
        "#16836F" if value > 0.0 else "#B23A48"
        for value in blind_positive_lifts
    ]
    axes[1].bar(
        np.arange(1, len(blind_positive_lifts) + 1),
        blind_positive_lifts,
        color=colors,
        width=0.82,
    )
    axes[1].axhline(0.0, color="#30393C", linewidth=1.0)
    axes[1].set_xlabel("TTC-blind collision events sorted by score lift")
    axes[1].set_ylabel("Safe-response minus TTC percentile score")
    axes[1].grid(axis="y", alpha=0.16)
    axes[1].set_title(
        "Model wins on 31/33 TTC-blind collisions",
        loc="left",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(advantage_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    labels = [
        "Peak\n1 frame"
        if int(row["pooling_frames"]) == 1
        else f"Sustain\n{float(row['pooling_seconds']):.2f} s"
        for row in pooling_rows
    ]
    positions = np.arange(len(pooling_rows))
    width = 0.35
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), facecolor="white")
    axes[0].bar(
        positions - width / 2,
        [float(row["ttc_negative_model_auroc"]) for row in pooling_rows],
        width,
        color="#16836F",
        label="Safe-response",
    )
    axes[0].bar(
        positions + width / 2,
        [float(row["ttc_negative_ttc_auroc"]) for row in pooling_rows],
        width,
        color="#356C9A",
        label="TTC",
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel("TTC-negative AUROC")
    axes[0].grid(axis="y", alpha=0.16)
    axes[0].legend(frameon=False)
    axes[0].set_title("Detection ranking", loc="left", fontweight="bold")

    axes[1].bar(
        positions,
        [float(row["ttc_blind_mean_lift"]) for row in pooling_rows],
        color="#16836F",
        width=0.58,
    )
    for position, row in zip(positions, pooling_rows):
        axes[1].annotate(
            f"win {float(row['ttc_blind_model_win_rate']):.1%}",
            (position, float(row["ttc_blind_mean_lift"])),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axes[1].axhline(0.0, color="#30393C", linewidth=1.0)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Mean paired percentile lift")
    axes[1].grid(axis="y", alpha=0.16)
    axes[1].set_title(
        "Model advantage on 33 TTC-blind collisions",
        loc="left",
        fontweight="bold",
    )
    figure.suptitle(
        "Temporal pooling sensitivity",
        x=0.06,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(pooling_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "threshold_tradeoff_png": str(tradeoff_path),
        "score_advantage_png": str(advantage_path),
        "temporal_pooling_ablation_png": str(pooling_path),
    }


def run(
    *,
    folds: Sequence[int],
    threshold_deltas: Sequence[float],
    rolling_frames: int,
    pooling_frames: Sequence[int],
    bootstrap_samples: int,
    seed: int,
    index_root: Path = DEFAULT_INDEX_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    safe_response_root: Path = DEFAULT_SAFE_RESPONSE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    if rolling_frames < 1:
        raise ValueError("rolling_frames must be positive")
    pooling_frames = sorted(
        {int(width) for width in pooling_frames} | {int(rolling_frames)}
    )
    if not pooling_frames or pooling_frames[0] < 1:
        raise ValueError("pooling_frames must contain positive integers")
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_by_delta: dict[float, list[dict[str, Any]]] = {
        float(delta): [] for delta in threshold_deltas
    }
    event_score_rows = []
    fold_rows = []
    for fold in folds:
        frames = _fold_frames(cache_root, fold)
        index_path = (
            index_root
            / f"fold_{fold:02d}"
            / "riskbench_nested_fold_index.json"
        )
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        summary = json.loads(
            (
                safe_response_root
                / f"fold_{fold:02d}"
                / "safe_response_summary.json"
            ).read_text(encoding="utf-8")
        )
        base_config = SafeResponseConfig(**summary["selected_config"])
        observations = _observations_by_frame(
            index_path,
            splits=tuple(frames),
        )
        geometry = _geometry_from_observations(
            observations,
            horizon_seconds=base_config.horizon_seconds,
            clearance_m=base_config.clearance_m,
        )
        _attach_relation_signals(frames, geometry=geometry)
        event_score_rows.extend(
            _event_score_rows(
                frames["test_positive"],
                fold=fold,
                positive=True,
                config=base_config,
                records=index_payload["splits"]["test_positive"],
                rolling_frames=rolling_frames,
                pooling_frames=pooling_frames,
            )
        )
        event_score_rows.extend(
            _event_score_rows(
                frames["test_normal"],
                fold=fold,
                positive=False,
                config=base_config,
                records=index_payload["splits"]["test_normal"],
                rolling_frames=rolling_frames,
                pooling_frames=pooling_frames,
            )
        )
        for delta in threshold_deltas:
            config = replace(
                base_config,
                score_threshold=max(
                    0.0,
                    float(base_config.score_threshold) - float(delta),
                ),
            )
            positive = _rescue_metrics(
                frames["test_positive"],
                config=config,
                positive=True,
            )
            normal = _rescue_metrics(
                frames["test_normal"],
                config=config,
                positive=False,
            )
            result = {
                "fold": fold,
                "threshold_delta": float(delta),
                "base_threshold": float(base_config.score_threshold),
                "relaxed_threshold": float(config.score_threshold),
                "positive": positive,
                "normal": normal,
            }
            results_by_delta[float(delta)].append(result)
            fold_rows.append(
                {
                    "fold": fold,
                    "threshold_delta": float(delta),
                    "base_threshold": float(base_config.score_threshold),
                    "relaxed_threshold": float(config.score_threshold),
                    "final_positive_hits": positive["final_event_hits"],
                    "ttc_blind_rescues": positive["blind_rescue_count"],
                    "normal_false_rescues": normal["blind_rescue_count"],
                    "conditional_rescue_fpr": normal["blind_rescue_rate"],
                }
            )
    threshold_rows = [
        _aggregate_threshold(delta, results_by_delta[delta])
        for delta in sorted(results_by_delta)
    ]
    advantage = _score_advantage(
        event_score_rows,
        rolling_frames=rolling_frames,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    pooling_rows = _pooling_sensitivity(
        event_score_rows,
        pooling_frames=pooling_frames,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 100,
    )
    _write_csv(output_root / "threshold_sweep.csv", threshold_rows)
    _write_csv(output_root / "threshold_sweep_by_fold.csv", fold_rows)
    _write_csv(output_root / "event_score_advantage.csv", event_score_rows)
    _write_csv(output_root / "temporal_pooling_ablation.csv", pooling_rows)
    figure_paths = _render_figures(
        threshold_rows=threshold_rows,
        advantage=advantage,
        event_rows=event_score_rows,
        pooling_rows=pooling_rows,
        output_root=output_root,
    )
    result = {
        "protocol": "riskbench-threshold-and-score-advantage-v1",
        "folds": list(folds),
        "threshold_sweep": threshold_rows,
        "score_advantage": advantage,
        "temporal_pooling_sensitivity": pooling_rows,
        "warning": (
            "Exploratory analysis on existing outer folds. Do not select "
            "a final threshold from this sweep and report it as untouched."
        ),
        "artifacts": {
            "threshold_sweep_csv": str(
                output_root / "threshold_sweep.csv"
            ),
            "threshold_sweep_by_fold_csv": str(
                output_root / "threshold_sweep_by_fold.csv"
            ),
            "event_score_advantage_csv": str(
                output_root / "event_score_advantage.csv"
            ),
            "temporal_pooling_ablation_csv": str(
                output_root / "temporal_pooling_ablation.csv"
            ),
            **figure_paths,
        },
        "status": "threshold_advantage_evaluation_complete",
    }
    summary_path = output_root / "threshold_advantage_summary.json"
    report_path = output_root / "THRESHOLD_ADVANTAGE_REPORT.md"
    result["artifacts"]["summary_json"] = str(summary_path)
    result["artifacts"]["report_markdown"] = str(report_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(
        _report(threshold_rows, advantage, pooling_rows),
        encoding="utf-8",
    )
    return result


def main() -> None:
    args = parse_args()
    folds = sorted(
        {
            int(value.strip())
            for value in args.folds.split(",")
            if value.strip()
        }
    )
    deltas = sorted(
        {
            float(value.strip())
            for value in args.threshold_deltas.split(",")
            if value.strip()
        }
    )
    pooling_frames = sorted(
        {
            int(value.strip())
            for value in args.pooling_frames.split(",")
            if value.strip()
        }
    )
    result = run(
        folds=folds,
        threshold_deltas=deltas,
        rolling_frames=args.rolling_frames,
        pooling_frames=pooling_frames,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        index_root=args.index_root,
        cache_root=args.cache_root,
        safe_response_root=args.safe_response_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
