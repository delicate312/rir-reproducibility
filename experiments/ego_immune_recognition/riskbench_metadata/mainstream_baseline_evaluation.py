"""Historical exploratory same-input baseline evaluation (superseded).

Do not use outputs from this module as paper results.  It evaluates an older
exact-variant holdout, uses fixed-epoch single-seed temporal classifiers, and
imports the rejected Pair-DAMP onset extension.  The formal replacement is
``mainstream_baseline_evaluation_v2.py`` and its controlled-baseline v2 result
directory.

The current metadata package contains ego-relative actor centers, but no actor
footprints, yaw, HD map, or full simulator trajectories.  Therefore the
trajectory baselines below are explicitly center-clearance approximations, not
claimed reproductions of the official RiskBench footprint collision checker.

GRU and Transformer models use development collision events for fitting,
safe-core events as negatives, and hard-safe validation events only for
matched-FPR threshold calibration.  Outer test events never affect fitting,
checkpoint selection, or thresholds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import pickle
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .core_hybrid_evaluation import (
    _usable_frames,
    event_alert,
    event_score,
    threshold_at_event_fpr,
)
from .family_grouped_nested_cv import WINDOW_FRAMES
from .pair_aligned_damp_evaluation import _attach_physical_percentile
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ego_immune_riskbench_pair_damp_v2_holdout"
)
DEFAULT_PAIR_CACHE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_v2_holdout"
    / "pair_frame_cache"
    / "graph_free_one_class"
)
DEFAULT_PAIR_DAMP_EVENTS = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_pair_damp_v2_holdout"
    / "v2_holdout_events.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_mainstream_baselines_v1"
)
DEFAULT_FPR_BUDGETS = (0.05, 0.10)
DEFAULT_SEED = 20260805
FEATURE_NAMES = (
    "actor_count",
    "min_distance",
    "distance_q25",
    "mean_distance",
    "max_speed",
    "max_closing_speed",
    "max_inverse_ttc",
    "cv_center_risk",
    "kalman_center_risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument(
        "--pair-cache-root", type=Path, default=DEFAULT_PAIR_CACHE_ROOT
    )
    parser.add_argument(
        "--pair-damp-events", type=Path, default=DEFAULT_PAIR_DAMP_EVENTS
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_index(index_root: Path, fold: int) -> dict[str, Any]:
    path = index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pair_frames(pair_cache_root: Path, fold: int) -> dict[str, Any]:
    with (pair_cache_root / f"fold_{fold:02d}.pkl").open("rb") as handle:
        return pickle.load(handle)


def _load_state(path: str | Path) -> dict[int, dict[str, np.ndarray]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(frame): {
            str(actor): np.asarray(position, dtype=np.float64)
            for actor, position in actors.items()
        }
        for frame, actors in payload.items()
    }


def _center_risk(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    horizon_seconds: float = 3.0,
    clearance_m: float = 3.0,
) -> tuple[float, float, float]:
    """Return smooth risk, closest distance, and time to closest approach."""
    speed_sq = float(np.dot(velocity, velocity))
    if speed_sq <= 1e-8:
        closest_time = 0.0
    else:
        closest_time = float(
            np.clip(
                -float(np.dot(position, velocity)) / speed_sq,
                0.0,
                horizon_seconds,
            )
        )
    closest = position + velocity * closest_time
    distance = float(np.linalg.norm(closest))
    proximity = math.exp(-0.5 * (distance / clearance_m) ** 2)
    urgency = math.exp(-closest_time / horizon_seconds)
    return float(proximity * urgency), distance, closest_time


class CenterKalman:
    """Minimal constant-velocity Kalman filter for one relative actor center."""

    def __init__(self, position: np.ndarray, dt: float) -> None:
        self.dt = float(dt)
        self.x = np.asarray([position[0], position[1], 0.0, 0.0], dtype=float)
        self.p = np.diag([1.0, 1.0, 25.0, 25.0])
        self.f = np.asarray(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        self.h = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float
        )
        self.q = np.diag([0.02, 0.02, 0.8, 0.8])
        self.r = np.diag([0.25, 0.25])

    def update(self, position: np.ndarray) -> None:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        innovation = np.asarray(position, dtype=float) - self.h @ self.x
        covariance = self.h @ self.p @ self.h.T + self.r
        gain = self.p @ self.h.T @ np.linalg.inv(covariance)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(4) - gain @ self.h) @ self.p

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()


def _robust_velocity(
    history: Sequence[tuple[int, np.ndarray]],
    *,
    dt: float,
    max_steps: int = 5,
) -> np.ndarray:
    recent = list(history)[-(max_steps + 1) :]
    estimates = []
    for (left_frame, left), (right_frame, right) in zip(recent, recent[1:]):
        delta = right_frame - left_frame
        if delta > 0:
            estimates.append((right - left) / (delta * dt))
    if not estimates:
        return np.zeros(2, dtype=float)
    return np.median(np.stack(estimates), axis=0)


def scenario_feature_trace(
    state: dict[int, dict[str, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Create a causal, actor-order-invariant feature vector per frame."""
    histories: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    filters: dict[str, CenterKalman] = {}
    output: dict[int, np.ndarray] = {}
    dt = RISK_BENCH_TIMESTEP_SECONDS
    for frame in sorted(state):
        actors = state[frame]
        distances = []
        speeds = []
        closings = []
        inverse_ttcs = []
        cv_risks = []
        kalman_risks = []
        for actor_id, position in actors.items():
            history = histories[actor_id]
            history.append((frame, position))
            history[:] = history[-6:]
            velocity = _robust_velocity(history, dt=dt)
            distance = float(np.linalg.norm(position))
            speed = float(np.linalg.norm(velocity))
            radial = (
                -float(np.dot(position, velocity)) / max(distance, 1e-6)
            )
            closing = max(radial, 0.0)
            inverse_ttc = closing / max(distance, 1e-6)
            cv_risk, _, _ = _center_risk(position, velocity)

            if actor_id not in filters:
                filters[actor_id] = CenterKalman(position, dt)
            else:
                filters[actor_id].update(position)
            kalman = filters[actor_id]
            kalman_risk, _, _ = _center_risk(kalman.position, kalman.velocity)

            distances.append(distance)
            speeds.append(speed)
            closings.append(closing)
            inverse_ttcs.append(inverse_ttc)
            cv_risks.append(cv_risk)
            kalman_risks.append(kalman_risk)

        if not distances:
            vector = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        else:
            vector = np.asarray(
                [
                    min(len(distances), 30) / 30.0,
                    min(distances),
                    float(np.quantile(distances, 0.25)),
                    float(np.mean(distances)),
                    max(speeds),
                    max(closings),
                    max(inverse_ttcs),
                    max(cv_risks),
                    max(kalman_risks),
                ],
                dtype=np.float32,
            )
        output[frame] = vector
    return output


def _right_padded_window(
    trace: dict[int, np.ndarray],
    end_frame: int,
    window_frames: int = WINDOW_FRAMES,
) -> tuple[np.ndarray, int]:
    selected = [trace[key] for key in sorted(trace) if key <= end_frame]
    selected = selected[-window_frames:]
    length = len(selected)
    output = np.zeros((window_frames, len(FEATURE_NAMES)), dtype=np.float32)
    if length:
        output[:length] = np.stack(selected)
    return output, length


def _normal_endpoints(trace: dict[int, np.ndarray]) -> list[int]:
    frames = sorted(trace)
    if not frames:
        return []
    candidates = np.linspace(0, len(frames) - 1, num=min(4, len(frames)), dtype=int)
    return sorted({frames[int(index)] for index in candidates})


def _positive_endpoints(record: dict[str, Any], trace: dict[int, np.ndarray]) -> list[int]:
    anchor = int(record["critical_frame"])
    frames = [frame for frame in sorted(trace) if anchor - WINDOW_FRAMES <= frame < anchor]
    return frames[::5] or frames[-1:]


class GRUBaseline(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            values,
            lengths.cpu().clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)


class TransformerBaseline(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, WINDOW_FRAMES, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(values.shape[1], device=values.device)[None, :]
        padding = steps >= lengths[:, None].clamp_min(1)
        encoded = self.encoder(
            self.input(values) + self.position[:, : values.shape[1]],
            src_key_padding_mask=padding,
        )
        last = (lengths.clamp_min(1) - 1).long()
        pooled = encoded[torch.arange(len(values), device=values.device), last]
        return self.head(pooled).squeeze(-1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _training_arrays(
    index: dict[str, Any],
    traces: dict[str, dict[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    lengths = []
    labels = []
    for split, label in (("train_normal", 0.0), ("validation_positive", 1.0)):
        for record in index["splits"][split]:
            trace = traces[record["scenario_id"]]
            endpoints = (
                _normal_endpoints(trace)
                if label == 0.0
                else _positive_endpoints(record, trace)
            )
            for endpoint in endpoints:
                window, length = _right_padded_window(trace, endpoint)
                if length:
                    windows.append(window)
                    lengths.append(length)
                    labels.append(label)
    return (
        np.stack(windows),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
    )


def _fit_standardizer(values: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.concatenate(
        [values[index, :length] for index, length in enumerate(lengths)], axis=0
    )
    mean = observed.mean(axis=0)
    scale = observed.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    output = (values - mean[None, None, :]) / scale[None, None, :]
    output[np.all(values == 0.0, axis=2)] = 0.0
    return output.astype(np.float32)


def train_temporal_model(
    kind: str,
    values: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    _set_seed(seed)
    model: nn.Module
    if kind == "gru":
        model = GRUBaseline(values.shape[-1])
    elif kind == "transformer":
        model = TransformerBaseline(values.shape[-1])
    else:
        raise ValueError(f"Unknown temporal baseline: {kind}")
    model.to(device)
    dataset = TensorDataset(
        torch.from_numpy(values),
        torch.from_numpy(lengths),
        torch.from_numpy(labels),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_values, batch_lengths, batch_labels in loader:
            batch_values = batch_values.to(device)
            batch_lengths = batch_lengths.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_values, batch_lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_values)
            count += len(batch_values)
        history.append({"epoch": epoch, "train_bce": total / max(count, 1)})
    return model, history


@torch.no_grad()
def _predict_windows(
    model: nn.Module,
    windows: Sequence[np.ndarray],
    lengths: Sequence[int],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> list[float]:
    if not windows:
        return []
    values = _standardize(np.stack(windows), mean, scale)
    model.eval()
    logits = model(
        torch.from_numpy(values).to(device),
        torch.as_tensor(lengths, dtype=torch.long, device=device),
    )
    return torch.sigmoid(logits).cpu().numpy().astype(float).tolist()


def _event_records(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"riskbench:{record['category']}:{record['scenario_id']}": record
        for records in index["splits"].values()
        for record in records
    }


def _event_frame_groups(pair_frames: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output = {}
    for split, split_frames in pair_frames.items():
        events = {}
        for frames in _group_dual_frames_by_event(split_frames).values():
            usable = _usable_frames(frames)
            # Keep events with no usable pre-anchor frame in the denominator.
            # Their causal score is zero rather than silently dropping them.
            if frames:
                events[str(frames[0]["event_uid"])] = usable
        output[split] = events
    return output


def _method_traces_for_fold(
    index: dict[str, Any],
    pair_frames: dict[str, Any],
    feature_traces: dict[str, dict[int, np.ndarray]],
    models: dict[str, nn.Module],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, list[tuple[dict[str, Any], float]]]],
    dict[str, dict[str, list[tuple[dict[str, Any], float]]]],
]:
    records = _event_records(index)
    groups = _event_frame_groups(pair_frames)
    method_traces: dict[str, dict[str, list[tuple[dict[str, Any], float]]]] = {
        method: defaultdict(dict) for method in ("cv", "kalman", "gru", "transformer")
    }
    ttc_traces: dict[str, dict[str, list[tuple[dict[str, Any], float]]]] = defaultdict(dict)
    for split, events in groups.items():
        for uid, frames in events.items():
            record = records[uid]
            features = feature_traces[record["scenario_id"]]
            ttc_traces[split][uid] = [
                (frame, float(frame["aggregations"]["physical_percentile"]))
                for frame in frames
            ]
            windows = []
            lengths = []
            cv_trace = []
            kalman_trace = []
            for frame in frames:
                source_frame = int(frame["source_frame_index"])
                vector = features[source_frame]
                cv_trace.append((frame, float(vector[7])))
                kalman_trace.append((frame, float(vector[8])))
                window, length = _right_padded_window(features, source_frame)
                windows.append(window)
                lengths.append(length)
            method_traces["cv"][split][uid] = cv_trace
            method_traces["kalman"][split][uid] = kalman_trace
            for kind, model in models.items():
                scores = _predict_windows(
                    model,
                    windows,
                    lengths,
                    mean=mean,
                    scale=scale,
                    device=device,
                )
                method_traces[kind][split][uid] = list(zip(frames, scores))
    return method_traces, ttc_traces


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    labels = np.asarray([int(row["positive"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def evaluate_fold(
    fold: int,
    *,
    index_root: Path,
    pair_cache_root: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index = _load_index(index_root, fold)
    pair_frames = _load_pair_frames(pair_cache_root, fold)
    _attach_physical_percentile(pair_frames)
    all_records = [record for records in index["splits"].values() for record in records]
    feature_traces = {
        record["scenario_id"]: scenario_feature_trace(_load_state(record["state_path"]))
        for record in all_records
    }
    train_values, train_lengths, train_labels = _training_arrays(index, feature_traces)
    mean, scale = _fit_standardizer(train_values, train_lengths)
    standardized = _standardize(train_values, mean, scale)
    models = {}
    histories = {}
    for offset, kind in enumerate(("gru", "transformer")):
        models[kind], histories[kind] = train_temporal_model(
            kind,
            standardized,
            train_lengths,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + fold * 100 + offset,
            device=device,
        )
    method_traces, ttc_traces = _method_traces_for_fold(
        index,
        pair_frames,
        feature_traces,
        models,
        mean=mean,
        scale=scale,
        device=device,
    )
    ttc_flags = {
        split: {
            uid: event_alert(trace, ALERT_PERCENTILE)[0]
            for uid, trace in events.items()
        }
        for split, events in ttc_traces.items()
    }
    rows = []
    thresholds = []
    for method, splits in method_traces.items():
        validation_blind = [
            uid
            for uid in splits["validation_normal"]
            if not ttc_flags["validation_normal"][uid]
        ]
        for budget in DEFAULT_FPR_BUDGETS:
            threshold, inner_fpr = threshold_at_event_fpr(
                [event_score(splits["validation_normal"][uid]) for uid in validation_blind],
                budget,
            )
            thresholds.append(
                {
                    "fold": fold,
                    "method": method,
                    "fpr_budget": budget,
                    "threshold": threshold,
                    "validation_ttc_negative_count": len(validation_blind),
                    "validation_incremental_fpr": inner_fpr,
                }
            )
            for split in ("test_normal", "test_positive"):
                for uid, trace in splits[split].items():
                    detected, lead = event_alert(trace, threshold)
                    ttc_detected, ttc_lead = event_alert(
                        ttc_traces[split][uid], ALERT_PERCENTILE
                    )
                    rows.append(
                        {
                            "fold": fold,
                            "method": method,
                            "fpr_budget": budget,
                            "split": split,
                            "event_uid": uid,
                            "positive": int(split == "test_positive"),
                            "score": event_score(trace),
                            "threshold": threshold,
                            "detected": int(detected),
                            "lead_seconds": "" if lead is None else lead,
                            "ttc_detected": int(ttc_detected),
                            "ttc_lead_seconds": "" if ttc_lead is None else ttc_lead,
                            "final_detected": int(ttc_detected or detected),
                            "blind_rescue": int((not ttc_detected) and detected),
                        }
                    )
    audit = {
        "fold": fold,
        "train_window_count": len(train_labels),
        "train_positive_window_count": int(train_labels.sum()),
        "train_negative_window_count": int(len(train_labels) - train_labels.sum()),
        "outer_test_event_count": sum(
            len(_group_dual_frames_by_event(pair_frames[split]))
            for split in ("test_normal", "test_positive")
        ),
        "feature_names": list(FEATURE_NAMES),
        "histories": histories,
    }
    return rows, thresholds, audit


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in sorted({str(row["method"]) for row in rows}):
        for budget in DEFAULT_FPR_BUDGETS:
            selected = [
                row for row in rows
                if row["method"] == method and float(row["fpr_budget"]) == budget
            ]
            positives = [row for row in selected if int(row["positive"])]
            normals = [row for row in selected if not int(row["positive"])]
            blind_positive = [row for row in positives if not int(row["ttc_detected"])]
            blind_normal = [row for row in normals if not int(row["ttc_detected"])]
            rescues = sum(int(row["blind_rescue"]) for row in blind_positive)
            false_rescues = sum(int(row["blind_rescue"]) for row in blind_normal)
            leads = [
                float(row["lead_seconds"])
                for row in blind_positive
                if int(row["blind_rescue"]) and row["lead_seconds"] != ""
            ]
            continuous = _binary_metrics(
                [
                    row for row in selected
                    if not int(row["ttc_detected"])
                ]
            )
            output.append(
                {
                    "method": method,
                    "fpr_budget": budget,
                    "positive_event_count": len(positives),
                    "hard_safe_event_count": len(normals),
                    "overall_recall": _safe_rate(
                        sum(int(row["final_detected"]) for row in positives),
                        len(positives),
                    ),
                    "overall_hard_safe_fpr": _safe_rate(
                        sum(int(row["final_detected"]) for row in normals), len(normals)
                    ),
                    "ttc_blind_positive_count": len(blind_positive),
                    "ttc_negative_hard_safe_count": len(blind_normal),
                    "blind_rescue_count": rescues,
                    "blind_rescue_recall": _safe_rate(rescues, len(blind_positive)),
                    "false_rescue_count": false_rescues,
                    "incremental_rescue_fpr": _safe_rate(false_rescues, len(blind_normal)),
                    "mean_blind_rescue_lead_seconds": (
                        float(np.mean(leads)) if leads else None
                    ),
                    "ttc_negative_auroc": continuous["auroc"],
                    "ttc_negative_average_precision": continuous["average_precision"],
                }
            )
    return output


def _pair_damp_reference(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    output = []
    for budget in DEFAULT_FPR_BUDGETS:
        selected = [row for row in rows if float(row["fpr_budget"]) == budget]
        positives = [row for row in selected if int(row["positive"])]
        normals = [row for row in selected if not int(row["positive"])]
        blind_positive = [row for row in positives if not int(row["ttc_detected"])]
        blind_normal = [row for row in normals if not int(row["ttc_detected"])]
        for method, key in (("ttc", None), ("pair_damp_v2", "onset_detected")):
            if key is None:
                hits = sum(int(row["ttc_detected"]) for row in positives)
                alerts = sum(int(row["ttc_detected"]) for row in normals)
                rescues = false_rescues = 0
            else:
                hits = sum(
                    int(int(row["ttc_detected"]) or int(row[key])) for row in positives
                )
                alerts = sum(
                    int(int(row["ttc_detected"]) or int(row[key])) for row in normals
                )
                rescues = sum(int(row[key]) for row in blind_positive)
                false_rescues = sum(int(row[key]) for row in blind_normal)
            output.append(
                {
                    "method": method,
                    "fpr_budget": budget,
                    "positive_event_count": len(positives),
                    "hard_safe_event_count": len(normals),
                    "overall_recall": _safe_rate(hits, len(positives)),
                    "overall_hard_safe_fpr": _safe_rate(alerts, len(normals)),
                    "ttc_blind_positive_count": len(blind_positive),
                    "ttc_negative_hard_safe_count": len(blind_normal),
                    "blind_rescue_count": rescues,
                    "blind_rescue_recall": _safe_rate(rescues, len(blind_positive)),
                    "false_rescue_count": false_rescues,
                    "incremental_rescue_fpr": _safe_rate(false_rescues, len(blind_normal)),
                    "mean_blind_rescue_lead_seconds": None,
                    "ttc_negative_auroc": None,
                    "ttc_negative_average_precision": None,
                }
            )
    return output


def _qcnet_readiness() -> dict[str, Any]:
    return {
        "baseline": "QCNet + collision checking",
        "status": "not_runnable_from_current_metadata_package",
        "result_produced": False,
        "current_inputs": [
            "ego-relative actor center positions at 20 Hz",
            "actor identifiers",
            "event critical frame and target audit metadata",
        ],
        "missing_required_inputs": [
            "actor footprint length/width and yaw",
            "global actor and ego trajectories in QCNet format",
            "HD map lane graph and map transforms",
            "official QCNet checkpoint or a leakage-safe training corpus",
        ],
        "official_riskbench_repository": "https://github.com/HCIS-Lab/RiskBench",
        "official_baseline_page": "https://hcis-lab.github.io/RiskBench/",
        "local_official_code_audit": (
            "/private/tmp/RiskBench-official2/risk_identification/Baselines/traj-based"
        ),
        "audit_finding": (
            "The inspected official RiskBench tree includes Kalman, SocialGAN, "
            "and MANTRA folders but no bundled QCNet implementation."
        ),
        "claim_boundary": (
            "No QCNet number is reported until the full trajectory/map assets "
            "and an official-compatible model are available."
        ),
    }


def _report(summary: Sequence[dict[str, Any]], qcnet: dict[str, Any]) -> str:
    lines = [
        "# Same-input mainstream baseline evaluation",
        "",
        "All rescue thresholds are calibrated on TTC-negative development hard-safe events. ",
        "Outer holdout events are used once for evaluation.",
        "",
        "| Method | FPR budget | Recall | Hard-safe FPR | TTC-blind rescue | Incremental FPR | Blind AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        auroc = row["ttc_negative_auroc"]
        lines.append(
            "| {method} | {budget:.0%} | {recall:.1%} | {fpr:.1%} | "
            "{rescue:.1%} ({rescues}/{blind}) | {incremental:.1%} | {auroc} |".format(
                method=row["method"],
                budget=float(row["fpr_budget"]),
                recall=float(row["overall_recall"]),
                fpr=float(row["overall_hard_safe_fpr"]),
                rescue=float(row["blind_rescue_recall"]),
                rescues=int(row["blind_rescue_count"]),
                blind=int(row["ttc_blind_positive_count"]),
                incremental=float(row["incremental_rescue_fpr"]),
                auroc="-" if auroc is None else f"{float(auroc):.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `cv` and `kalman` are same-input center-clearance baselines, not official footprint-collision reproductions.",
            "- `gru` and `transformer` are supervised development baselines and therefore use more positive supervision than one-class Pair-DAMP.",
            f"- QCNet status: `{qcnet['status']}`; no synthetic result was inserted.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_evaluation(
    *,
    index_root: Path = DEFAULT_INDEX_ROOT,
    pair_cache_root: Path = DEFAULT_PAIR_CACHE_ROOT,
    pair_damp_events: Path = DEFAULT_PAIR_DAMP_EVENTS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    folds: Iterable[int] = range(5),
    epochs: int = 10,
    batch_size: int = 64,
    seed: int = DEFAULT_SEED,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    rows = []
    thresholds = []
    audits = []
    for fold in folds:
        fold_rows, fold_thresholds, audit = evaluate_fold(
            int(fold),
            index_root=index_root,
            pair_cache_root=pair_cache_root,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        rows.extend(fold_rows)
        thresholds.extend(fold_thresholds)
        audits.append(audit)
        print(
            json.dumps(
                {
                    "fold": int(fold),
                    "status": "complete",
                    "event_rows": len(fold_rows),
                    "train_windows": audit["train_window_count"],
                }
            ),
            flush=True,
        )
    summary = _pair_damp_reference(pair_damp_events) + summarize(rows)
    qcnet = _qcnet_readiness()
    _write_csv(output_root / "baseline_events.csv", rows)
    _write_csv(output_root / "baseline_thresholds.csv", thresholds)
    _write_csv(output_root / "baseline_summary.csv", summary)
    (output_root / "training_audit.json").write_text(
        json.dumps(audits, indent=2), encoding="utf-8"
    )
    (output_root / "qcnet_readiness.json").write_text(
        json.dumps(qcnet, indent=2), encoding="utf-8"
    )
    (output_root / "BASELINE_REPORT.md").write_text(
        _report(summary, qcnet), encoding="utf-8"
    )
    result = {
        "protocol": "riskbench-same-input-mainstream-baselines-v1",
        "folds": list(folds),
        "epochs": epochs,
        "seed": seed,
        "feature_names": list(FEATURE_NAMES),
        "summary": summary,
        "qcnet": qcnet,
        "output_root": str(output_root),
        "status": "complete",
    }
    (output_root / "baseline_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    result = run_evaluation(
        index_root=args.index_root.expanduser().resolve(),
        pair_cache_root=args.pair_cache_root.expanduser().resolve(),
        pair_damp_events=args.pair_damp_events.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        folds=folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
