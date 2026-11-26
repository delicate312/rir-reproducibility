"""Leakage-controlled mainstream-style baselines for the formal RiskBench CV.

This evaluation deliberately separates three claim tiers:

1. TTC and the frozen final Pair-DAMP result are the primary physical and
   proposed-method references.
2. Center-CV, center-Kalman, and small supervised temporal event classifiers
   are controlled same-input baselines, not reproductions of published
   trajectory-prediction systems.
3. Published systems whose required inputs/checkpoints are unavailable are
   listed as not runnable and receive no synthetic result.

All temporal-model checkpoints are selected on a family-disjoint development
subsplit before validation-normal calibration and outer-test scoring begin.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import pickle
from pathlib import Path
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
from .mainstream_baseline_evaluation import (
    CenterKalman,
    FEATURE_NAMES,
    GRUBaseline,
    TransformerBaseline,
    _center_risk,
    _fit_standardizer,
    _load_state,
    _normal_endpoints,
    _positive_endpoints,
    _predict_windows,
    _right_padded_window,
    _robust_velocity,
    _set_seed,
    _standardize,
    scenario_feature_trace,
)
from .pair_aligned_damp_evaluation import _attach_physical_percentile
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_ROOT = (
    PROJECT_ROOT / "data/processed/ego_immune_riskbench_family_nested_cv_v1"
)
DEFAULT_PAIR_DAMP_ROOT = (
    PROJECT_ROOT
    / "results/ego_immune_riskbench_pair_damp_normal_only_multiseed_v1"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results/ego_immune_riskbench_controlled_baselines_v2"
)
DEFAULT_SEEDS = (20260812, 20260813, 20260814)
DEFAULT_FPR_BUDGETS = (0.05, 0.10)
MODEL_KINDS = ("gru", "transformer")
METHOD_NAMES = {
    "cv": "center_cv",
    "kalman": "center_kalman",
    "gru": "supervised_gru_event",
    "transformer": "supervised_transformer_event",
}
PROTOCOL = "riskbench-controlled-baselines-v2"
DEV_SPLIT_SALT = "riskbench-baseline-dev-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--pair-damp-root", type=Path, default=DEFAULT_PAIR_DAMP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_index(index_root: Path, fold: int) -> dict[str, Any]:
    path = index_root / f"fold_{fold:02d}/riskbench_nested_fold_index.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pair_frames(pair_damp_root: Path, seed: int, fold: int) -> dict[str, Any]:
    path = (
        pair_damp_root
        / f"seed_{seed}/pair_frame_cache/graph_free_one_class/fold_{fold:02d}.pkl"
    )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uid(record: dict[str, Any]) -> str:
    return f"riskbench:{record['category']}:{record['scenario_id']}"


def _record_map(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _uid(record): record
        for records in index["splits"].values()
        for record in records
    }


def _family_dev_split(
    records: Sequence[dict[str, Any]], validation_fraction: float = 0.20
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a fixed family-level model-train/model-selection partition."""
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        families[str(record["scenario_group_id"])].append(record)
    ordered = sorted(
        families,
        key=lambda family: hashlib.sha256(
            f"{DEV_SPLIT_SALT}:{family}".encode("utf-8")
        ).hexdigest(),
    )
    validation_count = max(1, int(round(len(ordered) * validation_fraction)))
    validation_families = set(ordered[:validation_count])
    training = [
        record
        for family, members in families.items()
        if family not in validation_families
        for record in members
    ]
    validation = [
        record
        for family, members in families.items()
        if family in validation_families
        for record in members
    ]
    return training, validation


def _arrays_for_records(
    normal_records: Sequence[dict[str, Any]],
    positive_records: Sequence[dict[str, Any]],
    traces: dict[str, dict[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    lengths: list[int] = []
    labels: list[float] = []
    for records, label in ((normal_records, 0.0), (positive_records, 1.0)):
        for record in records:
            trace = traces[str(record["scenario_id"])]
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
    if not windows:
        raise ValueError("Development split produced no temporal windows.")
    return (
        np.stack(windows),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
    )


def _new_model(kind: str, feature_dim: int) -> nn.Module:
    if kind == "gru":
        return GRUBaseline(feature_dim)
    if kind == "transformer":
        return TransformerBaseline(feature_dim)
    raise ValueError(f"Unknown temporal model: {kind}")


@torch.no_grad()
def _validation_stats(
    model: nn.Module,
    values: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float | None]:
    model.eval()
    logits = model(
        torch.from_numpy(values).to(device),
        torch.from_numpy(lengths).to(device),
    )
    loss = float(criterion(logits, torch.from_numpy(labels).to(device)).item())
    scores = torch.sigmoid(logits).cpu().numpy()
    auroc = (
        float(roc_auc_score(labels.astype(int), scores))
        if len(np.unique(labels)) == 2
        else None
    )
    return loss, auroc


def _train_checkpoint(
    *,
    kind: str,
    fold: int,
    seed: int,
    index: dict[str, Any],
    checkpoint_path: Path,
    history_path: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    normal_train, normal_select = _family_dev_split(index["splits"]["train_normal"])
    positive_train, positive_select = _family_dev_split(
        index["splits"]["validation_positive"]
    )
    development_records = [
        *normal_train,
        *normal_select,
        *positive_train,
        *positive_select,
    ]
    traces = {
        str(record["scenario_id"]): scenario_feature_trace(
            _load_state(record["state_path"])
        )
        for record in development_records
    }
    train_values, train_lengths, train_labels = _arrays_for_records(
        normal_train, positive_train, traces
    )
    select_values, select_lengths, select_labels = _arrays_for_records(
        normal_select, positive_select, traces
    )
    mean, scale = _fit_standardizer(train_values, train_lengths)
    train_values = _standardize(train_values, mean, scale)
    select_values = _standardize(select_values, mean, scale)

    model_seed = seed + fold * 100 + MODEL_KINDS.index(kind)
    _set_seed(model_seed)
    model = _new_model(kind, train_values.shape[-1]).to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_values),
        torch.from_numpy(train_lengths),
        torch.from_numpy(train_labels),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(model_seed),
    )
    positives = max(float(train_labels.sum()), 1.0)
    negatives = max(float(len(train_labels) - train_labels.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for values, lengths, labels in loader:
            values = values.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(values, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(values)
            total_count += len(values)
        validation_loss, validation_auroc = _validation_stats(
            model,
            select_values,
            select_lengths,
            select_labels,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_bce": total_loss / max(total_count, 1),
                "selection_bce": validation_loss,
                "selection_auroc": validation_auroc,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": PROTOCOL,
        "kind": kind,
        "fold": fold,
        "seed": seed,
        "model_seed": model_seed,
        "feature_names": list(FEATURE_NAMES),
        "state_dict": best_state,
        "mean": mean,
        "scale": scale,
        "best_epoch": best_epoch,
        "best_selection_bce": best_loss,
        "train_normal_families": sorted(
            str(record["scenario_group_id"]) for record in normal_train
        ),
        "selection_normal_families": sorted(
            str(record["scenario_group_id"]) for record in normal_select
        ),
        "train_positive_families": sorted(
            str(record["scenario_group_id"]) for record in positive_train
        ),
        "selection_positive_families": sorted(
            str(record["scenario_group_id"]) for record in positive_select
        ),
        "train_window_count": int(len(train_labels)),
        "selection_window_count": int(len(select_labels)),
    }
    torch.save(payload, checkpoint_path)
    _write_csv(history_path, history)
    return {
        "kind": kind,
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_selection_bce": best_loss,
        "selection_auroc": history[best_epoch - 1]["selection_auroc"],
        "epochs_run": len(history),
        "train_window_count": len(train_labels),
        "selection_window_count": len(select_labels),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "test_events_read_before_freeze": 0,
    }


def _load_model_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError(f"Incompatible checkpoint: {checkpoint_path}")
    model = _new_model(str(payload["kind"]), len(FEATURE_NAMES)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, np.asarray(payload["mean"]), np.asarray(payload["scale"])


def _event_groups(pair_frames: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, frames in pair_frames.items():
        output[split] = {}
        for event_frames in _group_dual_frames_by_event(frames).values():
            if event_frames:
                output[split][str(event_frames[0]["event_uid"])] = _usable_frames(
                    event_frames
                )
    return output


def _actor_center_risks(
    state: dict[int, dict[str, np.ndarray]],
) -> dict[int, dict[str, tuple[float, float]]]:
    histories: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    filters: dict[str, CenterKalman] = {}
    output: dict[int, dict[str, tuple[float, float]]] = {}
    for frame in sorted(state):
        output[frame] = {}
        for actor_id, position in state[frame].items():
            histories[actor_id].append((frame, position))
            histories[actor_id][:] = histories[actor_id][-6:]
            velocity = _robust_velocity(
                histories[actor_id], dt=RISK_BENCH_TIMESTEP_SECONDS
            )
            cv_risk, _, _ = _center_risk(position, velocity)
            if actor_id not in filters:
                filters[actor_id] = CenterKalman(
                    position, RISK_BENCH_TIMESTEP_SECONDS
                )
            else:
                filters[actor_id].update(position)
            kalman = filters[actor_id]
            kalman_risk, _, _ = _center_risk(kalman.position, kalman.velocity)
            output[frame][str(actor_id)] = (cv_risk, kalman_risk)
    return output


def _rank(scores: dict[str, float], targets: Sequence[str]) -> int | None:
    present = [target for target in targets if target in scores]
    if not present:
        return None
    target_score = max(scores[target] for target in present)
    return 1 + sum(score > target_score for score in scores.values())


def _score_fold(
    *,
    seed: int,
    fold: int,
    index_root: Path,
    pair_damp_root: Path,
    output_root: Path,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = _load_index(index_root, fold)
    records = _record_map(index)
    pair_frames = _load_pair_frames(pair_damp_root, seed, fold)
    _attach_physical_percentile(pair_frames)
    groups = _event_groups(pair_frames)
    evaluation_records = [
        record
        for split in ("validation_normal", "test_normal", "test_positive")
        for record in index["splits"][split]
    ]
    states = {
        str(record["scenario_id"]): _load_state(record["state_path"])
        for record in evaluation_records
    }
    features = {
        scenario_id: scenario_feature_trace(state)
        for scenario_id, state in states.items()
    }
    # The source indices label held-out positives as validation_positive in the
    # record payload; select them by membership above rather than payload label.
    actor_risks = {
        str(record["scenario_id"]): _actor_center_risks(
            states[str(record["scenario_id"])]
        )
        for record in index["splits"]["test_positive"]
    }

    models: dict[str, nn.Module] = {}
    means: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for kind in MODEL_KINDS:
        path = output_root / f"seed_{seed}/fold_{fold:02d}/{kind}_best.pt"
        models[kind], means[kind], scales[kind] = _load_model_checkpoint(path, device)

    traces: dict[str, dict[str, dict[str, list[tuple[dict[str, Any], float]]]]] = {
        method: defaultdict(dict) for method in METHOD_NAMES
    }
    ttc_traces: dict[str, dict[str, list[tuple[dict[str, Any], float]]]] = defaultdict(dict)
    for split, events in groups.items():
        if split not in ("validation_normal", "test_normal", "test_positive"):
            continue
        for event_uid, frames in events.items():
            record = records[event_uid]
            trace = features[str(record["scenario_id"])]
            ttc_traces[split][event_uid] = [
                (frame, float(frame["aggregations"]["physical_percentile"]))
                for frame in frames
            ]
            windows: list[np.ndarray] = []
            lengths: list[int] = []
            cv_trace: list[tuple[dict[str, Any], float]] = []
            kalman_trace: list[tuple[dict[str, Any], float]] = []
            for frame in frames:
                source_frame = int(frame["source_frame_index"])
                vector = trace[source_frame]
                cv_trace.append((frame, float(vector[7])))
                kalman_trace.append((frame, float(vector[8])))
                window, length = _right_padded_window(trace, source_frame)
                windows.append(window)
                lengths.append(length)
            traces["cv"][split][event_uid] = cv_trace
            traces["kalman"][split][event_uid] = kalman_trace
            for kind in MODEL_KINDS:
                scores = _predict_windows(
                    models[kind],
                    windows,
                    lengths,
                    mean=means[kind],
                    scale=scales[kind],
                    device=device,
                )
                traces[kind][split][event_uid] = list(zip(frames, scores))

    ttc_flags = {
        split: {
            event_uid: event_alert(trace, ALERT_PERCENTILE)[0]
            for event_uid, trace in events.items()
        }
        for split, events in ttc_traces.items()
    }
    rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    localization: list[dict[str, Any]] = []
    for short_method, split_traces in traces.items():
        method = METHOD_NAMES[short_method]
        calibration_uids = [
            event_uid
            for event_uid in split_traces["validation_normal"]
            if not ttc_flags["validation_normal"][event_uid]
        ]
        for budget in DEFAULT_FPR_BUDGETS:
            threshold, observed_fpr = threshold_at_event_fpr(
                [
                    event_score(split_traces["validation_normal"][event_uid])
                    for event_uid in calibration_uids
                ],
                budget,
            )
            thresholds.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "method": method,
                    "fpr_budget": budget,
                    "threshold": threshold,
                    "validation_ttc_negative_count": len(calibration_uids),
                    "validation_incremental_fpr": observed_fpr,
                }
            )
            for split in ("test_normal", "test_positive"):
                for event_uid, trace in split_traces[split].items():
                    record = records[event_uid]
                    detected, lead = event_alert(trace, threshold)
                    ttc_detected, ttc_lead = event_alert(
                        ttc_traces[split][event_uid], ALERT_PERCENTILE
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "method": method,
                            "fpr_budget": budget,
                            "event_uid": event_uid,
                            "scenario_group_id": record["scenario_group_id"],
                            "category": record["category"],
                            "scenario_id": record["scenario_id"],
                            "target_type": record.get("target_type_name", "none"),
                            "split": split,
                            "positive": int(split == "test_positive"),
                            "score": event_score(trace),
                            "threshold": threshold,
                            "detected": int(detected),
                            "lead_seconds": "" if lead is None else lead,
                            "ttc_score": event_score(ttc_traces[split][event_uid]),
                            "ttc_detected": int(ttc_detected),
                            "ttc_lead_seconds": "" if ttc_lead is None else ttc_lead,
                            "final_detected": int(ttc_detected or detected),
                            "blind_rescue": int((not ttc_detected) and detected),
                        }
                    )

            if short_method not in ("cv", "kalman"):
                continue
            risk_index = 0 if short_method == "cv" else 1
            for event_uid, event_frames in groups["test_positive"].items():
                record = records[event_uid]
                scenario_id = str(record["scenario_id"])
                targets = [str(value) for value in record["risk_actor_ids"]]
                per_frame: list[tuple[dict[str, Any], dict[str, float]]] = []
                event_scores: dict[str, float] = {}
                for frame in event_frames:
                    scores = {
                        actor: float(values[risk_index])
                        for actor, values in actor_risks[scenario_id][
                            int(frame["source_frame_index"])
                        ].items()
                    }
                    per_frame.append((frame, scores))
                    for actor, score in scores.items():
                        event_scores[actor] = max(event_scores.get(actor, 0.0), score)
                event_rank = _rank(event_scores, targets)
                alert_rank = None
                alert_lead = None
                alert_seen = False
                for frame, scores in per_frame:
                    if max(scores.values(), default=0.0) >= threshold:
                        alert_seen = True
                        alert_rank = _rank(scores, targets)
                        alert_lead = (
                            int(
                                frame.get(
                                    "eval_frames_to_anchor", frame["frames_to_anchor"]
                                )
                            )
                            * RISK_BENCH_TIMESTEP_SECONDS
                        )
                        break
                localization.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "method": method,
                        "fpr_budget": budget,
                        "event_uid": event_uid,
                        "target_type": record.get("target_type_name", "unknown"),
                        "target_ids": "|".join(targets),
                        "event_target_rank": "" if event_rank is None else event_rank,
                        "event_top1": int(event_rank is not None and event_rank <= 1),
                        "event_top3": int(event_rank is not None and event_rank <= 3),
                        "method_alerted": int(alert_seen),
                        "alert_target_rank": "" if alert_rank is None else alert_rank,
                        "alert_top1": int(alert_rank is not None and alert_rank <= 1),
                        "alert_top3": int(alert_rank is not None and alert_rank <= 3),
                        "alert_lead_seconds": "" if alert_lead is None else alert_lead,
                    }
                )
    return rows, thresholds, localization


def _formal_reference_rows(
    *,
    seed: int,
    pair_damp_root: Path,
    index_root: Path,
    folds: Iterable[int] = range(5),
) -> list[dict[str, Any]]:
    source = pair_damp_root / f"seed_{seed}/outer_test_events.csv"
    source_rows = list(csv.DictReader(source.open(encoding="utf-8")))
    metadata: dict[str, dict[str, Any]] = {}
    folds = tuple(int(value) for value in folds)
    for fold in folds:
        metadata.update(_record_map(_load_index(index_root, fold)))
    output: list[dict[str, Any]] = []
    for row in source_rows:
        budget = float(row["fpr_budget"])
        if budget not in DEFAULT_FPR_BUDGETS or int(row["fold"]) not in folds:
            continue
        record = metadata[str(row["event_uid"])]
        common = {
            "seed": seed,
            "fold": int(row["fold"]),
            "fpr_budget": budget,
            "event_uid": row["event_uid"],
            "scenario_group_id": record["scenario_group_id"],
            "category": row["category"],
            "scenario_id": row["scenario_id"],
            "target_type": row["target_type"],
            "split": row["split"],
            "positive": int(row["positive"]),
            "ttc_score": float(row["ttc_pool_1"]),
            "ttc_detected": int(row["ttc_detected"]),
            "ttc_lead_seconds": "",
        }
        ttc_detected = int(row["ttc_detected"])
        output.append(
            {
                **common,
                "method": "ttc",
                "score": float(row["ttc_pool_1"]),
                "threshold": ALERT_PERCENTILE,
                "detected": ttc_detected,
                "lead_seconds": "",
                "final_detected": ttc_detected,
                "blind_rescue": 0,
            }
        )
        baseline_detected = int(row["baseline_detected"])
        output.append(
            {
                **common,
                "method": "pair_damp_final",
                "score": float(row["baseline_score"]),
                "threshold": float(row["threshold"]),
                "detected": baseline_detected,
                "lead_seconds": row["baseline_lead_seconds"],
                "final_detected": int(ttc_detected or baseline_detected),
                "blind_rescue": int((not ttc_detected) and baseline_detected),
            }
        )
    return output


def _continuous_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    labels = np.asarray([int(row["positive"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "pauc_10": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "pauc_10": float(roc_auc_score(labels, scores, max_fpr=0.10)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _summarize_seed(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            for budget in DEFAULT_FPR_BUDGETS:
                selected = [
                    row
                    for row in rows
                    if int(row["seed"]) == seed
                    and row["method"] == method
                    and float(row["fpr_budget"]) == budget
                ]
                if not selected:
                    continue
                positives = [row for row in selected if int(row["positive"])]
                normals = [row for row in selected if not int(row["positive"])]
                blind_positive = [
                    row for row in positives if not int(row["ttc_detected"])
                ]
                blind_normal = [
                    row for row in normals if not int(row["ttc_detected"])
                ]
                blind_rows = [
                    row for row in selected if not int(row["ttc_detected"])
                ]
                continuous = _continuous_metrics(blind_rows)
                leads = [
                    float(row["lead_seconds"])
                    for row in blind_positive
                    if int(row["blind_rescue"]) and row["lead_seconds"] != ""
                ]
                output.append(
                    {
                        "seed": seed,
                        "method": method,
                        "fpr_budget": budget,
                        "positive_event_count": len(positives),
                        "hard_safe_event_count": len(normals),
                        "overall_recall": np.mean(
                            [int(row["final_detected"]) for row in positives]
                        ),
                        "overall_hard_safe_fpr": np.mean(
                            [int(row["final_detected"]) for row in normals]
                        ),
                        "ttc_blind_positive_count": len(blind_positive),
                        "ttc_negative_hard_safe_count": len(blind_normal),
                        "blind_rescue_recall": np.mean(
                            [int(row["blind_rescue"]) for row in blind_positive]
                        ),
                        "incremental_rescue_fpr": np.mean(
                            [int(row["blind_rescue"]) for row in blind_normal]
                        ),
                        "mean_blind_rescue_lead_seconds": (
                            float(np.mean(leads)) if leads else None
                        ),
                        "blind_auroc": continuous["auroc"],
                        "blind_pauc_10": continuous["pauc_10"],
                        "blind_average_precision": continuous["average_precision"],
                    }
                )
    return output


def _aggregate_seed_summary(seed_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "overall_recall",
        "overall_hard_safe_fpr",
        "blind_rescue_recall",
        "incremental_rescue_fpr",
        "mean_blind_rescue_lead_seconds",
        "blind_auroc",
        "blind_pauc_10",
        "blind_average_precision",
    )
    output: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["method"]), float(row["fpr_budget"])) for row in seed_rows}
    )
    for method, budget in keys:
        selected = [
            row
            for row in seed_rows
            if row["method"] == method and float(row["fpr_budget"]) == budget
        ]
        result: dict[str, Any] = {
            "method": method,
            "fpr_budget": budget,
            "seed_count": len(selected),
            "positive_event_count": selected[0]["positive_event_count"],
            "hard_safe_event_count": selected[0]["hard_safe_event_count"],
        }
        for metric in metrics:
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) not in (None, "")
            ]
            result[f"{metric}_mean"] = float(np.mean(values)) if values else None
            result[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ) if values else None
        output.append(result)
    return output


def _bootstrap_mean(
    values_by_family: dict[str, list[float]], replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    families = sorted(values_by_family)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(families, size=len(families), replace=True)
        values = [value for family in sampled for value in values_by_family[family]]
        draws[index] = float(np.mean(values))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _family_bootstrap_intervals(
    rows: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260812)
    for method in sorted({str(row["method"]) for row in rows}):
        for budget in DEFAULT_FPR_BUDGETS:
            selected = [
                row
                for row in rows
                if row["method"] == method and float(row["fpr_budget"]) == budget
            ]
            for metric, population, field in (
                ("overall_recall", "positive", "final_detected"),
                ("overall_hard_safe_fpr", "normal", "final_detected"),
                ("blind_rescue_recall", "blind_positive", "blind_rescue"),
                ("incremental_rescue_fpr", "blind_normal", "blind_rescue"),
            ):
                filtered = []
                for row in selected:
                    positive = bool(int(row["positive"]))
                    blind = not bool(int(row["ttc_detected"]))
                    include = (
                        (population == "positive" and positive)
                        or (population == "normal" and not positive)
                        or (population == "blind_positive" and positive and blind)
                        or (population == "blind_normal" and not positive and blind)
                    )
                    if include:
                        filtered.append(row)
                per_event: dict[tuple[str, str], list[float]] = defaultdict(list)
                for row in filtered:
                    per_event[(row["scenario_group_id"], row["event_uid"])].append(
                        float(row[field])
                    )
                values_by_family: dict[str, list[float]] = defaultdict(list)
                for (family, _), values in per_event.items():
                    values_by_family[family].append(float(np.mean(values)))
                low, high = _bootstrap_mean(values_by_family, replicates, rng)
                output.append(
                    {
                        "method": method,
                        "fpr_budget": budget,
                        "metric": metric,
                        "family_count": len(values_by_family),
                        "event_count": len(per_event),
                        "estimate": float(
                            np.mean([value for values in values_by_family.values() for value in values])
                        ),
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_replicates": replicates,
                    }
                )
    return output


def _paired_bootstrap(
    rows: Sequence[dict[str, Any]], replicates: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260813)
    comparison_methods = sorted(
        {str(row["method"]) for row in rows} - {"pair_damp_final"}
    )
    for budget in DEFAULT_FPR_BUDGETS:
        selected = [row for row in rows if float(row["fpr_budget"]) == budget]
        for comparator in comparison_methods:
            for metric, positive, blind_only, field in (
                ("overall_recall_delta", True, False, "final_detected"),
                ("incremental_fpr_delta", False, True, "blind_rescue"),
            ):
                event_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
                for row in selected:
                    if row["method"] not in ("pair_damp_final", comparator):
                        continue
                    if bool(int(row["positive"])) != positive:
                        continue
                    if blind_only and int(row["ttc_detected"]):
                        continue
                    event_values[(row["method"], row["scenario_group_id"], row["event_uid"])].append(
                        float(row[field])
                    )
                paired: dict[str, list[float]] = defaultdict(list)
                keys = {
                    (family, uid)
                    for method, family, uid in event_values
                    if method == "pair_damp_final"
                }
                for family, uid in keys:
                    left = event_values.get(("pair_damp_final", family, uid))
                    right = event_values.get((comparator, family, uid))
                    if left and right:
                        paired[family].append(float(np.mean(left) - np.mean(right)))
                families = sorted(paired)
                draws = np.empty(replicates, dtype=float)
                for index in range(replicates):
                    sampled = rng.choice(families, size=len(families), replace=True)
                    draws[index] = float(
                        np.mean([value for family in sampled for value in paired[family]])
                    )
                estimate = float(
                    np.mean([value for values in paired.values() for value in values])
                )
                # Plus-one correction prevents a finite bootstrap from
                # reporting the impossible value p=0.
                lower_tail = (float(np.sum(draws <= 0.0)) + 1.0) / (
                    replicates + 1.0
                )
                upper_tail = (float(np.sum(draws >= 0.0)) + 1.0) / (
                    replicates + 1.0
                )
                p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
                output.append(
                    {
                        "reference": "pair_damp_final",
                        "comparator": comparator,
                        "fpr_budget": budget,
                        "metric": metric,
                        "paired_family_count": len(families),
                        "estimate_pair_damp_minus_comparator": estimate,
                        "ci95_low": float(np.quantile(draws, 0.025)),
                        "ci95_high": float(np.quantile(draws, 0.975)),
                        "two_sided_bootstrap_p": p_value,
                        "bootstrap_replicates": replicates,
                    }
                )
    return output


def _localization_summary(
    controlled_rows: Sequence[dict[str, Any]], pair_damp_root: Path
) -> list[dict[str, Any]]:
    rows = list(controlled_rows)
    for seed in DEFAULT_SEEDS:
        path = pair_damp_root / f"seed_{seed}/localization/localization_events.csv"
        for row in csv.DictReader(path.open(encoding="utf-8")):
            if float(row["fpr_budget"]) not in DEFAULT_FPR_BUDGETS:
                continue
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "method": "pair_damp_final",
                    "fpr_budget": float(row["fpr_budget"]),
                    "event_target_rank": row["event_target_rank"],
                    "event_top1": int(row["event_top1"]),
                    "event_top3": int(row["event_top3"]),
                    "method_alerted": int(row["relation_alerted"]),
                    "alert_target_rank": row["alert_target_rank"],
                    "alert_top1": int(row["alert_top1"]),
                    "alert_top3": int(row["alert_top3"]),
                }
            )
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        for budget in DEFAULT_FPR_BUDGETS:
            for seed in DEFAULT_SEEDS:
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and float(row["fpr_budget"]) == budget
                    and int(row["seed"]) == seed
                ]
                eligible = [row for row in selected if row["event_target_rank"] != ""]
                alerted = [row for row in selected if int(row["method_alerted"])]
                output.append(
                    {
                        "seed": seed,
                        "method": method,
                        "fpr_budget": budget,
                        "positive_events": len(selected),
                        "localization_evaluable_events": len(eligible),
                        "event_top1": float(np.mean([row["event_top1"] for row in eligible])) if eligible else None,
                        "event_top3": float(np.mean([row["event_top3"] for row in eligible])) if eligible else None,
                        "alerted_events": len(alerted),
                        "first_alert_top1": float(np.mean([row["alert_top1"] for row in alerted])) if alerted else None,
                        "first_alert_top3": float(np.mean([row["alert_top3"] for row in alerted])) if alerted else None,
                    }
                )
    return output


def _method_contracts() -> list[dict[str, Any]]:
    return [
        {
            "method": "ttc",
            "claim_tier": "physical_rule",
            "same_center_input": True,
            "positive_training": False,
            "target_annotations_at_inference": False,
            "localization_available": False,
            "description": "Inverse-TTC percentile physical rule.",
        },
        {
            "method": "center_cv",
            "claim_tier": "controlled_same_input",
            "same_center_input": True,
            "positive_training": False,
            "target_annotations_at_inference": False,
            "localization_available": True,
            "description": "Constant-velocity center-clearance approximation; not official footprint collision checking.",
        },
        {
            "method": "center_kalman",
            "claim_tier": "controlled_same_input",
            "same_center_input": True,
            "positive_training": False,
            "target_annotations_at_inference": False,
            "localization_available": True,
            "description": "Kalman-smoothed center-clearance approximation; not an official RiskBench reproduction.",
        },
        {
            "method": "supervised_gru_event",
            "claim_tier": "controlled_supervised_classifier",
            "same_center_input": True,
            "positive_training": True,
            "target_annotations_at_inference": False,
            "localization_available": False,
            "description": "Small GRU on nine causal aggregate features; not a published trajectory architecture.",
        },
        {
            "method": "supervised_transformer_event",
            "claim_tier": "controlled_supervised_classifier",
            "same_center_input": True,
            "positive_training": True,
            "target_annotations_at_inference": False,
            "localization_available": False,
            "description": "Small Transformer on nine causal aggregate features; not a published trajectory architecture.",
        },
        {
            "method": "pair_damp_final",
            "claim_tier": "proposed_method",
            "same_center_input": True,
            "positive_training": False,
            "target_annotations_at_inference": False,
            "localization_available": True,
            "description": "Frozen graph-free CPA plus exact-pair DAMP baseline; onset extension excluded.",
        },
    ]


def _published_readiness() -> list[dict[str, Any]]:
    common_missing = [
        "actor footprint length/width and yaw",
        "global trajectories and coordinate transforms",
        "HD map/lane graph where required",
        "official compatible checkpoints or leakage-safe training corpus",
    ]
    return [
        {
            "method": name,
            "status": "not_runnable_from_current_metadata_package",
            "result_produced": False,
            "missing_inputs": common_missing,
            "claim_boundary": "No number is reported; controlled proxies are not named as this published method.",
        }
        for name in ("SocialGAN", "MANTRA", "QCNet")
    ]


def _report(
    aggregate: Sequence[dict[str, Any]],
    intervals: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
) -> str:
    ci = {
        (row["method"], float(row["fpr_budget"]), row["metric"]): row
        for row in intervals
    }
    lines = [
        "# Controlled baseline evaluation v2",
        "",
        "This report uses the same 147 positive and 273 hard-safe outer events as the formal Pair-DAMP evaluation. All supervised checkpoints were selected before outer-test scoring. Pair-DAMP uses the frozen `baseline_detected` result; the rejected onset extension is excluded.",
        "",
        "## Event results",
        "",
        "| Method | Budget | Recall (95% family CI) | TTC-blind rescue | Incremental FPR (95% CI) | Blind AUROC | Blind AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        method = str(row["method"])
        budget = float(row["fpr_budget"])
        recall_ci = ci[(method, budget, "overall_recall")]
        fpr_ci = ci[(method, budget, "incremental_rescue_fpr")]
        lines.append(
            "| {method} | {budget:.0%} | {recall:.1%} [{rl:.1%}, {rh:.1%}] | {rescue:.1%} | {fpr:.1%} [{fl:.1%}, {fh:.1%}] | {auroc} | {ap} |".format(
                method=method,
                budget=budget,
                recall=float(row["overall_recall_mean"]),
                rl=float(recall_ci["ci95_low"]),
                rh=float(recall_ci["ci95_high"]),
                rescue=float(row["blind_rescue_recall_mean"]),
                fpr=float(row["incremental_rescue_fpr_mean"]),
                fl=float(fpr_ci["ci95_low"]),
                fh=float(fpr_ci["ci95_high"]),
                auroc="-" if row["blind_auroc_mean"] is None else f"{float(row['blind_auroc_mean']):.3f}",
                ap="-" if row["blind_average_precision_mean"] is None else f"{float(row['blind_average_precision_mean']):.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- `center_cv` and `center_kalman` are center-clearance approximations, not official footprint-based RiskBench implementations.",
            "- `supervised_gru_event` and `supervised_transformer_event` are deliberately named as small event classifiers. They are not presented as published GRU/Transformer trajectory methods and they use positive development labels unavailable to one-class Pair-DAMP.",
            "- SocialGAN, MANTRA, and QCNet remain not runnable under the current metadata input contract; no invented values are inserted.",
            "- Family bootstrap intervals quantify event-family sampling uncertainty. Three-seed mean/std separately quantify model-initialization stability.",
            "",
            "## Paired Pair-DAMP differences",
            "",
            "See `paired_family_bootstrap.csv` for all effect sizes, confidence intervals, and two-sided bootstrap p-values.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_evaluation(
    *,
    index_root: Path = DEFAULT_INDEX_ROOT,
    pair_damp_root: Path = DEFAULT_PAIR_DAMP_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    folds: Iterable[int] = range(5),
    seeds: Iterable[int] = DEFAULT_SEEDS,
    max_epochs: int = 30,
    patience: int = 6,
    batch_size: int = 64,
    bootstrap_replicates: int = 5000,
    device_name: str = "cpu",
) -> dict[str, Any]:
    folds = tuple(int(value) for value in folds)
    seeds = tuple(int(value) for value in seeds)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    # Phase 1: train and freeze every checkpoint before reading any outer event.
    training_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            index = _load_index(index_root, fold)
            for kind in MODEL_KINDS:
                fold_root = output_root / f"seed_{seed}/fold_{fold:02d}"
                audit = _train_checkpoint(
                    kind=kind,
                    fold=fold,
                    seed=seed,
                    index=index,
                    checkpoint_path=fold_root / f"{kind}_best.pt",
                    history_path=fold_root / f"{kind}_history.csv",
                    max_epochs=max_epochs,
                    patience=patience,
                    batch_size=batch_size,
                    device=device,
                )
                training_rows.append(audit)
                print(json.dumps({"phase": "train", **audit}), flush=True)
    freeze = {
        "protocol": PROTOCOL,
        "checkpoint_count": len(training_rows),
        "seeds": list(seeds),
        "folds": list(folds),
        "checkpoint_selection": "minimum family-disjoint development BCE",
        "threshold_calibration_opened_before_freeze": False,
        "outer_test_opened_before_freeze": False,
        "checkpoints": training_rows,
    }
    freeze_path = output_root / "pre_outer_test_baseline_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    # Phase 2: calibration and outer evaluation only after the global freeze.
    event_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    for seed in seeds:
        event_rows.extend(
            _formal_reference_rows(
                seed=seed,
                pair_damp_root=pair_damp_root,
                index_root=index_root,
                folds=folds,
            )
        )
        for fold in folds:
            fold_rows, fold_thresholds, fold_localization = _score_fold(
                seed=seed,
                fold=fold,
                index_root=index_root,
                pair_damp_root=pair_damp_root,
                output_root=output_root,
                device=device,
            )
            event_rows.extend(fold_rows)
            threshold_rows.extend(fold_thresholds)
            localization_rows.extend(fold_localization)
            print(
                json.dumps(
                    {
                        "phase": "outer_evaluation",
                        "seed": seed,
                        "fold": fold,
                        "event_rows": len(fold_rows),
                    }
                ),
                flush=True,
            )

    seed_summary = _summarize_seed(event_rows)
    aggregate = _aggregate_seed_summary(seed_summary)
    intervals = _family_bootstrap_intervals(event_rows, bootstrap_replicates)
    paired = _paired_bootstrap(event_rows, bootstrap_replicates)
    localization_summary = _localization_summary(localization_rows, pair_damp_root)
    contracts = _method_contracts()
    readiness = _published_readiness()

    _write_csv(output_root / "all_seed_events.csv", event_rows)
    _write_csv(output_root / "all_seed_thresholds.csv", threshold_rows)
    _write_csv(output_root / "training_audit.csv", training_rows)
    _write_csv(output_root / "per_seed_summary.csv", seed_summary)
    _write_csv(output_root / "multi_seed_aggregate.csv", aggregate)
    _write_csv(output_root / "family_bootstrap_intervals.csv", intervals)
    _write_csv(output_root / "paired_family_bootstrap.csv", paired)
    _write_csv(output_root / "controlled_localization_events.csv", localization_rows)
    _write_csv(output_root / "localization_per_seed_summary.csv", localization_summary)
    _write_csv(output_root / "method_input_contracts.csv", contracts)
    (output_root / "published_method_readiness.json").write_text(
        json.dumps(readiness, indent=2), encoding="utf-8"
    )
    (output_root / "BASELINE_REPORT.md").write_text(
        _report(aggregate, intervals, paired), encoding="utf-8"
    )
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "seeds": list(seeds),
        "folds": list(folds),
        "positive_event_count_per_seed": 147,
        "hard_safe_event_count_per_seed": 273,
        "global_checkpoint_freeze": str(freeze_path),
        "method_contracts": contracts,
        "published_method_readiness": readiness,
        "aggregate": aggregate,
        "output_root": str(output_root),
    }
    (output_root / "baseline_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        index_root=args.index_root.expanduser().resolve(),
        pair_damp_root=args.pair_damp_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        folds=[int(value) for value in args.folds.split(",") if value.strip()],
        seeds=[int(value) for value in args.seeds.split(",") if value.strip()],
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
