"""Outer RiskBench evaluation of safely retrained Social-GAN checkpoints.

The published Social-GAN architecture is retrained separately in every fold on
safe-core families.  This script opens the outer events only after those five
checkpoints are fixed.  Predicted trajectories use the same soft center-
encounter risk function and matched hard-safe calibration as the CV/CA rollout
baselines, isolating the contribution of the learned motion predictor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .core_hybrid_evaluation import _usable_frames, event_alert, event_score, threshold_at_event_fpr
from .mainstream_baseline_evaluation import _load_state
from .mainstream_baseline_evaluation_v2 import (
    DEFAULT_FPR_BUDGETS, DEFAULT_INDEX_ROOT, DEFAULT_PAIR_DAMP_ROOT, DEFAULT_SEEDS,
    _aggregate_seed_summary, _family_bootstrap_intervals, _formal_reference_rows,
    _load_index, _load_pair_frames, _localization_summary, _paired_bootstrap,
    _record_map, _summarize_seed, _write_csv,
)
from .pair_aligned_damp_evaluation import _attach_physical_percentile
from .persistent_guarded_fusion import ALERT_PERCENTILE
from .probabilistic_rollout_evaluation import shared_soft_encounter_score
from .social_gan_clean_retrain import OBS_LEN, PRED_LEN, build_models, relative_to_absolute


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "results/riskbench_social_gan_verified_v2"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/riskbench_social_gan_verified_evaluation_v2"
PROTOCOL = "riskbench-social-gan-safe-only-clean-evaluation-v2"
METHOD = "social_gan_safe_only_retrained_shared_risk"
SAMPLES = 20
MAX_ACTORS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--pair-damp-root", type=Path, default=DEFAULT_PAIR_DAMP_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _history_scene(
    state: dict[int, dict[str, np.ndarray]], endpoint: int
) -> tuple[np.ndarray, list[str]] | None:
    frames = list(range(endpoint - OBS_LEN + 1, endpoint + 1))
    if any(frame not in state for frame in frames):
        return None
    common = set(str(actor) for actor in state[frames[0]])
    for frame in frames[1:]:
        common.intersection_update(str(actor) for actor in state[frame])
    current = state[endpoint]
    actors = sorted(
        common,
        key=lambda actor: (float(np.linalg.norm(current[actor])), actor),
    )[:MAX_ACTORS]
    if not actors:
        return None
    values = np.zeros((OBS_LEN, 1 + len(actors), 2), dtype=np.float32)
    for actor_index, actor in enumerate(actors, start=1):
        values[:, actor_index] = np.stack(
            [np.asarray(state[frame][actor], dtype=np.float32) for frame in frames]
        )
    return values, actors


def _stable_seed(fold: int, scenario_id: str, endpoint: int) -> int:
    digest = hashlib.sha256(f"{fold}:{scenario_id}:{endpoint}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


@torch.no_grad()
def frame_actor_scores(
    generator: torch.nn.Module,
    state: dict[int, dict[str, np.ndarray]],
    endpoint: int,
    *,
    fold: int,
    scenario_id: str,
    samples: int,
    device: torch.device,
) -> dict[str, float]:
    scene = _history_scene(state, endpoint)
    if scene is None:
        return {}
    values, actors = scene
    one_obs = torch.from_numpy(values).to(device)
    actor_count = one_obs.shape[1]
    # Each repeated block is an independent social scene. Social-GAN assigns
    # one independent global noise vector to every block, so this vectorizes
    # multimodal sampling without coupling actors from different samples.
    obs = one_obs.repeat(1, samples, 1)
    obs_rel = torch.zeros_like(obs)
    obs_rel[1:] = obs[1:] - obs[:-1]
    seq_start_end = torch.tensor(
        [[sample * actor_count, (sample + 1) * actor_count] for sample in range(samples)],
        device=device,
    )
    torch.manual_seed(_stable_seed(fold, scenario_id, endpoint))
    generator.eval()
    predicted_rel = generator(obs, obs_rel, seq_start_end)
    predicted = relative_to_absolute(predicted_rel, obs[-1])
    predicted = predicted.reshape(PRED_LEN, samples, actor_count, 2)
    relative_to_ego = predicted[:, :, 1:]  # Already ego-relative; synthetic origin fixed at zero.
    array = relative_to_ego.permute(1, 0, 2, 3).detach().cpu().numpy()
    weights = np.full(samples, 1.0 / samples, dtype=np.float64)
    return {
        actor: shared_soft_encounter_score(array[:, :, actor_index], weights)
        for actor_index, actor in enumerate(actors)
    }


def _rank(scores: dict[str, float], targets: Sequence[str]) -> int | None:
    present = [target for target in targets if target in scores]
    if not present:
        return None
    target_score = max(scores[target] for target in present)
    return 1 + sum(score > target_score for score in scores.values())


def _event_groups(pair_frames: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for split, frames in pair_frames.items():
        output[split] = {}
        for event_frames in _group_dual_frames_by_event(frames).values():
            if event_frames:
                output[split][str(event_frames[0]["event_uid"])] = _usable_frames(event_frames)
    return output


def load_generator(checkpoint_root: Path, fold: int, device: torch.device) -> torch.nn.Module:
    checkpoint_path = checkpoint_root / f"fold_{fold:02d}/generator_safe_selection_best.pt"
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator, _ = build_models(device)
    generator.load_state_dict(payload["state_dict"])
    generator.eval()
    return generator


def score_fold(
    *, fold: int, index_root: Path, pair_damp_root: Path,
    checkpoint_root: Path, samples: int, device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    index = _load_index(index_root, fold)
    records = _record_map(index)
    pair_frames = _load_pair_frames(pair_damp_root, DEFAULT_SEEDS[0], fold)
    _attach_physical_percentile(pair_frames)
    groups = _event_groups(pair_frames)
    generator = load_generator(checkpoint_root, fold, device)
    evaluation_records = [
        record for split in ("validation_normal", "test_normal", "test_positive")
        for record in index["splits"][split]
    ]
    states = {str(record["scenario_id"]): _load_state(record["state_path"]) for record in evaluation_records}
    traces: dict[str, dict[str, list[tuple[dict[str, Any], float]]]] = defaultdict(dict)
    actor_traces: dict[str, dict[str, list[tuple[dict[str, Any], dict[str, float]]]]] = defaultdict(dict)
    ttc_traces: dict[str, dict[str, list[tuple[dict[str, Any], float]]]] = defaultdict(dict)
    for split in ("validation_normal", "test_normal", "test_positive"):
        for uid, frames in groups[split].items():
            record = records[uid]
            state = states[str(record["scenario_id"])]
            traces[split][uid] = []
            actor_traces[split][uid] = []
            ttc_traces[split][uid] = []
            for frame in frames:
                endpoint = int(frame["source_frame_index"])
                scores = frame_actor_scores(
                    generator, state, endpoint, fold=fold,
                    scenario_id=str(record["scenario_id"]), samples=samples, device=device,
                )
                actor_traces[split][uid].append((frame, scores))
                traces[split][uid].append((frame, max(scores.values(), default=0.0)))
                ttc_traces[split][uid].append((frame, float(frame["aggregations"]["physical_percentile"])))
    ttc_flags = {
        split: {uid: event_alert(trace, ALERT_PERCENTILE)[0] for uid, trace in values.items()}
        for split, values in ttc_traces.items()
    }
    base_rows: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    localization: list[dict[str, Any]] = []
    for regime, budget in [(r, b) for r in ('complementary', 'standalone') for b in DEFAULT_FPR_BUDGETS]:
        calibration_uids = [uid for uid in traces["validation_normal"] if regime == "standalone" or not ttc_flags["validation_normal"][uid]]
        threshold, observed_fpr = threshold_at_event_fpr(
            [event_score(traces["validation_normal"][uid]) for uid in calibration_uids], budget
        )
        thresholds.append({
            "fold": fold, "method": METHOD, "fpr_budget": budget, "regime": regime,
            "threshold": threshold, "validation_ttc_negative_count": len(calibration_uids),
            "validation_incremental_fpr": observed_fpr,
        })
        for split in ("test_normal", "test_positive"):
            for uid, trace in traces[split].items():
                record = records[uid]
                detected, lead = event_alert(trace, threshold)
                ttc_detected, ttc_lead = event_alert(ttc_traces[split][uid], ALERT_PERCENTILE)
                base_rows.append({
                    "fold": fold, "method": METHOD, "fpr_budget": budget, "regime": regime,
                    "event_uid": uid, "scenario_group_id": record["scenario_group_id"],
                    "category": record["category"], "scenario_id": record["scenario_id"],
                    "target_type": record.get("target_type_name", "none"), "split": split,
                    "positive": int(split == "test_positive"), "score": event_score(trace),
                    "threshold": threshold, "detected": int(detected),
                    "observed_score_frames": sum(bool(x) for _, x in actor_traces[split][uid]),
                    "target_history_available": int(any(str(t) in x for _, x in actor_traces[split][uid] for t in record.get("risk_actor_ids", []))),
                    "lead_seconds": "" if lead is None else lead,
                    "ttc_score": event_score(ttc_traces[split][uid]),
                    "ttc_detected": int(ttc_detected),
                    "ttc_lead_seconds": "" if ttc_lead is None else ttc_lead,
                    "final_detected": int(detected if regime == "standalone" else (ttc_detected or detected)),
                    "blind_rescue": int((not ttc_detected) and detected),
                })
        for uid, frames in actor_traces["test_positive"].items():
            record = records[uid]
            targets = [str(value) for value in record.get("risk_actor_ids", [])]
            aggregate: dict[str, float] = {}
            for _, scores in frames:
                for actor, score in scores.items():
                    aggregate[actor] = max(aggregate.get(actor, 0.0), score)
            event_rank = _rank(aggregate, targets)
            alert_rank = None; alert_lead = None; alerted = False
            for (frame, frame_score), (_, actor_scores) in zip(traces["test_positive"][uid], frames):
                if frame_score >= threshold:
                    alerted = True; alert_rank = _rank(actor_scores, targets)
                    alert_lead = int(frame.get("eval_frames_to_anchor", frame["frames_to_anchor"])) * 0.05
                    break
            localization.append({
                "fold": fold, "method": METHOD, "fpr_budget": budget, "regime": regime,
                "event_uid": uid, "target_type": record.get("target_type_name", "unknown"),
                "target_ids": "|".join(targets),
                "event_target_rank": "" if event_rank is None else event_rank,
                "event_top1": int(event_rank is not None and event_rank <= 1),
                "event_top3": int(event_rank is not None and event_rank <= 3),
                "method_alerted": int(alerted),
                "alert_target_rank": "" if alert_rank is None else alert_rank,
                "alert_top1": int(alert_rank is not None and alert_rank <= 1),
                "alert_top3": int(alert_rank is not None and alert_rank <= 3),
                "alert_lead_seconds": "" if alert_lead is None else alert_lead,
            })
    return base_rows, thresholds, localization



if __name__ == "__main__":
    raise SystemExit("Use social_gan_verified_evaluation_runner for all 15 checkpoints; this module exposes score_fold only.")
