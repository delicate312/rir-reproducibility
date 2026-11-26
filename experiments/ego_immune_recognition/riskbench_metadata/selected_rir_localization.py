"""Recompute localization for the nested-selected final RIR decision."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .core_hybrid_evaluation import _usable_frames
from .damp_decision_necessity_ablation import (
    DEFAULT_INDEX_ROOT,
    DEFAULT_SELECTION_ROOT,
    DEFAULT_SOURCE_ROOT,
    _activation_fn,
    _activation_reference,
    _calibration,
    _cpa_reference,
    _load_frames,
    _merged_components,
    _selected_candidate,
)
from .damp_hard_safe_grid_selection import (
    PAIR_SUPPORT_FRAMES,
    SEVERE_CPA_PERCENTILE,
    _event_frames,
    _recompute_components,
)
from .pair_aligned_damp_evaluation import _percentile
from .pair_damp_freeze import PROJECT_ROOT
from .relation_geometry_rescue import (
    _geometry_from_observations,
    _observations_by_frame,
)


DEFAULT_THRESHOLDS = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_damp_necessity_v2_nested"
    / "fold_thresholds.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_selected_rir_localization_v2"
)
SEEDS = (20260812, 20260813, 20260814)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metadata(index_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        f"riskbench:{row['category']}:{row['scenario_id']}": {
            "target_ids": [str(value) for value in row["risk_actor_ids"]],
            "target_type": str(row.get("target_type_name", "unknown")),
        }
        for row in payload["splits"]["test_positive"]
    }


def _rank(scores: dict[str, float], targets: Sequence[str]) -> int | None:
    visible = [target for target in targets if target in scores]
    if not visible:
        return None
    target_score = max(float(scores[target]) for target in visible)
    return 1 + sum(float(value) > target_score for value in scores.values())


def _thresholds(path: Path) -> dict[tuple[int, int], float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (int(row["seed"]), int(row["fold"])): float(row["threshold"])
            for row in csv.DictReader(handle)
            if row["method"] == "rir_nested_selected"
        }


def _fold_rows(
    *, seed: int, fold: int, threshold: float
) -> list[dict[str, Any]]:
    frames_by_split = _load_frames(DEFAULT_SOURCE_ROOT, seed, fold)
    events = {
        split: _event_frames(frames_by_split[split])
        for split in ("validation_normal", "test_positive")
    }
    candidate = _selected_candidate(
        DEFAULT_SELECTION_ROOT, seed=seed, fold=fold
    )
    components = {
        split: _merged_components(
            split_events,
            _recompute_components(
                split_events,
                persistence_decay=candidate.persistence_decay,
                recovery_warmup=candidate.recovery_warmup,
            ),
        )
        for split, split_events in events.items()
    }
    calibration = _calibration(
        events["validation_normal"], components["validation_normal"]
    )
    activation = _activation_fn(
        "rir_nested_selected", calibration, candidate
    )
    activation_reference = _activation_reference(
        components["validation_normal"], activation
    )
    index_path = (
        DEFAULT_INDEX_ROOT
        / f"fold_{fold:02d}"
        / "riskbench_nested_fold_index.json"
    )
    observations = _observations_by_frame(
        index_path,
        splits=("validation_normal", "test_positive"),
    )
    geometry = _geometry_from_observations(
        observations, horizon_seconds=2.0, clearance_m=3.5
    )
    cpa_reference = _cpa_reference(events["validation_normal"], geometry)
    metadata = _metadata(index_path)
    output: list[dict[str, Any]] = []
    for uid, event_frames in events["test_positive"].items():
        histories: dict[str, list[tuple[int, float]]] = defaultdict(list)
        event_scores: dict[str, float] = {}
        first_alert_scores: dict[str, float] | None = None
        first_alert_lead: float | None = None
        for frame in _usable_frames(event_frames):
            position = int(frame["frame_position"])
            candidates: dict[str, float] = {}
            for target, geo in geometry.get((uid, position), {}).items():
                values = components["test_positive"][uid].get(
                    position, {}
                ).get(str(target))
                if values is None:
                    continue
                damp = _percentile(
                    activation_reference, activation(values)
                )
                cpa = _percentile(cpa_reference, float(geo["raw"]))
                aligned = math.sqrt(max(damp * cpa, 0.0))
                history = histories[str(target)]
                if history and position != history[-1][0] + 1:
                    history.clear()
                history.append((position, aligned))
                history[:] = history[-PAIR_SUPPORT_FRAMES:]
                if cpa >= SEVERE_CPA_PERCENTILE:
                    score = cpa
                elif len(history) == PAIR_SUPPORT_FRAMES:
                    score = float(np.mean([value for _, value in history]))
                else:
                    score = 0.0
                candidates[str(target)] = score
                event_scores[str(target)] = max(
                    event_scores.get(str(target), 0.0), score
                )
            if (
                first_alert_scores is None
                and max(candidates.values(), default=0.0) >= threshold
            ):
                first_alert_scores = candidates
                first_alert_lead = float(
                    frame.get(
                        "eval_frames_to_anchor", frame["frames_to_anchor"]
                    )
                ) * 0.05
        targets = metadata[uid]["target_ids"]
        event_rank = _rank(event_scores, targets)
        alert_rank = (
            None
            if first_alert_scores is None
            else _rank(first_alert_scores, targets)
        )
        output.append(
            {
                "seed": seed,
                "fold": fold,
                "event_uid": uid,
                "target_type": metadata[uid]["target_type"],
                "target_ids": "|".join(targets),
                "threshold": threshold,
                "localization_evaluable": int(event_rank is not None),
                "event_rank": "" if event_rank is None else event_rank,
                "event_top1": int(event_rank is not None and event_rank <= 1),
                "event_top3": int(event_rank is not None and event_rank <= 3),
                "relation_alerted": int(first_alert_scores is not None),
                "first_alert_rank": "" if alert_rank is None else alert_rank,
                "first_alert_top1": int(alert_rank is not None and alert_rank <= 1),
                "first_alert_top3": int(alert_rank is not None and alert_rank <= 3),
                "first_alert_lead_seconds": (
                    "" if first_alert_lead is None else first_alert_lead
                ),
            }
        )
    return output


def run(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    thresholds = _thresholds(DEFAULT_THRESHOLDS)
    rows = [
        row
        for seed in SEEDS
        for fold in range(5)
        for row in _fold_rows(
            seed=seed, fold=fold, threshold=thresholds[(seed, fold)]
        )
    ]
    per_seed = []
    for seed in SEEDS:
        selected = [row for row in rows if int(row["seed"]) == seed]
        eligible = [row for row in selected if int(row["localization_evaluable"])]
        alerted = [row for row in selected if int(row["relation_alerted"])]
        per_seed.append(
            {
                "seed": seed,
                "positive_events": len(selected),
                "evaluable_events": len(eligible),
                "relation_alerted_events": len(alerted),
                "event_top1": np.mean([row["event_top1"] for row in eligible]),
                "event_top3": np.mean([row["event_top3"] for row in eligible]),
                "first_alert_top1": np.mean(
                    [row["first_alert_top1"] for row in alerted]
                ),
                "first_alert_top3": np.mean(
                    [row["first_alert_top3"] for row in alerted]
                ),
            }
        )
    aggregate: dict[str, Any] = {
        "protocol": "nested-selected-rir-localization-v2",
        "seeds": list(SEEDS),
    }
    for metric in (
        "evaluable_events",
        "relation_alerted_events",
        "event_top1",
        "event_top3",
        "first_alert_top1",
        "first_alert_top3",
    ):
        values = np.asarray([float(row[metric]) for row in per_seed])
        aggregate[f"{metric}_mean"] = float(np.mean(values))
        aggregate[f"{metric}_std"] = float(np.std(values, ddof=1))
    _write_csv(output_root / "localization_events.csv", rows)
    _write_csv(output_root / "localization_per_seed.csv", per_seed)
    (output_root / "localization_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
