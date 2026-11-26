"""Calibrate a causal safe-response inhibitor for relaxed CPA rescue."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .complete_structure_ablation import DEFAULT_CACHE_ROOT
from .family_grouped_nested_cv import DEFAULT_INDEX_ROOT
from .relation_geometry_rescue import (
    DEFAULT_RESCUE_EVENT_FPR_BUDGET,
    _aggregate,
    _attach_relation_signals,
    _fold_frames,
    _geometry_from_observations,
    _observations_by_frame,
    _rescue_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_safe_response_rescue_v1"
)
HORIZONS_SECONDS = (2.0, 3.0)
CLEARANCES_M = (2.5, 3.5)
SIGNALS = ("cpa_only", "cpa_relation_geomean", "cpa_immune_geomean")
THRESHOLD_QUANTILES = (
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.97,
    0.98,
    0.99,
    0.995,
)
SUPPORT_RULES = ((1, 1), (2, 3), (2, 5), (3, 5))
MAX_RECOVERY_DROPS = (0.02, 0.05, 0.10, 0.20)
MINIMUM_CURRENT_DELTAS = (-0.05, -0.02, 0.0)


@dataclass(frozen=True)
class SafeResponseConfig:
    horizon_seconds: float
    clearance_m: float
    signal: str
    threshold_quantile: float
    score_threshold: float
    support_frames: int
    window_frames: int
    minimum_rise: float
    max_recovery_drop: float
    minimum_current_delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--rescue-event-fpr-budget",
        type=float,
        default=DEFAULT_RESCUE_EVENT_FPR_BUDGET,
    )
    parser.add_argument("--aggregate-existing", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_configs(
    frames: dict[str, list[dict[str, Any]]],
    *,
    horizon_seconds: float,
    clearance_m: float,
) -> list[SafeResponseConfig]:
    output = []
    for signal in SIGNALS:
        reference = np.asarray(
            [
                float(values[signal])
                for frame in frames["validation_normal"]
                for values in frame.get("geometry_relations", {}).values()
            ],
            dtype=float,
        )
        for quantile in THRESHOLD_QUANTILES:
            threshold = float(
                np.quantile(reference, quantile, method="higher")
            )
            for support, window in SUPPORT_RULES:
                for max_drop in MAX_RECOVERY_DROPS:
                    for minimum_delta in MINIMUM_CURRENT_DELTAS:
                        output.append(
                            SafeResponseConfig(
                                horizon_seconds=horizon_seconds,
                                clearance_m=clearance_m,
                                signal=signal,
                                threshold_quantile=quantile,
                                score_threshold=threshold,
                                support_frames=support,
                                window_frames=window,
                                minimum_rise=0.0,
                                max_recovery_drop=max_drop,
                                minimum_current_delta=minimum_delta,
                            )
                        )
    return output


def _selection_key(
    positive: dict[str, Any],
    normal: dict[str, Any],
    config: SafeResponseConfig,
) -> tuple[Any, ...]:
    return (
        int(positive["blind_rescue_count"]),
        -float(normal["blind_rescue_rate"]),
        float(config.threshold_quantile),
        -config.support_frames,
        -config.max_recovery_drop,
    )


def run_fold(
    *,
    fold: int,
    index_root: Path,
    cache_root: Path,
    output_root: Path,
    rescue_event_fpr_budget: float,
) -> dict[str, Any]:
    frames = _fold_frames(cache_root, fold)
    index_path = (
        index_root
        / f"fold_{fold:02d}"
        / "riskbench_nested_fold_index.json"
    )
    observations = _observations_by_frame(index_path, splits=tuple(frames))
    eligible = []
    candidates = []
    for horizon_seconds in HORIZONS_SECONDS:
        for clearance_m in CLEARANCES_M:
            geometry = _geometry_from_observations(
                observations,
                horizon_seconds=horizon_seconds,
                clearance_m=clearance_m,
            )
            _attach_relation_signals(frames, geometry=geometry)
            for config in _candidate_configs(
                frames,
                horizon_seconds=horizon_seconds,
                clearance_m=clearance_m,
            ):
                normal = _rescue_metrics(
                    frames["validation_normal"],
                    config=config,
                    positive=False,
                )
                positive = _rescue_metrics(
                    frames["validation_positive"],
                    config=config,
                    positive=True,
                )
                is_eligible = (
                    normal["blind_rescue_rate"]
                    <= rescue_event_fpr_budget + 1e-12
                )
                candidates.append(
                    {
                        "fold": fold,
                        **asdict(config),
                        "eligible": int(is_eligible),
                        "normal_rescue_event_fpr": normal[
                            "blind_rescue_rate"
                        ],
                        "positive_blind_rescues": positive[
                            "blind_rescue_count"
                        ],
                        "positive_ttc_blind_count": positive[
                            "ttc_negative_event_count"
                        ],
                    }
                )
                if is_eligible:
                    eligible.append((config, normal, positive))
    selected, inner_normal, inner_positive = max(
        eligible,
        key=lambda item: _selection_key(item[2], item[1], item[0]),
    )
    geometry = _geometry_from_observations(
        observations,
        horizon_seconds=selected.horizon_seconds,
        clearance_m=selected.clearance_m,
    )
    _attach_relation_signals(frames, geometry=geometry)
    outer_normal = _rescue_metrics(
        frames["test_normal"],
        config=selected,
        positive=False,
    )
    outer_positive = _rescue_metrics(
        frames["test_positive"],
        config=selected,
        positive=True,
    )
    fold_root = output_root / f"fold_{fold:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)
    _write_csv(fold_root / "safe_response_candidates.csv", candidates)
    _write_csv(
        fold_root / "safe_response_outer_events.csv",
        [
            {"fold": fold, **row}
            for row in outer_positive["events"] + outer_normal["events"]
        ],
    )
    result = {
        "fold": fold,
        "selected_config": asdict(selected),
        "inner_normal": {
            key: value for key, value in inner_normal.items() if key != "events"
        },
        "inner_positive": {
            key: value for key, value in inner_positive.items() if key != "events"
        },
        "outer_normal": {
            key: value for key, value in outer_normal.items() if key != "events"
        },
        "outer_positive": {
            key: value for key, value in outer_positive.items() if key != "events"
        },
    }
    path = fold_root / "safe_response_summary.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(path)
    return result


def run(
    *,
    folds: Sequence[int],
    index_root: Path = DEFAULT_INDEX_ROOT,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    rescue_event_fpr_budget: float = DEFAULT_RESCUE_EVENT_FPR_BUDGET,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        run_fold(
            fold=fold,
            index_root=index_root.expanduser().resolve(),
            cache_root=cache_root.expanduser().resolve(),
            output_root=output_root,
            rescue_event_fpr_budget=rescue_event_fpr_budget,
        )
        for fold in folds
    ]
    result = {
        "protocol": "riskbench-safe-response-rescue-exploratory-v1",
        "folds": list(folds),
        "ttc_preserved_by_construction": True,
        "safe_training_source": "inner hard-safe temporal morphology",
        "positive_selection_status": (
            "inner validation positives used for exploratory configuration "
            "selection; freeze before a new holdout"
        ),
        "inner_rescue_event_fpr_budget": rescue_event_fpr_budget,
        "aggregate": _aggregate(results),
        "fold_summaries": [result["summary_path"] for result in results],
    }
    path = output_root / "safe_response_rescue_summary.json"
    result["summary_path"] = str(path)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    if args.aggregate_existing:
        output_root = args.output_root.expanduser().resolve()
        fold_results = [
            json.loads(
                (
                    output_root
                    / f"fold_{fold:02d}"
                    / "safe_response_summary.json"
                ).read_text(encoding="utf-8")
            )
            for fold in folds
        ]
        result = {
            "protocol": "riskbench-safe-response-rescue-exploratory-v1",
            "folds": folds,
            "ttc_preserved_by_construction": True,
            "safe_training_source": "inner hard-safe temporal morphology",
            "positive_selection_status": (
                "inner validation positives used for exploratory "
                "configuration selection; freeze before a new holdout"
            ),
            "inner_rescue_event_fpr_budget": args.rescue_event_fpr_budget,
            "aggregate": _aggregate(fold_results),
            "fold_summaries": [
                str(
                    output_root
                    / f"fold_{fold:02d}"
                    / "safe_response_summary.json"
                )
                for fold in folds
            ],
        }
        path = output_root / "safe_response_rescue_summary.json"
        result["summary_path"] = str(path)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = run(
            folds=folds,
            index_root=args.index_root,
            cache_root=args.cache_root,
            output_root=args.output_root,
            rescue_event_fpr_budget=args.rescue_event_fpr_budget,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
