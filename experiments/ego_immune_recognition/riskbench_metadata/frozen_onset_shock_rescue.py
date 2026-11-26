"""Port the fixed relation-onset shock branch to frozen exact-pair DAMP.

This is a post-freeze sensitivity analysis.  It preserves the sealed baseline
and applies the previously fixed q=0.999 onset rule without threshold search.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..temporal_relation_train_eval import _group_dual_frames_by_event
from .complete_structure_ablation import BACKBONES
from .core_hybrid_evaluation import event_alert, event_score, threshold_at_event_fpr
from .pair_aligned_damp_evaluation import (
    _aligned_trace,
    _attach_aligned_geometry,
    _attach_pair_activation,
    _attach_physical_percentile,
    _component_by_target,
    _load_or_score_pair_frames,
)
from .pair_damp_confirmation import (
    DEFAULT_INDEX_ROOT,
    DEFAULT_OUTPUT_ROOT as DEFAULT_CONFIRMATION_ROOT,
    _event_traces,
    _verify_frozen_inputs,
)
from .pair_damp_freeze import (
    DEFAULT_FREEZE_PATH,
    FOLD_COUNT,
    FPR_BUDGETS,
    FROZEN_METHOD,
    PROJECT_ROOT,
)
from .persistent_guarded_fusion import ALERT_PERCENTILE
from .relation_geometry_rescue import (
    _geometry_from_observations,
    _observations_by_frame,
)
from .threshold_advantage_eval import maximum_rolling_mean


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "ego_immune_riskbench_pair_damp_onset_shock_v2"
)
ONSET_QUANTILE = 0.999
ONSET_QUANTILES = (0.95, 0.975, 0.99, 0.995, 0.999)
MAX_ADDITIONAL_VALIDATION_FPR = 0.02
MIN_ONSET_LEAD_SECONDS = 0.5
POOLING_FRAMES = (1, 5, 10)
ONSET_RAW_KEY = "onset_shock_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=DEFAULT_CONFIRMATION_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _outgoing_components(frames: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        values
        for frame in frames
        for values in frame.get("relation_components", {}).values()
        if int(values.get("ego_outgoing", 0))
    ]


def _attach_onset_percentile(
    frames: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    reference = np.sort(
        np.asarray(
            [
                float(values[ONSET_RAW_KEY])
                for values in _outgoing_components(frames["validation_normal"])
                if float(values[ONSET_RAW_KEY]) > 0.0
            ],
            dtype=float,
        )
    )
    if not reference.size:
        raise ValueError("No positive validation-normal onset-shock values.")
    recovery_reference = np.sort(
        np.asarray(
            [
                float(values["recovery_raw"])
                for values in _outgoing_components(frames["validation_normal"])
                if float(values["recovery_raw"]) > 0.0
            ],
            dtype=float,
        )
    )
    if not recovery_reference.size:
        raise ValueError("No positive validation-normal mature recovery values.")
    for split_frames in frames.values():
        for frame in split_frames:
            for values in frame.get("relation_components", {}).values():
                raw = float(values.get(ONSET_RAW_KEY, 0.0))
                values["onset_shock_percentile"] = (
                    0.0
                    if raw <= 0.0
                    else float(
                        np.searchsorted(reference, raw, side="right")
                        / len(reference)
                    )
                )
                recovery_raw = float(values.get("recovery_raw", 0.0))
                values["mature_recovery_percentile"] = (
                    0.0
                    if recovery_raw <= 0.0
                    else float(
                        np.searchsorted(
                            recovery_reference,
                            recovery_raw,
                            side="right",
                        )
                        / len(recovery_reference)
                    )
                )
    return {
        "positive_reference_count": int(reference.size),
        "raw_q99": float(np.quantile(reference, 0.99)),
        "raw_q999": float(np.quantile(reference, 0.999)),
        "mature_recovery_reference_count": int(recovery_reference.size),
        "mature_recovery_raw_q99": float(
            np.quantile(recovery_reference, 0.99)
        ),
    }


def _augment_trace(
    event_frames: Sequence[dict[str, Any]],
    *,
    onset_quantile: float = ONSET_QUANTILE,
) -> list[tuple[dict[str, Any], float]]:
    """Add only exact ego-target onset shock to the frozen baseline trace."""
    output = []
    for frame, baseline_score in _aligned_trace(event_frames):
        onset_score = _frame_aligned_onset(frame)
        lead_seconds = float(
            frame.get("eval_frames_to_anchor", frame["frames_to_anchor"])
        ) * 0.05
        shock_gate = (
            1.0
            if onset_score >= onset_quantile
            and lead_seconds >= MIN_ONSET_LEAD_SECONDS
            else 0.0
        )
        output.append((frame, max(float(baseline_score), shock_gate)))
    return output


def _frame_aligned_onset(frame: dict[str, Any]) -> float:
    components = _component_by_target(frame)
    return max(
        (
            float(components[target_id]["onset_shock_percentile"])
            for target_id in frame.get("aligned_geometry_relations", {})
            if target_id in components
        ),
        default=0.0,
    )


def _max_aligned_onset_context(
    trace: Sequence[tuple[dict[str, Any], float]],
) -> dict[str, Any]:
    best = {
        "max_aligned_onset_percentile": 0.0,
        "onset_raw": 0.0,
        "onset_frame_position": "",
        "onset_lead_seconds": "",
        "onset_target_id": "",
        "onset_cpa_percentile": 0.0,
        "onset_pair_damp_percentile": 0.0,
        "onset_pair_fused_score": 0.0,
        "baseline_at_onset": 0.0,
        "baseline_pre3_max": 0.0,
        "baseline_post3_max": 0.0,
    }
    best_recovery = {
        "max_aligned_mature_recovery_percentile": 0.0,
        "mature_recovery_raw": 0.0,
        "mature_recovery_lead_seconds": "",
        "mature_recovery_cpa_percentile": 0.0,
        "mature_recovery_pair_fused_score": 0.0,
    }
    for trace_index, (frame, baseline_score) in enumerate(trace):
        components = _component_by_target(frame)
        for target_id, geometry in frame.get(
            "aligned_geometry_relations", {}
        ).items():
            component = components.get(target_id)
            if component is None:
                continue
            percentile = float(component["onset_shock_percentile"])
            if percentile > float(best["max_aligned_onset_percentile"]):
                best = {
                    "max_aligned_onset_percentile": percentile,
                    "onset_raw": float(component.get(ONSET_RAW_KEY, 0.0)),
                    "onset_frame_position": int(frame["frame_position"]),
                    "onset_lead_seconds": float(
                        frame.get(
                            "eval_frames_to_anchor",
                            frame["frames_to_anchor"],
                        )
                    )
                    * 0.05,
                    "onset_target_id": target_id,
                    "onset_cpa_percentile": float(geometry["cpa_only"]),
                    "onset_pair_damp_percentile": float(
                        geometry["pair_damp_percentile"]
                    ),
                    "onset_pair_fused_score": float(
                        geometry["pair_aligned_geomean"]
                    ),
                    "baseline_at_onset": float(baseline_score),
                    "baseline_pre3_max": max(
                        (
                            float(score)
                            for _, score in trace[
                                max(0, trace_index - 2): trace_index + 1
                            ]
                        ),
                        default=0.0,
                    ),
                    "baseline_post3_max": max(
                        (
                            float(score)
                            for _, score in trace[trace_index: trace_index + 3]
                        ),
                        default=0.0,
                    ),
                }
            recovery_percentile = float(
                component.get("mature_recovery_percentile", 0.0)
            )
            lead_seconds = float(
                frame.get("eval_frames_to_anchor", frame["frames_to_anchor"])
            ) * 0.05
            if (
                recovery_percentile
                > float(best_recovery["max_aligned_mature_recovery_percentile"])
                and lead_seconds >= MIN_ONSET_LEAD_SECONDS
            ):
                best_recovery = {
                    "max_aligned_mature_recovery_percentile": recovery_percentile,
                    "mature_recovery_raw": float(
                        component.get("recovery_raw", 0.0)
                    ),
                    "mature_recovery_lead_seconds": lead_seconds,
                    "mature_recovery_cpa_percentile": float(
                        geometry["cpa_only"]
                    ),
                    "mature_recovery_pair_fused_score": float(
                        geometry["pair_aligned_geomean"]
                    ),
                }
    best.update(best_recovery)
    return best


def _onset_traces(
    frames: dict[str, list[dict[str, Any]]],
    *,
    onset_quantile: float = ONSET_QUANTILE,
) -> dict[str, dict[str, list[tuple[dict[str, Any], float]]]]:
    return {
        split: {
            str(event_uid): _augment_trace(
                event_frames,
                onset_quantile=onset_quantile,
            )
            for event_uid, event_frames in _group_dual_frames_by_event(
                split_frames
            ).items()
        }
        for split, split_frames in frames.items()
    }


def _fold_rows(
    *,
    fold: int,
    index_root: Path,
    cache_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    backbone = str(FROZEN_METHOD["backbone"])
    frames = _load_or_score_pair_frames(
        backbone=backbone,
        fold=fold,
        index_root=index_root,
        cache_root=cache_root,
    )
    _attach_physical_percentile(frames)
    _attach_pair_activation(frames)
    onset_calibration = _attach_onset_percentile(frames)
    index_path = index_root / f"fold_{fold:02d}" / "riskbench_nested_fold_index.json"
    observations = _observations_by_frame(
        index_path,
        splits=(
            "validation_normal",
            "validation_positive",
            "test_normal",
            "test_positive",
        ),
    )
    geometry = _geometry_from_observations(
        observations,
        horizon_seconds=float(FROZEN_METHOD["cpa_horizon_seconds"]),
        clearance_m=float(FROZEN_METHOD["cpa_clearance_m"]),
    )
    alignment = _attach_aligned_geometry(frames, geometry=geometry)
    baseline = _event_traces(frames, method="frozen_pair_damp")
    ttc = _event_traces(frames, method="ttc")
    onset_by_quantile = {
        quantile: _onset_traces(frames, onset_quantile=quantile)
        for quantile in ONSET_QUANTILES
    }
    ttc_flags = {
        split: {
            uid: event_alert(trace, ALERT_PERCENTILE)[0]
            for uid, trace in events.items()
        }
        for split, events in ttc.items()
    }
    validation_ttc_negative = {
        uid
        for uid, detected in ttc_flags["validation_normal"].items()
        if not detected
    }
    grouped_frames = {
        split: {
            str(event_index): event_frames
            for event_index, event_frames in _group_dual_frames_by_event(
                split_frames
            ).items()
        }
        for split, split_frames in frames.items()
    }
    rows = []
    for budget in FPR_BUDGETS:
        threshold, inner_fpr = threshold_at_event_fpr(
            [
                event_score(baseline["validation_normal"][uid])
                for uid in validation_ttc_negative
            ],
            budget,
        )
        validation_baseline_detected = {
            uid: event_alert(baseline["validation_normal"][uid], threshold)[0]
            for uid in validation_ttc_negative
        }
        eligible_quantiles = []
        for quantile in ONSET_QUANTILES:
            added = sum(
                event_alert(
                    onset_by_quantile[quantile]["validation_normal"][uid],
                    threshold,
                )[0]
                and not validation_baseline_detected[uid]
                for uid in validation_ttc_negative
            )
            added_fpr = added / max(len(validation_ttc_negative), 1)
            if added_fpr <= MAX_ADDITIONAL_VALIDATION_FPR + 1e-12:
                eligible_quantiles.append((quantile, added_fpr))
        selected_quantile, validation_added_fpr = (
            min(eligible_quantiles, key=lambda item: item[0])
            if eligible_quantiles
            else (ONSET_QUANTILE, 0.0)
        )
        onset = onset_by_quantile[selected_quantile]
        for split, positive in (("test_positive", True), ("test_normal", False)):
            for uid, baseline_trace in baseline[split].items():
                trace_frames = [frame for frame, _ in baseline_trace]
                source_frames = grouped_frames[split][uid]
                event_uid = str(source_frames[0]["event_uid"])
                onset_context = _max_aligned_onset_context(baseline_trace)
                ttc_detected = bool(ttc_flags[split][uid])
                baseline_detected, baseline_lead = event_alert(
                    baseline_trace, threshold
                )
                onset_detected, onset_lead = event_alert(onset[split][uid], threshold)
                row = {
                        "fold": fold,
                        "fpr_budget": budget,
                        "event_uid": event_uid,
                        "fold_event_index": uid,
                        "split": split,
                        "positive": int(positive),
                        "threshold": threshold,
                        "inner_rescue_fpr": inner_fpr,
                        "selected_onset_quantile": selected_quantile,
                        "validation_onset_added_fpr": validation_added_fpr,
                        "ttc_detected": int(ttc_detected),
                        "baseline_detected": int(baseline_detected),
                        "onset_detected": int(onset_detected),
                        "baseline_blind_rescue": int(
                            not ttc_detected and baseline_detected
                        ),
                        "onset_blind_rescue": int(
                            not ttc_detected and onset_detected
                        ),
                        "onset_added": int(
                            not ttc_detected
                            and onset_detected
                            and not baseline_detected
                        ),
                        **onset_context,
                        "baseline_score": event_score(baseline_trace),
                        "onset_score": event_score(onset[split][uid]),
                        "baseline_lead_seconds": (
                            "" if baseline_lead is None else baseline_lead
                        ),
                        "onset_lead_seconds": "" if onset_lead is None else onset_lead,
                    }
                traces = {
                    "ttc": [float(score) for _, score in ttc[split][uid]],
                    "baseline": [float(score) for _, score in baseline_trace],
                    "onset": [float(score) for _, score in onset[split][uid]],
                }
                for width in POOLING_FRAMES:
                    for method, values in traces.items():
                        row[f"{method}_pool_{width}"] = maximum_rolling_mean(
                            values,
                            width,
                        )
                    row[f"onset_vs_ttc_lift_{width}"] = (
                        row[f"onset_pool_{width}"] - row[f"ttc_pool_{width}"]
                    )
                    row[f"baseline_vs_ttc_lift_{width}"] = (
                        row[f"baseline_pool_{width}"] - row[f"ttc_pool_{width}"]
                    )
                rows.append(row)
    return rows, {"fold": fold, **onset_calibration, **alignment}


def _summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for budget in FPR_BUDGETS:
        selected = [row for row in rows if float(row["fpr_budget"]) == budget]
        positive = [row for row in selected if int(row["positive"])]
        normal = [row for row in selected if not int(row["positive"])]
        blind_positive = [row for row in positive if not int(row["ttc_detected"])]
        blind_normal = [row for row in normal if not int(row["ttc_detected"])]
        output.append(
            {
                "fpr_budget": budget,
                "positive_event_count": len(positive),
                "hard_safe_event_count": len(normal),
                "ttc_detected": sum(int(row["ttc_detected"]) for row in positive),
                "baseline_final_detected": sum(
                    int(row["ttc_detected"] or row["baseline_blind_rescue"])
                    for row in positive
                ),
                "onset_final_detected": sum(
                    int(row["ttc_detected"] or row["onset_blind_rescue"])
                    for row in positive
                ),
                "ttc_blind_positive_count": len(blind_positive),
                "baseline_blind_rescue": sum(
                    int(row["baseline_blind_rescue"]) for row in blind_positive
                ),
                "onset_blind_rescue": sum(
                    int(row["onset_blind_rescue"]) for row in blind_positive
                ),
                "onset_added_positive": sum(
                    int(row["onset_added"]) for row in blind_positive
                ),
                "ttc_negative_hard_safe_count": len(blind_normal),
                "baseline_false_rescue": sum(
                    int(row["baseline_blind_rescue"]) for row in blind_normal
                ),
                "onset_false_rescue": sum(
                    int(row["onset_blind_rescue"]) for row in blind_normal
                ),
                "onset_added_hard_safe": sum(
                    int(row["onset_added"]) for row in blind_normal
                ),
            }
        )
    return output


def run(
    *,
    freeze_path: Path = DEFAULT_FREEZE_PATH,
    index_root: Path = DEFAULT_INDEX_ROOT,
    confirmation_root: Path = DEFAULT_CONFIRMATION_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    freeze_path = freeze_path.expanduser().resolve()
    index_root = index_root.expanduser().resolve()
    confirmation_root = confirmation_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    frozen = _verify_frozen_inputs(freeze_path, index_root)
    if str(FROZEN_METHOD["backbone"]) not in BACKBONES:
        raise ValueError(FROZEN_METHOD["backbone"])
    rows = []
    audits = []
    cache_root = confirmation_root / "pair_frame_cache"
    for fold in range(FOLD_COUNT):
        fold_rows, audit = _fold_rows(
            fold=fold,
            index_root=index_root,
            cache_root=cache_root,
        )
        rows.extend(fold_rows)
        audits.append(audit)
    summaries = _summary(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    events_path = output_root / "onset_shock_events.csv"
    table_path = output_root / "onset_shock_table.csv"
    _write_csv(events_path, rows)
    _write_csv(table_path, summaries)
    result = {
        "protocol": "riskbench-exact-pair-onset-shock-post-freeze-v2",
        "evidence_tier": "post_freeze_sensitivity_not_independent_confirmation",
        "source_freeze_receipt_sha256": frozen["freeze_receipt_sha256"],
        "model_retrained": False,
        "baseline_modified": False,
        "onset_definition": (
            "recovery-shaped residual/damage drop within the first three valid "
            "predictions, restricted to exact ego-target CPA-aligned relations"
        ),
        "onset_quantile_candidates": list(ONSET_QUANTILES),
        "onset_quantile_selection": (
            "least strict candidate with <=2 percentage points additional "
            "event FPR on validation-normal TTC-negative events"
        ),
        "max_additional_validation_fpr": MAX_ADDITIONAL_VALIDATION_FPR,
        "minimum_onset_lead_seconds": MIN_ONSET_LEAD_SECONDS,
        "threshold_policy": "reuse_frozen_baseline_threshold_per_fold_and_budget",
        "summaries": summaries,
        "fold_audits": audits,
        "artifacts": {"events": str(events_path), "table": str(table_path)},
        "status": "exact_pair_onset_shock_sensitivity_complete",
    }
    summary_path = output_root / "onset_shock_summary.json"
    result["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(
                freeze_path=args.freeze_path,
                index_root=args.index_root,
                confirmation_root=args.confirmation_root,
                output_root=args.output_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
