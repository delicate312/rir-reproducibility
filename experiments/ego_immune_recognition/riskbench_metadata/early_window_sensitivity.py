"""Compare 1.0 s and 1.5 s early-alert windows without retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..danger_signal.diagnostics import false_alarm_morphology
from ..temporal_relation_train_eval import (
    _evaluate_dual_track,
    _group_dual_frames_by_event,
)
from .adapter import RISK_BENCH_TIMESTEP_SECONDS
from .onset_shock_ablation import (
    BASELINE_SCORE,
    DEFAULT_AUDIT_CSV,
    DEFAULT_FRAME_PATH,
    ONSET_RAW_SCORE,
    _load_frame_rows,
    _positive_percentile,
    _positive_reference,
)
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_early_window_sensitivity_v1"
)
WINDOWS = {"primary_1s": 20, "extended_1p5s": 30}
ONSET_QUANTILES = (0.90, 0.95, 0.999)
METHODS = {
    "ttc": "physical_percentile",
    "relation": BASELINE_SCORE,
    "relation_plus_onset_q900": "window_onset_q900",
    "relation_plus_onset_q950": "window_onset_q950",
    "relation_plus_onset_q999": "window_onset_q999",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-path", type=Path, default=DEFAULT_FRAME_PATH)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().resolve().open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _onset_score_name(quantile: float) -> str:
    return f"window_onset_q{int(round(quantile * 1000)):03d}"


def _attach_window_scores(
    frames_by_split: dict[str, list[dict[str, Any]]],
    onset_reference: np.ndarray,
) -> None:
    for frames in frames_by_split.values():
        for frame in frames:
            scores = frame["aggregations"]
            onset_percentile = _positive_percentile(
                onset_reference,
                float(scores[ONSET_RAW_SCORE]),
            )
            scores["window_onset_positive_percentile"] = onset_percentile
            for quantile in ONSET_QUANTILES:
                onset_gate = float(onset_percentile >= quantile)
                scores[_onset_score_name(quantile)] = max(
                    float(scores[BASELINE_SCORE]),
                    onset_gate,
                )


def _positive_event_detected(
    event_frames: Sequence[dict[str, Any]],
    score_key: str,
    lead_frames: int,
) -> bool:
    return any(
        0 < int(frame["frames_to_anchor"]) <= lead_frames
        and float(frame["aggregations"][score_key]) >= ALERT_PERCENTILE
        for frame in event_frames
    )


def _normal_event_detected(
    event_frames: Sequence[dict[str, Any]],
    score_key: str,
) -> bool:
    return any(
        float(frame["aggregations"][score_key]) >= ALERT_PERCENTILE
        for frame in event_frames
    )


def _max_in_window(
    event_frames: Sequence[dict[str, Any]],
    score_key: str,
    lead_frames: int,
) -> tuple[float, int | None]:
    usable = [
        frame
        for frame in event_frames
        if 0 < int(frame["frames_to_anchor"]) <= lead_frames
    ]
    if not usable:
        return 0.0, None
    best = max(
        usable,
        key=lambda frame: float(frame["aggregations"][score_key]),
    )
    return (
        float(best["aggregations"][score_key]),
        int(best["frames_to_anchor"]),
    )


def _metric_rows(
    frames_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for split in (
        "development_validation",
        "development_test",
        "confirmation",
    ):
        normal = frames_by_split[f"{split}_normal"]
        positive = frames_by_split[f"{split}_positive"]
        for method, score_key in METHODS.items():
            morphology = false_alarm_morphology(
                normal,
                score_key=score_key,
                threshold=ALERT_PERCENTILE,
            )
            for window, lead_frames in WINDOWS.items():
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "score_key": score_key,
                        "window": window,
                        "lead_frames": lead_frames,
                        "lead_seconds": (
                            lead_frames * RISK_BENCH_TIMESTEP_SECONDS
                        ),
                        **_evaluate_dual_track(
                            normal_frames=normal,
                            positive_frames=positive,
                            aggregation=score_key,
                            label_window=f"pre_anchor_{lead_frames}",
                            threshold=ALERT_PERCENTILE,
                            timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
                        ),
                        "normal_false_alarm_event_count": int(
                            round(
                                morphology["event_count"]
                                * morphology["event_false_alarm_rate"]
                            )
                        ),
                        "normal_false_alarm_event_rate": morphology[
                            "event_false_alarm_rate"
                        ],
                    }
                )
    return rows


def _event_rows(
    frames_by_split: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for split, frames in frames_by_split.items():
        positive = split.endswith("_positive")
        for event_frames in _group_dual_frames_by_event(frames).values():
            row: dict[str, Any] = {
                "split": split,
                "event_uid": str(event_frames[0]["event_uid"]),
                "positive": int(positive),
            }
            for method, score_key in METHODS.items():
                if positive:
                    for window, lead_frames in WINDOWS.items():
                        detected = _positive_event_detected(
                            event_frames,
                            score_key,
                            lead_frames,
                        )
                        maximum, maximum_fta = _max_in_window(
                            event_frames,
                            score_key,
                            lead_frames,
                        )
                        row[f"{method}_{window}_detected"] = int(detected)
                        row[f"{method}_{window}_max"] = maximum
                        row[f"{method}_{window}_max_fta"] = (
                            "" if maximum_fta is None else maximum_fta
                        )
                    row[f"{method}_extended_only"] = int(
                        row[f"{method}_extended_1p5s_detected"]
                        and not row[f"{method}_primary_1s_detected"]
                    )
                else:
                    row[f"{method}_normal_detected"] = int(
                        _normal_event_detected(event_frames, score_key)
                    )
            rows.append(row)
    return rows


def _audit_rows(
    events: Sequence[dict[str, Any]],
    audit_csv: Path,
) -> list[dict[str, Any]]:
    event_lookup = {
        (str(row["split"]), str(row["event_uid"])): row
        for row in events
    }
    output = []
    for audit in _read_csv(audit_csv):
        event = event_lookup[(audit["split"], audit["event_uid"])]
        output.append(
            {
                "audit_index": int(audit["audit_index"]),
                "split": audit["split"],
                "event_uid": audit["event_uid"],
                "failure_category": audit.get("failure_category", ""),
                "research_action": audit.get("research_action", ""),
                **{
                    key: value
                    for key, value in event.items()
                    if key not in {"split", "event_uid", "positive"}
                },
            }
        )
    return output


def _split_summary(
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in (
        "development_validation",
        "development_test",
        "confirmation",
    ):
        positives = [
            row
            for row in events
            if row["split"] == f"{split}_positive"
        ]
        normals = [
            row
            for row in events
            if row["split"] == f"{split}_normal"
        ]
        method_counts = {}
        for method in METHODS:
            method_counts[method] = {
                "detected_primary_1s": sum(
                    int(row[f"{method}_primary_1s_detected"])
                    for row in positives
                ),
                "detected_extended_1p5s": sum(
                    int(row[f"{method}_extended_1p5s_detected"])
                    for row in positives
                ),
                "extended_only_positive": sum(
                    int(row[f"{method}_extended_only"])
                    for row in positives
                ),
                "normal_false_alarm_events": sum(
                    int(row[f"{method}_normal_detected"])
                    for row in normals
                ),
            }
        comparisons = {}
        for window in WINDOWS:
            relation_only = [
                row["event_uid"]
                for row in positives
                if row[f"relation_{window}_detected"]
                and not row[f"ttc_{window}_detected"]
            ]
            ttc_only = [
                row["event_uid"]
                for row in positives
                if row[f"ttc_{window}_detected"]
                and not row[f"relation_{window}_detected"]
            ]
            comparisons[window] = {
                "relation_only_count": len(relation_only),
                "relation_only_event_uids": relation_only,
                "ttc_only_count": len(ttc_only),
                "ttc_only_event_uids": ttc_only,
                "both_count": sum(
                    int(
                        row[f"relation_{window}_detected"]
                        and row[f"ttc_{window}_detected"]
                    )
                    for row in positives
                ),
                "neither_count": sum(
                    int(
                        not row[f"relation_{window}_detected"]
                        and not row[f"ttc_{window}_detected"]
                    )
                    for row in positives
                ),
            }
        summary[split] = {
            "positive_event_count": len(positives),
            "normal_event_count": len(normals),
            "methods": method_counts,
            "relation_vs_ttc": comparisons,
        }
    return summary


def _render_readme(
    *,
    split_summary: dict[str, Any],
    audit: Sequence[dict[str, Any]],
    onset_reference: np.ndarray,
) -> str:
    audit1 = next(row for row in audit if row["audit_index"] == 1)
    audit20 = next(row for row in audit if row["audit_index"] == 20)
    lines = [
        "# Early-window sensitivity",
        "",
        "This is a no-retraining timing and threshold audit. The 1.0 s "
        "window remains the primary protocol; 1.5 s is an explicitly "
        "reported sensitivity track.",
        "",
        "## Key audited events",
        "",
        (
            "- Audit 1: onset/recovery evidence reaches the q90 branch "
            f"(1.0 s detected={audit1['relation_plus_onset_q900_primary_1s_detected']}, "
            f"1.5 s detected={audit1['relation_plus_onset_q900_extended_1p5s_detected']}), "
            "but the calibrated relation score itself remains sub-threshold."
        ),
        (
            "- Audit 20: relation detection changes from "
            f"{audit20['relation_primary_1s_detected']} to "
            f"{audit20['relation_extended_1p5s_detected']}; TTC changes from "
            f"{audit20['ttc_primary_1s_detected']} to "
            f"{audit20['ttc_extended_1p5s_detected']}."
        ),
        "",
        "## Calibration",
        "",
        (
            f"Onset positive-value reference n={len(onset_reference)}, "
            f"q90={np.quantile(onset_reference, 0.90):.4f}, "
            f"q95={np.quantile(onset_reference, 0.95):.4f}, "
            f"q999={np.quantile(onset_reference, 0.999):.4f}."
        ),
        "",
        "## Event counts",
        "",
    ]
    for split, values in split_summary.items():
        relation = values["methods"]["relation"]
        ttc = values["methods"]["ttc"]
        q900 = values["methods"]["relation_plus_onset_q900"]
        lines.extend(
            [
                f"### {split}",
                "",
                (
                    f"- Relation: {relation['detected_primary_1s']} -> "
                    f"{relation['detected_extended_1p5s']} / "
                    f"{values['positive_event_count']} positives."
                ),
                (
                    f"- TTC: {ttc['detected_primary_1s']} -> "
                    f"{ttc['detected_extended_1p5s']} / "
                    f"{values['positive_event_count']} positives."
                ),
                (
                    f"- Relation + onset q90: "
                    f"{q900['detected_primary_1s']} -> "
                    f"{q900['detected_extended_1p5s']} positives; "
                    f"{q900['normal_false_alarm_events']} normal events alert."
                ),
                "",
            ]
        )
    return "\n".join(lines)


def run_early_window_sensitivity(
    *,
    frame_path: Path = DEFAULT_FRAME_PATH,
    audit_csv: Path = DEFAULT_AUDIT_CSV,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    frame_path = frame_path.expanduser().resolve()
    audit_csv = audit_csv.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    frames = _load_frame_rows(frame_path)
    onset_reference = _positive_reference(
        frames["development_validation_normal"]
    )
    _attach_window_scores(frames, onset_reference)

    metrics = _metric_rows(frames)
    events = _event_rows(frames)
    audit = _audit_rows(events, audit_csv)
    split_summary = _split_summary(events)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "early_window_metrics.csv"
    events_path = output_root / "early_window_events.csv"
    audit_path = output_root / "early_window_audit20.csv"
    readme_path = output_root / "README.md"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    _write_csv(audit_path, audit)
    readme_path.write_text(
        _render_readme(
            split_summary=split_summary,
            audit=audit,
            onset_reference=onset_reference,
        ),
        encoding="utf-8",
    )

    extended_only = [
        {
            "split": row["split"],
            "event_uid": row["event_uid"],
            "relation_extended_only": row["relation_extended_only"],
            "ttc_extended_only": row["ttc_extended_only"],
            "relation_extended_max": row["relation_extended_1p5s_max"],
            "relation_extended_max_fta": row[
                "relation_extended_1p5s_max_fta"
            ],
            "ttc_extended_max": row["ttc_extended_1p5s_max"],
            "ttc_extended_max_fta": row["ttc_extended_1p5s_max_fta"],
        }
        for row in events
        if row["positive"]
        and (row["relation_extended_only"] or row["ttc_extended_only"])
    ]
    result = {
        "protocol": "riskbench-early-window-sensitivity-v1",
        "model_retrained": False,
        "primary_window_seconds": 1.0,
        "extended_window_seconds": 1.5,
        "threshold": ALERT_PERCENTILE,
        "onset_quantiles": list(ONSET_QUANTILES),
        "split_summary": split_summary,
        "extended_only_events": extended_only,
        "audit1": next(row for row in audit if row["audit_index"] == 1),
        "audit20": next(row for row in audit if row["audit_index"] == 20),
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "events_csv": str(events_path),
            "audit20_csv": str(audit_path),
            "readme": str(readme_path),
        },
        "status": "early_window_sensitivity_complete",
    }
    summary_path = output_root / "early_window_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(summary_path)
    return result


def main() -> None:
    args = parse_args()
    result = run_early_window_sensitivity(
        frame_path=args.frame_path,
        audit_csv=args.audit_csv,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
