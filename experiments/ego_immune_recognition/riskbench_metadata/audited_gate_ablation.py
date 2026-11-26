"""Test audited relation-gate candidates without retraining the backbone."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Sequence

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
    _load_frame_rows,
)
from .persistent_guarded_fusion import ALERT_PERCENTILE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ego_immune_riskbench_audited_gate_ablation_v1"
)

FrameRule = Callable[[dict[str, float]], bool]


def _near_sustain(values: dict[str, float]) -> bool:
    return (
        values["relation_percentile"] >= 0.94
        and values["persistence_percentile"] >= 0.85
        and values["peak_support_count"] >= 4
        and values["peak_relation_rise"] >= 0.15
    )


def _supported_peak_p80(values: dict[str, float]) -> bool:
    return (
        values["relation_percentile"] >= 0.95
        and values["persistence_percentile"] >= 0.80
        and values["peak_support_count"] >= 4
        and values["peak_relation_rise"] >= 0.15
    )


def _very_high_supported(values: dict[str, float]) -> bool:
    return (
        values["relation_percentile"] >= 0.97
        and values["peak_support_count"] >= 4
        and values["peak_relation_rise"] >= 0.15
    )


def _isolated_extreme_rise(values: dict[str, float]) -> bool:
    return (
        values["relation_percentile"] >= 0.96
        and values["peak_relation_rise"] >= 0.50
    )


RULES: dict[str, FrameRule] = {
    "near_sustain": _near_sustain,
    "supported_peak_p80": _supported_peak_p80,
    "very_high_supported": _very_high_supported,
    "isolated_extreme_rise": _isolated_extreme_rise,
}
METHODS = {
    "baseline": BASELINE_SCORE,
    "candidate_supported": "candidate_supported",
    "candidate_all": "candidate_all",
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
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _attach_candidate_scores(
    frames_by_split: dict[str, list[dict[str, Any]]],
) -> None:
    supported_rules = (
        "near_sustain",
        "supported_peak_p80",
        "very_high_supported",
    )
    for frames in frames_by_split.values():
        for frame in frames:
            values = frame["aggregations"]
            activations = {
                name: float(rule(values))
                for name, rule in RULES.items()
            }
            supported = any(activations[name] for name in supported_rules)
            all_candidates = supported or bool(
                activations["isolated_extreme_rise"]
            )
            baseline = float(values[BASELINE_SCORE])
            values.update(
                {
                    **{
                        f"audited_gate_{name}": active
                        for name, active in activations.items()
                    },
                    "candidate_supported": max(
                        baseline,
                        float(supported),
                    ),
                    "candidate_all": max(
                        baseline,
                        float(all_candidates),
                    ),
                }
            )


def _detected(
    event_frames: Sequence[dict[str, Any]],
    score_key: str,
    *,
    positive: bool,
) -> bool:
    return any(
        (not positive or int(frame["original_alert"]) == 1)
        and float(frame["aggregations"][score_key]) >= ALERT_PERCENTILE
        for frame in event_frames
    )


def _metric_rows(
    frames: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for split in (
        "development_validation",
        "development_test",
        "confirmation",
    ):
        normal = frames[f"{split}_normal"]
        positive = frames[f"{split}_positive"]
        for method, score_key in METHODS.items():
            morphology = false_alarm_morphology(
                normal,
                score_key=score_key,
                threshold=ALERT_PERCENTILE,
            )
            rows.append(
                {
                    "split": split,
                    "method": method,
                    **_evaluate_dual_track(
                        normal_frames=normal,
                        positive_frames=positive,
                        aggregation=score_key,
                        label_window="original",
                        threshold=ALERT_PERCENTILE,
                        timestep_seconds=RISK_BENCH_TIMESTEP_SECONDS,
                    ),
                    "normal_event_false_alarm_rate": morphology[
                        "event_false_alarm_rate"
                    ],
                    "normal_sustained_3_frame_false_alarm_rate": morphology[
                        "sustained_3_frame_event_false_alarm_rate"
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
                "event_uid": event_frames[0]["event_uid"],
                "positive": int(positive),
            }
            for method, score_key in METHODS.items():
                row[f"{method}_detected"] = int(
                    _detected(
                        event_frames,
                        score_key,
                        positive=positive,
                    )
                )
            usable = [
                frame
                for frame in event_frames
                if not positive or int(frame["original_alert"]) == 1
            ]
            for rule_name in RULES:
                row[f"{rule_name}_active"] = int(
                    any(
                        frame["aggregations"][
                            f"audited_gate_{rule_name}"
                        ]
                        >= 1.0
                        for frame in usable
                    )
                )
            rows.append(row)
    return rows


def _audit_rows(
    events: Sequence[dict[str, Any]],
    audit_csv: Path,
) -> list[dict[str, Any]]:
    lookup = {
        (row["split"], row["event_uid"]): row
        for row in events
    }
    rows = []
    for audit in _read_csv(audit_csv):
        event = lookup[(audit["split"], audit["event_uid"])]
        rows.append(
            {
                "audit_index": int(audit["audit_index"]),
                "split": audit["split"],
                "event_uid": audit["event_uid"],
                **{
                    key: value
                    for key, value in event.items()
                    if key not in {"split", "event_uid", "positive"}
                },
            }
        )
    return rows


def _summary_counts(
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
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
        result[split] = {
            method: {
                "positive_detected": sum(
                    int(row[f"{method}_detected"])
                    for row in positives
                ),
                "positive_added_vs_baseline": sum(
                    int(
                        row[f"{method}_detected"]
                        and not row["baseline_detected"]
                    )
                    for row in positives
                ),
                "normal_alerted": sum(
                    int(row[f"{method}_detected"])
                    for row in normals
                ),
                "normal_added_vs_baseline": sum(
                    int(
                        row[f"{method}_detected"]
                        and not row["baseline_detected"]
                    )
                    for row in normals
                ),
            }
            for method in METHODS
        }
    return result


def run_audited_gate_ablation(
    *,
    frame_path: Path = DEFAULT_FRAME_PATH,
    audit_csv: Path = DEFAULT_AUDIT_CSV,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    frame_path = frame_path.expanduser().resolve()
    audit_csv = audit_csv.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    frames = _load_frame_rows(frame_path)
    _attach_candidate_scores(frames)

    metrics = _metric_rows(frames)
    events = _event_rows(frames)
    audit = _audit_rows(events, audit_csv)
    counts = _summary_counts(events)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "audited_gate_metrics.csv"
    events_path = output_root / "audited_gate_events.csv"
    audit_path = output_root / "audited_gate_audit20.csv"
    _write_csv(metrics_path, metrics)
    _write_csv(events_path, events)
    _write_csv(audit_path, audit)

    result = {
        "protocol": "riskbench-audited-gate-ablation-v1",
        "model_retrained": False,
        "status_note": (
            "Exploratory rules derived from inspected misses. They must not "
            "be promoted as paper-facing results without a fresh holdout."
        ),
        "rules": {
            "near_sustain": (
                "relation>=.94 & persistence>=.85 & support>=4 & rise>=.15"
            ),
            "supported_peak_p80": (
                "relation>=.95 & persistence>=.80 & support>=4 & rise>=.15"
            ),
            "very_high_supported": (
                "relation>=.97 & support>=4 & rise>=.15"
            ),
            "isolated_extreme_rise": "relation>=.96 & rise>=.50",
        },
        "counts": counts,
        "audited_events": {
            str(index): next(
                row for row in audit if row["audit_index"] == index
            )
            for index in (5, 8, 13, 17)
        },
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "events_csv": str(events_path),
            "audit20_csv": str(audit_path),
        },
        "status": "audited_gate_ablation_complete",
    }
    summary_path = output_root / "audited_gate_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["summary_path"] = str(summary_path)
    return result


def main() -> None:
    args = parse_args()
    result = run_audited_gate_ablation(
        frame_path=args.frame_path,
        audit_csv=args.audit_csv,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
