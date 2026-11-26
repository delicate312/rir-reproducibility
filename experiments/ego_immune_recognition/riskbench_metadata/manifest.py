"""Build a leakage-auditable manifest for RiskBench metadata scenarios."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

from .adapter import (
    DEFAULT_METADATA_ROOT,
    RiskBenchScenarioRecord,
    discover_riskbench_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "ego_immune_riskbench_metadata_v0"
)


def record_to_row(record: RiskBenchScenarioRecord) -> dict[str, Any]:
    return {
        "scenario_id": record.scenario_id,
        "category": record.category,
        "pool_role": record.pool_role,
        "split": record.split,
        "scenario_group_id": record.scenario_group_id,
        "base_scenario_id": record.base_scenario_id,
        "weather": record.weather,
        "traffic_density": record.traffic_density,
        "risk_actor_ids": ";".join(record.risk_actor_ids),
        "risk_actor_count": len(record.risk_actor_ids),
        "critical_frame": record.critical_frame,
        "behavior_start_frame": (
            None if record.behavior_window is None else record.behavior_window[0]
        ),
        "behavior_end_frame": (
            None if record.behavior_window is None else record.behavior_window[1]
        ),
        "target_type_code": record.target_type_code,
        "target_type_name": record.target_type_name,
        "frame_count": record.frame_count,
        "actor_count": record.actor_count,
        "state_path": str(record.state_path),
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def summarize_manifest(records: Sequence[RiskBenchScenarioRecord]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record.scenario_group_id].add(record.split)
    leakage = {
        group_id: sorted(splits)
        for group_id, splits in groups.items()
        if len(splits) > 1
    }
    if leakage:
        raise ValueError(f"Augmentation-family leakage detected: {leakage}")

    frames = [record.frame_count for record in records]
    actors = [record.actor_count for record in records]
    category_group_count = Counter(
        (record.category, record.scenario_group_id) for record in records
    )
    variants_per_group = list(category_group_count.values())
    return {
        "protocol": "riskbench-metadata-three-pool-grouped-v0",
        "metadata_only": True,
        "scenario_count": len(records),
        "scenario_group_count": len(groups),
        "augmentation_group_leakage_count": len(leakage),
        "category_counts": _counter_dict(Counter(r.category for r in records)),
        "pool_role_counts": _counter_dict(Counter(r.pool_role for r in records)),
        "split_counts": _counter_dict(Counter(r.split for r in records)),
        "target_type_counts": _counter_dict(
            Counter(r.target_type_name for r in records)
        ),
        "frame_count": {
            "min": min(frames),
            "mean": mean(frames),
            "median": median(frames),
            "max": max(frames),
        },
        "actor_count": {
            "min": min(actors),
            "mean": mean(actors),
            "median": median(actors),
            "max": max(actors),
        },
        "weather_density_variants_per_group": {
            "min": min(variants_per_group),
            "mean": mean(variants_per_group),
            "median": median(variants_per_group),
            "max": max(variants_per_group),
        },
        "split_policy": {
            "unit": "category plus base scenario script",
            "augmentation_lock": (
                "All weather and traffic-density variants of one script stay "
                "in one split."
            ),
            "safe_core": "non-interactive train_normal only updates gradients",
            "hard_safe": "interactive is threshold calibration or test normal",
            "positive_ood": "collision is validation/test only and gradient-free",
            "obstacle": "excluded from v0 and retained for later static-risk work",
        },
    }


def write_manifest(
    records: Sequence[RiskBenchScenarioRecord],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [record_to_row(record) for record in records]
    manifest_path = output_root / "riskbench_metadata_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_manifest(records)
    summary["metadata_root"] = str(DEFAULT_METADATA_ROOT.resolve())
    summary["manifest_path"] = str(manifest_path)
    summary_path = output_root / "manifest_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_manifest(
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[list[RiskBenchScenarioRecord], dict[str, Any]]:
    records = discover_riskbench_records(metadata_root)
    return records, write_manifest(records, output_root)
