"""Complete a frozen DeepAccident evaluation with the physical OR RIR decision.

The relation-only scores are read from a completed external evaluation.  The
physical percentile reference is reconstructed from the pre-existing
RiskBench hard-safe validation-frame cache for the same frozen seed and fold.
No DeepAccident observation is used to fit either branch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from ..danger_signal.diagnostics import score_sequences
from ..riskbench_metadata import damp_decision_necessity_ablation as d
from ..riskbench_metadata.prospective_validation import SEEDS, digest, family_fold
from .adapter import load_view_records
from .frozen_pilot import (
    BUDGETS,
    PROJECT_ROOT,
    _input_receipt,
    _load_model,
    _normalize,
    _prepare_view,
    _tree_digest,
    _verify_frozen_runtime,
    _write_csv,
    _write_json,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data/processed/ego_immune_deepaccident_val01_v1"
    / "deepaccident_val01_focal_view_manifest.csv"
)
DEFAULT_BASE_PROTOCOL = DEFAULT_MANIFEST.with_name("external_validation_protocol_v1.json")
DEFAULT_ADDENDUM_PROTOCOL = DEFAULT_MANIFEST.with_name(
    "physical_union_addendum_protocol_v1.json"
)
DEFAULT_BASE_RESULTS = PROJECT_ROOT / "results/rir_deepaccident_val01_external_validation_v1"
DEFAULT_BUNDLE = PROJECT_ROOT / "results/rir_prospective_freeze_20260908_v2"
DEFAULT_RISKBENCH_SOURCE = (
    PROJECT_ROOT / "results/ego_immune_riskbench_pair_damp_normal_only_multiseed_v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/rir_deepaccident_val01_physical_union_addendum_v1"
PHYSICAL_THRESHOLD = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-protocol", type=Path, default=DEFAULT_BASE_PROTOCOL)
    parser.add_argument("--addendum-protocol", type=Path, default=DEFAULT_ADDENDUM_PROTOCOL)
    parser.add_argument("--base-results", type=Path, default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--riskbench-source", type=Path, default=DEFAULT_RISKBENCH_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _first_physical_crossing(
    trace: list[tuple[dict[str, Any], float]],
) -> tuple[bool, float, float | None]:
    score = d.event_score(trace)
    crossings = [
        frame
        for frame, value in trace
        if float(value) >= PHYSICAL_THRESHOLD
    ]
    if not crossings:
        return False, score, None
    first = min(crossings, key=lambda frame: int(frame["frame_position"]))
    return True, score, int(first["eval_frames_to_anchor"]) / 20.0


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(mean(int(row[key]) for row in rows)) if rows else 0.0


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for budget in BUDGETS:
        budget_rows = [row for row in rows if float(row["budget"]) == budget]
        seed_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in budget_rows:
            seed_rows[int(row["seed"])].append(row)
        seed_metrics = []
        for seed in SEEDS:
            current = seed_rows[seed]
            accident = [row for row in current if row["outcome"] == "accident"]
            normal = [row for row in current if row["outcome"] == "normal"]
            hidden = [row for row in accident if not int(row["physical_hit"])]
            physical_negative_normal = [
                row for row in normal if not int(row["physical_hit"])
            ]
            union_leads = [
                float(row["union_first_crossing_lead_seconds"])
                for row in accident
                if row["union_first_crossing_lead_seconds"] not in (None, "")
            ]
            seed_metrics.append(
                {
                    "seed": seed,
                    "physical_recall": _rate(accident, "physical_hit"),
                    "relation_recall": _rate(accident, "relation_hit"),
                    "union_recall": _rate(accident, "union_hit"),
                    "hidden_count": len(hidden),
                    "hidden_rescue": _rate(hidden, "relation_hit"),
                    "physical_normal_rate": _rate(normal, "physical_hit"),
                    "physical_negative_normal_count": len(physical_negative_normal),
                    "added_relation_rate": _rate(
                        physical_negative_normal, "relation_hit"
                    ),
                    "union_normal_rate": _rate(normal, "union_hit"),
                    "union_mean_lead_seconds": mean(union_leads) if union_leads else None,
                }
            )
        summary: dict[str, Any] = {
            "budget": budget,
            "view_count_per_seed": sum(
                row["outcome"] == "accident" for row in seed_rows[SEEDS[0]]
            ),
            "family_count": len(
                {
                    row["family"]
                    for row in seed_rows[SEEDS[0]]
                    if row["outcome"] == "accident"
                }
            ),
        }
        for key in (
            "physical_recall",
            "relation_recall",
            "union_recall",
            "hidden_rescue",
            "physical_normal_rate",
            "added_relation_rate",
            "union_normal_rate",
            "union_mean_lead_seconds",
        ):
            values = [float(row[key]) for row in seed_metrics if row[key] is not None]
            summary[f"{key}_mean"] = mean(values) if values else None
            summary[f"{key}_seed_sd"] = stdev(values) if len(values) > 1 else 0.0
        summary["hidden_accident_views_per_seed"] = [
            int(row["hidden_count"]) for row in seed_metrics
        ]
        summary["physical_negative_normal_views_per_seed"] = [
            int(row["physical_negative_normal_count"]) for row in seed_metrics
        ]
        output.append(summary)
    return output


def _markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# DeepAccident val01 physical-reference + RIR addendum",
        "",
        "This post-lock addendum completes the guarded system evaluation. The "
        "physical percentile definition and 0.95 threshold are inherited from "
        "the pre-existing RiskBench protocol; neither branch is recalibrated on "
        "DeepAccident.",
        "",
        "| Relation budget | Physical recall | Relation-only recall | Union overall recall | Rescue among physical-negative accidents | Physical normal alerts | Added relation candidates | Union normal candidates |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {float(row['budget']):.0%} | "
            f"{float(row['physical_recall_mean']):.1%} | "
            f"{float(row['relation_recall_mean']):.1%} | "
            f"{float(row['union_recall_mean']):.1%} | "
            f"{float(row['hidden_rescue_mean']):.1%} | "
            f"{float(row['physical_normal_rate_mean']):.1%} | "
            f"{float(row['added_relation_rate_mean']):.1%} | "
            f"{float(row['union_normal_rate_mean']):.1%} |"
        )
    lines.extend(
        [
            "",
            "The independent units are 11 accident-normal scenario families. "
            "The two focal views within a family and the three frozen seeds are "
            "not additional independent families. Because DeepAccident clips end "
            "1.0 s before collision, only the available 1.5-to-1.0 s interval is "
            "scored.",
            "",
            "This analysis was specified after viewing the relation-only result "
            "and must be described as a protocol-completion addendum rather than "
            "as preregistered evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    base_protocol = args.base_protocol.resolve()
    addendum_protocol = args.addendum_protocol.resolve()
    base_results = args.base_results.resolve()
    bundle = args.bundle.resolve()
    riskbench_source = args.riskbench_source.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ValueError(f"Refuse to overwrite addendum output: {output_root}")

    addendum = json.loads(addendum_protocol.read_text(encoding="utf-8"))
    if digest(base_protocol) != addendum["base_protocol_sha256"]:
        raise ValueError("Base external protocol changed after addendum lock.")
    if digest(base_results / "events.csv") != addendum["base_relation_events_sha256"]:
        raise ValueError("Relation-only event results changed after addendum lock.")
    if digest(base_results / "completion.json") != addendum["base_completion_sha256"]:
        raise ValueError("Relation-only completion receipt changed after addendum lock.")
    if float(addendum["physical_threshold"]) != PHYSICAL_THRESHOLD:
        raise ValueError("Unexpected physical threshold in addendum protocol.")

    runtime_hashes = _verify_frozen_runtime(bundle)
    records = load_view_records(manifest)
    family_count = len({record.pair_key for record in records})
    if len(records) != int(addendum["eligible_view_count"]):
        raise ValueError("Unexpected eligible view count.")
    if family_count != int(addendum["eligible_family_count"]):
        raise ValueError("Unexpected eligible family count.")

    relation_rows = _read_csv(base_results / "events.csv")
    relation = {
        (
            int(row["seed"]),
            float(row["budget"]),
            str(row["view_uid"]),
            str(row["outcome"]),
        ): row
        for row in relation_rows
        if row["method"] == "rir_all_safe_selected"
    }

    prepared = {f"{r.pair_key}:{r.focal_role}": _prepare_view(r) for r in records}
    model_cache: dict[tuple[int, int], tuple[Any, dict[str, Any], Any]] = {}
    physical_reference_cache: dict[tuple[int, int], Any] = {}
    output_rows: list[dict[str, Any]] = []
    used_riskbench_cache_paths: set[Path] = set()

    for view_index, record in enumerate(records, start=1):
        view_uid = f"{record.pair_key}:{record.focal_role}"
        fold = family_fold(record.pair_key)
        print(f"[{view_index}/{len(records)}] {view_uid} -> fold {fold}", flush=True)
        raw_sequences = prepared[view_uid]
        for seed in SEEDS:
            if (seed, fold) not in model_cache:
                model_cache[(seed, fold)] = _load_model(bundle, seed, fold)
            model, normalization, device = model_cache[(seed, fold)]
            if (seed, fold) not in physical_reference_cache:
                cache_path = (
                    riskbench_source
                    / f"seed_{seed}"
                    / "pair_frame_cache/graph_free_one_class"
                    / f"fold_{fold:02d}.pkl"
                )
                used_riskbench_cache_paths.add(cache_path)
                riskbench_frames = d._load_frames(riskbench_source, seed, fold)
                physical_reference_cache[(seed, fold)] = d._physical_reference(
                    d._event_frames(riskbench_frames["validation_normal"])
                )
            reference = physical_reference_cache[(seed, fold)]
            normalized = {
                outcome: _normalize(sequence, normalization)
                for outcome, sequence in raw_sequences.items()
            }
            scored = {
                outcome: score_sequences(
                    model,
                    [sequence],
                    device,
                    split=f"deepaccident_{outcome}",
                    group_count=4,
                    recovery_initialization_predictions=3,
                    include_relation_components=True,
                )
                for outcome, sequence in normalized.items()
            }
            physical = {
                outcome: _first_physical_crossing(d._physical_trace(frames, reference))
                for outcome, frames in scored.items()
            }
            for budget in BUDGETS:
                for outcome in ("accident", "normal"):
                    relation_row = relation[(seed, budget, view_uid, outcome)]
                    physical_hit, physical_score, physical_lead = physical[outcome]
                    relation_hit = bool(int(relation_row["hit"]))
                    relation_lead = _optional_float(
                        relation_row["first_crossing_lead_seconds"]
                    )
                    available_leads = [
                        lead
                        for lead in (physical_lead, relation_lead)
                        if lead is not None
                    ]
                    output_rows.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "budget": budget,
                            "family": record.pair_key,
                            "view_uid": view_uid,
                            "scenario_id": record.scenario_id,
                            "focal_role": record.focal_role,
                            "outcome": outcome,
                            "physical_score": physical_score,
                            "physical_threshold": PHYSICAL_THRESHOLD,
                            "physical_hit": int(physical_hit),
                            "physical_first_crossing_lead_seconds": physical_lead,
                            "relation_score": float(relation_row["score"]),
                            "relation_threshold": float(relation_row["threshold"]),
                            "relation_hit": int(relation_hit),
                            "relation_first_crossing_lead_seconds": relation_lead,
                            "relation_only_rescue": int(relation_hit and not physical_hit),
                            "union_hit": int(physical_hit or relation_hit),
                            "union_first_crossing_lead_seconds": (
                                max(available_leads) if available_leads else None
                            ),
                        }
                    )

    summaries = _summaries(output_rows)
    output_root.mkdir(parents=True)
    _write_csv(output_root / "union_events.csv", output_rows)
    _write_csv(output_root / "summary_metrics.csv", summaries)
    (output_root / "SUMMARY.md").write_text(_markdown(summaries), encoding="utf-8")
    receipt = _input_receipt(
        records,
        manifest=manifest,
        protocol=addendum_protocol,
        bundle=bundle,
        runtime_hashes=runtime_hashes,
    )
    receipt.update(
        {
            "physical_union_addendum_runner_sha256": digest(Path(__file__)),
            "base_protocol_sha256": digest(base_protocol),
            "base_relation_events_sha256": digest(base_results / "events.csv"),
            "physical_reference_cache_tree_sha256": _tree_digest(
                used_riskbench_cache_paths
            )[0],
            "physical_reference_cache_file_count": len(used_riskbench_cache_paths),
            "physical_threshold": PHYSICAL_THRESHOLD,
        }
    )
    _write_json(output_root / "input_receipt.json", receipt)
    _write_json(output_root / "protocol.json", addendum)
    _write_json(
        output_root / "completion.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed_post_lock_physical_union_addendum",
            "views": len(records),
            "families": family_count,
            "seeds": list(SEEDS),
            "budgets": list(BUDGETS),
            "event_rows": len(output_rows),
            "output_tree_sha256_excluding_completion": _tree_digest(
                path for path in output_root.iterdir() if path.name != "completion.json"
            )[0],
        },
    )
    print((output_root / "SUMMARY.md").read_text(encoding="utf-8"))
    print(f"output: {output_root}")


if __name__ == "__main__":
    main()
