"""Evaluate the predeclared physical-reference OR RIR decision externally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..riskbench_metadata import damp_decision_necessity_ablation as d
from ..riskbench_metadata.prospective_validation import SEEDS, digest, family_fold
from ..rule_baseline_dual_track_eval import rule_scores_for_graph
from .adapter import load_view_records
from .frozen_pilot import (
    BUDGETS,
    PROJECT_ROOT,
    _input_receipt,
    _prepare_view,
    _tree_digest,
    _verify_frozen_runtime,
    _write_csv,
    _write_json,
)
from .physical_union_addendum import (
    PHYSICAL_THRESHOLD,
    _first_physical_crossing,
    _optional_float,
    _read_csv,
    _summaries,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data/processed/ego_immune_deepaccident_val02_new_v1"
    / "deepaccident_val02_new_focal_view_manifest.csv"
)
DEFAULT_PROTOCOL = DEFAULT_MANIFEST.with_name("external_validation_protocol_v1.json")
DEFAULT_RELATION_RESULTS = (
    PROJECT_ROOT / "results/rir_deepaccident_val02_new_external_validation_v1"
)
DEFAULT_BUNDLE = PROJECT_ROOT / "results/rir_prospective_freeze_20260908_v2"
DEFAULT_RISKBENCH_SOURCE = (
    PROJECT_ROOT / "results/ego_immune_riskbench_pair_damp_normal_only_multiseed_v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/rir_deepaccident_val02_new_guarded_validation_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--relation-results", type=Path, default=DEFAULT_RELATION_RESULTS)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--riskbench-source", type=Path, default=DEFAULT_RISKBENCH_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _markdown(
    summaries: list[dict[str, Any]], *, title: str, families: int
) -> str:
    lines = [
        f"# {title}: guarded physical + RIR decision",
        "",
        "Both branches and their logical union were declared before this batch was scored. "
        "The physical percentile reference and threshold were inherited from RiskBench; "
        "DeepAccident did not fit or recalibrate either branch.",
        "",
        "| Relation budget | Physical recall | Relation-only recall | Union overall recall | RIR rescue among physical-negative accidents | Physical normal alerts | Added relation candidates | Union normal candidates |",
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
            f"The independent sample contains {families} previously unscored accident-normal "
            "scenario families. Multiple focal views and three frozen seeds are not treated as "
            "additional independent families. Only the available 1.5-to-1.0 s pre-collision "
            "interval is scored.",
            "",
        ]
    )
    return "\n".join(lines)


def _physical_frames(sequence: list[Any], *, split: str) -> list[dict[str, Any]]:
    """Build exactly the rule-score fields needed by the frozen TTC reference."""

    rows = []
    for frame_position, graph in enumerate(sequence):
        source_frame = int(graph.frame_index.view(-1)[0].item())
        critical_frame = int(graph.critical_frame.view(-1)[0].item())
        rows.append(
            {
                "split": split,
                "frame_position": frame_position,
                "source_frame_index": source_frame,
                "critical_frame": critical_frame,
                "eval_frames_to_anchor": critical_frame - source_frame,
                "frames_to_anchor": int(graph.frames_to_physical_danger.item()),
                "aggregations": rule_scores_for_graph(graph),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    relation_results = args.relation_results.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    riskbench_source = args.riskbench_source.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"Refuse to overwrite guarded-validation output: {output_root}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if digest(manifest) != protocol["input_manifest_sha256"]:
        raise ValueError("External manifest changed after protocol lock")
    if (bundle / "MANIFEST.sha256").read_text(encoding="utf-8").strip() != protocol[
        "frozen_bundle_manifest_sha256"
    ]:
        raise ValueError("Frozen model bundle changed after protocol lock")
    if "physical_or_rir" not in protocol["primary_methods"]:
        raise ValueError("Guarded union was not declared in the locked protocol")

    runtime_hashes = _verify_frozen_runtime(bundle)
    records = load_view_records(manifest)
    family_count = len({record.pair_key for record in records})
    if len(records) != int(protocol["eligible_view_count"]):
        raise ValueError("Unexpected eligible view count")
    if family_count != int(protocol["eligible_family_count"]):
        raise ValueError("Unexpected eligible family count")

    relation_rows = _read_csv(relation_results / "events.csv")
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
    physical_reference_cache: dict[tuple[int, int], Any] = {}
    used_riskbench_paths: set[Path] = set()
    output_rows: list[dict[str, Any]] = []

    for view_index, record in enumerate(records, start=1):
        view_uid = f"{record.pair_key}:{record.focal_role}"
        fold = family_fold(record.pair_key)
        print(f"[{view_index}/{len(records)}] {view_uid} -> fold {fold}", flush=True)
        raw_sequences = prepared[view_uid]
        physical_frames = {
            outcome: _physical_frames(sequence, split=f"deepaccident_{outcome}")
            for outcome, sequence in raw_sequences.items()
        }
        for seed in SEEDS:
            if (seed, fold) not in physical_reference_cache:
                cache_path = (
                    riskbench_source
                    / f"seed_{seed}"
                    / "pair_frame_cache/graph_free_one_class"
                    / f"fold_{fold:02d}.pkl"
                )
                used_riskbench_paths.add(cache_path)
                riskbench_frames = d._load_frames(riskbench_source, seed, fold)
                physical_reference_cache[(seed, fold)] = d._physical_reference(
                    d._event_frames(riskbench_frames["validation_normal"])
                )
            physical_reference = physical_reference_cache[(seed, fold)]
            physical = {
                outcome: _first_physical_crossing(
                    d._physical_trace(frames, physical_reference)
                )
                for outcome, frames in physical_frames.items()
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
    (output_root / "SUMMARY.md").write_text(
        _markdown(
            summaries,
            title=str(protocol["summary_title"]),
            families=family_count,
        ),
        encoding="utf-8",
    )
    receipt = _input_receipt(
        records,
        manifest=manifest,
        protocol=protocol_path,
        bundle=bundle,
        runtime_hashes=runtime_hashes,
    )
    receipt.update(
        {
            "guarded_runner_sha256": digest(Path(__file__)),
            "relation_events_sha256": digest(relation_results / "events.csv"),
            "physical_reference_cache_tree_sha256": _tree_digest(
                used_riskbench_paths
            )[0],
            "physical_threshold": PHYSICAL_THRESHOLD,
        }
    )
    _write_json(output_root / "input_receipt.json", receipt)
    _write_json(
        output_root / "completion.json",
        {
            "status": "completed_predeclared_guarded_external_validation",
            "views": len(records),
            "families": family_count,
            "seeds": list(SEEDS),
            "budgets": list(BUDGETS),
            "rows": len(output_rows),
            "output_tree_sha256_excluding_completion": _tree_digest(
                path for path in output_root.iterdir() if path.name != "completion.json"
            )[0],
        },
    )
    print((output_root / "SUMMARY.md").read_text(encoding="utf-8"))
    print(f"output: {output_root}")


if __name__ == "__main__":
    main()
