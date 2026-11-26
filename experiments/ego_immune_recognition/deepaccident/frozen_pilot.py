"""Run the sealed RIR models on the audited DeepAccident mini pairs.

This is a resource-limited external pilot.  DeepAccident labels define the
accident/normal pairing and the independent collision anchor, but do not fit
model parameters, normalization values, score references, or thresholds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ..danger_signal.diagnostics import score_sequences
from ..label_protocol import FrameDangerLabel, SequenceDangerLabels
from ..pyg_packager import package_pyg_sequence
from ..schema import EDGE_FEATURE_NAMES, RoadUserGraph
from ..temporal_relation_train_eval import (
    _NormalizedSequenceView,
    _load_training_module,
)
from ..riskbench_metadata import damp_decision_necessity_ablation as d
from ..riskbench_metadata.history_geometry_validation import pair_and_scene_scores
from ..riskbench_metadata.prospective_validation import (
    DEFAULT_BUNDLE,
    METHODS,
    SEEDS,
    digest,
    family_fold,
    verify,
)
from ..riskbench_metadata.relation_geometry_rescue import (
    _geometry_from_observations,
    _outgoing_ego_relations,
)
from ..riskbench_metadata.standalone_rir_audit import simple_controls
from .adapter import (
    DeepAccidentViewRecord,
    build_frozen_model_graphs,
    load_deepaccident_view,
    load_view_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data/processed/ego_immune_deepaccident_mini_v0"
    / "deepaccident_mini_focal_view_manifest.csv"
)
DEFAULT_PROTOCOL = DEFAULT_MANIFEST.with_name("external_pilot_protocol_v1.json")
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results/rir_deepaccident_mini_external_pilot_v1"
)
MODEL_HZ = 20.0
MAX_INPUT_FRAMES = 40
BUDGETS = (0.10, 0.05)
PRIMARY_METHODS = (
    "rir_all_safe_selected",
    "cpa_peak",
    "raw_residual_hybrid",
)
FROZEN_RUNTIME_FILES = (
    "02_alt_temporal_train.py",
    "relation_schema.py",
    "experiments/ego_immune_recognition/danger_signal/diagnostics.py",
    "experiments/ego_immune_recognition/temporal_relation_train_eval.py",
    "experiments/ego_immune_recognition/riskbench_metadata/prospective_validation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/damp_decision_necessity_ablation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/damp_hard_safe_grid_selection.py",
    "experiments/ego_immune_recognition/riskbench_metadata/standalone_rir_audit.py",
    "experiments/ego_immune_recognition/riskbench_metadata/history_geometry_validation.py",
    "experiments/ego_immune_recognition/riskbench_metadata/relation_geometry_rescue.py",
    "experiments/ego_immune_recognition/riskbench_metadata/core_hybrid_evaluation.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_protocol(protocol_path: Path, manifest_path: Path, bundle: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if digest(manifest_path) != protocol["input_manifest_sha256"]:
        raise ValueError("DeepAccident audit manifest changed after pilot lock.")
    if (bundle / "MANIFEST.sha256").read_text().strip() != protocol[
        "frozen_bundle_manifest_sha256"
    ]:
        raise ValueError("Frozen RIR bundle differs from the pilot protocol.")
    verify(bundle, expected_sha256=protocol["frozen_bundle_manifest_sha256"])
    return protocol


def _verify_frozen_runtime(bundle: Path) -> dict[str, str]:
    matches: dict[str, str] = {}
    for name in FROZEN_RUNTIME_FILES:
        current = PROJECT_ROOT / name
        frozen = bundle / "code" / name
        current_hash = digest(current)
        frozen_hash = digest(frozen)
        if current_hash != frozen_hash:
            raise ValueError(f"Frozen scoring dependency changed: {name}")
        matches[name] = current_hash
    return matches


def _tree_digest(paths: Iterable[Path]) -> tuple[str, int]:
    unique = sorted({path.expanduser().resolve() for path in paths})
    hasher = hashlib.sha256()
    for path in unique:
        hasher.update(str(path).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(digest(path).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest(), len(unique)


def _input_receipt(
    records: Sequence[DeepAccidentViewRecord],
    *,
    manifest: Path,
    protocol: Path,
    bundle: Path,
    runtime_hashes: dict[str, str],
) -> dict[str, Any]:
    label_paths = []
    for record in records:
        label_paths.extend(record.accident_label_root.glob("*.txt"))
        label_paths.extend(record.normal_label_root.glob("*.txt"))
    labels_sha256, label_file_count = _tree_digest(label_paths)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": digest(manifest),
        "protocol_path": str(protocol.resolve()),
        "protocol_sha256": digest(protocol),
        "freeze_bundle": str(bundle.resolve()),
        "freeze_manifest_sha256": (bundle / "MANIFEST.sha256").read_text().strip(),
        "used_label_file_count": label_file_count,
        "used_label_tree_sha256": labels_sha256,
        "deepaccident_adapter_sha256": digest(
            PROJECT_ROOT
            / "experiments/ego_immune_recognition/deepaccident/adapter.py"
        ),
        "pilot_runner_sha256": digest(Path(__file__)),
        "runtime_files_matching_freeze": runtime_hashes,
        "view_count": len(records),
        "family_count": len({record.pair_key for record in records}),
    }


def _frame_label(graph: RoadUserGraph, collision_model_frame: int) -> FrameDangerLabel:
    inverse_index = EDGE_FEATURE_NAMES.index("inverse_ttc_per_s")
    distance_index = EDGE_FEATURE_NAMES.index("distance_m")
    inverse = (
        0.0
        if graph.edge_count == 0
        else float(graph.edge_features[:, inverse_index].max(initial=0.0))
    )
    distance = (
        None
        if graph.edge_count == 0
        else float(graph.edge_features[:, distance_index].min(initial=np.inf))
    )
    lead = collision_model_frame - int(graph.frame_index)
    return FrameDangerLabel(
        frame_index=int(graph.frame_index),
        is_physical_danger=False,
        is_immune_alert=0 < lead <= 30,
        frames_to_physical_danger=lead,
        max_inverse_ttc_per_s=inverse,
        min_distance_m=distance,
        top_presented_edges=(),
    )


def _package_sequence(
    graphs: Sequence[RoadUserGraph],
    record: DeepAccidentViewRecord,
    outcome: str,
) -> list[Any]:
    collision_model_frame = int(record.collision_frame) * 2
    labels = SequenceDangerLabels(
        scenario_id=graphs[0].scenario_id,
        anchor_frame_position=None,
        frame_labels=tuple(
            _frame_label(graph, collision_model_frame) for graph in graphs
        ),
        source_metadata={
            "protocol": "deepaccident-independent-collision-anchor-v1",
            "collision_identity_used_as_model_input": False,
        },
    )
    packaged = list(package_pyg_sequence(graphs, labels).graphs)
    event_uid = f"deepaccident:{record.pair_key}:{record.focal_role}:{outcome}"
    for graph in packaged:
        graph.event_uid = event_uid
        graph.event_variant = outcome
        graph.critical_frame = torch.tensor([collision_model_frame], dtype=torch.long)
    return packaged


def _prepare_view(record: DeepAccidentViewRecord) -> dict[str, list[Any]]:
    result = {}
    for outcome in ("accident", "normal"):
        scenario = load_deepaccident_view(record, outcome=outcome)
        graphs = build_frozen_model_graphs(
            scenario,
            max_input_frames=MAX_INPUT_FRAMES,
        )
        result[outcome] = _package_sequence(graphs, record, outcome)
    if len(result["accident"]) != len(result["normal"]):
        raise ValueError(f"Unmatched model-clock horizons for {record.pair_key}")
    return result


def _load_model(bundle: Path, seed: int, fold: int) -> tuple[Any, dict[str, Any], torch.device]:
    checkpoint_path = bundle / "checkpoints" / f"{seed}_{fold}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training = _load_training_module()
    config = training.ModelConfig(**checkpoint["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training.build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["normalization"], device


def _normalize(sequence: Sequence[Any], normalization: dict[str, Any]) -> list[Any]:
    return _NormalizedSequenceView(
        [list(sequence)],
        normalization["x_mean"],
        normalization["x_std"],
        normalization["edge_mean"],
        normalization["edge_std"],
    )[0]


def _geometry(sequences: Sequence[Sequence[Any]]) -> dict[tuple[str, int], dict[str, dict[str, float]]]:
    observations = {}
    for sequence in sequences:
        for position, graph in enumerate(sequence):
            observations[(str(graph.event_uid), position)] = _outgoing_ego_relations(graph)
    return _geometry_from_observations(
        observations,
        horizon_seconds=2.0,
        clearance_m=3.5,
    )


def _frozen_pair_and_scene(
    bundle: Path,
    seed: int,
    fold: int,
    budget: float,
    events: dict[str, list[dict[str, Any]]],
    geometry: dict[tuple[str, int], dict[str, dict[str, float]]],
) -> tuple[
    dict[tuple[str, int, str], dict[str, float]],
    dict[str, dict[tuple[str, int], float]],
    dict[str, float],
]:
    base = bundle / "calibration" / f"{seed}_{fold}_{int(budget * 100)}"
    config = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    references = np.load(base.with_suffix(".npz"), allow_pickle=False)
    candidate = d.Candidate(**config["candidate"])
    components = d._merged_components(
        events,
        d._recompute_components(
            events,
            persistence_decay=candidate.persistence_decay,
            recovery_warmup=candidate.recovery_warmup,
        ),
    )
    activation = d._activation_fn(
        "rir_nested_selected", config["component_calibration"], candidate
    )
    pair, scene = pair_and_scene_scores(
        events,
        components,
        geometry,
        references["cpa"],
        activation,
        references["damp"],
        activation,
        references["damp"],
        references["raw"],
    )
    scores = simple_controls(events, geometry, references["cpa"], references["raw"])
    scores["rir_all_safe_selected"] = {
        key: values["rir"] for key, values in scene.items()
    }
    return pair, scores, {key: float(value) for key, value in config["thresholds"].items()}


def _event_score(
    frames: Sequence[dict[str, Any]],
    scores: dict[tuple[str, int], float],
) -> tuple[float, list[tuple[dict[str, Any], float]]]:
    usable = d._usable_frames(frames)
    trace = [
        (frame, float(scores[(str(frame["event_uid"]), int(frame["frame_position"]))]))
        for frame in usable
    ]
    return d.event_score(trace), trace


def _first_crossing(
    trace: Sequence[tuple[dict[str, Any], float]], threshold: float
) -> tuple[int | None, float | None]:
    crossings = [
        frame for frame, score in trace if float(score) >= float(threshold)
    ]
    if not crossings:
        return None, None
    first = min(crossings, key=lambda item: int(item["frame_position"]))
    lead_frames = int(first["eval_frames_to_anchor"])
    return int(first["frame_position"]), lead_frames / MODEL_HZ


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def _summaries(event_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    groups: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        groups[(float(row["budget"]), str(row["method"]), str(row["outcome"]))].append(row)
    for (budget, method, outcome), items in sorted(groups.items()):
        seed_rates = []
        for seed in SEEDS:
            seed_items = [item for item in items if int(item["seed"]) == seed]
            seed_rates.append(float(np.mean([int(item["hit"]) for item in seed_items])))
        family_values = []
        for family in sorted({str(item["family"]) for item in items}):
            family_items = [item for item in items if str(item["family"]) == family]
            family_values.append(float(np.mean([int(item["hit"]) for item in family_items])))
        seed_mean, seed_sd = _mean_sd(seed_rates)
        rows.append(
            {
                "budget": budget,
                "method": method,
                "outcome": outcome,
                "view_count": len({str(item["view_uid"]) for item in items}),
                "family_count": len(family_values),
                "view_rate_mean_over_seeds": seed_mean,
                "view_rate_seed_sd": seed_sd,
                "family_balanced_rate": float(np.mean(family_values)),
                "mean_event_score": float(np.mean([float(item["score"]) for item in items])),
                "mean_threshold": float(np.mean([float(item["threshold"]) for item in items])),
            }
        )
    return rows


def _paired_summaries(event_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {
        (int(row["seed"]), float(row["budget"]), str(row["method"]), str(row["view_uid"]), str(row["outcome"])): row
        for row in event_rows
    }
    summaries = []
    for budget in BUDGETS:
        for method in METHODS:
            pairs = []
            for seed in SEEDS:
                view_ids = sorted(
                    {
                        key[3]
                        for key in keyed
                        if key[0] == seed and key[1] == budget and key[2] == method
                    }
                )
                for view_uid in view_ids:
                    accident = keyed[(seed, budget, method, view_uid, "accident")]
                    normal = keyed[(seed, budget, method, view_uid, "normal")]
                    pairs.append((accident, normal))
            summaries.append(
                {
                    "budget": budget,
                    "method": method,
                    "seed_view_pairs": len(pairs),
                    "accident_only": sum(int(a["hit"]) and not int(n["hit"]) for a, n in pairs),
                    "both": sum(int(a["hit"]) and int(n["hit"]) for a, n in pairs),
                    "normal_only": sum(not int(a["hit"]) and int(n["hit"]) for a, n in pairs),
                    "neither": sum(not int(a["hit"]) and not int(n["hit"]) for a, n in pairs),
                    "mean_paired_score_margin": float(
                        np.mean([float(a["score"]) - float(n["score"]) for a, n in pairs])
                    ),
                    "positive_margin_fraction": float(
                        np.mean([float(a["score"]) > float(n["score"]) for a, n in pairs])
                    ),
                }
            )
    return summaries


def _summary_markdown(
    summaries: Sequence[dict[str, Any]],
    paired: Sequence[dict[str, Any]],
    localization: Sequence[dict[str, Any]],
    *,
    title: str,
    view_count: int,
    family_count: int,
    interpretation_boundary: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"This is a frozen external evaluation on {view_count} collision-involved "
        f"focal views from {family_count} scenario families. The recordings stop "
        "about 1.0 s before collision, so scoring uses only the available overlap "
        "with the frozen 1.5 s pre-anchor window.",
        "",
        "| Budget | Method | Accident hit rate | Matched-normal candidate rate | Family-balanced accident | Family-balanced normal |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    lookup = {
        (float(row["budget"]), str(row["method"]), str(row["outcome"])): row
        for row in summaries
    }
    for budget in BUDGETS:
        for method in PRIMARY_METHODS:
            accident = lookup[(budget, method, "accident")]
            normal = lookup[(budget, method, "normal")]
            lines.append(
                f"| {budget:.0%} | {method} | "
                f"{accident['view_rate_mean_over_seeds']:.1%} | "
                f"{normal['view_rate_mean_over_seeds']:.1%} | "
                f"{accident['family_balanced_rate']:.1%} | "
                f"{normal['family_balanced_rate']:.1%} |"
            )
    lines.extend(["", "## Paired view outcomes", ""])
    for row in paired:
        if float(row["budget"]) == 0.10 and row["method"] in PRIMARY_METHODS:
            lines.append(
                f"- {row['method']}: accident-only {row['accident_only']}, both {row['both']}, "
                f"normal-only {row['normal_only']}, neither {row['neither']} across "
                f"{row['seed_view_pairs']} seed-view pairs; positive score margin in "
                f"{row['positive_margin_fraction']:.1%}."
            )
    primary_localization = [
        row for row in localization if float(row["budget"]) == 0.10 and int(row["hit"])
    ]
    if primary_localization:
        lines.extend(
            [
                "",
                "## Localization",
                "",
                f"At the first 10% RIR crossing, the collision partner is top-ranked in "
                f"{sum(int(row['target_top1']) for row in primary_localization)}/"
                f"{len(primary_localization)} detected seed-view cases.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            interpretation_boundary,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise ValueError(f"Refuse to overwrite external-pilot output: {output_root}")
    protocol = _verify_protocol(protocol_path, manifest, bundle)
    runtime_hashes = _verify_frozen_runtime(bundle)
    records = load_view_records(manifest)
    family_count = len({record.pair_key for record in records})
    expected_views = int(protocol.get("eligible_view_count", 11))
    expected_families = int(protocol.get("eligible_family_count", 7))
    if len(records) != expected_views or family_count != expected_families:
        raise ValueError(
            "Protocol lock expects exactly "
            f"{expected_views} views from {expected_families} families, found "
            f"{len(records)} views from {family_count} families."
        )
    output_root.mkdir(parents=True)
    _write_json(
        output_root / "input_receipt.json",
        _input_receipt(
            records,
            manifest=manifest,
            protocol=protocol_path,
            bundle=bundle,
            runtime_hashes=runtime_hashes,
        ),
    )
    _write_json(output_root / "protocol.json", protocol)

    prepared = {f"{r.pair_key}:{r.focal_role}": _prepare_view(r) for r in records}
    event_rows: list[dict[str, Any]] = []
    localization_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    model_cache: dict[tuple[int, int], tuple[Any, dict[str, Any], torch.device]] = {}

    for view_index, record in enumerate(records, start=1):
        view_uid = f"{record.pair_key}:{record.focal_role}"
        fold = family_fold(record.pair_key)
        print(f"[{view_index}/{len(records)}] {view_uid} -> fold {fold}", flush=True)
        raw_sequences = prepared[view_uid]
        for seed in SEEDS:
            if (seed, fold) not in model_cache:
                model_cache[(seed, fold)] = _load_model(bundle, seed, fold)
            model, normalization, device = model_cache[(seed, fold)]
            normalized = {
                outcome: _normalize(sequence, normalization)
                for outcome, sequence in raw_sequences.items()
            }
            scored_frames = {
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
            events = {
                str(frames[0]["event_uid"]): frames
                for frames in scored_frames.values()
            }
            geometry = _geometry(list(normalized.values()))
            for budget in BUDGETS:
                pair_scores, method_scores, thresholds = _frozen_pair_and_scene(
                    bundle, seed, fold, budget, events, geometry
                )
                for outcome, frames in scored_frames.items():
                    event_uid = str(frames[0]["event_uid"])
                    for method in METHODS:
                        score, trace = _event_score(frames, method_scores[method])
                        threshold = thresholds[method]
                        first_position, lead_seconds = _first_crossing(trace, threshold)
                        event_rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "budget": budget,
                                "method": method,
                                "family": record.pair_key,
                                "view_uid": view_uid,
                                "scenario_id": record.scenario_id,
                                "focal_role": record.focal_role,
                                "outcome": outcome,
                                "score": score,
                                "threshold": threshold,
                                "hit": int(score >= threshold),
                                "first_crossing_frame_position": first_position,
                                "first_crossing_lead_seconds": lead_seconds,
                                "available_eval_frame_count": len(trace),
                                "last_observation_lead_seconds": (
                                    record.collision_frame
                                    - int(raw_sequences[outcome][-1].frame_index.item()) / 2
                                )
                                * 0.1,
                            }
                        )
                        if method == "rir_all_safe_selected":
                            for frame, value in trace:
                                position = int(frame["frame_position"])
                                frame_rows.append(
                                    {
                                        "seed": seed,
                                        "fold": fold,
                                        "budget": budget,
                                        "family": record.pair_key,
                                        "view_uid": view_uid,
                                        "outcome": outcome,
                                        "frame_position": position,
                                        "lead_seconds": int(frame["eval_frames_to_anchor"]) / MODEL_HZ,
                                        "rir_score": value,
                                        "rir_threshold": threshold,
                                    }
                                )
                    if outcome == "accident" and record.localization_eligible:
                        score, trace = _event_score(
                            frames, method_scores["rir_all_safe_selected"]
                        )
                        threshold = thresholds["rir_all_safe_selected"]
                        first_position, lead_seconds = _first_crossing(trace, threshold)
                        candidates = []
                        if first_position is not None:
                            for (uid, position, target), values in pair_scores.items():
                                if uid == event_uid and position == first_position:
                                    candidates.append((float(values["rir"]), str(target)))
                        candidates.sort(key=lambda item: (-item[0], item[1]))
                        predicted = "" if not candidates else candidates[0][1]
                        localization_rows.append(
                            {
                                "seed": seed,
                                "fold": fold,
                                "budget": budget,
                                "family": record.pair_key,
                                "view_uid": view_uid,
                                "scenario_id": record.scenario_id,
                                "focal_role": record.focal_role,
                                "target_actor_id": record.target_actor_id,
                                "hit": int(score >= threshold),
                                "first_crossing_lead_seconds": lead_seconds,
                                "predicted_top_actor_id": predicted,
                                "target_top1": int(
                                    bool(predicted)
                                    and predicted == str(record.target_actor_id)
                                ),
                                "candidate_actor_count": len(candidates),
                            }
                        )

    summaries = _summaries(event_rows)
    paired = _paired_summaries(event_rows)
    _write_csv(output_root / "events.csv", event_rows)
    _write_csv(output_root / "frame_scores_rir.csv", frame_rows)
    _write_csv(output_root / "localization.csv", localization_rows)
    _write_csv(output_root / "summary_metrics.csv", summaries)
    _write_csv(output_root / "paired_outcomes.csv", paired)
    (output_root / "SUMMARY.md").write_text(
        _summary_markdown(
            summaries,
            paired,
            localization_rows,
            title=str(protocol.get("summary_title", "DeepAccident mini frozen-RIR pilot")),
            view_count=len(records),
            family_count=family_count,
            interpretation_boundary=str(
                protocol.get(
                    "interpretation_boundary",
                    "The mini split is too small for a publication-level external "
                    "claim. Three seeds quantify model variation but do not increase "
                    "the number of independent scenario families.",
                )
            ),
        ),
        encoding="utf-8",
    )
    _write_json(
        output_root / "completion.json",
        {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "status": str(
                protocol.get("completion_status", "completed_external_pilot_descriptive_only")
            ),
            "views": len(records),
            "families": len({record.pair_key for record in records}),
            "seeds": list(SEEDS),
            "budgets": list(BUDGETS),
            "event_rows": len(event_rows),
            "localization_rows": len(localization_rows),
            "output_tree_sha256_excluding_completion": _tree_digest(
                path for path in output_root.iterdir() if path.name != "completion.json"
            )[0],
        },
    )
    print((output_root / "SUMMARY.md").read_text(encoding="utf-8"))
    print(f"output: {output_root}")


if __name__ == "__main__":
    main()
