"""Build and load an indexed, per-sequence cached RiskBench graph dataset."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from ..schema import AgentType
from .adapter import (
    DEFAULT_METADATA_ROOT,
    RISK_BENCH_GRAPH_INPUT_PROTOCOL,
    RiskBenchScenarioRecord,
)
from .manifest import build_manifest
from .smoke import _package_record


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "ego_immune_riskbench_metadata_lazy_v1"
)
DEFAULT_SPLIT_COUNTS = {
    "train_normal": 160,
    "validation_normal": 50,
    "test_normal": 50,
    "validation_positive": 50,
    "test_positive": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-normal", type=int, default=160)
    parser.add_argument("--validation-normal", type=int, default=50)
    parser.add_argument("--test-normal", type=int, default=50)
    parser.add_argument("--validation-positive", type=int, default=50)
    parser.add_argument("--test-positive", type=int, default=50)
    parser.add_argument("--pre-critical-frames", type=int, default=40)
    parser.add_argument("--post-critical-frames", type=int, default=20)
    parser.add_argument("--max-normal-frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def _hash_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _stratified_family_sample(
    records: Sequence[RiskBenchScenarioRecord],
    *,
    split: str,
    count: int,
    seed: int,
) -> list[RiskBenchScenarioRecord]:
    grouped: dict[str, list[RiskBenchScenarioRecord]] = {}
    for record in records:
        if record.split == split:
            grouped.setdefault(record.scenario_group_id, []).append(record)
    if len(grouped) < count:
        raise ValueError(
            f"Only {len(grouped)} independent families for {split}, requested {count}"
        )

    representatives = []
    for group_id, variants in grouped.items():
        representatives.append(
            min(
                variants,
                key=lambda record: _hash_rank(record.scenario_id, seed),
            )
        )
    by_type: dict[str, list[RiskBenchScenarioRecord]] = {}
    for record in representatives:
        by_type.setdefault(record.target_type_name, []).append(record)
    for type_records in by_type.values():
        type_records.sort(key=lambda record: _hash_rank(record.scenario_group_id, seed))

    selected = []
    type_names = sorted(by_type)
    while len(selected) < count:
        progressed = False
        for type_name in type_names:
            if by_type[type_name] and len(selected) < count:
                selected.append(by_type[type_name].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _record_payload(record: RiskBenchScenarioRecord) -> dict[str, Any]:
    return {
        "scenario_id": record.scenario_id,
        "category": record.category,
        "state_path": str(record.state_path.resolve()),
        "scenario_group_id": record.scenario_group_id,
        "base_scenario_id": record.base_scenario_id,
        "weather": record.weather,
        "traffic_density": record.traffic_density,
        "risk_actor_ids": list(record.risk_actor_ids),
        "critical_frame": record.critical_frame,
        "behavior_window": record.behavior_window,
        "target_type_code": record.target_type_code,
        "target_type_name": record.target_type_name,
        "target_agent_type": int(record.target_agent_type),
        "pool_role": record.pool_role,
        "split": record.split,
        "frame_count": record.frame_count,
        "actor_count": record.actor_count,
    }


def _record_from_payload(payload: dict[str, Any]) -> RiskBenchScenarioRecord:
    behavior = payload.get("behavior_window")
    return RiskBenchScenarioRecord(
        scenario_id=str(payload["scenario_id"]),
        category=str(payload["category"]),
        state_path=Path(str(payload["state_path"])),
        scenario_group_id=str(payload["scenario_group_id"]),
        base_scenario_id=str(payload["base_scenario_id"]),
        weather=str(payload["weather"]),
        traffic_density=str(payload["traffic_density"]),
        risk_actor_ids=tuple(str(value) for value in payload["risk_actor_ids"]),
        critical_frame=(
            None if payload.get("critical_frame") is None else int(payload["critical_frame"])
        ),
        behavior_window=(
            None if behavior is None else (int(behavior[0]), int(behavior[1]))
        ),
        target_type_code=str(payload["target_type_code"]),
        target_type_name=str(payload["target_type_name"]),
        target_agent_type=AgentType(int(payload["target_agent_type"])),
        pool_role=str(payload["pool_role"]),
        split=str(payload["split"]),
        frame_count=int(payload["frame_count"]),
        actor_count=int(payload["actor_count"]),
    )


def build_lazy_index(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    split_counts: dict[str, int] | None = None,
    pre_critical_frames: int = 40,
    post_critical_frames: int = 20,
    max_normal_frames: int = 80,
    seed: int = 20260721,
) -> dict[str, Any]:
    if max_normal_frames < 2:
        raise ValueError("max_normal_frames must be at least 2")
    split_counts = split_counts or DEFAULT_SPLIT_COUNTS
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records, manifest_summary = build_manifest(metadata_root, output_root)
    selected = {
        split: _stratified_family_sample(
            records,
            split=split,
            count=count,
            seed=seed,
        )
        for split, count in split_counts.items()
    }
    index_path = output_root / "riskbench_metadata_lazy_index.json"
    payload = {
        "format": "riskbench-metadata-lazy-v1",
        "protocol": "riskbench-metadata-three-pool-family-stratified-v1",
        "graph_input_protocol": RISK_BENCH_GRAPH_INPUT_PROTOCOL,
        "metadata_root": str(metadata_root.expanduser().resolve()),
        "cache_root": str((output_root / "sequence_cache").resolve()),
        "pre_critical_frames": pre_critical_frames,
        "post_critical_frames": post_critical_frames,
        "max_normal_frames": max_normal_frames,
        "seed": seed,
        "splits": {
            split: [_record_payload(record) for record in split_records]
            for split, split_records in selected.items()
        },
        "meta": {
            "dataset_name": "RiskBench metadata-only lazy ego immune recognition",
            "metadata_only": True,
            "full_sensor_archives_required": False,
            "gradient_allowed_splits": ["train_normal"],
            "positive_gradient_sequence_count": 0,
            "sampling_unit": "base scenario family",
            "sampling_order": "target-type-stratified deterministic hash",
            "graph_input_protocol": RISK_BENCH_GRAPH_INPUT_PROTOCOL,
            "gt_target_type_used_as_input": False,
        },
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "protocol": payload["protocol"],
        "index_path": str(index_path),
        "manifest_summary_path": manifest_summary["summary_path"],
        "split_counts": {split: len(values) for split, values in selected.items()},
        "total_sequence_count": sum(len(values) for values in selected.values()),
        "target_type_counts": {
            split: dict(Counter(record.target_type_name for record in values))
            for split, values in selected.items()
        },
        "unique_family_count": len(
            {
                record.scenario_group_id
                for values in selected.values()
                for record in values
            }
        ),
        "full_sensor_archives_required": False,
        "cache_policy": "one PyG sequence per file, generated on first access",
        "status": "lazy_index_ready_cache_not_materialized",
    }
    summary_path = output_root / "lazy_dataset_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class LazyRiskBenchSplit(Sequence[list[Any]]):
    def __init__(
        self,
        *,
        split: str,
        records: Sequence[dict[str, Any]],
        cache_root: Path,
        pre_critical_frames: int,
        post_critical_frames: int,
        max_normal_frames: int,
        seed: int,
        memory_cache_size: int = 2,
    ) -> None:
        self.split = split
        self.records = list(records)
        self.cache_root = cache_root / split
        self.pre_critical_frames = pre_critical_frames
        self.post_critical_frames = post_critical_frames
        self.max_normal_frames = max_normal_frames
        self.seed = seed
        self.memory_cache_size = memory_cache_size
        self._memory_cache: OrderedDict[int, list[Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, index: int) -> Path:
        scenario_id = str(self.records[index]["scenario_id"])
        token = hashlib.sha256(scenario_id.encode("utf-8")).hexdigest()[:20]
        return self.cache_root / f"{token}.pt"

    def _build(self, index: int) -> list[Any]:
        record = _record_from_payload(self.records[index])
        packaged = _package_record(
            record,
            pre_critical_frames=self.pre_critical_frames,
            post_critical_frames=self.post_critical_frames,
        )
        graphs = list(packaged.graphs)
        if record.critical_frame is None and len(graphs) > self.max_normal_frames:
            span = len(graphs) - self.max_normal_frames + 1
            start = int(_hash_rank(record.scenario_id, self.seed)[:12], 16) % span
            graphs = graphs[start : start + self.max_normal_frames]
        return graphs

    def __getitem__(self, index: int) -> list[Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index in self._memory_cache:
            value = self._memory_cache.pop(index)
            self._memory_cache[index] = value
            return value
        path = self._cache_path(index)
        if path.exists():
            value = torch.load(path, weights_only=False)
        else:
            value = self._build(index)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, path)
        self._memory_cache[index] = value
        while len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)
        return value

    def __iter__(self) -> Iterator[list[Any]]:
        for index in range(len(self)):
            yield self[index]


def load_lazy_package(
    index_path: Path,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = index_path.expanduser().resolve()
    payload = payload or json.loads(resolved.read_text(encoding="utf-8"))
    cache_root = Path(str(payload["cache_root"]))
    splits = {
        split: LazyRiskBenchSplit(
            split=split,
            records=records,
            cache_root=cache_root,
            pre_critical_frames=int(payload["pre_critical_frames"]),
            post_critical_frames=int(payload["post_critical_frames"]),
            max_normal_frames=int(payload["max_normal_frames"]),
            seed=int(payload["seed"]),
        )
        for split, records in payload["splits"].items()
    }
    splits["train_positive"] = []
    return {
        "splits": splits,
        "sequence_index": payload["splits"],
        "meta": {**payload["meta"], "lazy_index_path": str(resolved)},
    }


def main() -> None:
    args = parse_args()
    summary = build_lazy_index(
        metadata_root=args.metadata_root,
        output_root=args.output_root,
        split_counts={
            "train_normal": args.train_normal,
            "validation_normal": args.validation_normal,
            "test_normal": args.test_normal,
            "validation_positive": args.validation_positive,
            "test_positive": args.test_positive,
        },
        pre_critical_frames=args.pre_critical_frames,
        post_critical_frames=args.post_critical_frames,
        max_normal_frames=args.max_normal_frames,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
