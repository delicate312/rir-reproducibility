"""Audit a DeepAccident mini archive before building an RIR adapter.

The audit reads only the plain-text ``meta`` and per-frame ``label`` files.  It
does not load the pickle calibration files or any image/LiDAR payload.  The
result makes three dataset properties explicit:

1. whether an ``accident`` directory contains valid collision metadata;
2. which instrumented vehicle was actually involved in that collision; and
3. whether the other collision participant is observable before the anchor.

DeepAccident stores matched accident/normal runs under the same scenario name.
The generated manifest keeps that pairing for later external validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_MINI_ROOT = Path(
    "/Volumes/melancholia/RiskBench_download/DeepAccident_mini"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "ego_immune_deepaccident_mini_v0"
)
DEEPACCIDENT_TIMESTEP_SECONDS = 0.1

ROLE_TO_FOLDER = {
    "ego": "ego_vehicle",
    "ego_behind": "ego_vehicle_behind",
    "other": "other_vehicle",
    "other_behind": "other_vehicle_behind",
}
ROLE_TO_AGENT_INDEX = {
    "ego": 0,
    "ego_behind": 1,
    "other": 2,
    "other_behind": 3,
}


@dataclass(frozen=True)
class DeepAccidentObject:
    """One road-user annotation in a focal vehicle's coordinate frame."""

    class_name: str
    x_m: float
    y_m: float
    z_m: float
    length_m: float
    width_m: float
    height_m: float
    yaw_rad: float
    velocity_x_mps: float
    velocity_y_mps: float
    actor_id: int
    lidar_point_count: int
    camera_visible: bool


@dataclass(frozen=True)
class DeepAccidentFrameLabel:
    """Ego motion and annotated road users for one recorded frame."""

    ego_velocity_x_mps: float
    ego_velocity_y_mps: float
    objects: tuple[DeepAccidentObject, ...]


@dataclass(frozen=True)
class DeepAccidentMeta:
    """Scenario-level fields needed for collision-anchored evaluation."""

    scenario_id: str
    weather: str
    collision_actor_ids: tuple[int, ...]
    collision_actor_types: tuple[str, ...]
    collision_frame: int | None
    colliding_roles: tuple[str, ...]
    instrumented_actor_ids: tuple[int, ...]
    road_type: str
    another_vehicle_spawn_side: str
    ego_vehicle_direction: str
    other_vehicle_direction: str

    @property
    def has_collision(self) -> bool:
        return self.collision_frame is not None and len(self.collision_actor_ids) == 2


def _value_after_colon(line: str, field: str) -> str:
    if ":" not in line:
        raise ValueError(f"Malformed {field} line: {line!r}")
    return line.split(":", 1)[1].strip()


def parse_meta(path: Path) -> DeepAccidentMeta:
    """Parse one DeepAccident scenario metadata file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Expected seven metadata lines in {path}")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Malformed metadata header in {path}")
    weather = header[0]
    if header[1] == "-1":
        collision_actor_ids: tuple[int, ...] = ()
        collision_actor_types: tuple[str, ...] = ()
        collision_frame = None
    else:
        if len(header) < 6:
            raise ValueError(f"Incomplete collision metadata in {path}")
        collision_actor_ids = (int(header[1]), int(header[3]))
        collision_actor_types = (header[2], header[4])
        collision_frame = int(header[-1])

    colliding_roles = tuple(
        role
        for role in _value_after_colon(lines[1], "colliding agents").split()
        if role != "none"
    )
    instrumented_actor_ids = tuple(
        int(value)
        for value in _value_after_colon(lines[2], "agents id").split()
    )
    if len(instrumented_actor_ids) != 4:
        raise ValueError(f"Expected four instrumented actor IDs in {path}")

    return DeepAccidentMeta(
        scenario_id=path.stem,
        weather=weather,
        collision_actor_ids=collision_actor_ids,
        collision_actor_types=collision_actor_types,
        collision_frame=collision_frame,
        colliding_roles=colliding_roles,
        instrumented_actor_ids=instrumented_actor_ids,
        road_type=_value_after_colon(lines[3], "road type"),
        another_vehicle_spawn_side=_value_after_colon(
            lines[4], "another vehicle spawn side"
        ),
        ego_vehicle_direction=_value_after_colon(
            lines[5], "ego vehicle direction"
        ),
        other_vehicle_direction=_value_after_colon(
            lines[6], "other vehicle direction"
        ),
    )


def parse_frame_label(path: Path) -> DeepAccidentFrameLabel:
    """Parse one focal-view label without loading sensor or pickle files."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty frame label: {path}")
    ego_motion = lines[0].split()
    if len(ego_motion) != 2:
        raise ValueError(f"Malformed ego-motion header in {path}")

    objects: list[DeepAccidentObject] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 13:
            raise ValueError(
                f"Expected 13 object fields at {path}:{line_number}, "
                f"found {len(fields)}"
            )
        objects.append(
            DeepAccidentObject(
                class_name=fields[0],
                x_m=float(fields[1]),
                y_m=float(fields[2]),
                z_m=float(fields[3]),
                length_m=float(fields[4]),
                width_m=float(fields[5]),
                height_m=float(fields[6]),
                yaw_rad=float(fields[7]),
                velocity_x_mps=float(fields[8]),
                velocity_y_mps=float(fields[9]),
                actor_id=int(fields[10]),
                lidar_point_count=int(fields[11]),
                camera_visible=fields[12] == "True",
            )
        )
    return DeepAccidentFrameLabel(
        ego_velocity_x_mps=float(ego_motion[0]),
        ego_velocity_y_mps=float(ego_motion[1]),
        objects=tuple(objects),
    )


def _frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _label_paths(
    category_root: Path,
    role: str,
    scenario_id: str,
) -> list[Path]:
    folder = ROLE_TO_FOLDER[role]
    return sorted(
        (category_root / folder / "label" / scenario_id).glob("*.txt"),
        key=_frame_index,
    )


def _normal_category_name(accident_category_name: str) -> str:
    if not accident_category_name.endswith("_accident"):
        raise ValueError(accident_category_name)
    return accident_category_name[: -len("_accident")] + "_normal"


def _target_trace(
    label_paths: Iterable[Path],
    target_actor_id: int,
    collision_frame: int,
) -> dict[str, int | float | None]:
    present_frames: list[int] = []
    visible_frames: list[int] = []
    for label_path in label_paths:
        frame_index = _frame_index(label_path)
        frame = parse_frame_label(label_path)
        target = next(
            (obj for obj in frame.objects if obj.actor_id == target_actor_id),
            None,
        )
        if target is None:
            continue
        present_frames.append(frame_index)
        if target.camera_visible:
            visible_frames.append(frame_index)

    pre_anchor_present = [frame for frame in present_frames if frame < collision_frame]
    pre_anchor_visible = [frame for frame in visible_frames if frame < collision_frame]
    return {
        "target_first_present_frame": min(present_frames) if present_frames else None,
        "target_last_present_frame": max(present_frames) if present_frames else None,
        "target_pre_anchor_present_frames": len(pre_anchor_present),
        "target_pre_anchor_visible_frames": len(pre_anchor_visible),
        "earliest_target_lead_time_s": (
            None
            if not pre_anchor_present
            else (collision_frame - min(pre_anchor_present))
            * DEEPACCIDENT_TIMESTEP_SECONDS
        ),
        "latest_target_lead_time_s": (
            None
            if not pre_anchor_present
            else (collision_frame - max(pre_anchor_present))
            * DEEPACCIDENT_TIMESTEP_SECONDS
        ),
    }


def audit_dataset_roots(
    roots: Sequence[Path],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Audit one logical DeepAccident split stored across one or more parts.

    DeepAccident's archive parts are not self-contained: an accident member may
    be in one part while its matched normal member is in another.  The roots are
    therefore indexed together before any eligibility decision is made.
    """

    resolved_roots = tuple(root.expanduser().resolve() for root in roots)
    if not resolved_roots:
        raise ValueError("At least one DeepAccident root is required")
    for root in resolved_roots:
        if not root.exists():
            raise FileNotFoundError(root)

    category_index: dict[tuple[str, str], Path] = {}
    for root in resolved_roots:
        for category_root in sorted(path for path in root.iterdir() if path.is_dir()):
            meta_root = category_root / "meta"
            if not meta_root.exists():
                continue
            for meta_path in sorted(meta_root.glob("*.txt")):
                key = (category_root.name, meta_path.stem)
                if key in category_index:
                    raise ValueError(f"Duplicate DeepAccident entry across parts: {key}")
                category_index[key] = category_root

    rows: list[dict[str, object]] = []
    accident_entries = sorted(
        (category_name, scenario_id, category_root)
        for (category_name, scenario_id), category_root in category_index.items()
        if category_name.endswith("_accident")
    )
    scenario_pair_count = 0
    accident_without_collision: list[str] = []
    collision_scenarios: set[str] = set()

    for accident_category_name, scenario_id, accident_root in accident_entries:
        normal_category_name = _normal_category_name(accident_category_name)
        normal_root = category_index.get((normal_category_name, scenario_id))
        meta_path = accident_root / "meta" / f"{scenario_id}.txt"
        scenario_pair_count += 1
        meta = parse_meta(meta_path)
        normal_meta_path = (
            normal_root / "meta" / meta_path.name if normal_root is not None else None
        )
        paired_normal_exists = normal_meta_path is not None and normal_meta_path.exists()
        if not meta.has_collision:
            accident_without_collision.append(
                f"{accident_root.name}/{meta.scenario_id}"
            )
            continue
        collision_scenarios.add(f"{accident_category_name}:{meta.scenario_id}")

        for role in meta.colliding_roles:
            base: dict[str, object] = {
                "scenario_id": meta.scenario_id,
                "pair_key": (
                    f"{accident_root.name.removesuffix('_accident')}:"
                    f"{meta.scenario_id}"
                ),
                "accident_category": accident_root.name,
                "normal_category": normal_category_name,
                "focal_role": role,
                "focal_folder": ROLE_TO_FOLDER.get(role),
                "weather": meta.weather,
                "road_type": meta.road_type,
                "collision_frame": meta.collision_frame,
                "paired_normal_exists": paired_normal_exists,
            }
            if role not in ROLE_TO_FOLDER:
                base.update(
                    detection_eligible=False,
                    localization_eligible=False,
                    usable=False,
                    exclusion_reason="unknown_colliding_role",
                    localization_exclusion_reason="unknown_colliding_role",
                )
                rows.append(base)
                continue

            focal_actor_id = meta.instrumented_actor_ids[ROLE_TO_AGENT_INDEX[role]]
            base["focal_actor_id"] = focal_actor_id
            if focal_actor_id not in meta.collision_actor_ids:
                base.update(
                    detection_eligible=False,
                    localization_eligible=False,
                    usable=False,
                    exclusion_reason="role_actor_id_not_in_collision_pair",
                    localization_exclusion_reason=(
                        "role_actor_id_not_in_collision_pair"
                    ),
                )
                rows.append(base)
                continue

            target_actor_id = (
                meta.collision_actor_ids[1]
                if focal_actor_id == meta.collision_actor_ids[0]
                else meta.collision_actor_ids[0]
            )
            base["target_actor_id"] = target_actor_id
            target_index = meta.collision_actor_ids.index(target_actor_id)
            base["target_actor_type"] = meta.collision_actor_types[target_index]
            label_paths = _label_paths(accident_root, role, meta.scenario_id)
            normal_label_paths = (
                _label_paths(normal_root, role, meta.scenario_id)
                if normal_root is not None
                else []
            )
            base.update(
                accident_frame_count=len(label_paths),
                normal_frame_count=len(normal_label_paths),
                last_observed_frame=(
                    _frame_index(label_paths[-1]) if label_paths else None
                ),
                observation_to_collision_gap_s=(
                    None
                    if not label_paths
                    else (meta.collision_frame - _frame_index(label_paths[-1]))
                    * DEEPACCIDENT_TIMESTEP_SECONDS
                ),
            )
            trace = _target_trace(label_paths, target_actor_id, meta.collision_frame)
            base.update(trace)

            detection_exclusion_reason = ""
            if not paired_normal_exists or not normal_label_paths:
                detection_exclusion_reason = "missing_paired_normal_view"
            elif not label_paths:
                detection_exclusion_reason = "missing_accident_view"
            localization_eligible = (
                not detection_exclusion_reason
                and int(trace["target_pre_anchor_present_frames"] or 0) > 0
            )
            base.update(
                detection_eligible=not detection_exclusion_reason,
                localization_eligible=localization_eligible,
                usable=not detection_exclusion_reason,
                exclusion_reason=detection_exclusion_reason,
                localization_exclusion_reason=(
                    ""
                    if localization_eligible
                    else (
                        detection_exclusion_reason
                        or "collision_partner_not_annotated_before_anchor"
                    )
                ),
                accident_label_root=str(
                    accident_root
                    / ROLE_TO_FOLDER[role]
                    / "label"
                    / meta.scenario_id
                ),
                normal_label_root=(
                    ""
                    if normal_root is None
                    else str(
                        normal_root
                        / ROLE_TO_FOLDER[role]
                        / "label"
                        / meta.scenario_id
                    )
                ),
                accident_meta_path=str(meta_path),
                normal_meta_path=(
                    "" if normal_meta_path is None else str(normal_meta_path)
                ),
            )
            rows.append(base)

    usable_rows = [row for row in rows if row["usable"]]
    gap_values = [
        float(row["observation_to_collision_gap_s"])
        for row in usable_rows
        if row["observation_to_collision_gap_s"] is not None
    ]
    target_type_counts = {
        target_type: sum(
            row.get("target_actor_type") == target_type for row in usable_rows
        )
        for target_type in sorted(
            {str(row["target_actor_type"]) for row in usable_rows}
        )
    }
    summary: dict[str, object] = {
        "protocol": "deepaccident-multipart-text-label-audit-v1",
        "dataset_roots": [str(root) for root in resolved_roots],
        "timestep_seconds": DEEPACCIDENT_TIMESTEP_SECONDS,
        "matched_scenario_pairs": scenario_pair_count,
        "collision_scenarios": len(collision_scenarios),
        "accident_directories_without_collision_metadata": len(
            accident_without_collision
        ),
        "accident_without_collision_records": accident_without_collision,
        "collision_involved_focal_views": len(rows),
        "usable_paired_focal_views": len(usable_rows),
        "localization_eligible_focal_views": sum(
            bool(row.get("localization_eligible")) for row in usable_rows
        ),
        "excluded_focal_views": len(rows) - len(usable_rows),
        "exclusion_reasons": {
            reason: sum(row["exclusion_reason"] == reason for row in rows)
            for reason in sorted(
                {str(row["exclusion_reason"]) for row in rows if row["exclusion_reason"]}
            )
        },
        "usable_collision_scenarios": len(
            {str(row["scenario_id"]) for row in usable_rows}
        ),
        "usable_target_type_counts": target_type_counts,
        "observation_to_collision_gap_s": {
            "min": min(gap_values),
            "max": max(gap_values),
        },
        "all_usable_views_have_paired_normal": all(
            bool(row["paired_normal_exists"]) for row in usable_rows
        ),
        "all_usable_targets_present_before_anchor": all(
            int(row["target_pre_anchor_present_frames"]) > 0
            for row in usable_rows
        ),
        "all_usable_targets_camera_visible_before_anchor": all(
            int(row["target_pre_anchor_visible_frames"]) > 0
            for row in usable_rows
        ),
        "sensor_payload_required_for_rir_metadata_adapter": False,
        "next_step": (
            "Build an ego-relative text-label adapter, preserve matched "
            "accident/normal pairs, and use no RGB, LiDAR, future motion, or "
            "collision-partner identity as model input."
        ),
    }
    return rows, summary


def audit_mini_dataset(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Backward-compatible one-root audit."""

    return audit_dataset_roots([root])


def write_audit(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    output_root: Path,
    *,
    artifact_prefix: str = "deepaccident_mini",
) -> tuple[Path, Path]:
    """Write the focal-view manifest and machine-readable audit summary."""

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{artifact_prefix}_focal_view_manifest.csv"
    summary_path = output_root / f"{artifact_prefix}_audit_summary.json"
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(summary)
    summary["manifest_path"] = str(manifest_path)
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        help="DeepAccident archive root; repeat for a multipart split.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-prefix", default="deepaccident_mini")
    args = parser.parse_args()
    rows, summary = audit_dataset_roots(args.root or [DEFAULT_MINI_ROOT])
    manifest_path, summary_path = write_audit(
        rows,
        summary,
        args.output_root,
        artifact_prefix=args.artifact_prefix,
    )
    print(json.dumps(summary, indent=2))
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
