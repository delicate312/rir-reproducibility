from pathlib import Path

from experiments.ego_immune_recognition.deepaccident.adapter import (
    DeepAccidentViewRecord,
    causally_resample_scenario,
    load_deepaccident_view,
)


def _write_label(path: Path, target_x: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "0.0 0.0\n"
        "car 0 0 0 4.5 1.8 1.5 0 0 0 -100 100 True\n"
        f"car {target_x} 0 0 4.5 1.8 1.5 0 99 99 20 50 True\n",
        encoding="utf-8",
    )


def test_matched_adapter_uses_causal_difference_and_equal_horizon(
    tmp_path: Path,
) -> None:
    accident = tmp_path / "accident"
    normal = tmp_path / "normal"
    _write_label(accident / "scene_001.txt", 10.0)
    _write_label(accident / "scene_002.txt", 9.0)
    _write_label(normal / "scene_001.txt", 10.0)
    _write_label(normal / "scene_002.txt", 9.5)
    _write_label(normal / "scene_003.txt", 9.0)
    record = DeepAccidentViewRecord(
        scenario_id="scene",
        pair_key="pair:scene",
        focal_role="ego",
        focal_actor_id=10,
        target_actor_id=20,
        collision_frame=12,
        accident_label_root=accident,
        normal_label_root=normal,
        road_type="junction",
        weather="ClearNoon",
    )

    accident_scenario = load_deepaccident_view(record, outcome="accident")
    normal_scenario = load_deepaccident_view(record, outcome="normal")

    assert accident_scenario.frame_count == normal_scenario.frame_count == 2
    target = next(
        user
        for user in accident_scenario.frames[1].road_users
        if user.agent_id == "20"
    )
    assert target.velocity_xy_mps.tolist() == [-10.0, 0.0]
    assert target.velocity_xy_mps.tolist() != [99.0, 99.0]
    assert not accident_scenario.source_metadata["provided_velocity_used_as_input"]


def test_causal_resampling_uses_current_velocity_and_does_not_extend_recording(
    tmp_path: Path,
) -> None:
    accident = tmp_path / "accident"
    normal = tmp_path / "normal"
    _write_label(accident / "scene_001.txt", 10.0)
    _write_label(accident / "scene_002.txt", 9.0)
    _write_label(normal / "scene_001.txt", 10.0)
    _write_label(normal / "scene_002.txt", 9.5)
    record = DeepAccidentViewRecord(
        scenario_id="scene",
        pair_key="pair:scene",
        focal_role="ego",
        focal_actor_id=10,
        target_actor_id=20,
        collision_frame=12,
        accident_label_root=accident,
        normal_label_root=normal,
        road_type="junction",
        weather="ClearNoon",
    )

    native = load_deepaccident_view(record, outcome="accident")
    resampled = causally_resample_scenario(native)

    assert len(resampled.frames) == 3
    assert [frame.frame_index for frame in resampled.frames] == [2, 3, 4]
    assert [frame.timestamp_s for frame in resampled.frames] == [0.0, 0.05, 0.1]
    middle_target = next(
        user for user in resampled.frames[1].road_users if user.agent_id == "20"
    )
    # The first observation has no previous frame, so its causal velocity is
    # zero and the half-step is held at the current position.  In particular,
    # the adapter does not interpolate toward the unseen next position (9 m).
    assert middle_target.position_xy_m.tolist() == [10.0, 0.0]
    assert resampled.source_metadata["no_extension_after_last_observation"]
