from pathlib import Path

from experiments.ego_immune_recognition.deepaccident.mini_audit import (
    audit_dataset_roots,
    audit_mini_dataset,
    parse_frame_label,
    parse_meta,
)


def test_parse_meta_with_collision(tmp_path: Path) -> None:
    path = tmp_path / "Town01_type001_subtype0001_scenario00001.txt"
    path.write_text(
        "ClearNoon 10 car 20 pedestrian 1.0 opposite front 55\n"
        " colliding agents: ego none\n"
        " agents id: 10 11 12 13\n"
        " road_type: four-way junction\n"
        " another_vehicle_spawn_side: left\n"
        " ego_vehicle_direction: straight\n"
        " other_vehicle_direction: right\n",
        encoding="utf-8",
    )
    meta = parse_meta(path)
    assert meta.collision_actor_ids == (10, 20)
    assert meta.collision_frame == 55
    assert meta.colliding_roles == ("ego",)
    assert meta.instrumented_actor_ids == (10, 11, 12, 13)


def test_parse_frame_label(tmp_path: Path) -> None:
    path = tmp_path / "scenario_001.txt"
    path.write_text(
        "2.0 -1.0\n"
        "car 3 4 0 4.5 1.8 1.5 0.2 1.0 0.0 20 42 True\n",
        encoding="utf-8",
    )
    frame = parse_frame_label(path)
    assert frame.ego_velocity_x_mps == 2.0
    assert len(frame.objects) == 1
    assert frame.objects[0].actor_id == 20
    assert frame.objects[0].camera_visible


def test_audit_requires_existing_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    try:
        audit_mini_dataset(missing)
    except FileNotFoundError as error:
        assert error.args[0] == missing.resolve()
    else:
        raise AssertionError("Expected a missing dataset root to fail")


def test_multipart_audit_matches_pair_across_archive_parts(tmp_path: Path) -> None:
    part1 = tmp_path / "part1"
    part2 = tmp_path / "part2"
    accident = part1 / "type1_subtype1_accident"
    normal = part2 / "type1_subtype1_normal"
    scenario = "Town01_type001_subtype0001_scenario00001"
    for category, header in (
        (
            accident,
            "ClearNoon 10 car 20 car 1.0 opposite front 55\n"
            " colliding agents: ego none\n",
        ),
        (
            normal,
            "ClearNoon -1 none\n colliding agents: none\n",
        ),
    ):
        (category / "meta").mkdir(parents=True)
        (category / "meta" / f"{scenario}.txt").write_text(
            header
            + " agents id: 10 11 12 13\n"
            + " road_type: junction\n"
            + " another_vehicle_spawn_side: left\n"
            + " ego_vehicle_direction: straight\n"
            + " other_vehicle_direction: right\n",
            encoding="utf-8",
        )
        label = category / "ego_vehicle" / "label" / scenario / f"{scenario}_001.txt"
        label.parent.mkdir(parents=True)
        label.write_text(
            "0 0\ncar 3 0 0 4.5 1.8 1.5 0 0 0 20 10 True\n",
            encoding="utf-8",
        )

    rows, summary = audit_dataset_roots([part1, part2])
    assert len(rows) == 1
    assert rows[0]["paired_normal_exists"]
    assert rows[0]["detection_eligible"]
    assert rows[0]["localization_eligible"]
    assert summary["usable_paired_focal_views"] == 1
