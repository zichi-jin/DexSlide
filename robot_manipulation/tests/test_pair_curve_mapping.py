from __future__ import annotations

import json

import numpy as np
import pytest

from robot_manipulation.orca_control.direct_joint_mapping import (
    DirectJointCalibration,
    DirectJointMapper,
    DirectJointRule,
    TARGET_JOINTS,
)
from robot_manipulation.orca_control.orca_teleop_pair_calibrate.orca_teleop_interpolate import (
    build_pair_curve,
)
from robot_manipulation.orca_control.pair_curve_mapping import (
    PairCurveMapper,
    load_orca_teleop_mapper,
)

SOURCE_ORDER = tuple(f"source_{index}" for index in range(20))


def make_baseline_payload() -> dict[str, object]:
    joints: dict[str, dict[str, object]] = {}
    for index, target in enumerate(TARGET_JOINTS):
        joints[target] = {
            "source": SOURCE_ORDER[index],
            "kind": "swing" if target.endswith("abd") or target == "thumb_mcp" else "bend",
            "scale": 1.0,
            "bias_deg": 0.0,
            "target_min_deg": -50.0,
            "target_max_deg": 50.0,
            "motor_id": index + 1,
            "motor_min_rad": -1.0,
            "motor_max_rad": 1.0,
            "motor_scale_rad_per_deg": 0.01,
            "motor_bias_rad": 0.0,
        }
    return {
        "schema_version": 2,
        "status": "complete",
        "source_unit": "deg",
        "target_unit": "deg",
        "source_joint_order": list(SOURCE_ORDER),
        "target_joint_order": list(TARGET_JOINTS),
        "joints": joints,
    }


def make_pairs_payload() -> dict[str, object]:
    source_a = {joint: 0.0 for joint in SOURCE_ORDER}
    source_b = {joint: float(index + 1) for index, joint in enumerate(SOURCE_ORDER)}
    target_a = {joint: float(index) for index, joint in enumerate(TARGET_JOINTS)}
    target_b = {joint: float(20 + index) for index, joint in enumerate(TARGET_JOINTS)}
    return {
        "schema_version": 1,
        "source_unit": "deg",
        "target_unit": "deg",
        "source_joint_order": list(SOURCE_ORDER),
        "target_joint_order": list(TARGET_JOINTS),
        "pairs": [
            {
                "dexslide": {"joint_angles_deg": source_a},
                "orca": {"joint_positions_deg": target_a},
            },
            {
                "dexslide": {"joint_angles_deg": source_b},
                "orca": {"joint_positions_deg": target_b},
            },
        ],
    }


def test_pair_curve_exactly_passes_recorded_pairs(tmp_path) -> None:
    baseline = make_baseline_payload()
    pairs = make_pairs_payload()
    curve = build_pair_curve(pairs, baseline)
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(json.dumps(curve), encoding="utf-8")

    mapper = load_orca_teleop_mapper(curve_path)
    for pair in pairs["pairs"]:
        result = mapper.map(pair["dexslide"]["joint_angles_deg"])
        expected = pair["orca"]["joint_positions_deg"]
        assert result.target_positions_deg == expected


def test_pair_curve_keeps_existing_joint_motor_safety_limits() -> None:
    baseline = make_baseline_payload()
    pairs = make_pairs_payload()
    pairs["pairs"][0]["orca"]["joint_positions_deg"][TARGET_JOINTS[0]] = 80.0
    curve = build_pair_curve(pairs, baseline)
    mapper = PairCurveMapper.from_payload(curve)

    source = pairs["pairs"][0]["dexslide"]["joint_angles_deg"]
    result = mapper.map(source)

    assert result.target_positions_deg[TARGET_JOINTS[0]] == 50.0
    assert "joint_limit" in result.clip_reasons[TARGET_JOINTS[0]]


def test_pair_curve_rejects_conflicting_duplicate_glove_pose() -> None:
    baseline = make_baseline_payload()
    pairs = make_pairs_payload()
    pairs["pairs"][1]["dexslide"]["joint_angles_deg"] = dict(
        pairs["pairs"][0]["dexslide"]["joint_angles_deg"]
    )
    with pytest.raises(ValueError, match="identical DexSlide poses"):
        build_pair_curve(pairs, baseline)


def test_direct_mapper_project_targets_uses_same_safety_logic() -> None:
    calibration = DirectJointCalibration.from_payload(make_baseline_payload())
    mapper = DirectJointMapper(calibration)
    result = mapper.project_targets({joint: 80.0 for joint in TARGET_JOINTS})

    assert all(value == 50.0 for value in result.target_positions_deg.values())
    assert len(result.clip_reasons) == len(TARGET_JOINTS)
