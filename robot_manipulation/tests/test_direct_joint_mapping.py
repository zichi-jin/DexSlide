from __future__ import annotations

import numpy as np

from robot_manipulation.orca_control.direct_joint_mapping import (
    DirectJointCalibration,
    DirectJointMapper,
    DirectJointRule,
    TARGET_JOINTS,
)


SOURCE_ORDER = tuple(
    f"{finger}.{joint}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in ("DIP", "PIP", "MCP_front", "MCP_back")
)


def make_calibration() -> DirectJointCalibration:
    rules = {}
    for index, target in enumerate(TARGET_JOINTS):
        source = SOURCE_ORDER[index]
        rules[target] = DirectJointRule(
            target=target,
            source=source,
            kind="swing" if target == "thumb_mcp" else "bend",
            scale=1.0,
            bias_deg=0.0,
            target_min_deg=-10.0,
            target_max_deg=10.0,
            motor_id=index + 1,
            motor_min_rad=-1.0,
            motor_max_rad=1.0,
            motor_scale_rad_per_deg=0.1,
            motor_bias_rad=0.0,
        )
    return DirectJointCalibration(SOURCE_ORDER, TARGET_JOINTS, rules, {})


def test_mapper_accepts_fixed_20d_order_and_clips() -> None:
    mapper = DirectJointMapper(make_calibration())
    source = np.zeros(20, dtype=np.float64)
    source[0] = 20.0
    result = mapper.map(source)

    assert result.target_positions_deg[TARGET_JOINTS[0]] == 10.0
    assert "joint_limit" in result.clip_reasons[TARGET_JOINTS[0]]
    assert result.predicted_motor_positions_rad[1] == 1.0


def test_mapper_rejects_missing_source_joint() -> None:
    mapper = DirectJointMapper(make_calibration())
    try:
        mapper.map({SOURCE_ORDER[0]: 0.0})
    except ValueError as exc:
        assert "Missing DexSlide source joints" in str(exc)
    else:
        raise AssertionError("expected missing-source validation")
