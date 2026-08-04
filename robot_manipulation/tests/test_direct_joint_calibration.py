from __future__ import annotations

from pathlib import Path

from robot_manipulation.orca_control.direct_joint_calibration import (
    build_fingerwise_calibration,
    save_fingerwise_calibration,
)
from robot_manipulation.orca_control.direct_joint_mapping import DirectJointMapper, TARGET_JOINTS


STATIC_MAP = Path(__file__).resolve().parents[1] / "assets/orca_hand/configs/joint_map_dexslide_to_orcahand.json"
SOURCE_ORDER = tuple(
    f"{finger}.{joint}"
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in ("DIP", "PIP", "MCP_front", "MCP_back")
)


def test_fingerwise_endpoints_build_complete_mapping(tmp_path) -> None:
    endpoints = {
        finger: {
            "left": {source: -10.0 for source in SOURCE_ORDER},
            "right": {source: 10.0 for source in SOURCE_ORDER},
        }
        for finger in ("thumb", "index", "middle", "ring", "pinky")
    }
    payload = build_fingerwise_calibration(
        source_joint_order=SOURCE_ORDER,
        finger_endpoints_deg=endpoints,
        static_map_file=STATIC_MAP,
        joint_roms_deg={joint: (-20.0, 80.0) for joint in TARGET_JOINTS},
        joint_to_motor_map={joint: index + 1 for index, joint in enumerate(TARGET_JOINTS)},
        joint_inversion={"thumb_mcp": True},
        motor_calibration={
            "motor_limits": {index + 1: (0.0, 2.0) for index in range(16)},
            "joint_to_motor_ratios": {index + 1: 0.02 for index in range(16)},
        },
    )

    assert payload["schema_version"] == 2
    assert set(payload["joints"]) == set(TARGET_JOINTS)
    assert payload["joints"]["thumb_mcp"]["motor_scale_rad_per_deg"] == -0.02
    assert payload["joints"]["index_abd"]["target_min_deg"] == -20.0

    output = tmp_path / "direct_joint_teleop_calibration.json"
    save_fingerwise_calibration(output, payload)
    mapper = DirectJointMapper.from_file(output)
    result = mapper.map({source: 10.0 for source in SOURCE_ORDER})
    assert result.predicted_motor_positions_rad[1] == 0.0
