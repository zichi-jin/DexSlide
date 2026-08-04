from __future__ import annotations

from robot_manipulation.JAKA_control.paths import DEFAULT_WORKSPACE_MAPPING_FILE
from robot_manipulation.JAKA_control.workspace_mapping import load_workspace_axis_mapping


def test_default_workspace_mapping_loads_from_jaka_assets() -> None:
    mapping = load_workspace_axis_mapping(DEFAULT_WORKSPACE_MAPPING_FILE)
    assert mapping.robot_from_table_transform.shape == (4, 4)
    assert len(mapping.safe_start_pose_mmdeg) == 6
    assert mapping.safe_start_speed_mm_s == 60.0
    assert mapping.safe_start_pose_mmdeg == (100.0, -380.0, 200.0, 180.0, 0.0, 135.26)
    assert mapping.task_space_zero_pose_mmdeg == (100.0, -550.0, 120.0, -90.0, 45.0, -180.0)
    assert mapping.task_space_zero_speed_mm_s > 0.0
    assert mapping.teleop_workspace_min_mm.tolist() == [-300.0, -650.0, 120.0]
    assert mapping.teleop_workspace_max_mm.tolist() == [300.0, -450.0, 560.0]
