from __future__ import annotations

from robot_manipulation.JAKA_control.paths import DEFAULT_WORKSPACE_MAPPING_FILE
from robot_manipulation.JAKA_control.workspace_mapping import load_workspace_axis_mapping


def test_default_workspace_mapping_loads_from_jaka_assets() -> None:
    mapping = load_workspace_axis_mapping(DEFAULT_WORKSPACE_MAPPING_FILE)
    assert mapping.robot_from_table_transform.shape == (4, 4)
    assert len(mapping.safe_start_pose_mmdeg) == 6

