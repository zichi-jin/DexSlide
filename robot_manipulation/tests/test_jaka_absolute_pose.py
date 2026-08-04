from __future__ import annotations

import numpy as np

from robot_manipulation.JAKA_control.incremental_mapping import (
    TeleopAnchorState,
    build_desired_robot_transform,
)
from robot_manipulation.JAKA_control.workspace_mapping import load_workspace_axis_mapping


def test_position_uses_t0_anchor_and_rotation_is_absolute() -> None:
    mapping = load_workspace_axis_mapping(
        "robot_manipulation/assets/jaka/configs/workspace_axis_mapping.json"
    )
    anchor = TeleopAnchorState(
        glove_anchor_translation_m=np.array([0.10, 0.20, 0.30]),
        robot_anchor_translation_mm=np.array([100.0, -550.0, 120.0]),
        previous_desired_robot_transform=np.eye(4, dtype=np.float64),
        anchor_frame_idx=0,
    )
    glove = np.eye(4, dtype=np.float64)
    glove[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    glove[:3, 3] = [0.11, 0.18, 0.27]

    desired = build_desired_robot_transform(glove, anchor, mapping)

    reflection = np.diag([-1.0, 1.0, 1.0])
    expected_rotation = reflection @ glove[:3, :3] @ reflection
    np.testing.assert_allclose(desired[:3, 3], [90.0, -570.0, 90.0], atol=1e-9)
    np.testing.assert_allclose(desired[:3, :3], expected_rotation, atol=1e-9)
