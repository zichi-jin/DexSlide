from __future__ import annotations

import numpy as np

from robot_manipulation.JAKA_control.incremental_mapping import (
    clip_translation_to_workspace_mm,
)
from robot_manipulation.JAKA_control.teleop_endpoint import (
    JakaRobotAdapter,
    JakaTeleopConfig,
)
from robot_manipulation.JAKA_control.workspace_mapping import load_workspace_axis_mapping


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def servo_p(self, increment, move_mode, step_num):
        self.calls.append((increment, move_mode, step_num))
        return 0


def load_mapping():
    return load_workspace_axis_mapping(
        "robot_manipulation/assets/jaka/configs/workspace_axis_mapping.json"
    )


def test_workspace_clip_clamps_each_axis() -> None:
    mapping = load_mapping()

    clipped, was_clipped = clip_translation_to_workspace_mm(
        np.array([350.0, -300.0, 100.0]),
        mapping,
    )

    assert was_clipped is True
    np.testing.assert_allclose(clipped, [300.0, -450.0, 120.0])


def test_robot_adapter_clips_final_sdk_increment() -> None:
    mapping = load_mapping()
    robot = FakeRobot()
    adapter = JakaRobotAdapter(
        JakaTeleopConfig(enable_workspace_clip=True),
        mapping,
    )
    adapter.robot = robot
    adapter.move_mode_incr = 1
    adapter.last_tcp_pose = (299.0, -550.0, 200.0, 0.0, 0.0, 0.0)

    adapter.send_increment(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    sent_increment = np.asarray(robot.calls[0][0], dtype=np.float64)
    np.testing.assert_allclose(sent_increment[:3], [1.0, 0.0, 0.0])
    assert adapter.last_tcp_pose is not None
    assert adapter.last_tcp_pose[0] == 300.0


def test_robot_adapter_can_disable_workspace_clip() -> None:
    mapping = load_mapping()
    robot = FakeRobot()
    adapter = JakaRobotAdapter(
        JakaTeleopConfig(enable_workspace_clip=False),
        mapping,
    )
    adapter.robot = robot
    adapter.move_mode_incr = 1
    adapter.last_tcp_pose = (299.0, -550.0, 200.0, 0.0, 0.0, 0.0)

    adapter.send_increment(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    sent_increment = np.asarray(robot.calls[0][0], dtype=np.float64)
    np.testing.assert_allclose(sent_increment[:3], [5.0, 0.0, 0.0])
