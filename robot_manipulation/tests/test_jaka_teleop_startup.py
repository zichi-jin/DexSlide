from __future__ import annotations

import math
from types import SimpleNamespace

from robot_manipulation.JAKA_control.teleop_endpoint import JakaRobotAdapter, JakaTeleopConfig
from robot_manipulation.JAKA_control.workspace_mapping import load_workspace_axis_mapping


class FakeRobot:
    def __init__(
        self,
        safe_pose: tuple[float, float, float, float, float, float],
        task_zero_pose: tuple[float, float, float, float, float, float],
    ) -> None:
        self.pose = safe_pose
        self.task_zero_pose = task_zero_pose
        self.calls: list[tuple[object, ...]] = []

    def login(self): self.calls.append(("login",)); return 0
    def power_on(self): self.calls.append(("power_on",)); return 0
    def enable_robot(self): self.calls.append(("enable_robot",)); return 0
    def set_torque_sensor_mode(self, mode): self.calls.append(("sensor", mode)); return 0
    def zero_end_sensor(self): self.calls.append(("zero_sensor",)); return 0
    def set_ft_ctrl_frame(self, frame): self.calls.append(("frame", frame)); return 0
    def set_admit_ctrl_config(self, *args): self.calls.append(("admit", *args)); return 0
    def servo_move_enable(self, enabled): self.calls.append(("servo", enabled)); return 0
    def set_compliant_type(self, mode, value): self.calls.append(("compliant", mode, value)); return 0
    def get_actual_tcp_position(self): return 0, self.pose

    def linear_move(self, target, move_mode, blocking, speed):
        self.calls.append(("linear_move", target, move_mode, blocking, speed))
        self.pose = tuple(target)
        return 0

    def is_in_servomove(self): return 0, True

    def servo_p(self, increment, move_mode, step_num):
        self.calls.append(("servo_p", increment, move_mode, step_num))
        self.pose = self.task_zero_pose
        return 0


def test_task_space_zero_move_follows_successful_compliance(monkeypatch) -> None:
    mapping = load_workspace_axis_mapping("robot_manipulation/assets/jaka/configs/workspace_axis_mapping.json")
    safe_pose = (*mapping.safe_start_pose_mmdeg[:3], *[math.radians(value) for value in mapping.safe_start_pose_mmdeg[3:]])
    task_zero_pose = (
        *mapping.task_space_zero_pose_mmdeg[:3],
        *[math.radians(value) for value in mapping.task_space_zero_pose_mmdeg[3:]],
    )
    robot = FakeRobot(safe_pose, task_zero_pose)
    fake_jkrc = SimpleNamespace(MoveMode=SimpleNamespace(ABS=0, INCR=1), RC=lambda _: robot)
    monkeypatch.setattr("robot_manipulation.JAKA_control.teleop_endpoint.load_jkrc", lambda: fake_jkrc)
    monkeypatch.setattr("robot_manipulation.JAKA_control.teleop_endpoint.maybe_set_sensor_brand", lambda *args, **kwargs: None)
    monkeypatch.setattr("robot_manipulation.JAKA_control.teleop_endpoint.time.sleep", lambda _: None)
    monkeypatch.setattr("builtins.input", lambda _: "")

    adapter = JakaRobotAdapter(JakaTeleopConfig(use_saved_payload=False), mapping)
    pose = adapter.start()

    compliance_index = robot.calls.index(("compliant", 0, 1))
    servo_index = next(index for index, call in enumerate(robot.calls) if call[0] == "servo_p")
    assert compliance_index < servo_index
    assert not any(call[0] == "linear_move" for call in robot.calls)
    assert pose[:3] == mapping.task_space_zero_pose_mmdeg[:3]
