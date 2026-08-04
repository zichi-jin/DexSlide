"""Reusable JAKA endpoint for DexSlide wrist-pose incremental teleoperation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from dexslide.kinematics.transforms import make_transform
from dexslide.streaming import DexSlideSceneSample

from .incremental_mapping import (
    TeleopAnchorState,
    build_desired_robot_transform,
    build_servo_increment_from_desired_transforms,
    clip_translation_to_workspace_mm,
    compute_transform_delta,
    make_safe_start_pose_sdk,
    map_rotation_delta_to_robot_rad,
    map_translation_delta_to_robot_mm,
    pose_is_close,
    pose_mmrad_to_transform,
    transform_to_pose_mmrad,
)
from .paths import DEFAULT_PAYLOAD_CONFIG_FILE
from .sdk import ensure_ok, load_jkrc, maybe_set_sensor_brand, read_tcp_pose
from .workspace_mapping import WorkspaceAxisMapping


SERVO_COMMAND_PERIOD_S = 0.008


@dataclass(frozen=True)
class IdentifiedPayload:
    mass_kg: float
    centroid_mm: tuple[float, float, float]


@dataclass(frozen=True)
class JakaTeleopConfig:
    ip: str = "192.168.99.44"
    source_hand: str = "left"
    enable_workspace_clip: bool = True
    sensor_brand: int = 10
    force_damping_n: float = 1.0
    torque_damping_nm: float = 10.0
    translation_rebound_fk: float = 0.5
    rotation_rebound_fk: float = 10.0
    enable_rotation_compliance: bool = False
    use_saved_payload: bool = True
    servo_step_num: int = 1
    safe_start_position_tol_mm: float = 5.0
    safe_start_angle_tol_deg: float = 6.0
    power_off_on_exit: bool = False
    dry_run: bool = False


def format_pose_mmdeg(pose: tuple[float, float, float, float, float, float]) -> str:
    return (
        f"x={pose[0]:.1f} mm, y={pose[1]:.1f} mm, z={pose[2]:.1f} mm, "
        f"rx={math.degrees(pose[3]):.2f} deg, ry={math.degrees(pose[4]):.2f} deg, "
        f"rz={math.degrees(pose[5]):.2f} deg"
    )


def format_increment(delta: np.ndarray) -> str:
    values = np.asarray(delta, dtype=np.float64).reshape(6)
    return (
        f"dx={values[0]:.2f} mm, dy={values[1]:.2f} mm, dz={values[2]:.2f} mm, "
        f"drx={math.degrees(values[3]):.2f} deg, dry={math.degrees(values[4]):.2f} deg, "
        f"drz={math.degrees(values[5]):.2f} deg"
    )


def load_saved_payload_snapshot() -> IdentifiedPayload | None:
    if not DEFAULT_PAYLOAD_CONFIG_FILE.exists():
        return None
    try:
        payload_file = json.loads(DEFAULT_PAYLOAD_CONFIG_FILE.read_text(encoding="utf-8"))
        payload_raw = payload_file["payload"]
        mass_kg = float(payload_raw["mass_kg"])
        centroid_mm = tuple(float(value) for value in payload_raw["centroid_mm"])
        if payload_file.get("valid") is not True or mass_kg <= 0.0 or len(centroid_mm) != 3:
            return None
        return IdentifiedPayload(mass_kg=mass_kg, centroid_mm=centroid_mm)
    except Exception:
        return None


def write_payload(robot: object, payload: IdentifiedPayload) -> None:
    applied = False
    centroid = list(payload.centroid_mm)
    for method_name in ("set_torq_sensor_tool_payload", "set_payload"):
        setter = getattr(robot, method_name, None)
        if setter is not None:
            ensure_ok(setter(payload.mass_kg, centroid), method_name)
            applied = True
    if not applied:
        raise AttributeError("robot does not provide set_torq_sensor_tool_payload/set_payload")


class JakaRobotAdapter:
    """Thin lifecycle wrapper around the JAKA SDK."""

    def __init__(self, config: JakaTeleopConfig, mapping: WorkspaceAxisMapping) -> None:
        self.config = config
        self.mapping = mapping
        self.robot: object | None = None
        self.servo_enabled = False
        self.compliance_enabled = False
        self.move_mode_abs = 0
        self.move_mode_incr = 1
        self.last_tcp_pose: tuple[float, float, float, float, float, float] | None = None
        self.last_workspace_clip_warning_time = 0.0

    def start(self) -> tuple[float, float, float, float, float, float]:
        jkrc = load_jkrc()
        self.move_mode_abs = int(getattr(getattr(jkrc, "MoveMode", object()), "ABS", 0))
        self.move_mode_incr = int(getattr(getattr(jkrc, "MoveMode", object()), "INCR", 1))
        self.robot = jkrc.RC(str(self.config.ip).strip())
        robot = self.robot
        login = getattr(robot, "login", None)
        ensure_ok(login() if login is not None else robot.log_in(), "login")
        ensure_ok(robot.power_on(), "power_on")
        time.sleep(1.0)
        ensure_ok(robot.enable_robot(), "enable_robot")
        time.sleep(1.0)
        self._ensure_safe_start_pose()
        maybe_set_sensor_brand(robot, self.config.sensor_brand, default_brand=10, wait_s=1.0)
        ensure_ok(robot.set_torque_sensor_mode(1), "set_torque_sensor_mode")
        if self.config.use_saved_payload:
            payload = load_saved_payload_snapshot()
            if payload is not None:
                write_payload(robot, payload)
        zero_fn = getattr(robot, "zero_end_sensor", None)
        if zero_fn is None:
            raise AttributeError("robot does not provide zero_end_sensor")
        ensure_ok(zero_fn(), "zero_end_sensor")
        time.sleep(0.6)
        set_frame = getattr(robot, "set_ft_ctrl_frame", None)
        if set_frame is not None:
            ensure_ok(set_frame(0), "set_ft_ctrl_frame(tool)")
        for axis_index in range(6):
            translation_axis = axis_index < 3
            enabled = translation_axis or self.config.enable_rotation_compliance
            damping = self.config.force_damping_n if translation_axis else self.config.torque_damping_nm
            rebound = (
                self.config.translation_rebound_fk
                if translation_axis
                else self.config.rotation_rebound_fk
            )
            ensure_ok(
                robot.set_admit_ctrl_config(
                    axis_index, 1 if enabled else 0, damping, 0.0, 0, rebound if enabled else 0.0
                ),
                f"set_admit_ctrl_config(axis={axis_index})",
            )
        ensure_ok(robot.servo_move_enable(True), "servo_move_enable(on)")
        self.servo_enabled = True
        ensure_ok(robot.set_compliant_type(1, 0), "set_compliant_type(init)")
        time.sleep(4.0)
        ensure_ok(robot.set_compliant_type(0, 1), "set_compliant_type(constant_force)")
        self.compliance_enabled = True
        self._ensure_task_space_zero_pose()
        self.last_tcp_pose = read_tcp_pose(robot)
        return self.last_tcp_pose

    def _ensure_safe_start_pose(self) -> None:
        assert self.robot is not None
        self._move_to_confirmed_pose(
            target_pose=make_safe_start_pose_sdk(self.mapping.safe_start_pose_mmdeg),
            speed_mm_s=self.mapping.safe_start_speed_mm_s,
            label="safe_start",
            failure_message="机械臂未能可靠到达安全初始姿态，拒绝开启柔顺。",
        )

    def _ensure_task_space_zero_pose(self) -> None:
        assert self.robot is not None
        print(
            "[jaka] 导纳控制已成功开启，正在移动至任务空间零点："
            f"{self.mapping.task_space_zero_pose_mmdeg}"
        )
        reached = self._servo_to_task_space_zero(
            target_pose=make_safe_start_pose_sdk(self.mapping.task_space_zero_pose_mmdeg),
            speed_mm_s=self.mapping.task_space_zero_speed_mm_s,
        )
        if reached:
            print("[jaka] 任务空间零点已确认到达。")

    def _servo_to_task_space_zero(
        self,
        *,
        target_pose: tuple[float, float, float, float, float, float],
        speed_mm_s: float,
    ) -> bool:
        """Stay in compliant servo mode while interpolating to the task-space zero."""
        assert self.robot is not None
        current_pose = read_tcp_pose(self.robot)
        if pose_is_close(
            current_pose,
            target_pose,
            position_tol_mm=self.config.safe_start_position_tol_mm,
            angle_tol_deg=self.config.safe_start_angle_tol_deg,
        ):
            self.last_tcp_pose = current_pose
            return True

        checker = getattr(self.robot, "is_in_servomove", None)
        if checker is not None:
            status = checker()
            ensure_ok(status, "is_in_servomove")
            if isinstance(status, tuple) and len(status) >= 2 and not bool(status[1]):
                raise RuntimeError("导纳已开启但 servo motion 未激活，拒绝移动至任务空间零点。")

        current_transform = pose_mmrad_to_transform(current_pose)
        target_transform = pose_mmrad_to_transform(target_pose)
        translation, rotation = compute_transform_delta(current_transform, target_transform)
        distance_mm = float(np.linalg.norm(translation))
        rotation_deg = math.degrees(float(np.linalg.norm(rotation)))
        period_s = SERVO_COMMAND_PERIOD_S * self.config.servo_step_num
        translation_steps = math.ceil(distance_mm / (float(speed_mm_s) * period_s))
        rotation_steps = math.ceil(rotation_deg / max(self.mapping.max_rotation_step_deg, 1e-6))
        steps = max(1, translation_steps, rotation_steps)
        increment = np.concatenate([translation / steps, rotation / steps])
        print(
            f"[jaka] 任务零点 servo 插补：{steps} 步，"
            f"速度上限 {speed_mm_s:.1f} mm/s。"
        )
        next_tick = time.monotonic()
        for _ in range(steps):
            ensure_ok(
                self.robot.servo_p(
                    tuple(float(value) for value in increment),
                    self.move_mode_incr,
                    self.config.servo_step_num,
                ),
                "servo_p(task_space_zero)",
            )
            next_tick += period_s
            remaining = next_tick - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

        confirmed = read_tcp_pose(self.robot)
        if not pose_is_close(
            confirmed,
            target_pose,
            position_tol_mm=self.config.safe_start_position_tol_mm,
            angle_tol_deg=self.config.safe_start_angle_tol_deg,
        ):
            print(
                "[jaka] 任务空间零点未精确到达；导纳保持开启，由用户决定是否启动遥操。"
                f" current={format_pose_mmdeg(confirmed)} target={format_pose_mmdeg(target_pose)}"
            )
            self.last_tcp_pose = confirmed
            return False
        self.last_tcp_pose = confirmed
        return True

    def _move_to_confirmed_pose(
        self,
        *,
        target_pose: tuple[float, float, float, float, float, float],
        speed_mm_s: float,
        label: str,
        failure_message: str,
    ) -> None:
        assert self.robot is not None
        current_pose = read_tcp_pose(self.robot)
        if pose_is_close(
            current_pose,
            target_pose,
            position_tol_mm=self.config.safe_start_position_tol_mm,
            angle_tol_deg=self.config.safe_start_angle_tol_deg,
        ):
            self.last_tcp_pose = current_pose
            return
        ensure_ok(
            self.robot.linear_move(
                target_pose, self.move_mode_abs, True, float(speed_mm_s)
            ),
            f"linear_move({label})",
        )
        confirmed = read_tcp_pose(self.robot)
        if not pose_is_close(
            confirmed,
            target_pose,
            position_tol_mm=self.config.safe_start_position_tol_mm,
            angle_tol_deg=self.config.safe_start_angle_tol_deg,
        ):
            raise RuntimeError(
                failure_message
                +
                f" current={format_pose_mmdeg(confirmed)} target={format_pose_mmdeg(target_pose)}"
            )
        self.last_tcp_pose = confirmed

    def send_increment(self, delta: np.ndarray) -> None:
        assert self.robot is not None
        safe_delta = np.asarray(delta, dtype=np.float64).reshape(6).copy()
        if self.config.enable_workspace_clip:
            if self.last_tcp_pose is None:
                self.last_tcp_pose = read_tcp_pose(self.robot)
            current_translation = np.asarray(self.last_tcp_pose[:3], dtype=np.float64)
            requested_translation = current_translation + safe_delta[:3]
            clipped_translation, was_clipped = clip_translation_to_workspace_mm(
                requested_translation,
                self.mapping,
            )
            if was_clipped:
                safe_delta[:3] = clipped_translation - current_translation
                now = time.monotonic()
                if now - self.last_workspace_clip_warning_time >= 1.0:
                    print(
                        "[jaka] workspace clip at SDK boundary: "
                        f"requested={requested_translation.tolist()} mm "
                        f"clipped={clipped_translation.tolist()} mm"
                    )
                    self.last_workspace_clip_warning_time = now
        values = tuple(float(value) for value in safe_delta)
        ensure_ok(
            self.robot.servo_p(values, self.move_mode_incr, self.config.servo_step_num),
            "servo_p",
        )
        if self.last_tcp_pose is not None:
            last_transform = pose_mmrad_to_transform(self.last_tcp_pose)
            delta_transform = make_transform(
                cv2.Rodrigues(np.asarray(safe_delta[3:], dtype=np.float64).reshape(3, 1))[0],
                np.asarray(safe_delta[:3], dtype=np.float64).reshape(3),
            )
            self.last_tcp_pose = transform_to_pose_mmrad(delta_transform @ last_transform)

    def read_current_tcp_transform(self) -> np.ndarray:
        assert self.robot is not None
        self.last_tcp_pose = read_tcp_pose(self.robot)
        return pose_mmrad_to_transform(self.last_tcp_pose)

    def close(self) -> None:
        if self.robot is None:
            return
        try:
            if self.compliance_enabled:
                try:
                    ensure_ok(self.robot.set_compliant_type(0, 0), "set_compliant_type(reset)")
                    disable = getattr(self.robot, "disable_force_control", None)
                    if disable is not None:
                        ensure_ok(disable(), "disable_force_control")
                except Exception:
                    pass
            if self.servo_enabled:
                try:
                    ensure_ok(self.robot.servo_move_enable(False), "servo_move_enable(off)")
                except Exception:
                    pass
            if self.config.power_off_on_exit:
                for method_name in ("disable_robot", "power_off"):
                    method = getattr(self.robot, method_name, None)
                    if method is not None:
                        try:
                            ensure_ok(method(), method_name)
                        except Exception:
                            pass
        finally:
            logout = getattr(self.robot, "logout", None)
            try:
                ensure_ok(logout() if logout is not None else self.robot.log_out(), "logout")
            except Exception:
                pass


class JakaTeleopEndpoint:
    """Consume only wrist pose fields from a shared scene sample."""

    def __init__(self, config: JakaTeleopConfig, mapping: WorkspaceAxisMapping) -> None:
        self.config = config
        self.mapping = mapping
        self.robot = None if config.dry_run else JakaRobotAdapter(config, mapping)
        self.anchor_candidate: np.ndarray | None = None
        self.anchor_state: TeleopAnchorState | None = None
        self.previous_filtered: np.ndarray | None = None
        self.valid_streak = 0
        self.frame_idx = 0
        self.last_tracker_status = "starting"
        self.last_marker_ids: tuple[int, ...] = ()
        self._tracking_lost_reported = False
        self.last_workspace_clipped = False
        self.last_workspace_warning_time = 0.0
        self.start_before_scene = True
        self.requires_user_confirmation = True
        self.teleop_armed = False
        self.input_ready = False

    @property
    def name(self) -> str:
        return "jaka"

    def start(self) -> None:
        if self.robot is None:
            print("[jaka] dry-run mode")
            return
        pose = self.robot.start()
        print(f"[jaka] compliant teleop ready: {format_pose_mmdeg(pose)}")

    def wait_for_user_teleop_confirmation(self) -> None:
        """等待用户确认后，才允许将 DexSlide 输入接到 JAKA 输出。"""
        try:
            answer = input(
                "[jaka] DexSlide 信号已稳定。确认输入输出安全后按 Enter 接通遥操；输入 q 退出： "
            ).strip().lower()
        except EOFError as exc:
            raise RuntimeError("未收到用户的遥操启动确认。") from exc
        if answer in {"q", "quit", "exit"}:
            raise RuntimeError("用户取消启动遥操。")
        self.teleop_armed = True
        print("[jaka] 输入输出已接通，开始遥操。")

    def consume(self, sample: DexSlideSceneSample) -> None:
        hand = sample.hands[self.config.source_hand]
        self.frame_idx += 1
        if not hand.pose_valid:
            self.last_tracker_status = "table_or_hand_pose_lost"
            if self.anchor_state is None:
                self.anchor_candidate = None
                self.valid_streak = 0
            elif not self._tracking_lost_reported:
                print("[jaka] tracking lost -> pause but keep anchor")
                self._tracking_lost_reported = True
            return

        self._tracking_lost_reported = False
        current = np.asarray(hand.transform_table_hand, dtype=np.float64).reshape(4, 4)
        self.last_marker_ids = hand.marker_ids
        self.last_tracker_status = "ok"

        if self.previous_filtered is not None:
            delta_translation, delta_rotation = compute_transform_delta(self.previous_filtered, current)
            translation_mm = float(np.linalg.norm(map_translation_delta_to_robot_mm(delta_translation, self.mapping)))
            rotation_deg = math.degrees(
                float(np.linalg.norm(map_rotation_delta_to_robot_rad(delta_rotation, self.mapping)))
            )
            if (
                translation_mm > self.mapping.frame_jump_reject_mm
                or rotation_deg > self.mapping.frame_jump_reject_deg
            ):
                self.last_tracker_status = "frame_jump_rejected"
                self.previous_filtered = current
                if self.anchor_state is None:
                    self.anchor_candidate = current
                    self.valid_streak = 1
                return
        self.previous_filtered = current

        if self.anchor_state is None:
            self.anchor_candidate = current
            self.valid_streak += 1
            if self.valid_streak < self.mapping.anchor_stable_frames:
                return
            if not self.teleop_armed:
                self.input_ready = True
                self.last_tracker_status = "input_ready"
                return
            robot_pose = (
                np.eye(4, dtype=np.float64)
                if self.robot is None
                else self.robot.read_current_tcp_transform()
            )
            self.anchor_state = TeleopAnchorState(
                glove_anchor_translation_m=current[:3, 3].copy(),
                robot_anchor_translation_mm=np.asarray(
                    self.mapping.task_space_zero_pose_mmdeg[:3],
                    dtype=np.float64,
                ),
                previous_desired_robot_transform=robot_pose.copy(),
                anchor_frame_idx=self.frame_idx,
            )
            print(f"[jaka] position anchor acquired at frame {self.frame_idx}")
            return

        desired = build_desired_robot_transform(current, self.anchor_state, self.mapping)
        self.last_workspace_clipped = False
        if self.config.enable_workspace_clip:
            requested_translation = desired[:3, 3].copy()
            clipped_translation, was_clipped = clip_translation_to_workspace_mm(
                requested_translation,
                self.mapping,
            )
            if was_clipped:
                desired = desired.copy()
                desired[:3, 3] = clipped_translation
                self.last_workspace_clipped = True
                now = time.monotonic()
                if now - self.last_workspace_warning_time >= 1.0:
                    print(
                        "[jaka] workspace clip: "
                        f"requested={requested_translation.tolist()} mm "
                        f"target={clipped_translation.tolist()} mm"
                    )
                    self.last_workspace_warning_time = now
        increment = build_servo_increment_from_desired_transforms(
            self.anchor_state.previous_desired_robot_transform,
            desired,
            self.mapping,
        )
        if float(np.linalg.norm(increment)) > 0.0:
            if self.robot is None:
                print(f"[jaka dry-run] {format_increment(increment)}")
            else:
                self.robot.send_increment(increment)
        self.anchor_state = TeleopAnchorState(
            glove_anchor_translation_m=self.anchor_state.glove_anchor_translation_m,
            robot_anchor_translation_mm=self.anchor_state.robot_anchor_translation_mm,
            previous_desired_robot_transform=desired,
            anchor_frame_idx=self.anchor_state.anchor_frame_idx,
        )

    def status_lines(self) -> list[str]:
        return [
            f"state={'armed' if self.anchor_state is not None else 'waiting'}",
            "mapping=position_t0+rotation_absolute",
            f"workspace_clip={self.last_workspace_clipped}",
            f"input_ready={self.input_ready}",
            f"armed={self.teleop_armed}",
            f"tracker={self.last_tracker_status}",
            f"frame={self.frame_idx}",
            f"markers={list(self.last_marker_ids)}",
        ]

    def close(self) -> None:
        if self.robot is not None:
            self.robot.close()


__all__ = [
    "IdentifiedPayload",
    "JakaRobotAdapter",
    "JakaTeleopConfig",
    "JakaTeleopEndpoint",
    "format_increment",
    "format_pose_mmdeg",
    "load_saved_payload_snapshot",
    "write_payload",
]
