#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JAKA_SDK_ROOT = PROJECT_ROOT / "JAKA_control" / "JAKA_dependecies" / "x86_64-linux-gnu"
PAYLOAD_CONFIG_PATH = PROJECT_ROOT / "JAKA_control" / "config" / "jaka_s5_orcahand_payload.json"

DEFAULT_IP = "192.168.99.44"
DEFAULT_SENSOR_BRAND = 10
DEFAULT_FORCE_DAMPING_N = 7
DEFAULT_TORQUE_DAMPING_NM = 5
DEFAULT_REBOUND_FK = 0.5
DEFAULT_TRANSLATION_STEP_MM = 10
DEFAULT_ROTATION_STEP_DEG = 1
DEFAULT_SERVO_STEP_NUM = 1
DEFAULT_REFRESH_MS = 250
FT_CTRL_TOOL_FRAME = 0
SDK_POWER_ON_WAIT_S = 0.0
SDK_ENABLE_WAIT_S = 0.0
SDK_SENSOR_BRAND_WAIT_S = 2.0
SDK_COMPLIANCE_SWITCH_WAIT_S = 1.0
ZERO_SENSOR_SETTLE_S = 0.6


@dataclass(frozen=True)
class IdentifiedPayload:
    mass_kg: float
    centroid_mm: tuple[float, float, float]


def load_jkrc() -> object:
    lib_path = JAKA_SDK_ROOT / "libjakaAPI.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"libjakaAPI.so not found: {lib_path}")
    ctypes.CDLL(str(lib_path))
    if str(JAKA_SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(JAKA_SDK_ROOT))
    import jkrc  # type: ignore

    return jkrc


def result_code(result: object) -> int:
    if isinstance(result, tuple):
        return int(result[0])
    if isinstance(result, int):
        return int(result)
    raise RuntimeError(f"未知 SDK 返回值：{result!r}")


def ensure_ok(result: object, action: str) -> None:
    code = result_code(result)
    if code != 0:
        raise RuntimeError(f"{action} 失败，返回码：{code}，原始结果：{result!r}")


def maybe_set_sensor_brand(robot: object, sensor_brand: int) -> None:
    if sensor_brand == DEFAULT_SENSOR_BRAND:
        return
    setter = getattr(robot, "set_torsenosr_brand", None)
    if setter is None:
        return
    ensure_ok(setter(sensor_brand), "set_torsenosr_brand")
    time.sleep(SDK_SENSOR_BRAND_WAIT_S)


def read_tcp_pose(robot: object) -> tuple[float, float, float, float, float, float]:
    for method_name in ("get_actual_tcp_position", "get_tcp_position"):
        method = getattr(robot, method_name, None)
        if method is None:
            continue
        result = method()
        ensure_ok(result, method_name)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(f"{method_name} 返回值异常：{result!r}")
        return tuple(float(value) for value in result[1])
    raise AttributeError("robot does not provide get_actual_tcp_position/get_tcp_position")


def load_saved_payload_snapshot() -> IdentifiedPayload | None:
    if not PAYLOAD_CONFIG_PATH.exists():
        return None
    try:
        payload_file = json.loads(PAYLOAD_CONFIG_PATH.read_text(encoding="utf-8"))
        payload_raw = payload_file["payload"]
        mass_kg = float(payload_raw["mass_kg"])
        centroid_mm = tuple(float(value) for value in payload_raw["centroid_mm"])
        if payload_file.get("valid") is not True or mass_kg <= 0.0 or len(centroid_mm) != 3:
            return None
        return IdentifiedPayload(mass_kg=mass_kg, centroid_mm=centroid_mm)
    except Exception:
        return None


def write_payload(robot: object, payload: IdentifiedPayload) -> None:
    centroid = list(payload.centroid_mm)
    applied = False
    setter = getattr(robot, "set_torq_sensor_tool_payload", None)
    if setter is not None:
        ensure_ok(setter(payload.mass_kg, centroid), "set_torq_sensor_tool_payload")
        applied = True
    setter = getattr(robot, "set_payload", None)
    if setter is not None:
        ensure_ok(setter(payload.mass_kg, centroid), "set_payload")
        applied = True
    if not applied:
        raise AttributeError("robot does not provide set_torq_sensor_tool_payload/set_payload")


def format_pose_deg(pose: tuple[float, float, float, float, float, float]) -> str:
    return (
        f"x={pose[0]:.1f} mm   y={pose[1]:.1f} mm   z={pose[2]:.1f} mm\n"
        f"rx={math.degrees(pose[3]):.2f} deg   ry={math.degrees(pose[4]):.2f} deg   rz={math.degrees(pose[5]):.2f} deg"
    )


class JakaCompliantTeleopUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("JAKA Compliant Teleop")
        self.root.resizable(False, False)

        self.status_var = tk.StringVar(value="正在连接并进入柔顺模式...")
        self.pose_var = tk.StringVar(value="TCP：初始化中...")
        self.translation_step_var = tk.StringVar(value=f"{args.xyz_step_mm:g}")
        self.rotation_step_var = tk.StringVar(value=f"{args.rotation_step_deg:g}")

        self.controls: list[ttk.Button] = []
        self.robot: object | None = None
        self.robot_lock = threading.Lock()
        self.ready = False
        self.closing = False
        self.command_in_flight = False
        self.servo_enabled = False
        self.compliance_enabled = False
        self.move_mode_incr = 1

        self._build_ui()
        self._set_controls_enabled(False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.args.refresh_ms, self._poll_pose)
        threading.Thread(target=self._startup_robot, daemon=True).start()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, textvariable=self.status_var).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, textvariable=self.pose_var, justify="left").grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 12))

        ttk.Label(frame, text="平移步长 mm").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.translation_step_var, width=10).grid(row=2, column=1, sticky="w")
        ttk.Label(frame, text="旋转步长 deg").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.rotation_step_var, width=10).grid(row=3, column=1, sticky="w")

        axis_rows = (
            ("X", 0),
            ("Y", 1),
            ("Z", 2),
            ("Rx", 3),
            ("Ry", 4),
            ("Rz", 5),
        )
        start_row = 5
        for row_offset, (label, axis_index) in enumerate(axis_rows):
            row = start_row + row_offset
            ttk.Label(frame, text=label, width=6).grid(row=row, column=0, sticky="w")
            minus_btn = ttk.Button(frame, text="-", width=8, command=lambda i=axis_index: self._queue_increment(i, -1.0))
            plus_btn = ttk.Button(frame, text="+", width=8, command=lambda i=axis_index: self._queue_increment(i, 1.0))
            minus_btn.grid(row=row, column=1, padx=(0, 6), pady=2)
            plus_btn.grid(row=row, column=2, pady=2)
            self.controls.extend([minus_btn, plus_btn])

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for control in self.controls:
            control.configure(state=state)

    def _set_ready(self, pose: tuple[float, float, float, float, float, float]) -> None:
        self.ready = True
        self.pose_var.set(f"TCP：\n{format_pose_deg(pose)}")
        self.status_var.set("柔顺控制已就绪")
        self._set_controls_enabled(True)

    def _startup_failed(self, exc: Exception) -> None:
        self.status_var.set(f"初始化失败：{exc}")
        messagebox.showerror("JAKA Compliant Teleop", f"初始化失败：\n{exc}")

    def _startup_robot(self) -> None:
        try:
            jkrc = load_jkrc()
            self.move_mode_incr = int(getattr(getattr(jkrc, "MoveMode", object()), "INCR", 1))
            robot = jkrc.RC(str(self.args.ip).strip())
            self.robot = robot

            login = getattr(robot, "login", None)
            if login is not None:
                ensure_ok(login(), "login")
            else:
                ensure_ok(robot.log_in(), "log_in")

            ensure_ok(robot.power_on(), "power_on")
            time.sleep(SDK_POWER_ON_WAIT_S)
            ensure_ok(robot.enable_robot(), "enable_robot")
            time.sleep(SDK_ENABLE_WAIT_S)

            maybe_set_sensor_brand(robot, self.args.sensor_brand)
            ensure_ok(robot.set_torque_sensor_mode(1), "set_torque_sensor_mode")

            payload = load_saved_payload_snapshot()
            if payload is not None:
                write_payload(robot, payload)

            zero_fn = getattr(robot, "zero_end_sensor", None)
            if zero_fn is not None:
                ensure_ok(zero_fn(), "zero_end_sensor")
                time.sleep(ZERO_SENSOR_SETTLE_S)

            set_frame = getattr(robot, "set_ft_ctrl_frame", None)
            if set_frame is not None:
                ensure_ok(set_frame(FT_CTRL_TOOL_FRAME), "set_ft_ctrl_frame(tool)")

            for axis_index in range(6):
                damping = self.args.force_damping_n if axis_index < 3 else self.args.torque_damping_nm
                ensure_ok(
                    robot.set_admit_ctrl_config(axis_index, 1, damping, 0.0, 0, self.args.rebound_fk),
                    f"set_admit_ctrl_config(axis={axis_index})",
                )

            ensure_ok(robot.servo_move_enable(True), "servo_move_enable(on)")
            self.servo_enabled = True
            ensure_ok(robot.set_compliant_type(1, 0), "set_compliant_type(init)")
            time.sleep(SDK_COMPLIANCE_SWITCH_WAIT_S)
            ensure_ok(robot.set_compliant_type(0, 1), "set_compliant_type(constant_force)")
            self.compliance_enabled = True

            pose = read_tcp_pose(robot)
            self.root.after(0, lambda: self._set_ready(pose))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._startup_failed(exc))

    def _step_values(self) -> tuple[float, float]:
        translation_step = float(self.translation_step_var.get())
        rotation_step = float(self.rotation_step_var.get())
        if translation_step <= 0.0 or rotation_step <= 0.0:
            raise ValueError("步长必须大于 0")
        return translation_step, rotation_step

    def _queue_increment(self, axis_index: int, sign: float) -> None:
        if self.closing:
            return
        if not self.ready or self.robot is None:
            self.status_var.set("机器人还没就绪")
            return
        if self.command_in_flight:
            self.status_var.set("上一条指令还没完成")
            return

        try:
            translation_step, rotation_step = self._step_values()
        except Exception as exc:
            self.status_var.set(f"步长无效：{exc}")
            return

        delta = [0.0] * 6
        if axis_index < 3:
            delta[axis_index] = sign * translation_step
            unit = "mm"
            magnitude = translation_step
        else:
            delta[axis_index] = math.radians(sign * rotation_step)
            unit = "deg"
            magnitude = rotation_step

        axis_name = ("X", "Y", "Z", "Rx", "Ry", "Rz")[axis_index]
        direction = "+" if sign > 0 else "-"
        self.command_in_flight = True
        self._set_controls_enabled(False)
        self.status_var.set(f"{axis_name}{direction} {magnitude:g} {unit}，发送中...")
        threading.Thread(
            target=self._do_increment,
            args=(tuple(delta), f"{axis_name}{direction} {magnitude:g} {unit}"),
            daemon=True,
        ).start()

    def _do_increment(self, delta: tuple[float, float, float, float, float, float], label: str) -> None:
        try:
            assert self.robot is not None
            with self.robot_lock:
                ensure_ok(self.robot.servo_p(delta, self.move_mode_incr, self.args.servo_step_num), "servo_p")
                pose = read_tcp_pose(self.robot)
            self.root.after(0, lambda: self._finish_increment(label, pose, None))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._finish_increment(label, None, exc))

    def _finish_increment(
        self,
        label: str,
        pose: tuple[float, float, float, float, float, float] | None,
        exc: Exception | None,
    ) -> None:
        self.command_in_flight = False
        if not self.closing:
            self._set_controls_enabled(self.ready)
        if exc is not None:
            self.status_var.set(f"{label} 失败：{exc}")
            return
        if pose is not None:
            self.pose_var.set(f"TCP：\n{format_pose_deg(pose)}")
        self.status_var.set(f"{label} 完成")

    def _poll_pose(self) -> None:
        if self.closing:
            return
        if self.ready and not self.command_in_flight and self.robot is not None:
            locked = self.robot_lock.acquire(blocking=False)
            if locked:
                try:
                    pose = read_tcp_pose(self.robot)
                except Exception:
                    pose = None
                finally:
                    self.robot_lock.release()
                if pose is not None:
                    self.pose_var.set(f"TCP：\n{format_pose_deg(pose)}")
        self.root.after(self.args.refresh_ms, self._poll_pose)

    def _on_close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self._set_controls_enabled(False)
        self.status_var.set("正在退出...")
        threading.Thread(target=self._shutdown_robot, daemon=True).start()

    def _shutdown_robot(self) -> None:
        try:
            if self.robot is not None:
                with self.robot_lock:
                    if self.compliance_enabled:
                        try:
                            ensure_ok(self.robot.set_compliant_type(0, 0), "set_compliant_type(reset)")
                            disable_force_control = getattr(self.robot, "disable_force_control", None)
                            if disable_force_control is not None:
                                ensure_ok(disable_force_control(), "disable_force_control")
                        except Exception:
                            pass
                    if self.servo_enabled:
                        try:
                            ensure_ok(self.robot.servo_move_enable(False), "servo_move_enable(off)")
                        except Exception:
                            pass
                    if self.args.power_off_on_exit:
                        try:
                            disable_robot = getattr(self.robot, "disable_robot", None)
                            if disable_robot is not None:
                                ensure_ok(disable_robot(), "disable_robot")
                            power_off = getattr(self.robot, "power_off", None)
                            if power_off is not None:
                                ensure_ok(power_off(), "power_off")
                        except Exception:
                            pass
                    try:
                        logout = getattr(self.robot, "logout", None)
                        if logout is not None:
                            ensure_ok(logout(), "logout")
                        else:
                            ensure_ok(self.robot.log_out(), "log_out")
                    except Exception:
                        pass
        finally:
            self.root.after(0, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JAKA 六维柔顺末端控制 UI。")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--sensor-brand", type=int, default=DEFAULT_SENSOR_BRAND)
    parser.add_argument("--force-damping-n", type=float, default=DEFAULT_FORCE_DAMPING_N)
    parser.add_argument("--torque-damping-nm", type=float, default=DEFAULT_TORQUE_DAMPING_NM)
    parser.add_argument("--rebound-fk", type=float, default=DEFAULT_REBOUND_FK)
    parser.add_argument("--xyz-step-mm", type=float, default=DEFAULT_TRANSLATION_STEP_MM)
    parser.add_argument("--rotation-step-deg", type=float, default=DEFAULT_ROTATION_STEP_DEG)
    parser.add_argument("--servo-step-num", type=int, default=DEFAULT_SERVO_STEP_NUM)
    parser.add_argument("--refresh-ms", type=int, default=DEFAULT_REFRESH_MS)
    parser.add_argument("--power-off-on-exit", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.force_damping_n < 0.0 or args.torque_damping_nm < 0.0 or args.rebound_fk < 0.0:
        raise SystemExit("导纳参数不能小于 0")
    if args.xyz_step_mm <= 0.0 or args.rotation_step_deg <= 0.0:
        raise SystemExit("步长必须大于 0")
    if args.servo_step_num < 1:
        raise SystemExit("--servo-step-num 必须大于等于 1")
    if args.refresh_ms < 50:
        raise SystemExit("--refresh-ms 不能小于 50")

    app = JakaCompliantTeleopUI(args)
    app.run()


if __name__ == "__main__":
    main()
