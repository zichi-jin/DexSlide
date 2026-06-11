#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JAKA_SDK_ROOT = PROJECT_ROOT / "JAKA_control" / "JAKA_dependecies" / "x86_64-linux-gnu"

# 连接配置
DEFAULT_IP = "192.168.99.44"  # JAKA 控制器 IP
DEFAULT_SENSOR_BRAND = 10  # 10 表示当前这条内置力传感器路径

# 导纳 / 恒力柔顺参数。
# 这里调用的是 set_admit_ctrl_config(axis, 1, ftUser, 0.0, 0, ftReboundFK)。
# - ftUser 更像导纳公式里的速度 / 阻尼项系数，可粗略类比为 p'
# - ftReboundFK 更像位置回正项系数，可粗略类比为 p
# - 这组接口里没有单独暴露显式的惯性 / 加速度项，所以这里没有可直接单调的 p''
DEFAULT_FORCE_DAMPING_N = 1e-3  # xyz 平移轴的 ftUser；越大通常越硬
DEFAULT_TORQUE_DAMPING_NM = 1e-3  # rx/ry/rz 转动轴的 ftUser；越大通常越硬
DEFAULT_REBOUND_FK = 1.0  # 六轴共用的 ftReboundFK；越大越容易回正，也更容易感觉“拽回去”

# servo 轨迹参数。
# 手册里 servo 是 8 ms 一个插补周期；这里走 servo_p(INCR)，连续小步下发。
SERVO_CYCLE_S = 0.008
DEFAULT_SERVO_STEP_NUM = 1
DEFAULT_TRAJECTORY_PERIOD_S = 12.0  # 一轮轨迹的基准周期；越大越慢、越柔和
DEFAULT_XY_AMPLITUDE_MM = 35.0  # x / y 最大摆幅，不超过 40 mm
DEFAULT_Z_AMPLITUDE_MM = 25.0  # z 最大摆幅，不超过 40 mm
DEFAULT_ROTATION_AMPLITUDE_DEG = 12.0  # rx / ry / rz 最大摆幅，不超过 40 deg
MAX_TRANSLATION_AMPLITUDE_MM = 40.0
MAX_ROTATION_AMPLITUDE_DEG = 40.0

# payload 辨识与缓存
DEFAULT_PAYLOAD_RETURN_SPEED_DEG_S = 10.0  # 负载辨识完成后回原位的关节速度
DEFAULT_IDENTIFY_DELTA_DEG = 30.0  # 负载辨识时腕部 3 轴默认偏转角
PAYLOAD_IDENTIFY_POLL_S = 1.0  # 轮询辨识状态的间隔
PAYLOAD_CONFIG_PATH = PROJECT_ROOT / "JAKA_control" / "config" / "jaka_s5_orcahand_payload.json"
WRIST_JOINT_LIMITS_RAD = (
    math.radians(265.0),  # j4 上限
    math.radians(320.0),  # j5 上限
    math.radians(265.0),  # j6 上限
)

# 观察与打印
DEFAULT_HOLD_SECONDS = 120.0  # 轨迹运行总时长；你可以在这段时间里手推测试柔顺
DEFAULT_POLL_INTERVAL_S = 0.5  # 打印 TCP 和 force 的间隔
FORCE_SENSOR_SOURCE_EXTERNAL = 3  # 读取补偿后的 external force，而不是 raw force

# 控制参考系
FT_CTRL_TOOL_FRAME = 0  # 力控参考系使用 tool frame

# 控制器状态切换等待时间
SDK_POWER_ON_WAIT_S = 1
SDK_ENABLE_WAIT_S = 1
SDK_SENSOR_BRAND_WAIT_S = 2.0
SDK_COMPLIANCE_SWITCH_WAIT_S = 1.0
ZERO_SENSOR_SETTLE_S = 0.6


@dataclass(frozen=True)
class ForceReading:
    fx: float
    fy: float
    fz: float
    tx: float
    ty: float
    tz: float


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


def read_joint_position(robot: object) -> list[float]:
    result = robot.get_joint_position()
    ensure_ok(result, "get_joint_position")
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"get_joint_position 返回值异常：{result!r}")
    return [float(value) for value in result[1]]


def read_force_reading(robot: object, source_type: int = FORCE_SENSOR_SOURCE_EXTERNAL) -> ForceReading | None:
    getter = getattr(robot, "get_torque_sensor_data", None)
    if getter is None:
        return None
    result = getter(int(source_type))
    if result_code(result) != 0 or not isinstance(result, tuple) or len(result) < 2:
        return None
    payload = result[1]
    if not isinstance(payload, tuple) or len(payload) < 3:
        return None
    values = payload[2]
    if not isinstance(values, (tuple, list)) or len(values) < 6:
        return None
    return ForceReading(
        fx=float(values[0]),
        fy=float(values[1]),
        fz=float(values[2]),
        tx=float(values[3]),
        ty=float(values[4]),
        tz=float(values[5]),
    )


def format_pose(pose: tuple[float, float, float, float, float, float]) -> str:
    return (
        f"x={pose[0]:.3f} mm，y={pose[1]:.3f} mm，z={pose[2]:.3f} mm，"
        f"rx={pose[3]:.4f} rad，ry={pose[4]:.4f} rad，rz={pose[5]:.4f} rad"
    )


def format_force_reading(reading: ForceReading | None) -> str:
    if reading is None:
        return "力传感：未读取"
    return (
        f"fx={reading.fx:.2f} N，fy={reading.fy:.2f} N，fz={reading.fz:.2f} N，"
        f"tx={reading.tx:.3f} Nm，ty={reading.ty:.3f} Nm，tz={reading.tz:.3f} Nm"
    )


def format_joint_values(joints_rad: list[float]) -> str:
    joints_deg = [math.degrees(value) for value in joints_rad]
    return ", ".join(f"j{i + 1}={value:.1f} deg" for i, value in enumerate(joints_deg))


def format_offset(offset: tuple[float, float, float, float, float, float]) -> str:
    return (
        f"dx={offset[0]:.2f} mm，dy={offset[1]:.2f} mm，dz={offset[2]:.2f} mm，"
        f"drx={math.degrees(offset[3]):.2f} deg，dry={math.degrees(offset[4]):.2f} deg，"
        f"drz={math.degrees(offset[5]):.2f} deg"
    )


def choose_payload_identify_target(current_joints: list[float], delta_deg: float) -> list[float]:
    target = list(current_joints)
    delta_rad = math.radians(delta_deg)
    for local_index, joint_index in enumerate((3, 4, 5)):
        limit = WRIST_JOINT_LIMITS_RAD[local_index]
        plus_candidate = current_joints[joint_index] + delta_rad
        minus_candidate = current_joints[joint_index] - delta_rad
        plus_margin = limit - abs(plus_candidate)
        minus_margin = limit - abs(minus_candidate)
        if plus_margin >= minus_margin and plus_margin > 0.0:
            target[joint_index] = plus_candidate
        elif minus_margin > 0.0:
            target[joint_index] = minus_candidate
        else:
            target[joint_index] = max(-limit + 0.05, min(limit - 0.05, current_joints[joint_index]))
    return target


def parse_identified_payload(result: object) -> IdentifiedPayload:
    ensure_ok(result, "get_torq_sensor_payload_identify_result")
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"payload identify result 格式异常：{result!r}")

    payload = result[1]
    if not isinstance(payload, (tuple, list)):
        raise RuntimeError(f"payload identify payload 格式异常：{payload!r}")

    if len(payload) >= 5 and not isinstance(payload[2], (tuple, list)):
        return IdentifiedPayload(
            mass_kg=float(payload[1]),
            centroid_mm=(float(payload[2]), float(payload[3]), float(payload[4])),
        )

    if len(payload) >= 4 and not isinstance(payload[1], (tuple, list)):
        return IdentifiedPayload(
            mass_kg=float(payload[0]),
            centroid_mm=(float(payload[1]), float(payload[2]), float(payload[3])),
        )

    if len(payload) >= 3 and isinstance(payload[2], (tuple, list)) and len(payload[2]) >= 3:
        return IdentifiedPayload(
            mass_kg=float(payload[1]),
            centroid_mm=tuple(float(value) for value in payload[2][:3]),
        )

    raise RuntimeError(f"payload identify payload 解析失败：{payload!r}")


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


def save_payload_snapshot(payload: IdentifiedPayload, robot_ip: str) -> None:
    PAYLOAD_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAYLOAD_CONFIG_PATH.write_text(
        json.dumps(
            {
                "valid": True,
                "source": "scripts/jaka_admittance_motion_trial.py auto identify",
                "robot_ip": robot_ip,
                "tool_id": 1,
                "payload": {
                    "mass_kg": round(payload.mass_kg, 6),
                    "centroid_mm": [round(value, 6) for value in payload.centroid_mm],
                },
                "units": {
                    "mass_kg": "kg",
                    "centroid_mm": "mm in tool/flange payload frame",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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


def identify_and_apply_payload(
    robot: object,
    robot_ip: str,
    delta_deg: float,
    return_speed_deg_s: float,
) -> IdentifiedPayload:
    origin_joints = read_joint_position(robot)
    target_joints = choose_payload_identify_target(origin_joints, delta_deg)
    print(f"[payload] 当前关节：{format_joint_values(origin_joints)}")
    print(f"[payload] 辨识终点：{format_joint_values(target_joints)}")
    ensure_ok(robot.start_torq_sensor_payload_identify(target_joints), "start_torq_sensor_payload_identify")
    print("[payload] 已启动末端负载辨识，等待控制器执行辨识轨迹")

    while True:
        status_result = robot.get_torq_sensor_identify_staus()
        ensure_ok(status_result, "get_torq_sensor_identify_staus")
        if not isinstance(status_result, tuple) or len(status_result) < 2:
            raise RuntimeError(f"payload identify status 格式异常：{status_result!r}")
        status = int(status_result[1])
        print(f"[payload] 辨识状态：{status}")
        if status == 0:
            break
        if status == 2:
            raise RuntimeError("末端负载辨识失败，控制器返回 error 状态")
        time.sleep(PAYLOAD_IDENTIFY_POLL_S)

    payload_result = robot.get_torq_sensor_payload_identify_result()
    print(f"[payload] 原始辨识结果：{payload_result!r}")
    payload = parse_identified_payload(payload_result)
    print(
        f"[payload] 辨识结果：mass={payload.mass_kg:.4f} kg，"
        f"centroid=({payload.centroid_mm[0]:.3f}, {payload.centroid_mm[1]:.3f}, {payload.centroid_mm[2]:.3f}) mm"
    )
    write_payload(robot, payload)
    save_payload_snapshot(payload, robot_ip)
    print(f"[payload] 已写回控制器，并保存到 {PAYLOAD_CONFIG_PATH}")
    ensure_ok(robot.joint_move(origin_joints, 0, 1, return_speed_deg_s), "joint_move(return_origin)")
    print("[payload] 已返回辨识起点")
    return payload


def trajectory_offset(
    elapsed_s: float,
    xy_amplitude_mm: float,
    z_amplitude_mm: float,
    rotation_amplitude_rad: float,
    period_s: float,
) -> tuple[float, float, float, float, float, float]:
    phase = 2.0 * math.pi * elapsed_s / period_s
    return (
        xy_amplitude_mm * math.sin(phase),
        xy_amplitude_mm * math.sin(phase * 0.5),
        z_amplitude_mm * math.sin(phase * 1.5),
        rotation_amplitude_rad * math.sin(phase * 0.75),
        rotation_amplitude_rad * math.sin(phase * 1.25),
        rotation_amplitude_rad * math.sin(phase * 1.75),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JAKA 导纳 + servo_p 增量轨迹柔顺测试脚本。")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--sensor-brand", type=int, default=DEFAULT_SENSOR_BRAND)
    parser.add_argument("--skip-payload-identify", action="store_true")
    parser.add_argument("--force-payload-identify", action="store_true")
    parser.add_argument("--identify-delta-deg", type=float, default=DEFAULT_IDENTIFY_DELTA_DEG)
    parser.add_argument("--payload-return-speed-deg-s", type=float, default=DEFAULT_PAYLOAD_RETURN_SPEED_DEG_S)
    parser.add_argument("--force-damping-n", type=float, default=DEFAULT_FORCE_DAMPING_N)
    parser.add_argument("--torque-damping-nm", type=float, default=DEFAULT_TORQUE_DAMPING_NM)
    parser.add_argument("--rebound-fk", type=float, default=DEFAULT_REBOUND_FK)
    parser.add_argument("--hold-seconds", type=float, default=DEFAULT_HOLD_SECONDS)
    parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--trajectory-period-s", type=float, default=DEFAULT_TRAJECTORY_PERIOD_S)
    parser.add_argument("--xy-amplitude-mm", type=float, default=DEFAULT_XY_AMPLITUDE_MM)
    parser.add_argument("--z-amplitude-mm", type=float, default=DEFAULT_Z_AMPLITUDE_MM)
    parser.add_argument("--rotation-amplitude-deg", type=float, default=DEFAULT_ROTATION_AMPLITUDE_DEG)
    parser.add_argument("--servo-step-num", type=int, default=DEFAULT_SERVO_STEP_NUM)
    parser.add_argument("--power-off-on-exit", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.skip_payload_identify and args.force_payload_identify:
        raise SystemExit("--skip-payload-identify 与 --force-payload-identify 不能同时使用")

    for name in (
        "identify_delta_deg",
        "payload_return_speed_deg_s",
        "poll_interval_s",
        "trajectory_period_s",
    ):
        if getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} 必须大于 0")
    for name in ("force_damping_n", "torque_damping_nm", "rebound_fk", "hold_seconds"):
        if getattr(args, name) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} 不能小于 0")
    if args.servo_step_num < 1:
        raise SystemExit("--servo-step-num 必须大于等于 1")
    if not 0.0 <= args.xy_amplitude_mm <= MAX_TRANSLATION_AMPLITUDE_MM:
        raise SystemExit(f"--xy-amplitude-mm 必须在 0 到 {MAX_TRANSLATION_AMPLITUDE_MM:.1f} 之间")
    if not 0.0 <= args.z_amplitude_mm <= MAX_TRANSLATION_AMPLITUDE_MM:
        raise SystemExit(f"--z-amplitude-mm 必须在 0 到 {MAX_TRANSLATION_AMPLITUDE_MM:.1f} 之间")
    if not 0.0 <= args.rotation_amplitude_deg <= MAX_ROTATION_AMPLITUDE_DEG:
        raise SystemExit(f"--rotation-amplitude-deg 必须在 0 到 {MAX_ROTATION_AMPLITUDE_DEG:.1f} 之间")

    jkrc = load_jkrc()
    move_mode_incr = int(getattr(getattr(jkrc, "MoveMode", object()), "INCR", 1))
    robot = jkrc.RC(str(args.ip).strip())

    login = getattr(robot, "login", None)
    if login is not None:
        ensure_ok(login(), "login")
    else:
        ensure_ok(robot.log_in(), "log_in")

    servo_enabled = False
    compliance_enabled = False

    try:
        ensure_ok(robot.power_on(), "power_on")
        print(f"[setup] power_on 完成，等待 {SDK_POWER_ON_WAIT_S:.1f} s")
        time.sleep(SDK_POWER_ON_WAIT_S)

        ensure_ok(robot.enable_robot(), "enable_robot")
        print(f"[setup] enable_robot 完成，等待 {SDK_ENABLE_WAIT_S:.1f} s")
        time.sleep(SDK_ENABLE_WAIT_S)

        maybe_set_sensor_brand(robot, args.sensor_brand)
        ensure_ok(robot.set_torque_sensor_mode(1), "set_torque_sensor_mode")

        cached_payload = load_saved_payload_snapshot()
        if args.skip_payload_identify:
            if cached_payload is None:
                print("[payload] 已按参数跳过自动负载辨识；未发现本地完整 payload，直接沿用控制器当前 payload")
            else:
                write_payload(robot, cached_payload)
                print(
                    "[payload] 已按参数跳过自动负载辨识，并从本地缓存恢复 payload："
                    f"mass={cached_payload.mass_kg:.4f} kg，"
                    f"centroid=({cached_payload.centroid_mm[0]:.3f}, {cached_payload.centroid_mm[1]:.3f}, {cached_payload.centroid_mm[2]:.3f}) mm"
                )
        elif cached_payload is not None and not args.force_payload_identify:
            write_payload(robot, cached_payload)
            print(f"[payload] 发现本地完整 payload，已跳过自动辨识：{PAYLOAD_CONFIG_PATH}")
            print(
                f"[payload] 已恢复 payload：mass={cached_payload.mass_kg:.4f} kg，"
                f"centroid=({cached_payload.centroid_mm[0]:.3f}, {cached_payload.centroid_mm[1]:.3f}, {cached_payload.centroid_mm[2]:.3f}) mm"
            )
        else:
            identify_and_apply_payload(
                robot,
                robot_ip=str(args.ip).strip(),
                delta_deg=args.identify_delta_deg,
                return_speed_deg_s=args.payload_return_speed_deg_s,
            )

        zero_fn = getattr(robot, "zero_end_sensor", None)
        if zero_fn is None:
            print("[setup] 当前 SDK 未提供 zero_end_sensor，跳过显式置零")
        else:
            ensure_ok(zero_fn(), "zero_end_sensor")
            time.sleep(ZERO_SENSOR_SETTLE_S)

        set_frame = getattr(robot, "set_ft_ctrl_frame", None)
        if set_frame is not None:
            ensure_ok(set_frame(FT_CTRL_TOOL_FRAME), "set_ft_ctrl_frame(tool)")

        print("[setup] 进入导纳参数配置")
        print(
            f"[setup] ftUser：force={args.force_damping_n:.6g}，torque={args.torque_damping_nm:.6g}；"
            f"ftReboundFK={args.rebound_fk:.6g}"
        )
        for axis_index, axis_name in enumerate(("x", "y", "z", "rx", "ry", "rz")):
            damping = args.force_damping_n if axis_index < 3 else args.torque_damping_nm
            ensure_ok(
                robot.set_admit_ctrl_config(axis_index, 1, damping, 0.0, 0, args.rebound_fk),
                f"set_admit_ctrl_config({axis_name})",
            )

        print("[setup] 按手册顺序：先开 servo，再开恒力柔顺")
        ensure_ok(robot.servo_move_enable(True), "servo_move_enable(on)")
        servo_enabled = True
        ensure_ok(robot.set_compliant_type(1, 0), "set_compliant_type(init)")
        time.sleep(SDK_COMPLIANCE_SWITCH_WAIT_S)
        ensure_ok(robot.set_compliant_type(0, 1), "set_compliant_type(constant_force)")
        compliance_enabled = True

        current_pose = read_tcp_pose(robot)
        print(f"[setup] 当前 TCP：{format_pose(current_pose)}")
        print(f"[setup] 轨迹基准周期：{args.trajectory_period_s:.1f} s，servo step_num={args.servo_step_num}")
        print(
            f"[setup] 轨迹幅值：xy={args.xy_amplitude_mm:.1f} mm，z={args.z_amplitude_mm:.1f} mm，"
            f"rotation={args.rotation_amplitude_deg:.1f} deg"
        )
        print(f"[setup] external force：{format_force_reading(read_force_reading(robot))}")

        if args.hold_seconds > 0.0:
            print("[run] 已进入导纳 + servo_p 轨迹测试；现在可以在运动过程中手推机械臂")
            start_time = time.monotonic()
            next_tick = start_time
            next_report = start_time
            last_offset = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            rotation_amplitude_rad = math.radians(args.rotation_amplitude_deg)
            servo_period_s = SERVO_CYCLE_S * args.servo_step_num

            while True:
                now = time.monotonic()
                elapsed_s = now - start_time
                if elapsed_s >= args.hold_seconds:
                    break

                target_offset = trajectory_offset(
                    elapsed_s,
                    xy_amplitude_mm=args.xy_amplitude_mm,
                    z_amplitude_mm=args.z_amplitude_mm,
                    rotation_amplitude_rad=rotation_amplitude_rad,
                    period_s=args.trajectory_period_s,
                )
                delta = tuple(target - last for target, last in zip(target_offset, last_offset))
                ensure_ok(robot.servo_p(delta, move_mode_incr, args.servo_step_num), "servo_p")
                last_offset = target_offset

                if now >= next_report:
                    print(f"[run] 目标偏移：{format_offset(target_offset)}")
                    print(f"[run] 当前 TCP：{format_pose(read_tcp_pose(robot))}")
                    print(f"[run] external force：{format_force_reading(read_force_reading(robot))}")
                    next_report = now + args.poll_interval_s

                next_tick += servo_period_s
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
    finally:
        try:
            if compliance_enabled:
                ensure_ok(robot.set_compliant_type(0, 0), "set_compliant_type(reset)")
                disable_force_control = getattr(robot, "disable_force_control", None)
                if disable_force_control is not None:
                    ensure_ok(disable_force_control(), "disable_force_control")
                print("[cleanup] 恒力柔顺已关闭")
        except Exception as exc:
            print(f"[cleanup] 关闭恒力柔顺失败：{exc}")

        try:
            if servo_enabled:
                ensure_ok(robot.servo_move_enable(False), "servo_move_enable(off)")
                print("[cleanup] servo 模式已关闭")
        except Exception as exc:
            print(f"[cleanup] 关闭 servo 失败：{exc}")

        if args.power_off_on_exit:
            try:
                disable_robot = getattr(robot, "disable_robot", None)
                if disable_robot is not None:
                    ensure_ok(disable_robot(), "disable_robot")
                power_off = getattr(robot, "power_off", None)
                if power_off is not None:
                    ensure_ok(power_off(), "power_off")
                print("[cleanup] 机器人已 disable_robot + power_off")
            except Exception as exc:
                print(f"[cleanup] 下使能 / 断电失败：{exc}")

        try:
            logout = getattr(robot, "logout", None)
            if logout is not None:
                ensure_ok(logout(), "logout")
            else:
                ensure_ok(robot.log_out(), "log_out")
            print("[cleanup] SDK 已 logout")
        except Exception as exc:
            print(f"[cleanup] logout 失败：{exc}")


if __name__ == "__main__":
    main()
