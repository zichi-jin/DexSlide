#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JAKA_SDK_ROOT = PROJECT_ROOT / "JAKA_control" / "JAKA_dependecies" / "x86_64-linux-gnu"
DEFAULT_IP = "192.168.99.44"
DEFAULT_FORCE_THRESHOLD_N = 10.0
DEFAULT_SENSOR_BRAND = 10
AXIS_NAME_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "rx": 3,
    "ry": 4,
    "rz": 5,
}
ERR_OPERATION_TIMEOUT = -61


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


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


def call_method(robot: object, method_names: list[str], *args) -> object:
    for method_name in method_names:
        method = getattr(robot, method_name, None)
        if method is not None:
            return method(*args)
    raise AttributeError(f"robot does not provide any of: {method_names}")


def configure_axis(
    robot: object,
    axis_index: int,
    compliant_opt: int,
    ft_user: float,
    ft_constant: float,
    normal_track: int,
    ft_rebound_fk: float,
) -> None:
    result = robot.set_admit_ctrl_config(
        axis_index,
        compliant_opt,
        ft_user,
        ft_constant,
        normal_track,
        ft_rebound_fk,
    )
    ensure_ok(result, f"set_admit_ctrl_config(axis={axis_index})")


class FTxyz(ctypes.Structure):
    _fields_ = [
        ("fx", ctypes.c_double),
        ("fy", ctypes.c_double),
        ("fz", ctypes.c_double),
        ("tx", ctypes.c_double),
        ("ty", ctypes.c_double),
        ("tz", ctypes.c_double),
    ]


def call_set_compliance_condition(robot: object, linear_force_n: float, torque_nm: float) -> None:
    try:
        result = robot.set_compliance_condition(
            linear_force_n,
            linear_force_n,
            linear_force_n,
            torque_nm,
            torque_nm,
            torque_nm,
        )
    except TypeError:
        ft = FTxyz(
            linear_force_n,
            linear_force_n,
            linear_force_n,
            torque_nm,
            torque_nm,
            torque_nm,
        )
        result = robot.set_compliance_condition(ft)
    ensure_ok(result, "set_compliance_condition")


def selected_axis_indices(axis_names: Iterable[str]) -> list[int]:
    return [AXIS_NAME_TO_INDEX[name] for name in axis_names]


def get_sensor_brand(robot: object) -> int | None:
    getter = getattr(robot, "get_torsenosr_brand", None)
    if getter is None:
        getter = getattr(robot, "get_torsensor_brand", None)
    if getter is None:
        return None
    result = getter()
    if result_code(result) != 0:
        return None
    if not isinstance(result, tuple) or len(result) < 2:
        return None
    return int(result[1])


def maybe_set_sensor_brand(robot: object, sensor_brand: int) -> None:
    if sensor_brand == 10:
        print("skip set_torsenosr_brand: sensor_brand=10 表示内置传感器，官方文档说明无需调用此接口配置。")
        return

    current_brand = get_sensor_brand(robot)
    if current_brand == sensor_brand:
        print(f"skip set_torsenosr_brand: 当前已是目标型号 {sensor_brand}")
        return

    result = robot.set_torsenosr_brand(sensor_brand)
    code = result_code(result)
    if code == 0:
        return
    if code == ERR_OPERATION_TIMEOUT:
        raise RuntimeError(
            "set_torsenosr_brand 超时（-61）。这通常意味着当前控制器不接受此改动，"
            "或者你的机器人使用的是内置传感器，不该走外置型号配置。"
            f" 当前建议：对 S 系列直接使用 `--sensor-brand 10` 跳过此步骤。原始结果：{result!r}"
        )
    raise RuntimeError(f"set_torsenosr_brand 失败，返回码：{code}，原始结果：{result!r}")


def initialize_compliance_mode(
    robot: object,
    sensor_compensation: int,
    compliance_type: int,
) -> None:
    if sensor_compensation not in (0, 1):
        raise ValueError("sensor_compensation must be 0 or 1")
    if compliance_type not in (0, 1, 2):
        raise ValueError("compliance_type must be 0, 1, or 2")

    zero_fn = getattr(robot, "zero_end_sensor", None)
    if sensor_compensation == 1:
        if zero_fn is not None:
            ensure_ok(zero_fn(), "zero_end_sensor")
            time.sleep(0.6)
        ensure_ok(robot.set_compliant_type(1, 0), "set_compliant_type(init)")
        if compliance_type != 0:
            time.sleep(1.0)
            ensure_ok(robot.set_compliant_type(0, compliance_type), "set_compliant_type(type)")
    else:
        ensure_ok(robot.set_compliant_type(0, compliance_type), "set_compliant_type")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline JAKA admittance/compliance preset via Python SDK."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="Robot IP address")
    parser.add_argument(
        "--sensor-brand",
        type=int,
        default=DEFAULT_SENSOR_BRAND,
        help=(
            "Torque sensor brand. Use 10 for built-in flange sensor. "
            "For S5 this should usually stay 10, which will skip set_torsenosr_brand."
        ),
    )
    parser.add_argument(
        "--sensor-mode",
        type=int,
        choices=[0, 1],
        default=1,
        help="Torque sensor mode: 1 on, 0 off",
    )
    parser.add_argument(
        "--sensor-compensation",
        type=int,
        choices=[0, 1],
        default=1,
        help="set_compliant_type first-stage behavior; 1 means zero sensor and use actual external force display",
    )
    parser.add_argument(
        "--compliance-type",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="0 none, 1 constant force compliance, 2 speed compliance",
    )
    parser.add_argument(
        "--enable-admittance",
        type=int,
        choices=[0, 1],
        default=1,
        help="Whether to enable admittance/tool-drag force mode at the end",
    )
    parser.add_argument(
        "--force-threshold-n",
        type=non_negative_float,
        default=DEFAULT_FORCE_THRESHOLD_N,
        help="Compliance deadband threshold for fx/fy/fz",
    )
    parser.add_argument(
        "--torque-threshold-nm",
        type=non_negative_float,
        default=1.0,
        help="Compliance deadband threshold for tx/ty/tz",
    )
    parser.add_argument(
        "--active-axes",
        nargs="+",
        choices=list(AXIS_NAME_TO_INDEX.keys()),
        default=["z"],
        help="Axes using low-gain compliance config",
    )
    parser.add_argument(
        "--active-ft-user",
        type=positive_float,
        default=6.0,
        help="ftUser for compliant axes",
    )
    parser.add_argument(
        "--active-ft-constant",
        type=non_negative_float,
        default=0.0,
        help="ftConstant for compliant axes",
    )
    parser.add_argument(
        "--active-ft-rebound-fk",
        type=positive_float,
        default=60.0,
        help="ftReboundFK for compliant axes",
    )
    parser.add_argument(
        "--active-normal-track",
        type=int,
        default=0,
        help="ftNnormalTrack for compliant axes",
    )
    parser.add_argument(
        "--stiff-ft-user",
        type=positive_float,
        default=50.0,
        help="ftUser for non-compliant axes",
    )
    parser.add_argument(
        "--stiff-ft-constant",
        type=non_negative_float,
        default=0.0,
        help="ftConstant for non-compliant axes",
    )
    parser.add_argument(
        "--stiff-ft-rebound-fk",
        type=positive_float,
        default=120.0,
        help="ftReboundFK for non-compliant axes",
    )
    parser.add_argument(
        "--stiff-normal-track",
        type=int,
        default=0,
        help="ftNnormalTrack for non-compliant axes",
    )
    parser.add_argument(
        "--compliant-opt-active",
        type=int,
        default=1,
        help="opt argument for active axes in set_admit_ctrl_config",
    )
    parser.add_argument(
        "--compliant-opt-stiff",
        type=int,
        default=0,
        help="opt argument for non-active axes in set_admit_ctrl_config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned SDK calls without sending them",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    active_indices = set(selected_axis_indices(args.active_axes))

    plan = {
        "ip": args.ip,
        "sensor_brand": args.sensor_brand,
        "sensor_mode": args.sensor_mode,
        "sensor_compensation": args.sensor_compensation,
        "compliance_type": args.compliance_type,
        "enable_admittance": args.enable_admittance,
        "force_threshold": {
            "force_n": args.force_threshold_n,
            "torque_nm": args.torque_threshold_nm,
        },
        "axes": {},
    }
    for axis_name, axis_index in AXIS_NAME_TO_INDEX.items():
        if axis_index in active_indices:
            plan["axes"][axis_name] = {
                "opt": args.compliant_opt_active,
                "ft_user": args.active_ft_user,
                "ft_constant": args.active_ft_constant,
                "ft_normal_track": args.active_normal_track,
                "ft_rebound_fk": args.active_ft_rebound_fk,
            }
        else:
            plan["axes"][axis_name] = {
                "opt": args.compliant_opt_stiff,
                "ft_user": args.stiff_ft_user,
                "ft_constant": args.stiff_ft_constant,
                "ft_normal_track": args.stiff_normal_track,
                "ft_rebound_fk": args.stiff_ft_rebound_fk,
            }

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    jkrc = load_jkrc()
    robot = jkrc.RC(args.ip)

    try:
        login_fn = getattr(robot, "login", None)
        if login_fn is not None:
            ensure_ok(login_fn(), "login")
        else:
            ensure_ok(robot.log_in(), "log_in")

        ensure_ok(robot.power_on(), "power_on")
        ensure_ok(robot.enable_robot(), "enable_robot")
        maybe_set_sensor_brand(robot, args.sensor_brand)
        ensure_ok(robot.set_torque_sensor_mode(args.sensor_mode), "set_torque_sensor_mode")
        initialize_compliance_mode(robot, args.sensor_compensation, args.compliance_type)
        call_set_compliance_condition(robot, args.force_threshold_n, args.torque_threshold_nm)

        for axis_name, axis_index in AXIS_NAME_TO_INDEX.items():
            axis_cfg = plan["axes"][axis_name]
            configure_axis(
                robot,
                axis_index,
                axis_cfg["opt"],
                axis_cfg["ft_user"],
                axis_cfg["ft_constant"],
                axis_cfg["ft_normal_track"],
                axis_cfg["ft_rebound_fk"],
            )

        enable_result = call_method(robot, ["set_ft_ctrl_mode", "enable_admittance_ctrl"], args.enable_admittance)
        ensure_ok(enable_result, "enable_admittance/set_ft_ctrl_mode")
        print("admittance preset applied")
    finally:
        logout_fn = getattr(robot, "logout", None)
        if logout_fn is not None:
            logout_fn()
        else:
            robot.log_out()


if __name__ == "__main__":
    main()
