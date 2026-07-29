#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import math
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JAKA_SDK_ROOT = PROJECT_ROOT / "JAKA_control" / "JAKA_dependecies" / "x86_64-linux-gnu"

DEFAULT_IP = "192.168.99.44"
DEFAULT_SPEED_MM_S = 60.0
SDK_POWER_ON_WAIT_S = 4.0
SDK_ENABLE_WAIT_S = 4.0


def load_jkrc() -> object:
    lib_path = JAKA_SDK_ROOT / "libjakaAPI.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"libjakaAPI.so not found: {lib_path}")
    ctypes.CDLL(str(lib_path))
    if str(JAKA_SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(JAKA_SDK_ROOT))
    import jkrc  # type: ignore

    return jkrc


def ensure_ok(result: object, action: str) -> None:
    code = int(result[0]) if isinstance(result, tuple) else int(result)
    if code != 0:
        raise RuntimeError(f"{action} 失败，返回码：{code}，原始结果：{result!r}")


def read_tcp_pose(robot: object) -> tuple[float, float, float, float, float, float]:
    for method_name in ("get_actual_tcp_position", "get_tcp_position"):
        method = getattr(robot, method_name, None)
        if method is None:
            continue
        result = method()
        ensure_ok(result, method_name)
        return tuple(float(value) for value in result[1])
    raise AttributeError("robot does not provide get_actual_tcp_position/get_tcp_position")


def print_pose(robot: object) -> tuple[float, float, float, float, float, float]:
    pose = read_tcp_pose(robot)
    rx_deg = math.degrees(pose[3])
    ry_deg = math.degrees(pose[4])
    rz_deg = math.degrees(pose[5])
    print(
        "当前 TCP："
        f"x={pose[0]:.3f} mm，y={pose[1]:.3f} mm，z={pose[2]:.3f} mm，"
        f"rx={rx_deg:.3f} deg，ry={ry_deg:.3f} deg，rz={rz_deg:.3f} deg"
    )
    print(
        "可复用命令："
        f"move {pose[0]:.3f} {pose[1]:.3f} {pose[2]:.3f} {rx_deg:.3f} {ry_deg:.3f} {rz_deg:.3f}"
    )
    return pose


def run_linear_move(
    robot: object,
    target_pose: tuple[float, float, float, float, float, float],
    speed_mm_s: float,
    move_mode_abs: int,
    label: str,
) -> None:
    print(f"用户指令：{label}")
    print("移动中... 此时不接收新指令")
    ensure_ok(robot.linear_move(target_pose, move_mode_abs, True, speed_mm_s), "linear_move")
    print("移动完成")
    print_pose(robot)


def main() -> None:
    parser = argparse.ArgumentParser(description="JAKA linear 调位小工具。CLI 使用 mm + deg。")
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED_MM_S)
    parser.add_argument("--power-off-on-exit", action="store_true")
    args = parser.parse_args()

    if args.speed <= 0.0:
        raise SystemExit("--speed 必须大于 0")

    jkrc = load_jkrc()
    move_mode_abs = int(getattr(getattr(jkrc, "MoveMode", object()), "ABS", 0))
    robot = jkrc.RC(str(args.ip).strip())
    speed_mm_s = float(args.speed)

    login = getattr(robot, "login", None)
    if login is not None:
        ensure_ok(login(), "login")
    else:
        ensure_ok(robot.log_in(), "log_in")

    try:
        ensure_ok(robot.power_on(), "power_on")
        time.sleep(SDK_POWER_ON_WAIT_S)
        ensure_ok(robot.enable_robot(), "enable_robot")
        time.sleep(SDK_ENABLE_WAIT_S)

        print("已连接机械臂。命令只用 mm + deg。")
        print("命令：pose | move x y z rx ry rz | shift dx dy dz drx dry drz | speed [v] | quit")
        print_pose(robot)

        while True:
            try:
                line = input("linear> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in {"quit", "exit", "q"}:
                break

            if cmd in {"help", "h", "?"}:
                print("pose：查看当前位姿，并打印一条可复用 move 命令")
                print("move x y z rx ry rz：绝对移动；位置单位 mm，姿态单位 deg")
                print("shift dx dy dz drx dry drz：相对当前位姿偏移；位置单位 mm，姿态单位 deg")
                print("speed：查看当前速度；speed 20：把 linear 速度改成 20 mm/s")
                print("quit：退出")
                continue

            if cmd in {"pose", "p", "record"}:
                print_pose(robot)
                continue

            if cmd == "speed":
                if len(parts) == 1:
                    print(f"当前 linear 速度：{speed_mm_s:.3f} mm/s")
                    continue
                if len(parts) != 2:
                    print("用法：speed 20")
                    continue
                try:
                    speed_mm_s = float(parts[1])
                except ValueError:
                    print("speed 后面必须是数字")
                    continue
                if speed_mm_s <= 0.0:
                    print("speed 必须大于 0")
                    continue
                print(f"当前 linear 速度已改为：{speed_mm_s:.3f} mm/s")
                continue

            if cmd == "move":
                if len(parts) != 7:
                    print("用法：move x y z rx ry rz")
                    continue
                try:
                    x, y, z, rx_deg, ry_deg, rz_deg = [float(value) for value in parts[1:]]
                except ValueError:
                    print("move 后面 6 个值都必须是数字")
                    continue
                target_pose = (
                    x,
                    y,
                    z,
                    math.radians(rx_deg),
                    math.radians(ry_deg),
                    math.radians(rz_deg),
                )
                run_linear_move(robot, target_pose, speed_mm_s, move_mode_abs, line)
                continue

            if cmd == "shift":
                if len(parts) != 7:
                    print("用法：shift dx dy dz drx dry drz")
                    continue
                try:
                    dx, dy, dz, drx_deg, dry_deg, drz_deg = [float(value) for value in parts[1:]]
                except ValueError:
                    print("shift 后面 6 个值都必须是数字")
                    continue
                current_pose = read_tcp_pose(robot)
                target_pose = (
                    current_pose[0] + dx,
                    current_pose[1] + dy,
                    current_pose[2] + dz,
                    current_pose[3] + math.radians(drx_deg),
                    current_pose[4] + math.radians(dry_deg),
                    current_pose[5] + math.radians(drz_deg),
                )
                run_linear_move(robot, target_pose, speed_mm_s, move_mode_abs, line)
                continue

            print("未知命令。输入 help 查看说明。")
    finally:
        if args.power_off_on_exit:
            try:
                disable_robot = getattr(robot, "disable_robot", None)
                if disable_robot is not None:
                    ensure_ok(disable_robot(), "disable_robot")
                power_off = getattr(robot, "power_off", None)
                if power_off is not None:
                    ensure_ok(power_off(), "power_off")
            except Exception as exc:
                print(f"退出时断电失败：{exc}")

        try:
            logout = getattr(robot, "logout", None)
            if logout is not None:
                ensure_ok(logout(), "logout")
            else:
                ensure_ok(robot.log_out(), "log_out")
        except Exception as exc:
            print(f"logout 失败：{exc}")


if __name__ == "__main__":
    main()
