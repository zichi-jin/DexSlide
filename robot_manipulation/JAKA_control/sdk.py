"""Thin JAKA SDK adapter used by robot-specific entry points."""

from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
from typing import Any

from robot_manipulation.JAKA_control.paths import JAKA_SDK_DIR


def load_jkrc(sdk_root: Path = JAKA_SDK_DIR) -> object:
    lib_path = sdk_root / "libjakaAPI.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"libjakaAPI.so not found: {lib_path}")
    ctypes.CDLL(str(lib_path))
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))
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


def maybe_set_sensor_brand(robot: object, sensor_brand: int, *, default_brand: int, wait_s: float) -> None:
    if sensor_brand == default_brand:
        return
    setter = getattr(robot, "set_torsenosr_brand", None)
    if setter is None:
        return
    ensure_ok(setter(sensor_brand), "set_torsenosr_brand")
    time.sleep(float(wait_s))


__all__ = ["ensure_ok", "load_jkrc", "maybe_set_sensor_brand", "read_tcp_pose", "result_code"]

