"""Realtime serial helpers for DexSlide live pipelines."""

from __future__ import annotations

import atexit
import threading
from pathlib import Path

import numpy as np

from dexslide.paths import DEFAULT_GLOVE_CALIBRATION_FILE
from dexslide.serial_angles import AngleStreamReader, load_calibration, make_joint_order

_LISTENER_LOCK = threading.Lock()
_LISTENERS: dict[tuple[str, int, str, str], AngleStreamReader] = {}


def _listener_key(port: str, baud: int, mode: str, calib_file: Path) -> tuple[str, int, str, str]:
    return (port, int(baud), mode, str(calib_file.resolve()))


def live_listener(
    port: str,
    baud: int = 115200,
    mode: str = "raw",
    calib_file: str | Path = DEFAULT_GLOVE_CALIBRATION_FILE,
) -> AngleStreamReader:
    """Return a cached background listener for the given serial source."""

    calib_path = Path(calib_file).expanduser().resolve()
    key = _listener_key(port, baud, mode, calib_path)
    with _LISTENER_LOCK:
        reader = _LISTENERS.get(key)
        if reader is not None:
            return reader

        joint_order = make_joint_order()
        calibration = load_calibration(calib_path, joint_order)
        reader = AngleStreamReader(
            port=port,
            baud=baud,
            mode=mode,
            joint_order=joint_order,
            calibration=calibration,
        )
        reader.start()
        _LISTENERS[key] = reader
        return reader


def live_listen(
    port: str,
    baud: int = 115200,
    mode: str = "raw",
    calib_file: str | Path = DEFAULT_GLOVE_CALIBRATION_FILE,
    include_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, str]:
    """Read the latest DexSlide 20-DOF joint vector in radians."""

    reader = live_listener(
        port=port,
        baud=baud,
        mode=mode,
        calib_file=calib_file,
    )
    joint_angles, timestamp, raw_line = reader.snapshot_rad20()
    if include_meta:
        return joint_angles, timestamp, raw_line
    return joint_angles


def shutdown_live_listeners() -> None:
    """Stop all cached listeners."""

    with _LISTENER_LOCK:
        readers = list(_LISTENERS.values())
        _LISTENERS.clear()
    for reader in readers:
        reader.stop()


atexit.register(shutdown_live_listeners)
