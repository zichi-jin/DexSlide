"""Realtime serial helpers for DexSlide live pipelines."""

from __future__ import annotations

import atexit
import threading
from pathlib import Path

import numpy as np

from dexslide.communications import hand_joint_communication, resolve_joint_port
from dexslide.paths import DEFAULT_GLOVE_CALIBRATION_FILE
from dexslide.serial_angles import AngleStreamReader, load_calibration, make_joint_order

_LISTENER_LOCK = threading.Lock()
_LISTENERS: dict[tuple[str, int, str, str], AngleStreamReader] = {}


def _listener_key(port: str, baud: int, mode: str, calib_file: Path) -> tuple[str, int, str, str]:
    return (port, int(baud), mode, str(calib_file.resolve()))


def live_listener(
    port: str | None = None,
    baud: int | None = None,
    mode: str | None = None,
    calib_file: str | Path = DEFAULT_GLOVE_CALIBRATION_FILE,
    startup_timeout_sec: float | None = None,
    hand: str = "left",
) -> AngleStreamReader:
    """Return a cached background listener for the given serial source."""

    communication = hand_joint_communication(hand)
    resolved_port = resolve_joint_port(hand) if port is None else str(port).strip()
    if not resolved_port:
        raise ValueError("DexSlide serial port cannot be empty")
    resolved_baud = int(communication["baud"] if baud is None else baud)
    resolved_mode = str(communication["mode"] if mode is None else mode).strip().lower()
    resolved_timeout = float(
        communication["startup_timeout_sec"]
        if startup_timeout_sec is None
        else startup_timeout_sec
    )
    if resolved_mode not in {"raw", "angles"}:
        raise ValueError(f"Unsupported DexSlide stream mode: {resolved_mode!r}")

    calib_path = Path(calib_file).expanduser().resolve()
    key = _listener_key(resolved_port, resolved_baud, resolved_mode, calib_path)
    with _LISTENER_LOCK:
        reader = _LISTENERS.get(key)
        if reader is not None:
            if reader.running:
                reader.wait_for_first_sample(resolved_timeout)
                return reader
            _LISTENERS.pop(key, None)

        joint_order = make_joint_order()
        calibration = load_calibration(calib_path, joint_order)
        reader = AngleStreamReader(
            port=resolved_port,
            baud=resolved_baud,
            mode=resolved_mode,
            joint_order=joint_order,
            calibration=calibration,
        )
        reader.start()
        try:
            reader.wait_for_first_sample(resolved_timeout)
        except Exception:
            reader.stop()
            raise
        _LISTENERS[key] = reader
        return reader


def live_listen(
    port: str | None = None,
    baud: int | None = None,
    mode: str | None = None,
    calib_file: str | Path = DEFAULT_GLOVE_CALIBRATION_FILE,
    include_meta: bool = False,
    hand: str = "left",
) -> np.ndarray | tuple[np.ndarray, float, str]:
    """Read the latest DexSlide 20-DOF joint vector in radians."""

    reader = live_listener(
        port=port,
        baud=baud,
        mode=mode,
        calib_file=calib_file,
        hand=hand,
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
