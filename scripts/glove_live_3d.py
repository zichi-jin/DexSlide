#!/usr/bin/env python3
"""Compatibility wrapper for the DexSlide live 3D visualizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexslide.communications import hand_joint_communication, resolve_joint_port
from dexslide.paths import DEFAULT_GLOVE_CALIBRATION_FILE, DEFAULT_SKELETON_FILE
from dexslide.visualization.live_matplotlib import run_live_viewer


def main() -> None:
    communication = hand_joint_communication("left")
    parser = argparse.ArgumentParser(description="Realtime 3D hand reconstruction from DexSlide serial stream.")
    parser.add_argument("--port", default=resolve_joint_port("left"), help="Configured serial port")
    parser.add_argument("--baud", type=int, default=int(communication["baud"]), help="Baud rate")
    parser.add_argument(
        "--mode",
        choices=["raw", "angles"],
        default=str(communication["mode"]),
        help="raw: parse MCU I2C line; angles: parse ads_live_monitor --angles line",
    )
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE), help="Path to skeleton JSON")
    parser.add_argument("--calib-file", default=str(DEFAULT_GLOVE_CALIBRATION_FILE), help="Glove ADC calibration JSON")
    parser.add_argument("--hand", choices=["auto", "left", "right"], default="left")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    run_live_viewer(
        port=args.port,
        baud=args.baud,
        mode=args.mode,
        skeleton_file=args.skeleton_file,
        calib_file=args.calib_file,
        hand=args.hand,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
