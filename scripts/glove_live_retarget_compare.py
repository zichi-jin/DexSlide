#!/usr/bin/env python3
"""Live overlay of DexSlide human reconstruction and retargeted Orca hand."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexslide.paths import (
    DEFAULT_GLOVE_CALIBRATION_FILE,
    DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE,
    DEFAULT_SKELETON_FILE,
)
from dexslide.serial_angles import pick_default_port
from dexslide.visualization.live_retarget_compare import run_live_retarget_compare_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime overlay of reconstructed human hand and retargeted Orca hand."
    )
    parser.add_argument("--port", default=pick_default_port(), help="Serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument(
        "--mode",
        choices=["raw", "angles"],
        default="raw",
        help="raw: parse MCU I2C line; angles: parse ads_live_monitor --angles line",
    )
    parser.add_argument("--calib-file", default=str(DEFAULT_GLOVE_CALIBRATION_FILE))
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--retarget-config", default=str(DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE))
    parser.add_argument(
        "--hand",
        choices=["left", "right", "auto"],
        default="auto",
        help="Physical glove/skeleton side used for human-hand reconstruction.",
    )
    parser.add_argument(
        "--mirror-reconstruction",
        action="store_true",
        help="Mirror the reconstructed human landmarks across the palm to treat a left glove as a right hand.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--align", choices=["rigid", "similarity"], default="similarity")
    parser.add_argument(
        "--dex-retargeting-python",
        default=None,
        help=(
            "Optional optimizer fallback interpreter. The compare viewer itself still needs "
            "`requirements-retargeting.txt` installed in the current DexSlide environment."
        ),
    )
    args = parser.parse_args()

    if not args.port:
        raise SystemExit("No serial port found. Use --port /dev/ttyACM0")

    run_live_retarget_compare_viewer(
        port=args.port,
        baud=args.baud,
        mode=args.mode,
        calib_file=args.calib_file,
        skeleton_file=args.skeleton_file,
        retarget_config=args.retarget_config,
        dex_retargeting_python=args.dex_retargeting_python,
        fps=args.fps,
        hand=args.hand,
        mirror_reconstruction=args.mirror_reconstruction,
        align_mode=args.align,
    )


if __name__ == "__main__":
    main()
