#!/usr/bin/env python3
"""Standalone direct-joint DexSlide→OrcaHand teleoperation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROBOT_MANIP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROBOT_MANIP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dexslide.paths import DEFAULT_DEXSLIDE_STREAMING_FILE
from dexslide.recording import DexSlideRecorder
from dexslide.streaming import DexSlideScene
from dexslide.visualization import DexSlideARViewer
from robot_manipulation.orca_control.paths import (
    DEFAULT_DIRECT_JOINT_CALIBRATION_FILE,
    DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE,
)
from robot_manipulation.orca_control.teleop_endpoint import (
    OrcaTeleopConfig,
    OrcaTeleopEndpoint,
)
from robot_manipulation.teleop.runtime import DexSlideTeleopRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-config", default=str(DEFAULT_DEXSLIDE_STREAMING_FILE))
    parser.add_argument("--source-hand", default="left")
    parser.add_argument("--calibration-file", default=str(DEFAULT_DIRECT_JOINT_CALIBRATION_FILE))
    parser.add_argument("--orca-config", default=str(DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE))
    parser.add_argument("--loop-hz", type=float, default=None, help="可选的遥操循环限速；默认跟随 DexSlide API 可用速率")
    parser.add_argument("--max-sample-age-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--show-overlay", action="store_true")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--session-id", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.loop_hz is not None and args.loop_hz <= 0.0) or args.max_sample_age_sec <= 0.0:
        raise SystemExit("loop and sample-age values must be positive")
    scene = DexSlideScene.from_file(args.stream_config)
    endpoint = OrcaTeleopEndpoint(
        OrcaTeleopConfig(
            source_hand=args.source_hand,
            calibration_file=args.calibration_file,
            orca_config_file=args.orca_config,
            loop_hz=args.loop_hz,
            max_sample_age_sec=args.max_sample_age_sec,
            dry_run=args.dry_run,
            mock=args.mock,
        )
    )
    recorder = (
        DexSlideRecorder(args.save_dir, scene, session_id=args.session_id or None)
        if args.save_dir
        else None
    )
    viewer = DexSlideARViewer(scene) if args.show_overlay else None
    DexSlideTeleopRuntime(
        scene,
        [endpoint],
        loop_hz=args.loop_hz,
        recorder=recorder,
        viewer=viewer,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
