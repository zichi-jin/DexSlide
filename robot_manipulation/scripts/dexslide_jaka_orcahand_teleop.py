#!/usr/bin/env python3
"""Combined one-arm DexSlide→JAKA wrist + OrcaHand finger teleoperation."""

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
from robot_manipulation.JAKA_control.paths import DEFAULT_WORKSPACE_MAPPING_FILE
from robot_manipulation.JAKA_control.teleop_endpoint import JakaTeleopConfig, JakaTeleopEndpoint
from robot_manipulation.JAKA_control.workspace_mapping import load_workspace_axis_mapping
from robot_manipulation.orca_control.paths import (
    DEFAULT_DIRECT_JOINT_CALIBRATION_FILE,
    DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE,
)
from robot_manipulation.orca_control.teleop_endpoint import OrcaTeleopConfig, OrcaTeleopEndpoint
from robot_manipulation.teleop.runtime import DexSlideTeleopRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-config", default=str(DEFAULT_DEXSLIDE_STREAMING_FILE))
    parser.add_argument("--source-hand", default="left")
    parser.add_argument("--mapping-file", default=str(DEFAULT_WORKSPACE_MAPPING_FILE))
    parser.add_argument("--orca-calibration-file", default=str(DEFAULT_DIRECT_JOINT_CALIBRATION_FILE))
    parser.add_argument("--orca-config", default=str(DEFAULT_ORCAHAND_V1_RIGHT_CONFIG_FILE))
    parser.add_argument("--jaka-ip", default="192.168.99.44")
    parser.add_argument(
        "--no-workspace-clip",
        action="store_true",
        help="关闭 JAKA 遥操阶段的 XYZ workspace 安全裁剪",
    )
    parser.add_argument("--loop-hz", type=float, default=None, help="可选的遥操循环限速；默认跟随 DexSlide API 可用速率")
    parser.add_argument("--print-interval-s", type=float, default=1.0)
    parser.add_argument("--jaka-dry-run", action="store_true")
    parser.add_argument("--orca-dry-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-orca", action="store_true")
    parser.add_argument("--show-overlay", action="store_true")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--session-id", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.loop_hz is not None and args.loop_hz <= 0.0) or args.print_interval_s <= 0.0:
        raise SystemExit("loop and print intervals must be positive")
    dry_run = bool(args.dry_run)
    scene = DexSlideScene.from_file(args.stream_config)
    mapping = load_workspace_axis_mapping(args.mapping_file)
    jaka = JakaTeleopEndpoint(
        JakaTeleopConfig(
            ip=args.jaka_ip,
            source_hand=args.source_hand,
            enable_workspace_clip=not args.no_workspace_clip,
            dry_run=dry_run or args.jaka_dry_run,
        ),
        mapping,
    )
    orca = OrcaTeleopEndpoint(
        OrcaTeleopConfig(
            source_hand=args.source_hand,
            calibration_file=args.orca_calibration_file,
            orca_config_file=args.orca_config,
            loop_hz=args.loop_hz,
            dry_run=dry_run or args.orca_dry_run,
            mock=args.mock_orca,
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
        [jaka, orca],
        loop_hz=args.loop_hz,
        print_interval_s=args.print_interval_s,
        recorder=recorder,
        viewer=viewer,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
