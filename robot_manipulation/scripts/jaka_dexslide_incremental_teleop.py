#!/usr/bin/env python3
"""Standalone DexSlide wrist-pose incremental teleoperation for JAKA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROBOT_MANIP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROBOT_MANIP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dexslide.paths import DEFAULT_DEXSLIDE_STREAMING_FILE
from dexslide.kinematics.transforms import make_transform
from dexslide.recording import DexSlideRecorder
from dexslide.streaming import DexSlideScene
from dexslide.visualization import DexSlideARViewer
from robot_manipulation.JAKA_control.incremental_mapping import *  # noqa: F401,F403
from robot_manipulation.JAKA_control.paths import DEFAULT_WORKSPACE_MAPPING_FILE
from robot_manipulation.JAKA_control.teleop_endpoint import (
    JakaTeleopConfig,
    JakaTeleopEndpoint,
)
from robot_manipulation.JAKA_control.workspace_mapping import (
    load_workspace_axis_mapping,
)
from robot_manipulation.teleop.runtime import DexSlideTeleopRuntime

DEFAULT_MAPPING_FILE = DEFAULT_WORKSPACE_MAPPING_FILE



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-config", default=str(DEFAULT_DEXSLIDE_STREAMING_FILE))
    parser.add_argument("--mapping-file", default=str(DEFAULT_MAPPING_FILE))
    parser.add_argument("--source-hand", default="left")
    parser.add_argument(
        "--no-workspace-clip",
        action="store_true",
        help="关闭 JAKA 遥操阶段的 XYZ workspace 安全裁剪",
    )
    parser.add_argument("--ip", default="192.168.99.44")
    parser.add_argument("--sensor-brand", type=int, default=10)
    parser.add_argument("--force-damping-n", type=float, default=1.0)
    parser.add_argument("--torque-damping-nm", type=float, default=10.0)
    parser.add_argument("--translation-rebound-fk", type=float, default=0.5)
    parser.add_argument("--rotation-rebound-fk", type=float, default=10.0)
    parser.add_argument("--enable-rotation-compliance", action="store_true")
    parser.add_argument("--no-saved-payload", action="store_true")
    parser.add_argument("--servo-step-num", type=int, default=1)
    parser.add_argument("--safe-start-position-tol-mm", type=float, default=5.0)
    parser.add_argument("--safe-start-angle-tol-deg", type=float, default=6.0)
    parser.add_argument("--power-off-on-exit", action="store_true")
    parser.add_argument("--loop-hz", type=float, default=None, help="可选的遥操循环限速；默认跟随 DexSlide API 可用速率")
    parser.add_argument("--print-interval-s", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-overlay", action="store_true")
    parser.add_argument("--overlay-window-name", default="DexSlide Teleop AR")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--session-id", default="")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (args.loop_hz is not None and args.loop_hz <= 0.0) or args.print_interval_s <= 0.0:
        raise SystemExit("loop and print intervals must be positive")
    if args.servo_step_num < 1:
        raise SystemExit("--servo-step-num must be >= 1")
    if args.safe_start_position_tol_mm <= 0.0 or args.safe_start_angle_tol_deg <= 0.0:
        raise SystemExit("safe-start tolerances must be positive")


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)
    scene = DexSlideScene.from_file(args.stream_config)
    mapping = load_workspace_axis_mapping(args.mapping_file)
    endpoint = JakaTeleopEndpoint(
        JakaTeleopConfig(
            ip=args.ip,
            source_hand=args.source_hand,
            enable_workspace_clip=not args.no_workspace_clip,
            sensor_brand=args.sensor_brand,
            force_damping_n=args.force_damping_n,
            torque_damping_nm=args.torque_damping_nm,
            translation_rebound_fk=args.translation_rebound_fk,
            rotation_rebound_fk=args.rotation_rebound_fk,
            enable_rotation_compliance=args.enable_rotation_compliance,
            use_saved_payload=not args.no_saved_payload,
            servo_step_num=args.servo_step_num,
            safe_start_position_tol_mm=args.safe_start_position_tol_mm,
            safe_start_angle_tol_deg=args.safe_start_angle_tol_deg,
            power_off_on_exit=args.power_off_on_exit,
            dry_run=args.dry_run,
        ),
        mapping,
    )
    recorder = (
        DexSlideRecorder(args.save_dir, scene, session_id=args.session_id or None)
        if args.save_dir
        else None
    )
    viewer = (
        DexSlideARViewer(scene, window_name=args.overlay_window_name)
        if args.show_overlay
        else None
    )
    DexSlideTeleopRuntime(
        scene,
        [endpoint],
        loop_hz=args.loop_hz,
        print_interval_s=args.print_interval_s,
        recorder=recorder,
        viewer=viewer,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
