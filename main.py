"""
DexSlide — Main entry point for PC-side software.

Usage:
    python main.py calibrate-skeleton                           # Offline A4 + manual marking (default)
    python main.py calibrate-skeleton --show-debug              # Show per-image mm skeleton debug window
    python main.py run                                          # Uses global communications config
    python main.py raw                                          # Uses global communications config
"""

import argparse
import json
import time
from pathlib import Path

from dexslide.communications import (
    camera_communication,
    hand_joint_communication,
    resolve_camera_source,
    resolve_joint_port,
)
from dexslide.paths import (
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_GLOVE_CALIBRATION_FILE,
    DEFAULT_RESULTS_FILE,
    DEFAULT_SKELETON_FILE,
)

def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_group_hand_map(raw: str | None) -> dict[str, str] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    path = Path(text).expanduser()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise SystemExit(f"Invalid --aruco-group-hand-map file, expected dict: {path}")
        return {str(k): str(v) for k, v in obj.items()}

    mapping: dict[str, str] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(
                "Invalid --aruco-group-hand-map format. "
                "Use 'group1:hand1,group2:hand2' or a JSON file path."
            )
        group, hand_slot = chunk.split(":", 1)
        group = group.strip()
        hand_slot = hand_slot.strip()
        if not group or not hand_slot:
            raise SystemExit(
                "Invalid --aruco-group-hand-map entry. "
                "Use 'group1:hand1,group2:hand2'."
            )
        mapping[group] = hand_slot
    return mapping or None


def cmd_calibrate_skeleton(args):
    """Phase 1: Offline skeleton extraction from photos with manual A4 marking."""
    from dexslide.calibration.offline_a4_bone_mm import run_offline_pipeline

    print("Starting offline skeleton calibration (A4 + manual 4-corner marking)...")
    run_offline_pipeline(
        input_dir=Path(args.input_dir),
        output_json=Path(args.results_file),
        skeleton_json=Path(args.skeleton_file),
        min_conf=float(args.min_conf),
        show_debug=bool(args.show_debug),
        reuse_a4=bool(args.reuse_a4),
        skeleton_aggregate=str(args.skeleton_aggregate),
    )
    print(f"\nSkeleton saved to: {args.skeleton_file}")
    print(f"Detailed results saved to: {args.results_file}")
    print(
        "\nNext: calibrate the STM32 ADC-to-angle mapping with "
        "`python scripts/glove_calibrate.py --port <PORT> --out <DexSlide>/assets/calibration/glove_calibration.json` "
        "from the dexslide_infra repository if `glove_calibration.json` is not ready yet."
    )

    if not args.no_skeleton_plot:
        try:
            from dexslide.visualization.skeleton_plot import show_skeleton_plot

            print("Opening 2D skeleton review plot...")
            skeleton = _load_json(args.skeleton_file)
            show_skeleton_plot(skeleton)
        except ImportError:
            print("WARNING: matplotlib not available, skip skeleton plot.")
        except Exception as ex:
            print(f"WARNING: failed to show skeleton plot: {ex}")


def cmd_run(args):
    """Real-time reconstruction and visualization."""
    from dexslide.visualization.live_matplotlib import run_live_viewer

    skeleton_file = Path(args.skeleton_file)
    calib_file = Path(args.calib_file)
    if not skeleton_file.exists():
        raise SystemExit(
            f"Skeleton file not found: {skeleton_file}\n"
            "Run `python main.py calibrate-skeleton` first."
        )
    if not calib_file.exists() and args.mode == "raw":
        raise SystemExit(
            f"Glove angle calibration file not found: {calib_file}\n"
            "Run `python scripts/glove_calibrate.py --port <PORT> "
            "--out <DexSlide>/assets/calibration/glove_calibration.json` "
            "from the dexslide_infra repository "
            "to record the 20-joint ADC mapping, then verify it with "
            "`python scripts/ads_live_monitor.py --port <PORT> --angles "
            "--calib-file <DexSlide>/assets/calibration/glove_calibration.json`."
        )
    if not args.port:
        raise SystemExit("No serial port configured in assets/dexslide_communications.json")

    aruco_group_hand_map = _parse_group_hand_map(args.aruco_group_hand_map)
    aruco_tracker = None
    try:
        if args.aruco_enable:
            if not args.aruco_camera_intrinsics:
                raise SystemExit(
                    "--aruco-enable requires --aruco-camera-intrinsics "
                    "(path to camera intrinsics json)."
                )
            if not args.aruco_yaml:
                raise SystemExit(
                    "--aruco-enable requires --aruco-yaml "
                    "(path to ArUco config yaml)."
                )
            aruco_intr_path = Path(args.aruco_camera_intrinsics).expanduser()
            aruco_yaml_path = Path(args.aruco_yaml).expanduser()
            if not aruco_intr_path.exists():
                raise SystemExit(f"ArUco intrinsics file not found: {aruco_intr_path}")
            if not aruco_yaml_path.exists():
                raise SystemExit(f"ArUco config yaml not found: {aruco_yaml_path}")
            if args.aruco_fusion_groups_yaml:
                groups_path = Path(args.aruco_fusion_groups_yaml).expanduser()
                if not groups_path.exists():
                    raise SystemExit(f"ArUco fusion group yaml not found: {groups_path}")
            from dexslide.vision.aruco_pose_tracker import ArucoPoseTracker

            aruco_tracker = ArucoPoseTracker(
                source=args.aruco_source,
                camera_intrinsics=aruco_intr_path,
                aruco_yaml=aruco_yaml_path,
                offset_scale=float(args.aruco_offset_scale),
                merge_pos_threshold=float(args.aruco_merge_pos_threshold),
                fusion_groups_yaml=args.aruco_fusion_groups_yaml,
                warning_cooldown_sec=float(args.aruco_warning_cooldown_sec),
                width=args.aruco_width,
                height=args.aruco_height,
                fps=args.aruco_fps,
                buffer_size=int(args.aruco_buffer_size),
                num_workers=int(args.aruco_num_workers),
                refine_subpix=True,
            )
            aruco_tracker.start()
            print(
                "ArUco pose tracker enabled. "
                f"source={args.aruco_source}, hand_slot={args.aruco_hand_slot}, "
                f"hand_group={args.aruco_hand_group or 'auto'}"
            )

        run_live_viewer(
            port=args.port,
            baud=args.baud,
            mode=args.mode,
            skeleton_file=skeleton_file,
            calib_file=calib_file,
            hand=args.hand,
            fps=args.fps,
            aruco_pose_tracker=aruco_tracker,
            aruco_hand_group=args.aruco_hand_group,
            aruco_group_hand_map=aruco_group_hand_map,
            aruco_hand_slot=args.aruco_hand_slot,
            aruco_pose_hold_sec=float(args.aruco_pose_hold_sec),
        )
    finally:
        if aruco_tracker is not None:
            aruco_tracker.stop()


def cmd_raw_monitor(args):
    """Debug: print raw serial values to console."""
    from dexslide.serial_reader import SerialReader, joint_label

    print(f"Raw monitor on {args.port}... (Ctrl+C to stop)")
    with SerialReader(args.port, args.baud) as reader:
        try:
            while True:
                frame = reader.read_frame()
                if frame is None:
                    continue
                labels = [f"{joint_label(i)}={frame[i]:5d}" for i in range(20)]
                # Print in 5 rows (one per finger)
                for row in range(5):
                    print("  ".join(labels[row * 4:(row + 1) * 4]))
                print("---")
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    joint_communication = hand_joint_communication("left")
    camera_intrinsics = _load_json(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE)
    parser = argparse.ArgumentParser(description="DexSlide Exoskeleton Data Glove")
    subparsers = parser.add_subparsers(dest="command")

    # Phase 1
    p_skel = subparsers.add_parser(
        "calibrate-skeleton",
        help="Offline skeleton extraction (A4 + manual 4-point marking)",
    )
    default_input_dir = Path(__file__).resolve().parent / "assets" / "photos"
    p_skel.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help="Input image directory (default: assets/photos)",
    )
    p_skel.add_argument(
        "--results-file",
        "--output-json",
        dest="results_file",
        default=str(DEFAULT_RESULTS_FILE),
        help="Detailed per-image output JSON",
    )
    p_skel.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE), help="Output file")
    p_skel.add_argument(
        "--min-conf",
        type=float,
        default=0.35,
        help="Minimum MediaPipe landmark confidence [0..1]",
    )
    p_skel.add_argument(
        "--reuse-a4",
        action="store_true",
        help="Reuse first image's A4 corner points for all images",
    )
    p_skel.add_argument(
        "--show-debug",
        action="store_true",
        help="Show mm debug plot for each image (mouse-only control)",
    )
    p_skel.add_argument(
        "--skeleton-aggregate",
        choices=["median", "mean"],
        default="mean",
        help="How to aggregate multi-image measurements into skeleton.json",
    )
    p_skel.add_argument(
        "--no-skeleton-plot",
        action="store_true",
        help="Skip skeleton review plot after calibration",
    )
    p_skel.set_defaults(func=cmd_calibrate_skeleton)

    # Real-time reconstruction
    p_run = subparsers.add_parser("run", help="Real-time hand reconstruction")
    p_run.add_argument("--port", default=resolve_joint_port("left"), help="Serial port")
    p_run.add_argument("--baud", type=int, default=int(joint_communication["baud"]), help="Baud rate")
    p_run.add_argument("--mode", choices=["raw", "angles"], default=str(joint_communication["mode"]))
    p_run.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    p_run.add_argument("--calib-file", default=str(DEFAULT_GLOVE_CALIBRATION_FILE))
    p_run.add_argument("--hand", choices=["auto", "left", "right"], default="left")
    p_run.add_argument("--fps", type=float, default=30.0)
    p_run.add_argument(
        "--aruco-enable",
        action="store_true",
        help="Enable realtime ArUco marker pose tracking and inject as hand pose.",
    )
    p_run.add_argument(
        "--aruco-source",
        default=resolve_camera_source("primary"),
        help="ArUco capture source. Defaults to assets/dexslide_communications.json.",
    )
    p_run.add_argument(
        "--aruco-camera-intrinsics",
        default=str(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE),
        help="Camera intrinsics json for ArUco pose estimation.",
    )
    p_run.add_argument(
        "--aruco-yaml",
        default=None,
        help="ArUco config yaml (dictionary + marker_size_map).",
    )
    p_run.add_argument(
        "--aruco-fusion-groups-yaml",
        default=None,
        help="Optional fusion-group yaml. If omitted, one implicit group 'all' is used.",
    )
    p_run.add_argument(
        "--aruco-hand-group",
        default=None,
        help="Optional group name used as the current viewer hand pose source.",
    )
    p_run.add_argument(
        "--aruco-group-hand-map",
        default=None,
        help=(
            "Optional mapping for future multi-hand routing, format: "
            "'group1:hand1,group2:hand2' or a JSON file path."
        ),
    )
    p_run.add_argument(
        "--aruco-hand-slot",
        choices=["hand1", "hand2"],
        default="hand1",
        help="Current viewer hand slot label used with --aruco-group-hand-map.",
    )
    p_run.add_argument(
        "--aruco-offset-scale",
        type=float,
        default=0.0,
        help="Offset (meters) along marker center negative z-axis.",
    )
    p_run.add_argument(
        "--aruco-merge-pos-threshold",
        type=float,
        default=0.03,
        help="Max pairwise distance (m) for per-group position fusion.",
    )
    p_run.add_argument(
        "--aruco-warning-cooldown-sec",
        type=float,
        default=1.0,
        help="Cooldown seconds for repeated per-group disagree warnings.",
    )
    p_run.add_argument(
        "--aruco-pose-hold-sec",
        type=float,
        default=0.5,
        help="Drop ArUco pose if snapshot age exceeds this threshold (seconds).",
    )
    p_run.add_argument("--aruco-width", type=int, default=int(camera_intrinsics["image_width"]), help="Requested ArUco capture width.")
    p_run.add_argument("--aruco-height", type=int, default=int(camera_intrinsics["image_height"]), help="Requested ArUco capture height.")
    p_run.add_argument("--aruco-fps", type=float, default=float(camera_intrinsics["fps"]), help="Requested ArUco capture FPS.")
    p_run.add_argument(
        "--aruco-buffer-size",
        type=int,
        default=2,
        help="ArUco capture buffer size.",
    )
    p_run.add_argument(
        "--aruco-num-workers",
        type=int,
        default=1,
        help="OpenCV thread count for ArUco tracker.",
    )
    p_run.set_defaults(func=cmd_run)

    # Debug
    p_raw = subparsers.add_parser("raw", help="Print raw serial values")
    p_raw.add_argument("--port", default=resolve_joint_port("left"), help="Serial port")
    p_raw.add_argument("--baud", type=int, default=int(joint_communication["baud"]), help="Baud rate")
    p_raw.set_defaults(func=cmd_raw_monitor)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
