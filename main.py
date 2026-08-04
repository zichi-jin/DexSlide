"""
DexSlide — Main entry point for PC-side software.

Usage:
    python main.py calibrate-skeleton                           # Offline A4 + manual marking (default)
    python main.py calibrate-skeleton --show-debug              # Show per-image mm skeleton debug window
    python main.py run                                          # Plot3D compatibility alias
    python main.py raw                                          # Uses global communications config
"""

import argparse
from contextlib import ExitStack, redirect_stdout
import json
import sys
import time
from pathlib import Path

import numpy as np

from dexslide.communications import hand_joint_communication, resolve_joint_port
from dexslide.paths import (
    DEFAULT_DEXSLIDE_STREAMING_FILE,
    DEFAULT_RESULTS_FILE,
    DEFAULT_SKELETON_FILE,
)


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    """Compatibility alias for the scene-backed Plot3D stream."""

    print(
        "[run] compatibility alias: using DexSlideScene with the Plot3D viewer; "
        "camera, serial and calibration settings come from --config.",
        file=sys.stderr,
    )
    cmd_stream(
        argparse.Namespace(
            config=args.config,
            joint_unit=args.joint_unit,
            joint_mode=args.joint_mode,
            pose_filter_enabled=getattr(args, "pose_filter_enabled", None),
            rate_hz=args.rate_hz,
            duration_sec=args.duration_sec,
            max_samples=args.max_samples,
            no_stdout=True,
            show_overlay=False,
            show_plot3d=True,
            plot_fps=args.plot_fps,
            plot_range_m=args.plot_range_m,
            no_skeleton=args.no_skeleton,
            save_dir=None,
            session_id=None,
            chunk_size=1000,
        )
    )


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


def cmd_stream(args):
    """Emit realtime scene samples and attach optional consumers."""
    from dexslide.recording import DexSlideRecorder
    from dexslide.streaming import DexSlideScene
    from dexslide.visualization import DexSlideARViewer, DexSlidePlot3DViewer

    if args.rate_hz is not None and args.rate_hz <= 0.0:
        raise SystemExit("--rate-hz must be positive")
    if args.duration_sec is not None and args.duration_sec <= 0.0:
        raise SystemExit("--duration-sec must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if args.plot_fps <= 0.0:
        raise SystemExit("--plot-fps must be positive")
    if args.plot_range_m <= 0.0:
        raise SystemExit("--plot-range-m must be positive")

    started_at = time.monotonic()
    emitted = 0
    with ExitStack() as stack:
        scene = DexSlideScene.from_file(
            args.config,
            joint_unit=args.joint_unit,
            joint_mode=args.joint_mode,
            pose_filter_enabled=getattr(args, "pose_filter_enabled", None),
        )
        with redirect_stdout(sys.stderr):
            scene.start()
        camera_cfg = scene.config.get("camera", {})
        print(
            "[stream] "
            f"config={Path(args.config).resolve()} "
            f"joint_unit={scene.joint_unit} joint_mode={scene.joint_mode} "
            f"pose_filter={'enabled' if scene.pose_filter_enabled else 'disabled'} "
            f"camera_source={camera_cfg.get('source', '<auto>')} "
            f"requested={camera_cfg.get('width')}x{camera_cfg.get('height')}@{camera_cfg.get('fps')} "
            f"fourcc={camera_cfg.get('fourcc') or 'default'} "
            f"hands={','.join(scene.hand_ids)} "
            f"recording={'enabled' if args.save_dir else 'disabled'} "
            f"ar_viewer={'enabled' if args.show_overlay else 'disabled'} "
            f"plot3d_viewer={'enabled' if args.show_plot3d else 'disabled'}",
            file=sys.stderr,
        )
        stack.callback(scene.close)
        recorder = None
        if args.save_dir:
            recorder = stack.enter_context(
                DexSlideRecorder(
                    args.save_dir,
                    scene,
                    session_id=args.session_id,
                    chunk_size=args.chunk_size,
                )
            )
            print(f"[recording] session={recorder.session_dir}", file=sys.stderr)
        ar_viewer = (
            DexSlideARViewer(scene, show_skeleton=not args.no_skeleton)
            if args.show_overlay
            else None
        )
        if ar_viewer is not None:
            stack.callback(ar_viewer.close)
        plot3d_viewer = (
            DexSlidePlot3DViewer(
                scene,
                show_skeleton=not args.no_skeleton,
                plot_range_m=args.plot_range_m,
                max_refresh_hz=args.plot_fps,
            )
            if args.show_plot3d
            else None
        )
        if plot3d_viewer is not None:
            stack.callback(plot3d_viewer.close)

        try:
            for sample in scene.samples(rate_hz=args.rate_hz):
                if not args.no_stdout:
                    print(_compact_stream_payload(sample), flush=True)
                if recorder is not None:
                    recorder.write(sample)
                if ar_viewer is not None and not ar_viewer.update(sample):
                    break
                if plot3d_viewer is not None and not plot3d_viewer.update(sample):
                    break
                emitted += 1
                if args.max_samples is not None and emitted >= args.max_samples:
                    break
                if (
                    args.duration_sec is not None
                    and time.monotonic() - started_at >= args.duration_sec
                ):
                    break
        except KeyboardInterrupt:
            pass


def _compact_stream_payload(sample) -> str:
    """Serialize the primary hand as the small terminal/teleop JSONL payload."""

    hand = next(iter(sample.hands.values()))
    transform = np.asarray(hand.transform_table_hand, dtype=np.float64).reshape(4, 4)
    payload = {
        "timestamp": round(float(sample.timestamp), 3),
        "T_hand": transform.tolist() if np.isfinite(transform).all() else None,
        "joint_angles": [int(round(float(value))) for value in hand.joint_angles],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def main():
    joint_communication = hand_joint_communication("left")
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

    # Compatibility alias for the scene-backed Plot3D stream.
    p_run = subparsers.add_parser(
        "run",
        help="Compatibility alias for `stream --show-plot3d`",
    )
    p_run.add_argument("--config", default=str(DEFAULT_DEXSLIDE_STREAMING_FILE))
    p_run.add_argument("--joint-unit", choices=["deg", "rad"], default=None)
    p_run.add_argument("--joint-mode", choices=["dexalign", "raw"], default=None)
    run_pose_filter_group = p_run.add_mutually_exclusive_group()
    run_pose_filter_group.add_argument(
        "--pose-filter",
        dest="pose_filter_enabled",
        action="store_true",
        help="Enable all realtime pose smoothing layers, overriding the streaming config.",
    )
    run_pose_filter_group.add_argument(
        "--no-pose-filter",
        dest="pose_filter_enabled",
        action="store_false",
        help="Disable all realtime pose smoothing layers, overriding the streaming config.",
    )
    p_run.add_argument(
        "--rate-hz",
        "--fps",
        dest="rate_hz",
        type=float,
        default=None,
        help="Optional scene sample-rate limit; --fps is retained as an alias.",
    )
    p_run.add_argument("--duration-sec", type=float, default=None)
    p_run.add_argument("--max-samples", type=int, default=None)
    p_run.add_argument("--plot-fps", type=float, default=20.0)
    p_run.add_argument("--plot-range-m", type=float, default=0.45)
    p_run.add_argument("--no-skeleton", action="store_true")
    p_run.set_defaults(func=cmd_run, pose_filter_enabled=None)

    # Independent realtime scene stream.
    p_stream = subparsers.add_parser(
        "stream",
        help="Run the realtime multi-hand scene stream",
    )
    p_stream.add_argument("--config", default=str(DEFAULT_DEXSLIDE_STREAMING_FILE))
    p_stream.add_argument("--joint-unit", choices=["deg", "rad"], default=None)
    p_stream.add_argument("--joint-mode", choices=["dexalign", "raw"], default=None)
    stream_pose_filter_group = p_stream.add_mutually_exclusive_group()
    stream_pose_filter_group.add_argument(
        "--pose-filter",
        dest="pose_filter_enabled",
        action="store_true",
        help="Enable all realtime pose smoothing layers, overriding the streaming config.",
    )
    stream_pose_filter_group.add_argument(
        "--no-pose-filter",
        dest="pose_filter_enabled",
        action="store_false",
        help="Disable all realtime pose smoothing layers, overriding the streaming config.",
    )
    p_stream.add_argument(
        "--rate-hz",
        type=float,
        default=None,
        help="Optional output/sample rate limit. Omit to run at the available processing rate.",
    )
    p_stream.add_argument("--duration-sec", type=float, default=None)
    p_stream.add_argument("--max-samples", type=int, default=None)
    stdout_group = p_stream.add_mutually_exclusive_group()
    stdout_group.add_argument(
        "--stdout",
        dest="no_stdout",
        action="store_false",
        help="Opt in to compact realtime JSONL samples on stdout.",
    )
    stdout_group.add_argument(
        "--no-stdout",
        dest="no_stdout",
        action="store_true",
        help="Suppress realtime sample output (the default).",
    )
    p_stream.add_argument("--show-overlay", action="store_true")
    p_stream.add_argument(
        "--show-plot3d",
        action="store_true",
        help="Show skeleton and body/wrist poses in the table-frame Plot3D viewer.",
    )
    p_stream.add_argument(
        "--plot-fps",
        type=float,
        default=20.0,
        help="Maximum Plot3D refresh rate; acquisition continues independently.",
    )
    p_stream.add_argument(
        "--plot-range-m",
        type=float,
        default=0.45,
        help="Initial Plot3D half-range around the table origin in meters.",
    )
    p_stream.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Do not draw the optional reconstructed hand skeleton in either viewer.",
    )
    p_stream.add_argument("--save-dir", default=None)
    p_stream.add_argument("--session-id", default=None)
    p_stream.add_argument("--chunk-size", type=int, default=1000)
    p_stream.set_defaults(func=cmd_stream, no_stdout=True, pose_filter_enabled=None)

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
