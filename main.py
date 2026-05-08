"""
DexSlide — Main entry point for PC-side software.

Usage:
    python main.py calibrate-skeleton                           # Offline A4 + manual marking (default)
    python main.py calibrate-skeleton --show-debug              # Show per-image mm skeleton debug window
    python main.py run --port /dev/ttyACM0
    python main.py raw --port /dev/ttyACM0
"""

import argparse
import json
import time
from pathlib import Path

from dexslide.paths import (
    DEFAULT_GLOVE_CALIBRATION_FILE,
    DEFAULT_RESULTS_FILE,
    DEFAULT_SKELETON_FILE,
)
from dexslide.serial_angles import pick_default_port

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
        raise SystemExit("No serial port found. Use --port /dev/ttyACM0")

    run_live_viewer(
        port=args.port,
        baud=args.baud,
        mode=args.mode,
        skeleton_file=skeleton_file,
        calib_file=calib_file,
        hand=args.hand,
        fps=args.fps,
    )


def cmd_raw_monitor(args):
    """Debug: print raw serial values to console."""
    from dexslide.serial_reader import SerialReader, joint_label

    print(f"Raw monitor on {args.port}... (Ctrl+C to stop)")
    with SerialReader(args.port) as reader:
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
    p_run.add_argument("--port", default=pick_default_port(), help="Serial port")
    p_run.add_argument("--baud", type=int, default=115200, help="Baud rate")
    p_run.add_argument("--mode", choices=["raw", "angles"], default="raw")
    p_run.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    p_run.add_argument("--calib-file", default=str(DEFAULT_GLOVE_CALIBRATION_FILE))
    p_run.add_argument("--hand", choices=["auto", "left", "right"], default="left")
    p_run.add_argument("--fps", type=float, default=30.0)
    p_run.set_defaults(func=cmd_run)

    # Debug
    p_raw = subparsers.add_parser("raw", help="Print raw serial values")
    p_raw.add_argument("--port", required=True, help="Serial port")
    p_raw.set_defaults(func=cmd_raw_monitor)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
