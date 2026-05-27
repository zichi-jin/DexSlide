#!/usr/bin/env python3
"""TASK-017 style 30-second live localization smoke test.

Runs realsense_online against a provided atlas, counts stdout pose lines,
and writes a compact JSON summary for quick pass/fail inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BINARY = Path(
    "/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online"
)
VOCAB = Path("/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt")
SETTINGS = Path("/data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 30-second live localization smoke test with a provided map_atlas.osa."
    )
    parser.add_argument("atlas", help="Path to map_atlas.osa")
    parser.add_argument("--duration", type=int, default=30, help="Live run duration in seconds.")
    parser.add_argument(
        "--min-pose-lines",
        type=int,
        default=750,
        help="Minimum pose lines required to pass. Default scales the old 60s >=1500 rule to 30s.",
    )
    parser.add_argument(
        "--exposure-us",
        type=int,
        default=0,
        help="RealSense exposure in microseconds. 0 = auto-exposure; override for a fixed indoor setup.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional summary JSON path. Defaults to /tmp/test_live_30s_YYYYMMDD_HHMMSS.json.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional raw log path. Defaults to /tmp/test_live_30s_YYYYMMDD_HHMMSS.log.",
    )
    parser.add_argument(
        "--show-rgb",
        action="store_true",
        help="Show the incoming RGB frames in an OpenCV window while the test is running.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_pose_lines(output: str) -> tuple[int, float | None]:
    pose_count = 0
    first_ts = None
    last_ts = None
    for line in output.splitlines():
        if not line.startswith("pose "):
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[1])
        except ValueError:
            continue
        pose_count += 1
        if first_ts is None:
            first_ts = ts
        last_ts = ts

    pose_hz = None
    if pose_count >= 2 and first_ts is not None and last_ts is not None:
        span = max(last_ts - first_ts, 1e-6)
        pose_hz = (pose_count - 1) / span
    return pose_count, pose_hz


def classify_output(output: str, exit_code: int, pose_count: int, min_pose_lines: int) -> tuple[bool, str]:
    if "No RealSense device found" in output:
        return False, "no_device"
    if "Load map file not found" in output:
        return False, "atlas_not_found"
    if exit_code not in (0, 124):
        return False, f"process_exit_{exit_code}"
    if "Atlas loaded from:" in output and pose_count == 0:
        return False, "no_pose"
    if pose_count < min_pose_lines:
        return False, "insufficient_pose_lines"
    return True, "ok"


def main() -> int:
    args = parse_args()
    atlas = Path(args.atlas).expanduser().resolve()

    try:
        require_file(atlas, "atlas")
        require_file(BINARY, "realsense_online binary")
        require_file(VOCAB, "vocabulary")
        require_file(SETTINGS, "settings yaml")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = Path(args.output_json) if args.output_json else Path(f"/tmp/test_live_30s_{stamp}.json")
    log_path = Path(args.log_path) if args.log_path else Path(f"/tmp/test_live_30s_{stamp}.log")

    timeout_seconds = max(1, int(math.ceil(args.duration)))
    cmd = [
        "timeout",
        str(timeout_seconds),
        str(BINARY),
        "-v",
        str(VOCAB),
        "-s",
        str(SETTINGS),
        "-l",
        str(atlas),
        "--exposure_us",
        str(args.exposure_us),
        "--run_seconds",
        str(timeout_seconds),
        "--publisher",
        "stdout",
    ]
    if args.show_rgb:
        cmd.append("--show_rgb")

    output_lines: list[str] = []
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
        proc.wait()

    output_text = "".join(output_lines)
    log_path.write_text(output_text, encoding="utf-8")

    pose_count, pose_hz = parse_pose_lines(output_text)
    passed, reason = classify_output(output_text, proc.returncode, pose_count, args.min_pose_lines)
    summary = {
        "atlas": str(atlas),
        "duration_seconds": timeout_seconds,
        "min_pose_lines": args.min_pose_lines,
        "pose_lines": pose_count,
        "estimated_pose_hz": pose_hz,
        "exit_code": proc.returncode,
        "pass": passed,
        "reason": reason,
        "log_path": str(log_path),
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if proc.returncode not in (0, 124):
        return 1
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
