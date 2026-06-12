#!/usr/bin/env python3
"""TASK-037 ATE regression check for native playback vs Docker baseline.

Runs realsense_online in playback mode against a recorded session, captures
stdout pose lines, aligns them to the batch trajectory CSV, and reports ATE.
Uses numpy only; no scipy dependency.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


BINARY = Path(
    "/home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online"
)
SETTINGS = Path("/home/jzq/MyJob/DexSlide/umi_mono/config/RealSense_D435i_online.yaml")
VOCAB = Path("/home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare native playback poses against Docker baseline CSV.")
    parser.add_argument("--recording", required=True, help="Session directory with raw_video.mp4, imu_data.json, demos/mapping assets.")
    parser.add_argument("--duration", type=float, default=60.0, help="Max playback runtime in seconds.")
    parser.add_argument("--tolerance_cm", type=float, default=2.0, help="ATE tolerance in centimeters.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_pose_stdout(output: str) -> List[Tuple[float, np.ndarray]]:
    poses: List[Tuple[float, np.ndarray]] = []
    for line in output.splitlines():
        if not line.startswith("pose "):
            continue
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            t = float(parts[1])
            xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
        except ValueError:
            continue
        poses.append((t, xyz))
    return poses


def parse_trajectory_csv(path: Path) -> List[Tuple[float, np.ndarray]]:
    poses: List[Tuple[float, np.ndarray]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalized = line.replace(",", " ")
            tokens = normalized.split()
            numeric: List[float] = []
            for token in tokens:
                try:
                    numeric.append(float(token))
                except ValueError:
                    numeric = []
                    break
            if len(numeric) < 4:
                continue
            poses.append((numeric[0], np.array(numeric[1:4], dtype=np.float64)))
    return poses


def compute_ate_cm(
    estimate: Sequence[Tuple[float, np.ndarray]],
    reference: Sequence[Tuple[float, np.ndarray]],
    max_dt: float = 0.05,
) -> Tuple[float, int]:
    if not estimate or not reference:
        raise ValueError("empty trajectory input")

    ref_times = np.array([item[0] for item in reference], dtype=np.float64)
    ref_xyz = np.array([item[1] for item in reference], dtype=np.float64)

    matched_distances_m: List[float] = []
    for t_est, xyz_est in estimate:
        idx = int(np.searchsorted(ref_times, t_est))
        candidates: List[int] = []
        if idx < len(ref_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda i: abs(ref_times[i] - t_est))
        if abs(ref_times[best_idx] - t_est) > max_dt:
            continue
        matched_distances_m.append(float(np.linalg.norm(xyz_est - ref_xyz[best_idx])))

    if not matched_distances_m:
        raise ValueError("no time-aligned trajectory pairs within tolerance")

    distances = np.array(matched_distances_m, dtype=np.float64)
    ate_cm = float(np.sqrt(np.mean(np.square(distances))) * 100.0)
    return ate_cm, len(matched_distances_m)


def main() -> int:
    args = parse_args()
    recording = Path(args.recording)
    playback_video = recording / "raw_video.mp4"
    playback_imu = recording / "imu_data.json"
    atlas = recording / "demos" / "mapping" / "map_atlas.osa"
    baseline_csv = recording / "demos" / "mapping" / "camera_trajectory.csv"

    try:
        require_file(playback_video, "playback video")
        require_file(playback_imu, "playback imu json")
        require_file(atlas, "map atlas")
        require_file(baseline_csv, "baseline trajectory csv")
        require_file(SETTINGS, "settings yaml")
        require_file(VOCAB, "vocabulary")
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

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
        "--publisher",
        "stdout",
        "--playback_video",
        str(playback_video),
        "--playback_imu",
        str(playback_imu),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode not in (0, 124):
        sys.stderr.write(proc.stdout)
        return 1

    estimate = parse_pose_stdout(proc.stdout)
    reference = parse_trajectory_csv(baseline_csv)
    try:
        ate_cm, matched = compute_ate_cm(estimate, reference)
    except ValueError as exc:
        print(f"TASK-037 FAIL: {exc}", file=sys.stderr)
        return 1

    note = ""
    if len(estimate) > 1 and len(reference) > 1:
        est_span = float(np.linalg.norm(estimate[-1][1] - estimate[0][1]))
        ref_span = float(np.linalg.norm(reference[-1][1] - reference[0][1]))
        if ref_span > 1e-6:
            ratio = est_span / ref_span
            if ratio < 0.9 or ratio > 1.1:
                note = " Note: ATE may include scale ambiguity."

    if ate_cm <= args.tolerance_cm:
        print(
            f"TASK-037 PASS: ATE = {ate_cm:.3f} cm (tolerance {args.tolerance_cm:.3f}, matched {matched}){note}"
        )
        return 0

    print(
        f"TASK-037 FAIL: ATE = {ate_cm:.3f} cm (tolerance {args.tolerance_cm:.3f}, matched {matched}){note}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
