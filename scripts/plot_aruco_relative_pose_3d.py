#!/usr/bin/env python3
"""Realtime 3D plot for direct ArUco relative pose tracking."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexslide.communications import camera_communication, resolve_camera_source
from dexslide.paths import (
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE,
    DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE,
)
from dexslide.vision import DirectArucoTracker


COLOR_CYCLE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def _parse_target_marker_ids(raw_value: str) -> list[int] | None:
    text = str(raw_value).strip()
    if not text:
        return None
    marker_ids: list[int] = []
    for chunk in text.replace(";", ",").split(","):
        item = chunk.strip()
        if not item:
            continue
        marker_ids.append(int(item))
    return marker_ids or None


def _matrix_from_pose_dict(pose_dict):
    if pose_dict is None:
        return None
    return np.asarray(pose_dict["matrix"], dtype=np.float64)


def _append_history(history: deque[np.ndarray], position: np.ndarray | None) -> None:
    if position is None:
        history.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))
        return
    history.append(np.asarray(position, dtype=np.float64).reshape(3))


def _draw_axes(ax, transform: np.ndarray, scale: float, prefix: str) -> None:
    origin = np.asarray(transform[:3, 3], dtype=np.float64)
    rot = np.asarray(transform[:3, :3], dtype=np.float64)
    for axis_index, color in enumerate(("r", "g", "b")):
        axis_vec = rot[:, axis_index] * scale
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            axis_vec[0],
            axis_vec[1],
            axis_vec[2],
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.18,
        )
        ax.text(
            origin[0] + axis_vec[0],
            origin[1] + axis_vec[1],
            origin[2] + axis_vec[2],
            f"{prefix}{'xyz'[axis_index]}",
            color=color,
            fontsize=8,
        )


def _get_current_view(ax, default_elev: float, default_azim: float) -> tuple[float, float, float]:
    elev = float(getattr(ax, "elev", default_elev))
    azim = float(getattr(ax, "azim", default_azim))
    roll = float(getattr(ax, "roll", 0.0))
    return elev, azim, roll


def _restore_view(ax, elev: float, azim: float, roll: float) -> None:
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def _set_equal_limits(ax, points: list[np.ndarray], default_radius: float) -> None:
    if not points:
        radius = float(default_radius)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)
        return

    stacked = np.vstack(points)
    finite_rows = np.isfinite(stacked).all(axis=1)
    if not np.any(finite_rows):
        radius = float(default_radius)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_zlim(-radius, radius)
        return

    pts = stacked[finite_rows]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, float(default_radius))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def main() -> None:
    camera_intrinsics = json.loads(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Plot direct ArUco relative pose in a table/world frame.")
    parser.add_argument("--source", default=resolve_camera_source("primary"), help="Configured capture source")
    parser.add_argument(
        "--camera-intrinsics",
        default=str(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE),
        help="Camera intrinsics JSON path",
    )
    parser.add_argument(
        "--table-aruco-yaml",
        default=str(DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE),
        help="Reference/table ArUco YAML path",
    )
    parser.add_argument(
        "--target-aruco-yaml",
        default=str(DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE),
        help="Target ArUco YAML path",
    )
    parser.add_argument("--table-marker-id", type=int, default=0, help="Fixed table/reference marker id")
    parser.add_argument(
        "--target-marker-ids",
        default="",
        help="Comma-separated target marker ids. Empty means track all non-table ids that are detected.",
    )
    parser.add_argument("--width", type=int, default=int(camera_intrinsics["image_width"]), help="Capture width override")
    parser.add_argument("--height", type=int, default=int(camera_intrinsics["image_height"]), help="Capture height override")
    parser.add_argument("--fps", type=float, default=float(camera_intrinsics["fps"]), help="Capture fps override")
    parser.add_argument("--plot-fps", type=float, default=15.0, help="Matplotlib refresh rate")
    parser.add_argument("--history-size", type=int, default=300, help="Trajectory history length")
    parser.add_argument("--axis-length", type=float, default=0.06, help="Axis glyph size in meters")
    parser.add_argument("--range-padding", type=float, default=0.20, help="Minimum plot half-range in meters")
    parser.add_argument("--view-elev", type=float, default=24.0, help="3D view elevation")
    parser.add_argument("--view-azim", type=float, default=-58.0, help="3D view azimuth")
    parser.add_argument("--buffer-size", type=int, default=2, help="VideoCapture buffer size")
    parser.add_argument("--num-workers", type=int, default=1, help="OpenCV thread count")
    parser.add_argument(
        "--no-refine-subpix",
        action="store_true",
        help="Disable ArUco corner subpixel refinement",
    )
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    target_marker_ids = _parse_target_marker_ids(args.target_marker_ids)
    tracker = DirectArucoTracker(
        source=args.source,
        camera_intrinsics=args.camera_intrinsics,
        table_aruco_yaml=args.table_aruco_yaml,
        table_marker_id=args.table_marker_id,
        target_marker_ids=target_marker_ids,
        target_aruco_yaml=args.target_aruco_yaml,
        width=args.width,
        height=args.height,
        fps=args.fps,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        refine_subpix=not args.no_refine_subpix,
    )
    tracker.start()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure("Direct ArUco Relative Pose", figsize=(9, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    camera_history: deque[np.ndarray] = deque(maxlen=args.history_size)
    target_histories: dict[int, deque[np.ndarray]] = {}
    last_frame_idx = -1
    latest_camera_pose: np.ndarray | None = None
    latest_target_poses: dict[int, np.ndarray] = {}

    def update(_frame_idx: int):
        nonlocal last_frame_idx, latest_camera_pose, latest_target_poses

        snap = tracker.snapshot()
        err = tracker.error()
        if snap is not None and int(snap["frame_idx"]) != last_frame_idx:
            last_frame_idx = int(snap["frame_idx"])

            latest_camera_pose = _matrix_from_pose_dict(snap.get("camera_in_table"))
            _append_history(
                camera_history,
                None if latest_camera_pose is None else latest_camera_pose[:3, 3],
            )

            current_target_poses: dict[int, np.ndarray] = {}
            consumed_target_ids: set[int] = set()
            for raw_target_id, target_data in snap.get("targets", {}).items():
                target_id = int(raw_target_id)
                consumed_target_ids.add(target_id)
                target_histories.setdefault(target_id, deque(maxlen=args.history_size))
                target_pose = _matrix_from_pose_dict(target_data.get("target_in_table"))
                if target_pose is not None:
                    current_target_poses[target_id] = target_pose
                    _append_history(target_histories[target_id], target_pose[:3, 3])
                else:
                    _append_history(target_histories[target_id], None)

            for target_id, history in target_histories.items():
                if target_id not in consumed_target_ids:
                    _append_history(history, None)

            latest_target_poses = current_target_poses

        view_elev, view_azim, view_roll = _get_current_view(ax, args.view_elev, args.view_azim)
        ax.cla()
        ax.set_title("Direct ArUco Relative Pose")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.grid(True, alpha=0.25)

        world_transform = np.eye(4, dtype=np.float64)
        _draw_axes(ax, world_transform, args.axis_length, "w")
        ax.scatter(
            [0.0],
            [0.0],
            [0.0],
            color="k",
            marker="s",
            s=48,
            edgecolors="w",
            linewidths=0.8,
            label="table/world origin",
        )

        plotted_points: list[np.ndarray] = [np.zeros((1, 3), dtype=np.float64)]

        if camera_history:
            camera_points = np.asarray(camera_history, dtype=np.float64)
            ax.plot(
                camera_points[:, 0],
                camera_points[:, 1],
                camera_points[:, 2],
                color="0.35",
                linewidth=1.8,
                alpha=0.8,
                label="camera traj",
            )
            plotted_points.append(camera_points)
        if latest_camera_pose is not None:
            _draw_axes(ax, latest_camera_pose, args.axis_length, "c")
            cam_pos = latest_camera_pose[:3, 3]
            ax.scatter(
                [cam_pos[0]],
                [cam_pos[1]],
                [cam_pos[2]],
                color="deeppink",
                s=48,
                edgecolors="k",
                linewidths=0.6,
                label="camera",
            )
            plotted_points.append(cam_pos.reshape(1, 3))

        for color_index, target_id in enumerate(sorted(target_histories.keys())):
            color = COLOR_CYCLE[color_index % len(COLOR_CYCLE)]
            target_points = np.asarray(target_histories[target_id], dtype=np.float64)
            ax.plot(
                target_points[:, 0],
                target_points[:, 1],
                target_points[:, 2],
                color=color,
                linewidth=1.8,
                alpha=0.85,
                label=f"target {target_id} traj",
            )
            plotted_points.append(target_points)
            current_pose = latest_target_poses.get(target_id)
            if current_pose is None:
                continue
            _draw_axes(ax, current_pose, args.axis_length, f"t{target_id}-")
            pos = current_pose[:3, 3]
            ax.scatter([pos[0]], [pos[1]], [pos[2]], color=color, s=40, label=f"target {target_id}")
            plotted_points.append(pos.reshape(1, 3))

        _set_equal_limits(ax, plotted_points, args.range_padding)
        _restore_view(ax, view_elev, view_azim, view_roll)

        status_lines = []
        if snap is None:
            status_lines.append("waiting for first frame...")
        else:
            status_lines.append(f"frame={snap['frame_idx']}  image={snap['image_size']}")
            status_lines.append(
                f"detected_ids={snap['detected_ids']}  table_detected={snap['table_detected']}  world_targets={snap['n_world_targets']}"
            )
        if err:
            status_lines.append(f"error={err}")
        ax.text2D(0.02, 0.98, "\n".join(status_lines), transform=ax.transAxes, va="top", fontsize=10)
        ax.legend(loc="upper left")

    anim = FuncAnimation(
        fig,
        update,
        interval=max(1, int(1000.0 / max(args.plot_fps, 1.0))),
        cache_frame_data=False,
    )
    fig._anim = anim  # type: ignore[attr-defined]

    try:
        plt.tight_layout()
        plt.show()
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
