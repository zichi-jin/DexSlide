#!/usr/bin/env python3
"""Realtime camera overlay for the direct table-target ArUco pose chain."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexslide.paths import (
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE,
    DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE,
)
from dexslide.vision.aruco_pose_tracker import (
    _convert_fisheye_intrinsics_resolution,
    _parse_aruco_config,
    _parse_capture_source,
    _parse_fisheye_intrinsics,
)
from dexslide.world_pose.direct_aruco_tracker import (
    _build_direct_aruco_frame_result,
    _detect_relevant_aruco_tags,
    _normalize_target_marker_ids,
)


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


def _project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intr: dict[str, np.ndarray],
) -> np.ndarray:
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    k = np.asarray(intr["K"], dtype=np.float64)
    d = np.asarray(intr["D"], dtype=np.float64)
    if d.size == 4:
        img_points, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)
    else:
        img_points, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
    return img_points.reshape(-1, 2)


def _draw_axes_overlay(
    image: np.ndarray,
    intr: dict[str, np.ndarray],
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length: float,
    label: str,
    label_color: tuple[int, int, int],
) -> None:
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    projected = _project_points(axis_points, rvec, tvec, intr)
    origin = tuple(np.round(projected[0]).astype(int))
    axis_colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for idx, axis_color in enumerate(axis_colors, start=1):
        endpoint = tuple(np.round(projected[idx]).astype(int))
        cv2.line(image, origin, endpoint, axis_color, 2, lineType=cv2.LINE_AA)
        cv2.circle(image, endpoint, 3, axis_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, origin, 4, label_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (origin[0] + 6, origin[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        label_color,
        2,
        lineType=cv2.LINE_AA,
    )


def _draw_marker_outline(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int],
    text: str,
) -> None:
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    anchor = tuple(np.round(pts[0]).astype(int))
    cv2.putText(
        image,
        text,
        (anchor[0] + 4, anchor[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def _draw_hud(image: np.ndarray, lines: list[str]) -> None:
    y = 26
    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (24, 24, 24),
            1,
            lineType=cv2.LINE_AA,
        )
        y += 24


def main() -> None:
    parser = argparse.ArgumentParser(description="Show direct ArUco pose overlay on the live camera feed.")
    parser.add_argument("--source", default="0", help="Capture source, e.g. 0 or /dev/video4")
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
    parser.add_argument("--width", type=int, default=None, help="Capture width override")
    parser.add_argument("--height", type=int, default=None, help="Capture height override")
    parser.add_argument("--fps", type=float, default=None, help="Capture fps override")
    parser.add_argument("--buffer-size", type=int, default=1, help="VideoCapture buffer size")
    parser.add_argument("--num-workers", type=int, default=1, help="OpenCV thread count")
    parser.add_argument("--table-axis-length", type=float, default=0.08, help="Table axis glyph length in meters")
    parser.add_argument("--target-axis-length", type=float, default=0.04, help="Target axis glyph length in meters")
    parser.add_argument(
        "--no-refine-subpix",
        action="store_true",
        help="Disable ArUco corner subpixel refinement",
    )
    parser.add_argument(
        "--strict-detector",
        action="store_true",
        help="Disable the motion-tolerant detector profile and use the stricter default detector settings.",
    )
    args = parser.parse_args()

    cv2.setNumThreads(max(1, int(args.num_workers)))

    target_marker_ids = _normalize_target_marker_ids(
        _parse_target_marker_ids(args.target_marker_ids),
        table_marker_id=args.table_marker_id,
    )

    with open(args.table_aruco_yaml, "r", encoding="utf-8") as handle:
        table_cfg = _parse_aruco_config(yaml.safe_load(handle))
    same_cfg = Path(args.table_aruco_yaml).resolve() == Path(args.target_aruco_yaml).resolve()
    if same_cfg:
        target_cfg = table_cfg
    else:
        with open(args.target_aruco_yaml, "r", encoding="utf-8") as handle:
            target_cfg = _parse_aruco_config(yaml.safe_load(handle))

    with open(args.camera_intrinsics, "r", encoding="utf-8") as handle:
        raw_intr = _parse_fisheye_intrinsics(json.load(handle))

    cap_source = _parse_capture_source(args.source)
    if isinstance(cap_source, str) and cap_source.startswith("/dev/video"):
        cap = cv2.VideoCapture(cap_source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(cap_source)

    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    if args.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(args.fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(args.buffer_size))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open capture source: {args.source}")

    intr: dict[str, np.ndarray] | None = None
    intr_resolution: tuple[int, int] | None = None
    frame_idx = 0
    last_time = time.monotonic()
    fps_ema = 0.0

    window_name = "Direct ArUco Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                time.sleep(0.01)
                continue

            height, width = frame_bgr.shape[:2]
            resolution = (int(width), int(height))
            if intr is None or intr_resolution != resolution:
                intr = _convert_fisheye_intrinsics_resolution(raw_intr, target_resolution=resolution)
                intr_resolution = resolution

            assert intr is not None
            tag_dict = _detect_relevant_aruco_tags(
                frame_bgr=frame_bgr,
                intr=intr,
                table_cfg=table_cfg,
                target_cfg=target_cfg,
                table_marker_id=args.table_marker_id,
                target_marker_ids=target_marker_ids,
                same_cfg=same_cfg,
                refine_subpix=not args.no_refine_subpix,
                motion_tolerant=not args.strict_detector,
            )
            frame_result = _build_direct_aruco_frame_result(
                frame_idx=frame_idx,
                image_size=[width, height],
                table_marker_id=args.table_marker_id,
                target_marker_ids=target_marker_ids,
                tag_dict=tag_dict,
            )

            vis = frame_bgr.copy()
            table_tag = tag_dict.get(int(args.table_marker_id))
            if table_tag is not None:
                _draw_marker_outline(vis, table_tag["corners"], (0, 215, 255), f"table:{args.table_marker_id}")
                _draw_axes_overlay(
                    vis,
                    intr,
                    table_tag["rvec"],
                    table_tag["tvec"],
                    axis_length=float(args.table_axis_length),
                    label=f"table:{args.table_marker_id}",
                    label_color=(0, 215, 255),
                )

            for raw_target_id, target_data in frame_result["targets"].items():
                target_id = int(raw_target_id)
                tag = tag_dict.get(target_id)
                if tag is None:
                    continue
                _draw_marker_outline(vis, tag["corners"], (80, 255, 80), f"target:{target_id}")
                _draw_axes_overlay(
                    vis,
                    intr,
                    tag["rvec"],
                    tag["tvec"],
                    axis_length=float(args.target_axis_length),
                    label=f"target:{target_id}",
                    label_color=(80, 255, 80),
                )

            now = time.monotonic()
            dt = max(now - last_time, 1e-6)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema <= 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)
            last_time = now

            hud_lines = [
                f"frame={frame_idx}  det={frame_result['detected_ids']}  table={frame_result['table_detected']}  world_targets={frame_result['n_world_targets']}",
                f"fps~{fps_ema:.1f}  detector={'motion' if not args.strict_detector else 'strict'}  subpix={not args.no_refine_subpix}",
                "keys: q / Esc quit",
            ]
            for raw_target_id, target_data in sorted(frame_result["targets"].items(), key=lambda item: int(item[0])):
                pose = target_data.get("target_in_table")
                if pose is None:
                    hud_lines.append(f"target {raw_target_id}: detected={target_data['detected']}  world_pose=unavailable")
                    continue
                pos = pose["position_m"]
                hud_lines.append(
                    f"target {raw_target_id}: xyz=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m"
                )
            _draw_hud(vis, hud_lines)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
