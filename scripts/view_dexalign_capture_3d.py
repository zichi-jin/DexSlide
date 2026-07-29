#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexslide.retargeting.human_model import HUMAN_LANDMARK_NAMES  # noqa: E402
from dexslide.calibration.dexalign.collect_alignment_dataset import (  # noqa: E402
    MarkerPoseObservation,
    deproject_keypoints_with_realsense,
    make_realsense_marker_pose_estimator,
    next_realsense_frame,
    start_realsense_pipeline,
)
from dexslide.calibration.dexalign.io_utils import load_marker2hand_asset_mm  # noqa: E402
from dexslide.calibration.dexalign.live_preview import (  # noqa: E402
    _compose_camera_overlay,
    _draw_axes,
    _draw_hand,
    _estimate_bbox_xyxy,
    _set_axes_equal,
    drain_camera_preview_commands,
    enqueue_camera_preview_frame,
    start_camera_preview_process,
    stop_camera_preview_process,
)
from dexslide.calibration.landmark_detector import LandmarkDetector  # noqa: E402
from dexslide.communications import camera_communication, resolve_realsense_serial  # noqa: E402
from dexslide.paths import (  # noqa: E402
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE,
    DEFAULT_LEFT_MARKER_TO_WRIST_FILE,
    DEFAULT_LEFT_TAGS_TO_MARKER_FILE,
)
from dexslide.vision.aruco_pose_tracker import _detect_localize_aruco_tags, _parse_aruco_config  # noqa: E402
from dexslide.world_pose.direct_aruco_tracker import _build_direct_aruco_frame_result  # noqa: E402
from dexslide.world_pose.hand_cube_overlay import (  # noqa: E402
    invert_transform,
    make_transform,
    transform_points,
    transform_to_rvec_tvec,
)

THUMB_BASE_INDEX = 1


def _build_parser() -> argparse.ArgumentParser:
    camera = camera_communication("primary")
    camera_intrinsics = json.loads(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Realtime 3D preview for DexAlign observation quality.")
    parser.add_argument("--marker-body-config", default=str(DEFAULT_LEFT_TAGS_TO_MARKER_FILE))
    parser.add_argument("--marker2hand-file", default=str(DEFAULT_LEFT_MARKER_TO_WRIST_FILE))
    parser.add_argument("--camera-serial", default=resolve_realsense_serial("primary"))
    parser.add_argument("--width", type=int, default=int(camera_intrinsics["image_width"]))
    parser.add_argument("--height", type=int, default=int(camera_intrinsics["image_height"]))
    parser.add_argument("--fps", type=int, default=int(camera_intrinsics["fps"]))
    parser.add_argument("--depth-window-radius", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--body-outlier-threshold-mm", type=float, default=20.0)
    parser.add_argument("--body-reprojection-threshold-px", type=float, default=5.0)
    parser.add_argument("--strict-detector", action="store_true")
    parser.add_argument("--no-refine-subpix", action="store_true")
    parser.add_argument(
        "--enable-table-frame",
        action="store_true",
        help="Transform 3D keypoints and poses from camera frame into the realtime table ArUco frame.",
    )
    parser.add_argument(
        "--table-aruco-yaml",
        default=str(DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE),
        help="Reference/table ArUco YAML path used when --enable-table-frame is set.",
    )
    parser.add_argument("--table-marker-id", type=int, default=0, help="Fixed table/reference marker id.")
    parser.add_argument("--camera-window-name", default="DexAlign Camera Preview")
    return parser


def _format_marker2hand_pose_text(marker2hand_m: np.ndarray) -> str:
    transform = np.asarray(marker2hand_m, dtype=np.float64).reshape(4, 4)
    rvec_rad, t_m = transform_to_rvec_tvec(transform)
    rvec_deg = np.rad2deg(np.asarray(rvec_rad, dtype=np.float64).reshape(3))
    return (
        "marker2hand_6d = "
        f"t_m=[{t_m[0]:+.3f}, {t_m[1]:+.3f}, {t_m[2]:+.3f}] "
        f"rvec_deg=[{rvec_deg[0]:+.1f}, {rvec_deg[1]:+.1f}, {rvec_deg[2]:+.1f}]"
    )


def _format_point_mm_text(point_mm: np.ndarray) -> str:
    point = np.asarray(point_mm, dtype=np.float64).reshape(3)
    if not np.isfinite(point).all():
        return "[nan, nan, nan]"
    return f"[{point[0]:+7.1f}, {point[1]:+7.1f}, {point[2]:+7.1f}]"


def _format_point_m_text(point_m: np.ndarray) -> str:
    point = np.asarray(point_m, dtype=np.float64).reshape(3)
    if not np.isfinite(point).all():
        return "[nan, nan, nan]"
    return f"[{point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f}]"


def _build_keypoint_text_lines(
    keypoints_xyz_mm: np.ndarray,
    valid_mask: np.ndarray,
    *,
    frame_name: str,
) -> list[str]:
    keypoints = np.asarray(keypoints_xyz_mm, dtype=np.float64).reshape(21, 3)
    valid = np.asarray(valid_mask, dtype=bool).reshape(21)
    lines = [f"mediapipe_keypoints_xyz_{frame_name}_mm:"]
    for index, name in enumerate(HUMAN_LANDMARK_NAMES):
        state = "ok" if valid[index] else "na"
        lines.append(f"{index:02d} {name:<13} {state} {_format_point_mm_text(keypoints[index])}")
    return lines


def _emit_terminal_keypoint_stream(
    frame_idx: int,
    frame_mode: str,
    frame_name: str,
    marker_translation_mm: np.ndarray,
    marker2hand_pose_text: str,
    keypoints_xyz_mm: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    marker_translation = np.asarray(marker_translation_mm, dtype=np.float64).reshape(3)
    lines = [
        "DexAlign Capture 3D Terminal Stream",
        f"frame_idx = {frame_idx}",
        f"frame_mode = {frame_mode}",
        f"marker_t_{frame_name}_mm = {_format_point_mm_text(marker_translation)}",
        marker2hand_pose_text,
        *_build_keypoint_text_lines(keypoints_xyz_mm, valid_mask, frame_name=frame_name),
    ]
    block = "\n".join(lines)
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[H\x1b[J" + block + "\n")
    else:
        sys.stdout.write(block + "\n")
    sys.stdout.flush()


def _safe_unit(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return None
    return value / norm


def _estimate_runtime_hand_pose_from_keypoints(
    keypoints_camera_m: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray | None:
    keypoints = np.asarray(keypoints_camera_m, dtype=np.float64).reshape(21, 3)
    valid = np.asarray(valid_mask, dtype=bool).reshape(21)
    palm_indices = (0, 5, 9, 13, 17)
    if not np.all(valid[list(palm_indices)]):
        return None

    wrist = keypoints[0]
    index_mcp = keypoints[5]
    middle_mcp = keypoints[9]
    ring_mcp = keypoints[13]
    pinky_mcp = keypoints[17]
    if not np.isfinite(np.stack([wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0)).all():
        return None

    palm_center = np.mean(np.stack([index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0), axis=0)
    x_axis = _safe_unit(palm_center - wrist)
    if x_axis is None:
        return None

    lateral = _safe_unit(pinky_mcp - index_mcp)
    if lateral is None:
        return None

    z_axis = _safe_unit(np.cross(x_axis, lateral))
    if z_axis is None:
        return None

    y_axis = _safe_unit(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None

    z_axis = _safe_unit(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None

    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return make_transform(rotation, wrist)


def _load_aruco_cfg(yaml_path: str | Path) -> dict[str, Any]:
    with open(Path(yaml_path).expanduser().resolve(), "r", encoding="utf-8") as handle:
        return _parse_aruco_config(yaml.safe_load(handle))


def _estimate_camera_in_table_transform(
    *,
    frame_bgr: np.ndarray,
    intr_dict: dict[str, np.ndarray],
    table_cfg: dict[str, Any],
    table_marker_id: int,
    refine_subpix: bool,
    motion_tolerant: bool,
) -> np.ndarray | None:
    tag_dict = _detect_localize_aruco_tags(
        img_bgr=frame_bgr,
        aruco_dict=table_cfg["aruco_dict"],
        marker_size_map=table_cfg["marker_size_map"],
        fisheye_intr_dict=intr_dict,
        refine_subpix=bool(refine_subpix),
        motion_tolerant=bool(motion_tolerant),
        corner_refine_mode="apriltag" if refine_subpix else "none",
    )
    frame_result = _build_direct_aruco_frame_result(
        frame_idx=0,
        image_size=frame_bgr.shape[:2],
        table_marker_id=int(table_marker_id),
        target_marker_ids=[],
        tag_dict=tag_dict,
        time_wall=0.0,
    )
    camera_in_table = frame_result.get("camera_in_table")
    if camera_in_table is None:
        return None
    return np.asarray(camera_in_table["matrix"], dtype=np.float64).reshape(4, 4)


def _resolve_display_frame(
    *,
    enable_table_frame: bool,
    camera_in_table_m: np.ndarray | None,
) -> tuple[np.ndarray | None, str, str]:
    if not enable_table_frame:
        return None, "camera", "camera"
    if camera_in_table_m is None:
        return None, "camera", "camera_fallback"
    return np.asarray(camera_in_table_m, dtype=np.float64).reshape(4, 4), "table", "table"


def _transform_keypoints_to_frame(
    keypoints_camera_m: np.ndarray,
    transform_camera_frame: np.ndarray | None,
) -> np.ndarray:
    keypoints = np.asarray(keypoints_camera_m, dtype=np.float64).reshape(-1, 3).copy()
    if transform_camera_frame is None:
        return keypoints
    finite_mask = np.isfinite(keypoints).all(axis=1)
    if np.any(finite_mask):
        keypoints[finite_mask] = transform_points(transform_camera_frame, keypoints[finite_mask])
    return keypoints


def _transform_pose_to_frame(
    pose_camera_frame: np.ndarray | None,
    transform_camera_frame: np.ndarray | None,
) -> np.ndarray | None:
    if pose_camera_frame is None:
        return None
    pose = np.asarray(pose_camera_frame, dtype=np.float64).reshape(4, 4)
    if transform_camera_frame is None:
        return pose.copy()
    return np.asarray(transform_camera_frame, dtype=np.float64).reshape(4, 4) @ pose


def _capture_thumb_base_local_point(
    runtime_hand_pose_m: np.ndarray | None,
    keypoints_camera_m: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray | None:
    if runtime_hand_pose_m is None:
        return None
    keypoints = np.asarray(keypoints_camera_m, dtype=np.float64).reshape(21, 3)
    valid = np.asarray(valid_mask, dtype=bool).reshape(21)
    if not bool(valid[THUMB_BASE_INDEX]):
        return None
    thumb_base_camera = keypoints[THUMB_BASE_INDEX]
    if not np.isfinite(thumb_base_camera).all():
        return None
    return transform_points(
        invert_transform(np.asarray(runtime_hand_pose_m, dtype=np.float64).reshape(4, 4)),
        thumb_base_camera[None, :],
    )[0]


def _apply_thumb_base_freeze(
    keypoints_camera_m: np.ndarray,
    valid_mask: np.ndarray,
    runtime_hand_pose_m: np.ndarray | None,
    frozen_thumb_base_local_m: np.ndarray | None,
    *,
    last_frozen_thumb_base_camera_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    keypoints = np.asarray(keypoints_camera_m, dtype=np.float64).reshape(21, 3).copy()
    valid = np.asarray(valid_mask, dtype=bool).reshape(21).copy()
    if frozen_thumb_base_local_m is None:
        return keypoints, valid, None, "live"

    if runtime_hand_pose_m is None:
        if last_frozen_thumb_base_camera_m is None:
            return keypoints, valid, None, "frozen_wait_palm"
        keypoints[THUMB_BASE_INDEX] = np.asarray(last_frozen_thumb_base_camera_m, dtype=np.float64).reshape(3)
        valid[THUMB_BASE_INDEX] = np.isfinite(keypoints[THUMB_BASE_INDEX]).all()
        return keypoints, valid, keypoints[THUMB_BASE_INDEX].copy(), "frozen_hold_last"

    thumb_base_camera = transform_points(
        np.asarray(runtime_hand_pose_m, dtype=np.float64).reshape(4, 4),
        np.asarray(frozen_thumb_base_local_m, dtype=np.float64).reshape(1, 3),
    )[0]
    keypoints[THUMB_BASE_INDEX] = thumb_base_camera
    valid[THUMB_BASE_INDEX] = np.isfinite(thumb_base_camera).all()
    return keypoints, valid, thumb_base_camera.copy(), "frozen_tracking"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    preview = start_camera_preview_process(str(args.camera_window_name))

    detector = LandmarkDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )
    table_cfg = _load_aruco_cfg(args.table_aruco_yaml) if args.enable_table_frame else None
    _initial_marker, _result_marker, active_marker_mm = load_marker2hand_asset_mm(args.marker2hand_file)
    active_marker_m = np.asarray(active_marker_mm, dtype=np.float64).reshape(4, 4).copy()
    active_marker_m[:3, 3] *= 0.001

    pipeline = None
    last_pose_mm: np.ndarray | None = None
    frame_idx = 0
    last_frame_time = time.perf_counter()
    fps_ema = 0.0
    try:
        pipeline, align, intr_dict = start_realsense_pipeline(
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            camera_serial=str(args.camera_serial),
        )
        pose_estimator = make_realsense_marker_pose_estimator(
            marker_body_config_path=args.marker_body_config,
            intr_dict=intr_dict,
            outlier_threshold_mm=float(args.body_outlier_threshold_mm),
            reprojection_threshold_px=float(args.body_reprojection_threshold_px),
            refine_subpix=not args.no_refine_subpix,
            motion_tolerant=not args.strict_detector,
        )

        plt.ion()
        fig = plt.figure("DexAlign Capture 3D", figsize=(8.5, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        info_text = fig.text(0.02, 0.02, "", family="monospace", fontsize=10)
        thumb_base_state: dict[str, Any] = {
            "active": False,
            "local_m": None,
            "last_camera_m": None,
            "mode": "live",
        }

        def _set_thumb_base_freeze(active: bool, *, runtime_hand_pose_m: np.ndarray | None, keypoints_camera_m: np.ndarray, valid_mask: np.ndarray) -> None:
            active = bool(active)
            if not active:
                thumb_base_state["active"] = False
                thumb_base_state["local_m"] = None
                thumb_base_state["last_camera_m"] = None
                thumb_base_state["mode"] = "live"
                print("[dexalign-preview] thumb_base=LIVE")
                fig.canvas.draw_idle()
                return

            frozen_local_m = _capture_thumb_base_local_point(runtime_hand_pose_m, keypoints_camera_m, valid_mask)
            if frozen_local_m is None:
                thumb_base_state["mode"] = "freeze_failed"
                print("[dexalign-preview] thumb_base freeze failed: palm pose or thumb_base unavailable")
                fig.canvas.draw_idle()
                return
            thumb_base_state["active"] = True
            thumb_base_state["local_m"] = np.asarray(frozen_local_m, dtype=np.float64).reshape(3).copy()
            thumb_base_state["last_camera_m"] = np.asarray(
                keypoints_camera_m[THUMB_BASE_INDEX],
                dtype=np.float64,
            ).reshape(3).copy()
            thumb_base_state["mode"] = "frozen_tracking"
            print("[dexalign-preview] thumb_base=FROZEN_RELATIVE_TO_PALM")
            fig.canvas.draw_idle()

        def _on_key(event: Any) -> None:
            key = "" if event.key is None else str(event.key).lower()
            if key not in {" ", "space"}:
                return
            _set_thumb_base_freeze(
                not bool(thumb_base_state["active"]),
                runtime_hand_pose_m=thumb_base_state.get("runtime_hand_pose_m"),
                keypoints_camera_m=thumb_base_state.get("keypoints_camera_m", np.full((21, 3), np.nan, dtype=np.float64)),
                valid_mask=thumb_base_state.get("valid_mask", np.zeros(21, dtype=bool)),
            )

        fig.canvas.mpl_connect("key_press_event", _on_key)
        print("[dexalign-preview] 聚焦 3D 窗口后按 SPACE：冻结 / 解冻 thumb_base 相对 palm 的局部位置。")

        while plt.fignum_exists(fig.number) and not preview.stop_event.is_set():
            for command in drain_camera_preview_commands(preview):
                if command == "space":
                    _set_thumb_base_freeze(
                        not bool(thumb_base_state["active"]),
                        runtime_hand_pose_m=thumb_base_state.get("runtime_hand_pose_m"),
                        keypoints_camera_m=thumb_base_state.get(
                            "keypoints_camera_m",
                            np.full((21, 3), np.nan, dtype=np.float64),
                        ),
                        valid_mask=thumb_base_state.get(
                            "valid_mask",
                            np.zeros(21, dtype=bool),
                        ),
                    )
            frame = next_realsense_frame(pipeline, align)
            if frame is None:
                continue
            ax.cla()

            detection = detector.detect(frame.color_bgr)
            marker_pose: MarkerPoseObservation | None = pose_estimator(frame, last_pose_mm)
            keypoints_camera_mm = np.full((21, 3), np.nan, dtype=np.float64)
            valid_mask = np.zeros(21, dtype=bool)
            depth_mm = np.full(21, np.nan, dtype=np.float64)
            if detection is not None:
                keypoints_2d, _confidence = detection
                deprojection = deproject_keypoints_with_realsense(
                    frame,
                    np.asarray(keypoints_2d, dtype=np.float64),
                    window_radius=int(args.depth_window_radius),
                )
                keypoints_camera_mm = deprojection.keypoints_camera_mm
                valid_mask = deprojection.valid_mask
                depth_mm = deprojection.depth_mm

            camera_in_table_m = None
            if table_cfg is not None:
                camera_in_table_m = _estimate_camera_in_table_transform(
                    frame_bgr=frame.color_bgr,
                    intr_dict=intr_dict,
                    table_cfg=table_cfg,
                    table_marker_id=int(args.table_marker_id),
                    refine_subpix=not args.no_refine_subpix,
                    motion_tolerant=not args.strict_detector,
                )

            keypoints_camera_m = keypoints_camera_mm * 0.001
            runtime_hand_pose_m = _estimate_runtime_hand_pose_from_keypoints(keypoints_camera_m, valid_mask)
            thumb_base_state["runtime_hand_pose_m"] = None if runtime_hand_pose_m is None else np.asarray(runtime_hand_pose_m, dtype=np.float64).reshape(4, 4).copy()
            thumb_base_state["keypoints_camera_m"] = np.asarray(keypoints_camera_m, dtype=np.float64).reshape(21, 3).copy()
            thumb_base_state["valid_mask"] = np.asarray(valid_mask, dtype=bool).reshape(21).copy()
            keypoints_camera_effective_m, valid_mask_effective, frozen_thumb_base_camera_m, thumb_base_mode = _apply_thumb_base_freeze(
                keypoints_camera_m,
                valid_mask,
                runtime_hand_pose_m,
                thumb_base_state["local_m"] if bool(thumb_base_state["active"]) else None,
                last_frozen_thumb_base_camera_m=thumb_base_state.get("last_camera_m"),
            )
            thumb_base_state["mode"] = thumb_base_mode
            if frozen_thumb_base_camera_m is not None:
                thumb_base_state["last_camera_m"] = np.asarray(frozen_thumb_base_camera_m, dtype=np.float64).reshape(3).copy()
            marker_pose_m = None
            wrist_pose_m = None
            if marker_pose is not None:
                marker_pose_m = np.asarray(marker_pose.camera_T_marker_mm, dtype=np.float64).reshape(4, 4).copy()
                marker_pose_m[:3, 3] *= 0.001
                wrist_pose_m = runtime_hand_pose_m if runtime_hand_pose_m is not None else (marker_pose_m @ active_marker_m)

            transform_camera_display_m, display_frame_name, display_frame_mode = _resolve_display_frame(
                enable_table_frame=bool(args.enable_table_frame),
                camera_in_table_m=camera_in_table_m,
            )
            keypoints_display_m = _transform_keypoints_to_frame(keypoints_camera_effective_m, transform_camera_display_m)
            marker_pose_display_m = _transform_pose_to_frame(marker_pose_m, transform_camera_display_m)
            wrist_pose_display_m = _transform_pose_to_frame(wrist_pose_m, transform_camera_display_m)

            _draw_axes(
                ax,
                np.eye(4, dtype=np.float64),
                label=display_frame_name,
                length_m=0.05,
                colors=("#dc2626", "#16a34a", "#2563eb"),
            )
            if marker_pose_display_m is not None:
                _draw_axes(ax, marker_pose_display_m, label="marker", length_m=0.04, colors=("#ef4444", "#22c55e", "#3b82f6"))
            if wrist_pose_display_m is not None:
                _draw_axes(ax, wrist_pose_display_m, label="wrist", length_m=0.035, colors=("#f97316", "#84cc16", "#0ea5e9"))
            if marker_pose is not None:
                last_pose_mm = np.asarray(marker_pose.camera_T_marker_mm, dtype=np.float64).reshape(4, 4).copy()
            _draw_hand(ax, keypoints_display_m, valid_mask_effective)

            if marker_pose_display_m is not None:
                marker_translation_m = np.asarray(marker_pose_display_m[:3, 3], dtype=np.float64).reshape(3)
            else:
                marker_translation_m = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            runtime_marker2hand_m = None
            if marker_pose is not None and runtime_hand_pose_m is not None:
                runtime_marker2hand_m = invert_transform(marker_pose_m) @ runtime_hand_pose_m
            marker2hand_pose_text = (
                _format_marker2hand_pose_text(runtime_marker2hand_m)
                if runtime_marker2hand_m is not None
                else "marker2hand_6d = NA"
            )
            wrist_point_m = keypoints_display_m[0] if valid_mask_effective[0] else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            thumb_base_point_m = (
                keypoints_display_m[THUMB_BASE_INDEX]
                if valid_mask_effective[THUMB_BASE_INDEX]
                else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            )

            reference_points = keypoints_display_m[np.isfinite(keypoints_display_m).all(axis=1)]
            if marker_pose_display_m is not None:
                reference_points = np.vstack([reference_points, marker_pose_display_m[:3, 3][None, :]])
            if transform_camera_display_m is not None:
                reference_points = np.vstack([reference_points, np.zeros((1, 3), dtype=np.float64)])
            if reference_points.size == 0:
                reference_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
            _set_axes_equal(ax, reference_points)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(f"DexAlign Capture 3D Preview ({display_frame_mode})")
            now = time.perf_counter()
            dt = max(now - last_frame_time, 1e-6)
            instant_fps = 1.0 / dt
            fps_ema = instant_fps if frame_idx == 0 else (0.85 * fps_ema + 0.15 * instant_fps)
            last_frame_time = now
            frame_idx += 1
            info_text.set_text(
                "\n".join(
                    [
                        f"frame_mode = {display_frame_mode}",
                        f"marker_t_{display_frame_name}_m = {_format_point_m_text(marker_translation_m)}",
                        marker2hand_pose_text,
                        f"wrist_xyz_{display_frame_name}_m = {_format_point_m_text(wrist_point_m)}",
                        f"thumb_base_mode = {thumb_base_mode}",
                        f"thumb_base_xyz_{display_frame_name}_m = {_format_point_m_text(thumb_base_point_m)}",
                        f"fps = {fps_ema:.1f}",
                    ]
                )
            )
            _emit_terminal_keypoint_stream(
                frame_idx=frame_idx,
                frame_mode=display_frame_mode,
                frame_name=display_frame_name,
                marker_translation_mm=marker_translation_m * 1000.0,
                marker2hand_pose_text=marker2hand_pose_text,
                keypoints_xyz_mm=keypoints_display_m * 1000.0,
                valid_mask=valid_mask_effective,
            )
            camera_overlay = _compose_camera_overlay(
                frame.color_bgr,
                detector=detector,
                detection=detection,
                marker_pose=marker_pose,
                valid_mask=valid_mask,
                depth_mm=depth_mm,
                fps_ema=fps_ema,
                frame_idx=frame_idx,
            )
            enqueue_camera_preview_frame(preview, camera_overlay)
            plt.pause(0.001)
    finally:
        stop_camera_preview_process(preview)
        detector.close()
        if pipeline is not None:
            pipeline.stop()
        plt.ioff()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
