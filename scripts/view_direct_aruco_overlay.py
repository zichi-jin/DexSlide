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

from dexslide.calibration.calibrate_marker_wrist_offset import (
    _default_output_config_path as _default_wrist_align_output_config_path,
    _default_output_report_path as _default_wrist_align_output_report_path,
    _camera_points_to_body_points,
    _deproject_keypoint_points,
    _estimate_bbox_xyxy,
    _estimate_body_pose_in_camera_from_tag_dict,
    _rs_intrinsics_to_opencv_dict,
)
from dexslide.calibration.landmark_detector import LandmarkDetector
from dexslide.calibration.palm_triangle_alignment import (
    PALM_TRIANGLE_LANDMARK_INDICES,
    PALM_TRIANGLE_LANDMARK_NAMES,
    apply_body_to_wrist_transform,
    average_body_to_wrist_transforms,
    capture_body_to_wrist_transform_sample,
    estimate_body_to_wrist_transform_from_triangles,
    save_body_to_wrist_alignment_outputs,
    select_palm_triangle_points,
)
from dexslide.communications import (
    camera_communication,
    hand_joint_communication,
    resolve_camera_source,
    resolve_joint_port,
    resolve_realsense_serial,
)
from dexslide.live import live_listener
from dexslide.paths import (
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE,
    DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE,
    DEFAULT_GLOVE_CALIBRATION_FILE,
    DEFAULT_LEFT_MARKER_TO_WRIST_FILE,
    DEFAULT_SKELETON_FILE,
    DEXALIGN_CALIBRATION_DIR,
    DIRECT_ARUCO_CALIBRATION_DIR,
)
from dexslide.retargeting.human_model import DexSlideHumanModel
from dexslide.visualization.aruco_overlay import (
    draw_axes as _draw_axes_overlay,
    draw_hud as _draw_hud,
    draw_marker_outline as _draw_marker_outline,
    draw_projected_hand,
    marker_id_to_color as _marker_id_to_color,
)
from dexslide.vision.aruco_pose_tracker import (
    _convert_fisheye_intrinsics_resolution,
    _parse_aruco_config,
    _parse_capture_source,
    _parse_fisheye_intrinsics,
)
from dexslide.kinematics.transforms import (
    make_transform,
    rotmat_to_quaternion_xyzw,
    rvec_tvec_to_transform,
    transform_points,
    transform_to_rvec_tvec,
)
from dexslide.world_pose.direct_aruco_tracker import (
    _build_direct_aruco_frame_result,
    _detect_relevant_aruco_tags,
    _normalize_target_marker_ids,
)
from dexslide.world_pose.hand_cube_overlay import (
    CubePoseEstimate,
    HandCubeOverlayConfig,
    compose_overlay_joint_angles,
    marker_to_wrist_asset_transforms,
    resolve_marker_body_tag_pose_branches,
    try_load_hand_cube_overlay_config,
)
from dexslide.world_pose.marker_body_pose_tracker import MarkerBodyPoseTracker

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - runtime dependency
    rs = None

def _latest_dexalign_result_path(filename: str) -> Path | None:
    base_dir = Path(DEXALIGN_CALIBRATION_DIR).expanduser().resolve()
    if not base_dir.exists():
        return None

    candidates: list[Path] = []
    for session_dir in base_dir.iterdir():
        if not session_dir.is_dir():
            continue
        candidate = session_dir / str(filename)
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _default_overlay_skeleton_file() -> Path:
    optimized = _latest_dexalign_result_path("optimized_skeleton.json")
    return optimized if optimized is not None else DEFAULT_SKELETON_FILE


def _default_overlay_marker2hand_file() -> Path:
    optimized = _latest_dexalign_result_path("optimized_marker2hand.json")
    return optimized if optimized is not None else DEFAULT_LEFT_MARKER_TO_WRIST_FILE


def _default_overlay_joint_calibration_file() -> str:
    optimized = _latest_dexalign_result_path("optimized_joint_calibration.json")
    return "" if optimized is None else str(optimized)


def _load_overlay_joint_calibration(path: str | Path | None) -> tuple[np.ndarray, np.ndarray, Path | None]:
    if path is None:
        return np.ones(20, dtype=np.float64), np.zeros(20, dtype=np.float64), None
    text = str(path).strip()
    if not text:
        return np.ones(20, dtype=np.float64), np.zeros(20, dtype=np.float64), None

    config_path = Path(text).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid joint calibration JSON `{config_path}`: expected object root.")

    joint_scale = np.asarray(payload.get("joint_scale"), dtype=np.float64).reshape(-1)
    joint_bias = np.asarray(payload.get("joint_bias_rad"), dtype=np.float64).reshape(-1)
    if joint_scale.shape[0] != 20 or joint_bias.shape[0] != 20:
        raise ValueError(
            f"Invalid joint calibration JSON `{config_path}`: expected 20 scales and 20 biases, got "
            f"{joint_scale.shape[0]} and {joint_bias.shape[0]}."
        )
    return joint_scale, joint_bias, config_path


def _apply_overlay_joint_calibration(
    joint_angles: np.ndarray,
    *,
    joint_scale: np.ndarray,
    joint_bias: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(joint_angles, dtype=np.float64).reshape(20)
    scale = np.asarray(joint_scale, dtype=np.float64).reshape(20)
    bias = np.asarray(joint_bias, dtype=np.float64).reshape(20)
    return (scale * raw) + bias


def _marker_id_compact_text(marker_ids: list[int]) -> str:
    return "-" if not marker_ids else "".join(str(int(marker_id)) for marker_id in marker_ids)


def _default_hand_overlay_config_path(hand: str) -> Path:
    requested = str(hand).strip().lower()
    if requested in {"left", "right"}:
        return DIRECT_ARUCO_CALIBRATION_DIR / f"{requested}_tags2marker.json"

    fallback_candidates = [
        DIRECT_ARUCO_CALIBRATION_DIR / "left_tags2marker.json",
        DIRECT_ARUCO_CALIBRATION_DIR / "right_tags2marker.json",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate
    return fallback_candidates[0]


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


def _iter_capture_candidates(source: str | int) -> list[int | str]:
    requested = _parse_capture_source(source)
    return [requested]


def _capture_backend_attempts(source: int | str) -> list[tuple[str, int | None]]:
    if isinstance(source, str) and source.startswith("/dev/"):
        return [("v4l2", cv2.CAP_V4L2), ("default", None)]
    if isinstance(source, int):
        return [("default", None), ("v4l2", cv2.CAP_V4L2)]
    return [("default", None)]


def _configure_capture_device(
    cap: cv2.VideoCapture,
    *,
    width: int | None,
    height: int | None,
    fps: float | None,
    buffer_size: int,
) -> None:
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))


def _open_capture_with_fallback(
    *,
    source: str | int,
    width: int | None,
    height: int | None,
    fps: float | None,
    buffer_size: int,
    purpose: str,
) -> tuple[cv2.VideoCapture, int | str]:
    requested = _parse_capture_source(source)
    tried: list[str] = []
    for candidate in _iter_capture_candidates(source):
        for backend_label, backend in _capture_backend_attempts(candidate):
            tried.append(f"{candidate}[{backend_label}]")
            if backend is None:
                cap = cv2.VideoCapture(candidate)
            else:
                cap = cv2.VideoCapture(candidate, backend)
            _configure_capture_device(
                cap,
                width=width,
                height=height,
                fps=fps,
                buffer_size=buffer_size,
            )
            if not cap.isOpened():
                cap.release()
                continue

            probe_ok = False
            frame_shape: tuple[int, ...] | None = None
            for _ in range(5):
                ok, frame_bgr = cap.read()
                if ok and frame_bgr is not None:
                    probe_ok = True
                    frame_shape = tuple(int(value) for value in frame_bgr.shape)
                    break
                time.sleep(0.03)
            if not probe_ok:
                cap.release()
                continue

            shape_desc = "x".join(str(value) for value in frame_shape) if frame_shape is not None else "unknown"
            print(
                f"[camera] requested={requested}  selected={candidate}  "
                f"backend={backend_label}  frame_shape={shape_desc}"
            )
            return cap, candidate

    tried_desc = ", ".join(tried) if tried else "<none>"
    raise RuntimeError(
        f"Failed to open capture source for {purpose}: requested={requested}, tried=[{tried_desc}]"
    )


def _ensure_hand_cube_config(
    *,
    config_path: Path,
    hand: str,
) -> HandCubeOverlayConfig:
    cfg = try_load_hand_cube_overlay_config(config_path)
    if cfg is None:
        raise SystemExit(
            "未找到可用的 marker body 配置文件："
            f"{config_path}\n"
            "请先准备 tags->marker JSON，并写入 18 个 marker 面的几何关系；"
            "当前版本已移除运行时逐面交互配置。"
        )
    requested = str(hand).strip().lower()
    if requested in {"left", "right"}:
        cfg.hand = requested
    elif str(cfg.hand).strip().lower() not in {"left", "right"}:
        cfg.hand = "left"
    return cfg


def _resolve_target_marker_ids_for_overlay(
    parsed_target_ids: list[int] | None,
    hand_overlay_enabled: bool,
    hand_overlay_config: HandCubeOverlayConfig | None,
    table_marker_id: int,
) -> list[int] | None:
    if hand_overlay_enabled:
        configured = [] if hand_overlay_config is None else hand_overlay_config.marker_ids()
        selected = configured if parsed_target_ids is None else parsed_target_ids
        return _normalize_target_marker_ids(selected, table_marker_id)
    return _normalize_target_marker_ids(parsed_target_ids, table_marker_id)


def _emit_runtime_status(
    *,
    frame_idx: int,
    frame_result: dict[str, object],
    cube_pose: CubePoseEstimate | None,
    raw_joint_stamp: float,
    glove_port: str,
) -> None:
    detected = _marker_id_compact_text(list(frame_result.get("detected_ids", [])))
    body_ids = _marker_id_compact_text([] if cube_pose is None else cube_pose.source_marker_ids)
    spread_mm = -1.0 if cube_pose is None else float(cube_pose.max_position_deviation_m) * 1000.0
    reproj_px = -1.0 if cube_pose is None else float(cube_pose.mean_reprojection_error_px)
    solver_mode = "-" if cube_pose is None else str(cube_pose.solver_mode)
    hand_age_ms = (time.time() - raw_joint_stamp) * 1000.0 if raw_joint_stamp > 0.0 else -1.0
    line = (
        f"frame={frame_idx} table={'Y' if frame_result.get('table_detected') else 'N'} "
        f"detected={detected} body={body_ids} spread={spread_mm:.1f}mm reproj={reproj_px:.2f}px solver={solver_mode} "
        f"hand_port={glove_port} hand_age={hand_age_ms:.1f}ms "
        "keys=q"
    )
    sys.stdout.write("\r" + line.ljust(180))
    sys.stdout.flush()


def _emit_marker_body_diagnostic(
    *,
    report: object | None,
    position_threshold_mm: float,
    rotation_threshold_deg: float,
) -> None:
    if report is None or not getattr(report, "items", None):
        return

    suspicious = [
        item
        for item in report.items
        if float(item.peer_position_error_m) * 1000.0 >= float(position_threshold_mm)
        or float(item.peer_rotation_error_deg) >= float(rotation_threshold_deg)
    ]
    visible_text = ",".join(str(marker_id) for marker_id in report.marker_ids)
    if suspicious:
        details: list[str] = []
        for item in suspicious[:3]:
            reproj_text = (
                "-"
                if item.reprojection_mean_error_px is None
                else f"{float(item.reprojection_mean_error_px):.2f}px"
            )
            fused_pos_mm = -1.0 if item.fused_position_error_m is None else float(item.fused_position_error_m) * 1000.0
            fused_rot_deg = -1.0 if item.fused_rotation_error_deg is None else float(item.fused_rotation_error_deg)
            details.append(
                f"id{item.marker_id}:peer={float(item.peer_position_error_m)*1000.0:.1f}mm/{float(item.peer_rotation_error_deg):.1f}deg "
                f"fused={fused_pos_mm:.1f}mm/{fused_rot_deg:.1f}deg reproj={reproj_text}"
            )
        print(
            "\n[diag] visible=["
            f"{visible_text}] suspicious=[{','.join(str(item.marker_id) for item in suspicious)}] "
            + " | ".join(details)
        )
        return

    worst = report.items[0]
    print(
        "\n[diag] visible=["
        f"{visible_text}] consistent "
        f"worst=id{worst.marker_id} peer={float(worst.peer_position_error_m)*1000.0:.1f}mm/{float(worst.peer_rotation_error_deg):.1f}deg"
    )


def _compute_overlay_joint_angles(
    raw_joint_angles: np.ndarray,
    cfg: HandCubeOverlayConfig,
) -> np.ndarray:
    return compose_overlay_joint_angles(
        raw_joint_angles=np.asarray(raw_joint_angles, dtype=np.float64),
        joint_zero_rad=np.asarray(cfg.joint_zero_rad, dtype=np.float64),
        joint_base_render_rad=np.asarray(cfg.joint_base_render_rad, dtype=np.float64),
    )


def _optional_path_arg(raw: str | None) -> Path | None:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _resolve_camera_body_transform(
    *,
    transform_camera_table: np.ndarray | None,
    cube_pose: CubePoseEstimate | None,
    camera_body_pose: object | None,
) -> tuple[np.ndarray | None, str]:
    if cube_pose is not None and transform_camera_table is not None:
        return (
            np.asarray(transform_camera_table, dtype=np.float64).reshape(4, 4)
            @ np.asarray(cube_pose.transform_table_cube, dtype=np.float64).reshape(4, 4),
            "table_body",
        )
    if camera_body_pose is not None:
        return (
            np.asarray(camera_body_pose.transform_camera_body, dtype=np.float64).reshape(4, 4).copy(),
            "camera_body",
        )
    return None, "none"


def _apply_runtime_body_to_wrist_transform(
    cfg: HandCubeOverlayConfig,
    samples_body_to_wrist: list[np.ndarray],
) -> np.ndarray | None:
    mean_transform = average_body_to_wrist_transforms(samples_body_to_wrist)
    if mean_transform is None:
        return None
    return apply_body_to_wrist_transform(cfg, mean_transform)


def _use_realsense_backend(*, camera_backend: str, wrist_align_enabled: bool, use_realsense_rgb: bool) -> bool:
    return bool(str(camera_backend).strip().lower() == "realsense" or wrist_align_enabled or use_realsense_rgb)


def main() -> None:
    joint_communication = hand_joint_communication("left")
    camera = camera_communication("primary")
    camera_intrinsics = json.loads(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Show direct ArUco marker-body overlay on the live camera feed."
    )
    parser.add_argument(
        "--source",
        default=resolve_camera_source("primary"),
        help="OpenCV capture source. Ignored when the configured camera backend is RealSense.",
    )
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
        help="Target ArUco YAML path. Ignored when --enable-hand-overlay loads marker sizes from the tags->marker JSON.",
    )
    parser.add_argument("--table-marker-id", type=int, default=0, help="Fixed table/reference marker id")
    parser.add_argument(
        "--target-marker-ids",
        default="",
        help=(
            "Comma-separated target marker ids. Empty means track all detected non-table ids, "
            "or all configured marker-body ids when --enable-hand-overlay is active."
        ),
    )
    parser.add_argument("--width", type=int, default=int(camera_intrinsics["image_width"]), help="Capture width override")
    parser.add_argument("--height", type=int, default=int(camera_intrinsics["image_height"]), help="Capture height override")
    parser.add_argument("--fps", type=float, default=float(camera_intrinsics["fps"]), help="Capture fps override")
    parser.add_argument("--camera-serial", default=resolve_realsense_serial("primary"), help="RealSense serial number")
    parser.add_argument("--buffer-size", type=int, default=2, help="VideoCapture buffer size")
    parser.add_argument("--num-workers", type=int, default=2, help="OpenCV thread count")
    parser.add_argument("--table-axis-length", type=float, default=0.08, help="Table axis glyph length in meters")
    parser.add_argument("--target-axis-length", type=float, default=0.04, help="Target axis glyph length in meters")
    parser.add_argument(
        "--body-axis-length",
        "--cube-axis-length",
        dest="cube_axis_length",
        type=float,
        default=0.05,
        help="Marker-body axis glyph length in meters.",
    )
    parser.add_argument("--hand-axis-length", type=float, default=0.04, help="Hand wrist axis glyph length in meters")
    parser.add_argument(
        "--no-refine-subpix",
        action="store_true",
        help="Disable ArUco corner subpixel refinement",
    )
    parser.add_argument(
        "--corner-refine-mode",
        choices=["apriltag", "subpix", "contour", "none"],
        default="",
        help="Corner refinement mode. Empty means apriltag by default, or none when --no-refine-subpix is set.",
    )
    parser.add_argument(
        "--strict-detector",
        action="store_true",
        help="Disable the motion-tolerant detector profile and use the stricter default detector settings.",
    )
    parser.add_argument(
        "--enable-hand-overlay",
        action="store_true",
        help="Overlay the live DexSlide skeleton by binding it to the tracked multi-face ArUco marker body.",
    )
    parser.add_argument("--glove-port", default=resolve_joint_port("left"), help="DexSlide serial port")
    parser.add_argument("--glove-baud", type=int, default=int(joint_communication["baud"]), help="DexSlide serial baud rate")
    parser.add_argument(
        "--glove-mode",
        choices=["raw", "angles"],
        default=str(joint_communication["mode"]),
        help="DexSlide stream mode",
    )
    parser.add_argument("--glove-calib-file", default=str(DEFAULT_GLOVE_CALIBRATION_FILE))
    parser.add_argument("--skeleton-file", default=str(_default_overlay_skeleton_file()))
    parser.add_argument(
        "--joint-calibration-file",
        default=_default_overlay_joint_calibration_file(),
        help="DexAlign joint calibration JSON. Defaults to the latest optimized result when available.",
    )
    parser.add_argument(
        "--hand",
        choices=["auto", "left", "right"],
        default="left",
        help="Hand side. 'auto' reuses the hand recorded in the tags->marker JSON, or falls back to the first available config.",
    )
    parser.add_argument(
        "--marker2hand-file",
        default=str(_default_overlay_marker2hand_file()),
        help="Marker->wrist JSON file. Defaults to the latest DexAlign optimized result when available.",
    )
    parser.add_argument(
        "--hand-overlay-config",
        default="",
        help="Tags->marker JSON file. The paired marker->wrist JSON is resolved automatically.",
    )
    parser.add_argument(
        "--body-outlier-threshold-mm",
        "--cube-outlier-threshold-mm",
        dest="cube_outlier_threshold_mm",
        type=float,
        default=20.0,
        help="Reject marker-body seed poses whose centroid disagrees with the median by more than this threshold.",
    )
    parser.add_argument(
        "--body-smoothing",
        "--cube-smoothing",
        dest="cube_smoothing",
        type=float,
        default=0.9,
        help="Exponential smoothing weight for the newly estimated marker-body pose. 1.0 disables smoothing.",
    )
    parser.add_argument(
        "--body-reprojection-threshold-px",
        "--cube-reprojection-threshold-px",
        dest="cube_reprojection_threshold_px",
        type=float,
        default=1,
        help="Drop the worst marker and re-solve when its mean reprojection error exceeds this threshold in pixels.",
    )
    parser.add_argument(
        "--body-pose-solver",
        "--cube-pose-solver",
        dest="body_pose_solver",
        choices=["joint_pnp", "marker_average"],
        default="joint_pnp",
        help="Marker-body pose solver. `joint_pnp` jointly optimizes all visible marker corners; `marker_average` averages per-marker body poses.",
    )
    parser.add_argument(
        "--show-target-axes",
        action="store_true",
        help="Draw each detected target marker's local xyz axes. This is mainly for offline tags geometry editing.",
    )
    parser.add_argument(
        "--diagnose-marker-body",
        action="store_true",
        help="Print per-marker marker-body consistency diagnostics to the terminal.",
    )
    parser.add_argument(
        "--diagnose-position-threshold-mm",
        type=float,
        default=12.0,
        help="Flag a marker when its single-marker body pose disagrees with peers by at least this position threshold.",
    )
    parser.add_argument(
        "--diagnose-rotation-threshold-deg",
        type=float,
        default=20.0,
        help="Flag a marker when its single-marker body pose disagrees with peers by at least this rotation threshold.",
    )
    parser.add_argument(
        "--enable-mediapipe-wrist-align",
        action="store_true",
        help="Use RealSense RGB-D + MediaPipe palm triangle as a live calibration step for the full body->wrist pose.",
    )
    parser.add_argument(
        "--use-realsense-rgb",
        action="store_true",
        help="Force the RealSense RGB stream as the live color backend without enabling MediaPipe wrist alignment.",
    )
    parser.add_argument(
        "--wrist-align-output-config",
        default="",
        help="Optional marker->wrist JSON output path for the calibrated body->wrist transform.",
    )
    parser.add_argument(
        "--wrist-align-output-report",
        default="",
        help="Optional JSON report path for captured wrist-alignment samples.",
    )
    parser.add_argument(
        "--wrist-align-depth-window-radius",
        type=int,
        default=2,
        help="Median depth sampling radius around each MediaPipe palm-triangle keypoint pixel.",
    )
    parser.add_argument(
        "--wrist-align-min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence for wrist alignment.",
    )
    parser.add_argument(
        "--wrist-align-min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence for wrist alignment.",
    )
    args = parser.parse_args()

    corner_refine_mode = (
        str(args.corner_refine_mode).strip().lower()
        if str(args.corner_refine_mode).strip()
        else ("none" if args.no_refine_subpix else "apriltag")
    )

    cv2.setNumThreads(max(1, int(args.num_workers)))

    hand_overlay_enabled = bool(args.enable_hand_overlay)
    wrist_align_enabled = bool(args.enable_mediapipe_wrist_align)
    realsense_backend_enabled = _use_realsense_backend(
        camera_backend=str(camera["backend"]),
        wrist_align_enabled=wrist_align_enabled,
        use_realsense_rgb=bool(args.use_realsense_rgb),
    )
    marker_body_tracking_enabled = bool(args.enable_hand_overlay or args.diagnose_marker_body)
    hand_overlay_cfg_path = (
        Path(args.hand_overlay_config).expanduser().resolve()
        if str(args.hand_overlay_config).strip()
        else _default_hand_overlay_config_path(args.hand)
    )
    wrist_align_output_config_path = _optional_path_arg(args.wrist_align_output_config)
    wrist_align_output_report_path = _optional_path_arg(args.wrist_align_output_report)

    if wrist_align_enabled and not hand_overlay_enabled:
        raise SystemExit("--enable-mediapipe-wrist-align 需要同时启用 --enable-hand-overlay。")
    if realsense_backend_enabled and rs is None:
        raise SystemExit("启用 RealSense 后端需要 pyrealsense2。")

    parsed_target_ids = _parse_target_marker_ids(args.target_marker_ids)

    hand_overlay_cfg: HandCubeOverlayConfig | None = None
    resolved_hand = str(args.hand).strip().lower()
    if marker_body_tracking_enabled:
        if hand_overlay_enabled and not args.glove_port:
            raise SystemExit("assets/dexslide_communications.json 中未配置手套串口。")
        hand_overlay_cfg = _ensure_hand_cube_config(
            config_path=hand_overlay_cfg_path,
            hand=args.hand,
        )
        marker2hand_path = Path(args.marker2hand_file).expanduser().resolve()
        _initial_marker2hand, _result_marker2hand, active_marker2hand = marker_to_wrist_asset_transforms(marker2hand_path)
        if active_marker2hand is None:
            raise SystemExit(f"未找到可用的 marker2hand 结果：{marker2hand_path}")
        hand_overlay_cfg.set_body_to_wrist_transform(active_marker2hand)
        resolved_hand = str(hand_overlay_cfg.hand).strip().lower()
        joint_zero = np.asarray(hand_overlay_cfg.joint_zero_rad, dtype=np.float64)
        joint_base = np.asarray(hand_overlay_cfg.joint_base_render_rad, dtype=np.float64)
        print(f"[hand-overlay] config={hand_overlay_cfg_path}")
        print(f"[hand-overlay] marker2hand={marker2hand_path}")
        print(f"[hand-overlay] skeleton={Path(args.skeleton_file).expanduser().resolve()}")
        print(f"[hand-overlay] hand={resolved_hand}")
        print(
            "[hand-overlay] marker_ids="
            f"{hand_overlay_cfg.marker_ids()} aruco_bound_mm={hand_overlay_cfg.aruco_bound_size_m * 1000.0:.1f} "
            f"marker_square_mm={hand_overlay_cfg.marker_square_size_m * 1000.0:.1f}"
        )
        print(f"[hand-overlay] body_pose_solver={args.body_pose_solver}")
        print(
            "[hand-overlay] marker_to_wrist_translation_m="
            f"{[round(float(x), 6) for x in hand_overlay_cfg.cube_to_wrist_translation_m]}"
        )
        print(
            "[hand-overlay] marker_to_wrist_quaternion_xyzw="
            f"{[round(float(x), 6) for x in hand_overlay_cfg.cube_to_wrist_quaternion_xyzw]}"
        )
        print(
            "[hand-overlay] joint_zero_abs_max_deg="
            f"{float(np.max(np.abs(np.rad2deg(joint_zero)))):.2f}  "
            "joint_base_abs_max_deg="
            f"{float(np.max(np.abs(np.rad2deg(joint_base)))):.2f}"
        )

    if wrist_align_enabled and hand_overlay_cfg is not None:
        wrist_align_output_config_path = (
            wrist_align_output_config_path
            if wrist_align_output_config_path is not None
            else _default_wrist_align_output_config_path(hand_overlay_cfg_path)
        )
        wrist_align_output_report_path = (
            wrist_align_output_report_path
            if wrist_align_output_report_path is not None
            else _default_wrist_align_output_report_path(wrist_align_output_config_path)
        )
        print(f"[wrist-align] output_config={wrist_align_output_config_path}")
        print(f"[wrist-align] output_report={wrist_align_output_report_path}")
        print(
            "[wrist-align] triangle_landmarks="
            f"{list(zip(PALM_TRIANGLE_LANDMARK_NAMES, PALM_TRIANGLE_LANDMARK_INDICES))}"
        )

    target_marker_ids = _resolve_target_marker_ids_for_overlay(
        parsed_target_ids=parsed_target_ids,
        hand_overlay_enabled=marker_body_tracking_enabled,
        hand_overlay_config=hand_overlay_cfg,
        table_marker_id=args.table_marker_id,
    )

    with open(args.table_aruco_yaml, "r", encoding="utf-8") as handle:
        table_cfg = _parse_aruco_config(yaml.safe_load(handle))
    if marker_body_tracking_enabled and hand_overlay_cfg is not None:
        same_cfg = False
        target_cfg = hand_overlay_cfg.build_target_aruco_config()
    else:
        same_cfg = Path(args.table_aruco_yaml).resolve() == Path(args.target_aruco_yaml).resolve()
        if same_cfg:
            target_cfg = table_cfg
        else:
            with open(args.target_aruco_yaml, "r", encoding="utf-8") as handle:
                target_cfg = _parse_aruco_config(yaml.safe_load(handle))

    raw_intr = None
    if not realsense_backend_enabled:
        with open(args.camera_intrinsics, "r", encoding="utf-8") as handle:
            raw_intr = _parse_fisheye_intrinsics(json.load(handle))

    glove_reader = None
    human_model = None
    overlay_joint_scale = np.ones(20, dtype=np.float64)
    overlay_joint_bias = np.zeros(20, dtype=np.float64)
    source_triangle_wrist_m = None
    marker_body_tracker = None
    wrist_detector = None
    if hand_overlay_enabled:
        glove_reader = live_listener(
            port=args.glove_port,
            baud=args.glove_baud,
            mode=args.glove_mode,
            calib_file=args.glove_calib_file,
        )
        human_model = DexSlideHumanModel(args.skeleton_file, hand=resolved_hand)
        overlay_joint_scale, overlay_joint_bias, loaded_joint_calibration_path = _load_overlay_joint_calibration(
            args.joint_calibration_file
        )
        if loaded_joint_calibration_path is not None:
            print(f"[overlay] joint calibration: {loaded_joint_calibration_path}")
        source_triangle_wrist_m = select_palm_triangle_points(
            human_model.landmarks_from_angles(np.zeros(20, dtype=np.float64))
        )
    if wrist_align_enabled:
        wrist_detector = LandmarkDetector(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=float(args.wrist_align_min_detection_confidence),
            min_tracking_confidence=float(args.wrist_align_min_tracking_confidence),
        )
    if hand_overlay_cfg is not None:
        marker_body_tracker = MarkerBodyPoseTracker(
            hand_overlay_cfg,
            pose_solver=str(args.body_pose_solver),
            smoothing_alpha=float(args.cube_smoothing),
            outlier_threshold_m=float(args.cube_outlier_threshold_mm) * 0.001,
            reprojection_error_threshold_px=float(args.cube_reprojection_threshold_px),
            enable_diagnostics=bool(args.diagnose_marker_body),
        )

    cap = None
    rs_pipeline = None
    rs_align = None
    rs_pipeline_started = False
    intr: dict[str, np.ndarray] | None = None
    intr_resolution: tuple[int, int] | None = None
    if realsense_backend_enabled:
        rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(str(args.camera_serial))
        rs_fps = int(round(float(args.fps)))
        rs_config.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, rs_fps)
        if wrist_align_enabled:
            rs_config.enable_stream(rs.stream.depth, int(args.width), int(args.height), rs.format.z16, rs_fps)
            rs_align = rs.align(rs.stream.color)
        rs_profile = rs_pipeline.start(rs_config)
        rs_pipeline_started = True
        rs_color_profile = rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = _rs_intrinsics_to_opencv_dict(rs_color_profile.get_intrinsics())
        intr_resolution = (int(intr["DIM"][0]), int(intr["DIM"][1]))
        if wrist_align_enabled:
            print("[wrist-align] using RealSense RGB-D backend; ignoring --source and --camera-intrinsics")
        else:
            print("[camera] using RealSense RGB backend; MediaPipe wrist align disabled")
    else:
        cap, _selected_source = _open_capture_with_fallback(
            source=args.source,
            width=args.width,
            height=args.height,
            fps=args.fps,
            buffer_size=args.buffer_size,
            purpose="main loop",
        )
    frame_idx = 0
    last_time = time.monotonic()
    fps_ema = 0.0
    last_status_emit = 0.0
    last_diag_emit = 0.0
    last_cube_pose: CubePoseEstimate | None = None
    last_camera_body_transform: np.ndarray | None = None
    wrist_align_samples_body_to_wrist: list[np.ndarray] = []
    base_body_to_wrist_transform = None if hand_overlay_cfg is None else hand_overlay_cfg.cube_to_wrist_transform().copy()

    window_name = "Direct ArUco Overlay"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            depth_frame = None
            if realsense_backend_enabled:
                assert rs_pipeline is not None
                frames = rs_pipeline.wait_for_frames()
                if wrist_align_enabled:
                    assert rs_align is not None
                    frames = rs_align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                else:
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                frame_bgr = np.asanyarray(color_frame.get_data())
                height, width = frame_bgr.shape[:2]
                resolution = (int(width), int(height))
                if intr is None or intr_resolution != resolution:
                    color_intr = rs.video_stream_profile(color_frame.profile).get_intrinsics()
                    intr = _rs_intrinsics_to_opencv_dict(color_intr)
                    intr_resolution = resolution
            else:
                assert cap is not None
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    time.sleep(0.01)
                    continue

                height, width = frame_bgr.shape[:2]
                resolution = (int(width), int(height))
                if intr is None or intr_resolution != resolution:
                    assert raw_intr is not None
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
                corner_refine_mode=corner_refine_mode,
            )
            if hand_overlay_cfg is not None:
                reference_camera_body = None
                table_tag_for_reference = tag_dict.get(int(args.table_marker_id))
                if table_tag_for_reference is not None and last_cube_pose is not None:
                    transform_camera_table_reference = rvec_tvec_to_transform(
                        table_tag_for_reference["rvec"],
                        table_tag_for_reference["tvec"],
                    )
                    reference_camera_body = (
                        transform_camera_table_reference @ last_cube_pose.transform_table_cube
                    )
                resolve_marker_body_tag_pose_branches(
                    tag_dict,
                    hand_overlay_cfg,
                    reference_camera_body=reference_camera_body,
                )
            camera_body_pose = None
            if hand_overlay_cfg is not None:
                camera_body_pose = _estimate_body_pose_in_camera_from_tag_dict(
                    tag_dict,
                    hand_overlay_cfg,
                    np.asarray(intr["K"], dtype=np.float64),
                    reference_camera_body=last_camera_body_transform,
                    outlier_threshold_m=float(args.cube_outlier_threshold_mm) * 0.001,
                    reprojection_error_threshold_px=float(args.cube_reprojection_threshold_px),
                )
                if camera_body_pose is not None:
                    last_camera_body_transform = camera_body_pose.transform_camera_body.copy()
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
                _draw_marker_outline(vis, table_tag["corners"], (0, 215, 255), None)
                _draw_axes_overlay(
                    vis,
                    intr,
                    table_tag["rvec"],
                    table_tag["tvec"],
                    axis_length=float(args.table_axis_length),
                    label=None,
                    label_color=(0, 215, 255),
                )

            for raw_target_id, target_data in frame_result["targets"].items():
                target_id = int(raw_target_id)
                tag = tag_dict.get(target_id)
                if tag is None:
                    continue
                color = _marker_id_to_color(target_id)
                _draw_marker_outline(vis, tag["corners"], color, None)
                if args.show_target_axes:
                    _draw_axes_overlay(
                        vis,
                        intr,
                        tag["rvec"],
                        tag["tvec"],
                        axis_length=float(args.target_axis_length),
                        label=str(target_id),
                        label_color=color,
                    )

            tracker_result = None
            if marker_body_tracker is not None:
                tracker_result = marker_body_tracker.update(
                    frame_result=frame_result,
                    camera_matrix=np.asarray(intr["K"], dtype=np.float64),
                )
            cube_pose = None if tracker_result is None else tracker_result.smoothed_pose
            if cube_pose is not None:
                last_cube_pose = cube_pose
            transform_camera_table = None
            table_pose = frame_result.get("table_in_camera")
            if table_pose is not None:
                transform_camera_table = np.asarray(table_pose["matrix"], dtype=np.float64).reshape(4, 4)
            transform_camera_body_live, body_pose_source = _resolve_camera_body_transform(
                transform_camera_table=transform_camera_table,
                cube_pose=cube_pose,
                camera_body_pose=camera_body_pose,
            )

            raw_joint_angles = np.zeros(20, dtype=np.float64)
            raw_joint_stamp = 0.0
            overlay_joint_angles = np.zeros(20, dtype=np.float64)
            if glove_reader is not None and hand_overlay_cfg is not None:
                raw_joint_angles, raw_joint_stamp, _raw_line = glove_reader.snapshot_rad20()
                overlay_joint_angles = _compute_overlay_joint_angles(raw_joint_angles, cfg=hand_overlay_cfg)
                overlay_joint_angles = _apply_overlay_joint_calibration(
                    overlay_joint_angles,
                    joint_scale=overlay_joint_scale,
                    joint_bias=overlay_joint_bias,
                )

            transform_camera_wrist_live = None
            if transform_camera_body_live is not None and hand_overlay_cfg is not None:
                transform_camera_wrist_live = transform_camera_body_live @ hand_overlay_cfg.cube_to_wrist_transform()

            if human_model is not None and transform_camera_wrist_live is not None:
                landmarks_wrist_m = human_model.landmarks_from_angles(overlay_joint_angles)
                camera_landmarks_m = transform_points(
                    transform_camera_wrist_live,
                    landmarks_wrist_m,
                )
                draw_projected_hand(
                    vis,
                    intr,
                    camera_landmarks_m,
                    draw_axes_enabled=False,
                    axis_rvec=None,
                    axis_tvec=None,
                    axis_length_m=float(args.hand_axis_length),
                )

            if transform_camera_body_live is not None:
                rvec_cube, tvec_cube = transform_to_rvec_tvec(transform_camera_body_live)
                _draw_axes_overlay(
                    vis,
                    intr,
                    rvec_cube,
                    tvec_cube,
                    axis_length=float(args.cube_axis_length),
                    label=None,
                    label_color=(20, 20, 20),
                )

            triangle_camera_xyz = None
            triangle_body_xyz = None
            current_body_to_wrist_transform = None
            wrist_conf = float("nan")
            if (
                wrist_align_enabled
                and wrist_detector is not None
                and depth_frame is not None
                and source_triangle_wrist_m is not None
            ):
                detection = wrist_detector.detect(frame_bgr)
                if detection is not None:
                    keypoints_2d, confidence = detection
                    wrist_conf = float(np.mean(confidence)) if confidence.size else float("nan")
                    vis = wrist_detector.draw_landmarks(vis, keypoints_2d, confidence)
                    bbox = _estimate_bbox_xyxy(keypoints_2d, frame_bgr.shape)
                    cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (80, 220, 255), 2)
                    triangle_camera_xyz = _deproject_keypoint_points(
                        depth_frame,
                        keypoints_2d,
                        landmark_indices=PALM_TRIANGLE_LANDMARK_INDICES,
                        window_radius=int(args.wrist_align_depth_window_radius),
                    )
                    triangle_colors = (
                        (0, 255, 255),
                        (255, 180, 80),
                        (80, 255, 180),
                    )
                    for landmark_idx, landmark_name, color in zip(
                        PALM_TRIANGLE_LANDMARK_INDICES,
                        PALM_TRIANGLE_LANDMARK_NAMES,
                        triangle_colors,
                    ):
                        pt_uv = tuple(
                            np.round(np.asarray(keypoints_2d[int(landmark_idx)], dtype=np.float64)).astype(int)
                        )
                        cv2.circle(vis, pt_uv, 6, color, -1)
                        cv2.putText(
                            vis,
                            landmark_name,
                            (pt_uv[0] + 8, pt_uv[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                    if triangle_camera_xyz is not None and transform_camera_body_live is not None:
                        triangle_body_xyz = _camera_points_to_body_points(
                            transform_camera_body_live,
                            triangle_camera_xyz,
                        )
                        current_body_to_wrist_transform = estimate_body_to_wrist_transform_from_triangles(
                            source_triangle_wrist_m,
                            triangle_body_xyz,
                        )
                        if current_body_to_wrist_transform is not None:
                            transform_camera_wrist_candidate = (
                                transform_camera_body_live @ current_body_to_wrist_transform
                            )
                            rvec_wrist, tvec_wrist = transform_to_rvec_tvec(transform_camera_wrist_candidate)
                            _draw_axes_overlay(
                                vis,
                                intr,
                                rvec_wrist,
                                tvec_wrist,
                                axis_length=float(args.hand_axis_length),
                                label="mp",
                                label_color=(32, 32, 32),
                            )

            now = time.monotonic()
            dt = max(now - last_time, 1e-6)
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema <= 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)
            last_time = now
            if hand_overlay_enabled and hand_overlay_cfg is not None and (now - last_status_emit) >= 0.08:
                _emit_runtime_status(
                    frame_idx=frame_idx,
                    frame_result=frame_result,
                    cube_pose=cube_pose,
                    raw_joint_stamp=raw_joint_stamp,
                    glove_port=str(args.glove_port),
                )
                last_status_emit = now
            if (
                args.diagnose_marker_body
                and tracker_result is not None
                and (now - last_diag_emit) >= 0.6
            ):
                _emit_marker_body_diagnostic(
                    report=tracker_result.consistency_report,
                    position_threshold_mm=float(args.diagnose_position_threshold_mm),
                    rotation_threshold_deg=float(args.diagnose_rotation_threshold_deg),
                )
                last_diag_emit = now

            if wrist_align_enabled:
                hud_lines = [
                    f"wrist_align=on body_source={body_pose_source} samples={len(wrist_align_samples_body_to_wrist)} mp_conf={wrist_conf:.3f}",
                    "keys: space=set-now  a=append-avg  r=reset  w=write  q=quit",
                ]
                if triangle_camera_xyz is not None:
                    wrist_cam = triangle_camera_xyz[0]
                    hud_lines.append(
                        f"triangle_wrist_cam=({wrist_cam[0]:+.3f}, {wrist_cam[1]:+.3f}, {wrist_cam[2]:+.3f}) m"
                    )
                if triangle_body_xyz is not None:
                    wrist_body = triangle_body_xyz[0]
                    hud_lines.append(
                        f"triangle_wrist_body=({wrist_body[0]:+.3f}, {wrist_body[1]:+.3f}, {wrist_body[2]:+.3f}) m"
                    )
                if current_body_to_wrist_transform is not None:
                    pose_t = current_body_to_wrist_transform[:3, 3]
                    pose_q = rotmat_to_quaternion_xyzw(current_body_to_wrist_transform[:3, :3])
                    hud_lines.append(
                        f"candidate_t=({pose_t[0]:+.3f}, {pose_t[1]:+.3f}, {pose_t[2]:+.3f}) q=({pose_q[0]:+.2f}, {pose_q[1]:+.2f}, {pose_q[2]:+.2f}, {pose_q[3]:+.2f})"
                    )
                if wrist_align_samples_body_to_wrist:
                    sample_mean = average_body_to_wrist_transforms(wrist_align_samples_body_to_wrist)
                    if sample_mean is not None:
                        sample_t = sample_mean[:3, 3]
                        sample_q = rotmat_to_quaternion_xyzw(sample_mean[:3, :3])
                        hud_lines.append(
                            f"mean_t=({sample_t[0]:+.3f}, {sample_t[1]:+.3f}, {sample_t[2]:+.3f}) q=({sample_q[0]:+.2f}, {sample_q[1]:+.2f}, {sample_q[2]:+.2f}, {sample_q[3]:+.2f})"
                        )
                _draw_hud(vis, hud_lines)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" ") and wrist_align_enabled:
                if current_body_to_wrist_transform is None or hand_overlay_cfg is None:
                    print("\n[wrist-align] capture skipped: need palm-triangle depth and marker-body pose")
                else:
                    mean_transform = capture_body_to_wrist_transform_sample(
                        wrist_align_samples_body_to_wrist,
                        current_body_to_wrist_transform,
                        replace_existing=True,
                    )
                    apply_body_to_wrist_transform(hand_overlay_cfg, mean_transform)
                    print(
                        "\n[wrist-align] set current sample body_to_wrist_t_m="
                        f"{[round(float(x), 6) for x in current_body_to_wrist_transform[:3, 3]]} active_mean_t_m="
                        f"{[round(float(x), 6) for x in mean_transform[:3, 3]]}"
                    )
            elif key == ord("a") and wrist_align_enabled:
                if current_body_to_wrist_transform is None or hand_overlay_cfg is None:
                    print("\n[wrist-align] append skipped: need palm-triangle depth and marker-body pose")
                else:
                    capture_body_to_wrist_transform_sample(
                        wrist_align_samples_body_to_wrist,
                        current_body_to_wrist_transform,
                        replace_existing=False,
                    )
                    mean_transform = _apply_runtime_body_to_wrist_transform(
                        hand_overlay_cfg,
                        wrist_align_samples_body_to_wrist,
                    )
                    assert mean_transform is not None
                    print(
                        "\n[wrist-align] appended sample #"
                        f"{len(wrist_align_samples_body_to_wrist)} body_to_wrist_t_m="
                        f"{[round(float(x), 6) for x in current_body_to_wrist_transform[:3, 3]]} mean_t_m="
                        f"{[round(float(x), 6) for x in mean_transform[:3, 3]]}"
                    )
            elif key == ord("r") and wrist_align_enabled:
                wrist_align_samples_body_to_wrist.clear()
                if hand_overlay_cfg is not None and base_body_to_wrist_transform is not None:
                    hand_overlay_cfg.set_body_to_wrist_transform(base_body_to_wrist_transform)
                print("\n[wrist-align] cleared samples and restored original body->wrist transform")
            elif key == ord("w") and wrist_align_enabled:
                if not wrist_align_samples_body_to_wrist or hand_overlay_cfg is None:
                    print("\n[wrist-align] write skipped: no captured samples")
                else:
                    mean_transform, std_translation = save_body_to_wrist_alignment_outputs(
                        input_config_path=hand_overlay_cfg_path,
                        output_config_path=wrist_align_output_config_path,
                        output_report_path=wrist_align_output_report_path,
                        cfg=hand_overlay_cfg,
                        samples_body_to_wrist=np.stack(wrist_align_samples_body_to_wrist, axis=0),
                        initial_guess_transform=base_body_to_wrist_transform,
                    )
                    print(
                        "\n[wrist-align] saved body_to_wrist_t_m="
                        f"{[round(float(x), 6) for x in mean_transform[:3, 3]]} std_t_m="
                        f"{[round(float(x), 6) for x in std_translation]}"
                    )
                    print(f"[wrist-align] wrote config: {wrist_align_output_config_path}")
                    print(f"[wrist-align] wrote report: {wrist_align_output_report_path}")
            frame_idx += 1
    finally:
        if cap is not None:
            cap.release()
        if wrist_detector is not None:
            wrist_detector.close()
        if rs_pipeline_started and rs_pipeline is not None:
            rs_pipeline.stop()
        cv2.destroyAllWindows()
        if hand_overlay_enabled and hand_overlay_cfg is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
