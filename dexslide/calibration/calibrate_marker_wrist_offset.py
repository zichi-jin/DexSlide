from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

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
from dexslide.communications import camera_communication, resolve_realsense_serial
from dexslide.paths import (
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_HAND_MARKER_BODY_LEFT_CONFIG_FILE,
    DEFAULT_SKELETON_FILE,
    DIRECT_ARUCO_CALIBRATION_DIR,
)
from dexslide.retargeting.human_model import DexSlideHumanModel
from dexslide.kinematics.transforms import (
    invert_transform,
    make_transform,
    rvec_tvec_to_transform,
    rotmat_to_quaternion_xyzw,
    transform_points,
    transform_to_rvec_tvec,
)
from dexslide.vision.aruco_pose_tracker import _detect_localize_aruco_tags
from dexslide.visualization.aruco_overlay import draw_axes, draw_marker_outline
from dexslide.vision.hand_cube_overlay import (
    HandCubeOverlayConfig,
    _build_marker_observations,
    _compute_marker_reprojection_errors,
    _seed_camera_body_pose,
    _solve_body_pose_camera_from_observations,
    resolve_marker_body_tag_pose_branches,
    resolve_hand_overlay_asset_paths,
)

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - runtime dependency
    rs = None


VIEW_CHOICES = ("color", "depth", "split")


@dataclass(frozen=True)
class BodyPoseEstimate:
    transform_camera_body: np.ndarray
    source_marker_ids: tuple[int, ...]
    max_position_deviation_m: float
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    resolved_tag_dict: dict[int, dict[str, Any]]


def _default_hand_overlay_config_path(hand: str) -> Path:
    requested = str(hand).strip().lower()
    if requested in {"left", "right"}:
        return DIRECT_ARUCO_CALIBRATION_DIR / f"{requested}_tags2marker.json"
    return DEFAULT_HAND_MARKER_BODY_LEFT_CONFIG_FILE


def _default_output_config_path(config_path: Path) -> Path:
    asset_paths = resolve_hand_overlay_asset_paths(config_path)
    return asset_paths["marker_to_wrist"]


def _default_output_report_path(config_path: Path) -> Path:
    asset_paths = resolve_hand_overlay_asset_paths(config_path)
    return asset_paths["marker_to_wrist_dataset"]


def _optional_path_arg(raw: str | None) -> Path | None:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _rs_intrinsics_to_opencv_dict(intr: Any) -> dict[str, np.ndarray]:
    return {
        "DIM": np.array([int(intr.width), int(intr.height)], dtype=np.int64),
        "K": np.array(
            [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "D": np.asarray(list(intr.coeffs), dtype=np.float64).reshape(-1, 1),
    }


def _make_camera_frame_result_from_tag_dict(tag_dict: dict[int, dict[str, Any]]) -> dict[str, Any]:
    targets: dict[str, dict[str, Any]] = {}
    for marker_id, tag in tag_dict.items():
        targets[str(int(marker_id))] = {
            "detected": True,
            "target_in_camera": {
                "matrix": rvec_tvec_to_transform(tag["rvec"], tag["tvec"]).tolist(),
            },
            "undistorted_corners": np.asarray(
                tag.get("undistorted_corners", tag.get("corners")),
                dtype=np.float64,
            ).reshape(4, 2).tolist(),
        }
    return {"targets": targets}


def _estimate_body_pose_in_camera_from_tag_dict(
    tag_dict: dict[int, dict[str, Any]],
    config: HandCubeOverlayConfig,
    camera_matrix: np.ndarray,
    *,
    reference_camera_body: np.ndarray | None = None,
    outlier_threshold_m: float,
    reprojection_error_threshold_px: float,
) -> BodyPoseEstimate | None:
    resolved_tag_dict: dict[int, dict[str, Any]] = copy.deepcopy(tag_dict)
    resolve_marker_body_tag_pose_branches(
        resolved_tag_dict,
        config,
        reference_camera_body=reference_camera_body,
    )

    frame_result = _make_camera_frame_result_from_tag_dict(resolved_tag_dict)
    observations = _build_marker_observations(frame_result, config)
    if not observations:
        return None

    seed_camera_body, active_observations, seed_spread_m = _seed_camera_body_pose(
        observations,
        outlier_threshold_m=outlier_threshold_m,
    )
    if seed_camera_body is None or not active_observations:
        return None

    final_transform_camera_body = seed_camera_body
    final_observations = list(active_observations)
    final_errors = _compute_marker_reprojection_errors(
        final_observations,
        transform_camera_body=final_transform_camera_body,
        camera_matrix=camera_matrix,
    )

    while active_observations:
        solved_transform = _solve_body_pose_camera_from_observations(
            active_observations,
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            initial_transform=seed_camera_body,
            reprojection_error_threshold_px=reprojection_error_threshold_px,
        )
        if solved_transform is None:
            break

        current_errors = _compute_marker_reprojection_errors(
            active_observations,
            transform_camera_body=solved_transform,
            camera_matrix=camera_matrix,
        )
        final_transform_camera_body = solved_transform
        final_observations = list(active_observations)
        final_errors = current_errors

        if len(active_observations) <= 1:
            break

        worst_error = max(
            current_errors,
            key=lambda item: (float(item["mean_error_px"]), float(item["max_error_px"])),
        )
        if float(worst_error["mean_error_px"]) <= float(reprojection_error_threshold_px):
            break

        active_observations = [
            obs for obs in active_observations if obs.marker_id != int(worst_error["marker_id"])
        ]
        seed_camera_body, active_observations, seed_spread_m = _seed_camera_body_pose(
            active_observations,
            outlier_threshold_m=outlier_threshold_m,
        )
        if seed_camera_body is None or not active_observations:
            break

    max_position_deviation_m = seed_spread_m
    if final_observations:
        final_positions = np.stack(
            [obs.transform_camera_body_single[:3, 3] for obs in final_observations],
            axis=0,
        )
        diffs = final_positions - final_transform_camera_body[:3, 3][None, :]
        max_position_deviation_m = float(np.max(np.linalg.norm(diffs, axis=-1)))

    return BodyPoseEstimate(
        transform_camera_body=np.asarray(final_transform_camera_body, dtype=np.float64).reshape(4, 4).copy(),
        source_marker_ids=tuple(int(obs.marker_id) for obs in final_observations),
        max_position_deviation_m=float(max_position_deviation_m),
        mean_reprojection_error_px=(
            0.0 if not final_errors else float(np.mean([err["mean_error_px"] for err in final_errors]))
        ),
        max_reprojection_error_px=(
            0.0 if not final_errors else float(np.max([err["max_error_px"] for err in final_errors]))
        ),
        resolved_tag_dict=resolved_tag_dict,
    )


def _sample_depth_m(
    depth_frame: Any,
    u: int,
    v: int,
    *,
    window_radius: int,
) -> float | None:
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    samples: list[float] = []
    for dv in range(-int(window_radius), int(window_radius) + 1):
        for du in range(-int(window_radius), int(window_radius) + 1):
            uu = int(np.clip(u + du, 0, max(0, width - 1)))
            vv = int(np.clip(v + dv, 0, max(0, height - 1)))
            z_m = float(depth_frame.get_distance(uu, vv))
            if z_m > 0.0:
                samples.append(z_m)
    if not samples:
        return None
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _deproject_keypoint_points(
    depth_frame: Any,
    keypoints_2d: np.ndarray,
    *,
    landmark_indices: tuple[int, ...] = PALM_TRIANGLE_LANDMARK_INDICES,
    window_radius: int,
) -> np.ndarray | None:
    keypoints = np.asarray(keypoints_2d, dtype=np.float64).reshape(-1, 2)
    if keypoints.shape[0] <= max(int(idx) for idx in landmark_indices):
        return None

    intr = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    points_camera_xyz: list[np.ndarray] = []
    for landmark_idx in landmark_indices:
        uv = keypoints[int(landmark_idx)]
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
        z_m = _sample_depth_m(depth_frame, u, v, window_radius=window_radius)
        if z_m is None or z_m <= 0.0:
            return None
        point_xyz = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))
        points_camera_xyz.append(np.asarray(point_xyz, dtype=np.float64).reshape(3))
    return np.asarray(points_camera_xyz, dtype=np.float64).reshape(len(landmark_indices), 3)


def _deproject_wrist_point(
    depth_frame: Any,
    wrist_uv: tuple[float, float],
    *,
    window_radius: int,
) -> np.ndarray | None:
    triangle_points = _deproject_keypoint_points(
        depth_frame,
        np.asarray([wrist_uv], dtype=np.float64).reshape(1, 2),
        landmark_indices=(0,),
        window_radius=window_radius,
    )
    if triangle_points is None:
        return None
    return np.asarray(triangle_points[0], dtype=np.float64).reshape(3)


def _camera_points_to_body_points(
    transform_camera_body: np.ndarray,
    camera_points_xyz: np.ndarray,
) -> np.ndarray:
    points = np.asarray(camera_points_xyz, dtype=np.float64).reshape(-1, 3)
    camera_points_h = np.ones((points.shape[0], 4), dtype=np.float64)
    camera_points_h[:, :3] = points
    body_points_h = (invert_transform(transform_camera_body) @ camera_points_h.T).T
    return np.asarray(body_points_h[:, :3], dtype=np.float64).reshape(points.shape[0], 3)


def _wrist_body_point(
    transform_camera_body: np.ndarray,
    wrist_camera_xyz: np.ndarray,
) -> np.ndarray:
    wrist_body = _camera_points_to_body_points(transform_camera_body, wrist_camera_xyz)
    return np.asarray(wrist_body[0], dtype=np.float64).reshape(3)


def _estimate_bbox_xyxy(
    keypoints_2d: np.ndarray,
    image_shape: tuple[int, ...],
    padding_px: int = 24,
) -> tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    pts = np.asarray(keypoints_2d, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return (0, 0, max(0, width - 1), max(0, height - 1))
    x0 = int(np.floor(float(np.min(pts[:, 0])))) - int(padding_px)
    y0 = int(np.floor(float(np.min(pts[:, 1])))) - int(padding_px)
    x1 = int(np.ceil(float(np.max(pts[:, 0])))) + int(padding_px)
    y1 = int(np.ceil(float(np.max(pts[:, 1])))) + int(padding_px)
    x0 = int(np.clip(x0, 0, max(0, width - 1)))
    y0 = int(np.clip(y0, 0, max(0, height - 1)))
    x1 = int(np.clip(x1, 0, max(0, width - 1)))
    y1 = int(np.clip(y1, 0, max(0, height - 1)))
    return (x0, y0, x1, y1)


def _draw_projected_axes(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    transform_camera_body: np.ndarray,
    *,
    axis_length_m: float,
    label: str,
) -> None:
    rvec, tvec = transform_to_rvec_tvec(transform_camera_body)
    draw_axes(
        image_bgr,
        {
            "K": np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            "D": np.zeros((1, 5), dtype=np.float64),
        },
        rvec,
        tvec,
        axis_length_m,
        label,
        (255, 255, 255),
    )


def _draw_marker_outlines(image_bgr: np.ndarray, tag_dict: dict[int, dict[str, Any]]) -> None:
    for marker_id, tag in sorted(tag_dict.items(), key=lambda item: int(item[0])):
        corners = np.asarray(tag.get("corners"), dtype=np.float64).reshape(4, 2)
        draw_marker_outline(image_bgr, corners, (0, 215, 255), str(int(marker_id)))


def _draw_status_lines(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image_bgr.copy()
    line_height = 24
    box_height = max(36, 12 + line_height * len(lines))
    cv2.rectangle(out, (8, 8), (900, box_height), (15, 15, 15), -1)
    cv2.rectangle(out, (8, 8), (900, box_height), (90, 90, 90), 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (18, 32 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return out


def _compose_display_frame(color_bgr: np.ndarray, depth_bgr: np.ndarray, *, view: str) -> np.ndarray:
    if view == "color":
        return color_bgr
    if view == "depth":
        return depth_bgr
    if view != "split":
        raise ValueError(f"Unsupported view: {view}")
    return np.hstack([color_bgr, depth_bgr])


def _save_alignment_outputs(
    *,
    input_config_path: Path,
    output_config_path: Path,
    output_report_path: Path,
    cfg: HandCubeOverlayConfig,
    samples_body_to_wrist: np.ndarray,
    initial_guess_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return save_body_to_wrist_alignment_outputs(
        input_config_path=input_config_path,
        output_config_path=output_config_path,
        output_report_path=output_report_path,
        cfg=cfg,
        samples_body_to_wrist=samples_body_to_wrist,
        initial_guess_transform=initial_guess_transform,
    )


def _build_parser() -> argparse.ArgumentParser:
    camera = camera_communication("primary")
    camera_intrinsics = json.loads(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Calibrate marker-body to wrist pose using the MediaPipe palm triangle."
    )
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--skeleton-file", default=str(DEFAULT_SKELETON_FILE))
    parser.add_argument("--camera-serial", default=resolve_realsense_serial("primary"))
    parser.add_argument("--width", type=int, default=int(camera_intrinsics["image_width"]))
    parser.add_argument("--height", type=int, default=int(camera_intrinsics["image_height"]))
    parser.add_argument("--fps", type=int, default=int(camera_intrinsics["fps"]))
    parser.add_argument("--view", choices=VIEW_CHOICES, default="split")
    parser.add_argument("--depth-alpha", type=float, default=0.55)
    parser.add_argument("--marker-body-config", default=None, help="Tags->marker JSON path.")
    parser.add_argument("--out-config", default=None, help="Output marker->wrist JSON path.")
    parser.add_argument("--out-report", default=None, help="Output marker->wrist dataset JSON path.")
    parser.add_argument("--depth-window-radius", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--body-outlier-threshold-mm", type=float, default=20.0)
    parser.add_argument("--body-reprojection-threshold-px", type=float, default=5.0)
    parser.add_argument("--axis-length-m", type=float, default=0.04)
    parser.add_argument("--no-refine-subpix", action="store_true")
    parser.add_argument("--strict-detector", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if rs is None:
        raise SystemExit("pyrealsense2 is required. Install librealsense Python bindings first.")

    marker_body_config_path = _optional_path_arg(args.marker_body_config)
    config_path = (
        marker_body_config_path
        if marker_body_config_path is not None
        else _default_hand_overlay_config_path(args.hand).resolve()
    )
    if not config_path.exists():
        raise SystemExit(f"Marker body config not found: {config_path}")
    overlay_cfg = HandCubeOverlayConfig.load(config_path)
    resolved_hand = str(overlay_cfg.hand).strip().lower() or str(args.hand).strip().lower()
    human_model = DexSlideHumanModel(args.skeleton_file, hand=resolved_hand)
    source_triangle_wrist_m = select_palm_triangle_points(
        human_model.landmarks_from_angles(np.zeros(20, dtype=np.float64))
    )
    target_cfg = overlay_cfg.build_target_aruco_config()

    out_config_path = _optional_path_arg(args.out_config)
    output_config_path = (
        out_config_path
        if out_config_path is not None
        else _default_output_config_path(config_path)
    )
    out_report_path = _optional_path_arg(args.out_report)
    output_report_path = (
        out_report_path
        if out_report_path is not None
        else _default_output_report_path(output_config_path)
    )

    pipeline = rs.pipeline()
    pipeline_config = rs.config()
    pipeline_config.enable_device(str(args.camera_serial))
    pipeline_config.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, int(args.fps))
    pipeline_config.enable_stream(rs.stream.depth, int(args.width), int(args.height), rs.format.z16, int(args.fps))

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    detector = LandmarkDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )

    sample_body_to_wrist_transforms: list[np.ndarray] = []
    base_body_to_wrist_transform = overlay_cfg.body_to_wrist_transform().copy()
    last_body_transform: np.ndarray | None = None
    intr_dict: dict[str, np.ndarray] | None = None
    camera_matrix: np.ndarray | None = None
    frame_idx = 0
    last_frame_time = time.perf_counter()
    fps_ema = 0.0
    pipeline_started = False

    window_name = "Marker Wrist Alignment"

    try:
        pipeline_profile = pipeline.start(pipeline_config)
        pipeline_started = True
        color_profile = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr_dict = _rs_intrinsics_to_opencv_dict(color_profile.get_intrinsics())
        camera_matrix = np.asarray(intr_dict["K"], dtype=np.float64).reshape(3, 3)
        print(
            "[align] controls: space=capture sample, w=write output, r=reset samples, "
            "c=color, d=depth, s=split, q/ESC=quit"
        )
        print(f"[align] input_config={config_path}")
        print(f"[align] output_config={output_config_path}")
        print(f"[align] output_report={output_report_path}")
        print(
            "[align] triangle_landmarks="
            f"{list(zip(PALM_TRIANGLE_LANDMARK_NAMES, PALM_TRIANGLE_LANDMARK_INDICES))}"
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth_bgr = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_overlay = cv2.addWeighted(
                color_bgr,
                1.0 - float(args.depth_alpha),
                depth_bgr,
                float(args.depth_alpha),
                0.0,
            )

            detection = detector.detect(color_bgr)
            color_overlay = color_bgr.copy()

            triangle_camera_xyz: np.ndarray | None = None
            triangle_body_xyz: np.ndarray | None = None
            current_body_to_wrist_transform: np.ndarray | None = None
            hand_conf = float("nan")
            if detection is not None:
                keypoints_2d, confidence = detection
                hand_conf = float(np.mean(confidence)) if confidence.size else float("nan")
                color_overlay = detector.draw_landmarks(color_overlay, keypoints_2d, confidence)
                depth_overlay = detector.draw_landmarks(depth_overlay, keypoints_2d, confidence)
                bbox = _estimate_bbox_xyxy(keypoints_2d, color_bgr.shape)
                cv2.rectangle(color_overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (80, 220, 255), 2)
                cv2.rectangle(depth_overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (80, 220, 255), 2)
                triangle_camera_xyz = _deproject_keypoint_points(
                    depth_frame,
                    keypoints_2d,
                    landmark_indices=PALM_TRIANGLE_LANDMARK_INDICES,
                    window_radius=int(args.depth_window_radius),
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
                    cv2.circle(color_overlay, pt_uv, 6, color, -1)
                    cv2.circle(depth_overlay, pt_uv, 6, color, -1)
                    cv2.putText(
                        color_overlay,
                        landmark_name,
                        (pt_uv[0] + 8, pt_uv[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            assert intr_dict is not None
            assert camera_matrix is not None
            tag_dict = _detect_localize_aruco_tags(
                img_bgr=color_bgr,
                aruco_dict=target_cfg["aruco_dict"],
                marker_size_map=target_cfg["marker_size_map"],
                fisheye_intr_dict=intr_dict,
                refine_subpix=not args.no_refine_subpix,
                motion_tolerant=not args.strict_detector,
                corner_refine_mode="apriltag" if not args.no_refine_subpix else "none",
            )

            body_pose = _estimate_body_pose_in_camera_from_tag_dict(
                tag_dict,
                overlay_cfg,
                camera_matrix,
                reference_camera_body=last_body_transform,
                outlier_threshold_m=float(args.body_outlier_threshold_mm) * 0.001,
                reprojection_error_threshold_px=float(args.body_reprojection_threshold_px),
            )
            if body_pose is not None:
                last_body_transform = body_pose.transform_camera_body.copy()
                if triangle_camera_xyz is not None:
                    triangle_body_xyz = _camera_points_to_body_points(
                        body_pose.transform_camera_body,
                        triangle_camera_xyz,
                    )
                    current_body_to_wrist_transform = estimate_body_to_wrist_transform_from_triangles(
                        source_triangle_wrist_m,
                        triangle_body_xyz,
                    )
                _draw_projected_axes(
                    color_overlay,
                    camera_matrix,
                    body_pose.transform_camera_body,
                    axis_length_m=float(args.axis_length_m),
                    label="body",
                )
                _draw_projected_axes(
                    depth_overlay,
                    camera_matrix,
                    body_pose.transform_camera_body,
                    axis_length_m=float(args.axis_length_m),
                    label="body",
                )
                _draw_marker_outlines(color_overlay, body_pose.resolved_tag_dict)
                _draw_marker_outlines(depth_overlay, body_pose.resolved_tag_dict)
                if current_body_to_wrist_transform is not None:
                    transform_camera_wrist = body_pose.transform_camera_body @ current_body_to_wrist_transform
                    _draw_projected_axes(
                        color_overlay,
                        camera_matrix,
                        transform_camera_wrist,
                        axis_length_m=float(args.axis_length_m),
                        label="wrist",
                    )
                    _draw_projected_axes(
                        depth_overlay,
                        camera_matrix,
                        transform_camera_wrist,
                        axis_length_m=float(args.axis_length_m),
                        label="wrist",
                    )

            now = time.perf_counter()
            dt = max(now - last_frame_time, 1e-6)
            instant_fps = 1.0 / dt
            fps_ema = instant_fps if frame_idx == 0 else (0.85 * fps_ema + 0.15 * instant_fps)
            last_frame_time = now
            frame_idx += 1

            if body_pose is None:
                marker_status = "body=NA"
                marker_detail = f"markers={len(tag_dict)}"
            else:
                marker_status = (
                    f"body=OK markers={list(body_pose.source_marker_ids)} "
                    f"reproj={body_pose.mean_reprojection_error_px:.2f}/{body_pose.max_reprojection_error_px:.2f}px"
                )
                marker_detail = f"spread={body_pose.max_position_deviation_m * 1000.0:.1f}mm"

            triangle_cam_text = "triangle_cam=NA"
            if triangle_camera_xyz is not None:
                triangle_cam_text = (
                    "triangle_cam_m="
                    f"[{triangle_camera_xyz[0, 0]:+.3f}, {triangle_camera_xyz[0, 1]:+.3f}, {triangle_camera_xyz[0, 2]:+.3f}] "
                    f"[{triangle_camera_xyz[1, 0]:+.3f}, {triangle_camera_xyz[1, 1]:+.3f}, {triangle_camera_xyz[1, 2]:+.3f}] "
                    f"[{triangle_camera_xyz[2, 0]:+.3f}, {triangle_camera_xyz[2, 1]:+.3f}, {triangle_camera_xyz[2, 2]:+.3f}]"
                )

            body_to_wrist_text = "body_to_wrist=NA"
            if current_body_to_wrist_transform is not None:
                body_to_wrist_t = current_body_to_wrist_transform[:3, 3]
                body_to_wrist_q = rotmat_to_quaternion_xyzw(current_body_to_wrist_transform[:3, :3])
                body_to_wrist_text = (
                    f"body_to_wrist_t_m=[{body_to_wrist_t[0]:+.3f}, {body_to_wrist_t[1]:+.3f}, {body_to_wrist_t[2]:+.3f}] "
                    f"q_xyzw=[{body_to_wrist_q[0]:+.3f}, {body_to_wrist_q[1]:+.3f}, {body_to_wrist_q[2]:+.3f}, {body_to_wrist_q[3]:+.3f}]"
                )

            sample_text = "samples=0"
            if sample_body_to_wrist_transforms:
                mean_transform = average_body_to_wrist_transforms(sample_body_to_wrist_transforms)
                assert mean_transform is not None
                sample_mean_t = mean_transform[:3, 3]
                sample_text = (
                    f"samples={len(sample_body_to_wrist_transforms)} mean_body_to_wrist_t_m="
                    f"[{sample_mean_t[0]:+.3f}, {sample_mean_t[1]:+.3f}, {sample_mean_t[2]:+.3f}]"
                )

            status_lines = [
                f"view={args.view} fps={fps_ema:.1f} frame={frame_idx} hand_conf={hand_conf:.3f}",
                marker_status,
                marker_detail,
                triangle_cam_text,
                body_to_wrist_text,
                sample_text,
                "keys: space=set-now  a=append-avg  r=reset  w=write  q=quit",
            ]
            display = _compose_display_frame(color_overlay, depth_overlay, view=args.view)
            display = _draw_status_lines(display, status_lines)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                args.view = "color"
            elif key == ord("d"):
                args.view = "depth"
            elif key == ord("s"):
                args.view = "split"
            elif key == ord("r"):
                sample_body_to_wrist_transforms.clear()
                print("[align] cleared captured samples")
            elif key == ord(" "):
                if current_body_to_wrist_transform is None:
                    print("[align] capture skipped: need depth for wrist/index_mcp/middle_mcp and marker body pose")
                else:
                    mean_transform = capture_body_to_wrist_transform_sample(
                        sample_body_to_wrist_transforms,
                        current_body_to_wrist_transform,
                        replace_existing=True,
                    )
                    print(
                        "[align] set current sample body_to_wrist_t_m="
                        f"{[round(float(x), 6) for x in current_body_to_wrist_transform[:3, 3]]} active_mean_t_m="
                        f"{[round(float(x), 6) for x in mean_transform[:3, 3]]}"
                    )
            elif key == ord("a"):
                if current_body_to_wrist_transform is None:
                    print("[align] append skipped: need depth for wrist/index_mcp/middle_mcp and marker body pose")
                else:
                    mean_transform = capture_body_to_wrist_transform_sample(
                        sample_body_to_wrist_transforms,
                        current_body_to_wrist_transform,
                        replace_existing=False,
                    )
                    print(
                        "[align] appended sample "
                        f"#{len(sample_body_to_wrist_transforms)} body_to_wrist_t_m="
                        f"{[round(float(x), 6) for x in current_body_to_wrist_transform[:3, 3]]} mean_t_m="
                        f"{[round(float(x), 6) for x in mean_transform[:3, 3]]}"
                    )
            elif key == ord("w"):
                if not sample_body_to_wrist_transforms:
                    print("[align] write skipped: no captured samples")
                else:
                    mean_transform, std_translation = _save_alignment_outputs(
                        input_config_path=config_path,
                        output_config_path=output_config_path,
                        output_report_path=output_report_path,
                        cfg=overlay_cfg,
                        samples_body_to_wrist=np.stack(sample_body_to_wrist_transforms, axis=0),
                        initial_guess_transform=base_body_to_wrist_transform,
                    )
                    print(
                        "[align] saved body_to_wrist_t_m="
                        f"{[round(float(x), 6) for x in mean_transform[:3, 3]]} std_t_m="
                        f"{[round(float(x), 6) for x in std_translation]}"
                    )
                    print(f"[align] wrote config: {output_config_path}")
                    print(f"[align] wrote report: {output_report_path}")
    finally:
        detector.close()
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
