from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dexslide.calibration.landmark_detector import LandmarkDetector
from dexslide.communications import (
    camera_communication,
    hand_joint_communication,
    resolve_joint_port,
    resolve_realsense_serial,
)
from dexslide.live import live_listener
from dexslide.paths import (
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_GLOVE_CALIBRATION_FILE,
    DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE,
    DEFAULT_LEFT_MARKER_TO_WRIST_FILE,
    DEFAULT_LEFT_TAGS_TO_MARKER_FILE,
    DEXALIGN_CALIBRATION_DIR,
)
from dexslide.vision.aruco_pose_tracker import _detect_localize_aruco_tags
from dexslide.vision.marker_body_model import HandCubeOverlayConfig

from dexslide.calibration.calibrate_marker_wrist_offset import BodyPoseEstimate, _estimate_body_pose_in_camera_from_tag_dict
from dexslide.calibration.dexalign.io_utils import ensure_session_dir, load_marker2hand_asset_mm, save_alignment_dataset
from dexslide.calibration.dexalign.live_preview import (
    _compose_camera_overlay,
    _draw_axes,
    _draw_hand,
    _set_axes_equal,
    drain_camera_preview_commands,
    enqueue_camera_preview_frame,
    start_camera_preview_process,
    stop_camera_preview_process,
)
from dexslide.calibration.dexalign.types import AlignmentDataset, AlignmentFrame, NUM_KEYPOINTS

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - 运行时依赖
    rs = None


REQUIRED_KEYPOINT_INDICES = (0, 5, 9)


@dataclass(frozen=True)
class RuntimeFrame:
    timestamp: float
    color_bgr: np.ndarray
    depth_frame: Any


@dataclass(frozen=True)
class KeypointDeprojection:
    keypoints_camera_mm: np.ndarray
    valid_mask: np.ndarray
    depth_mm: np.ndarray


@dataclass(frozen=True)
class MarkerPoseObservation:
    camera_T_marker_mm: np.ndarray
    marker_ids_used: tuple[int, ...] = ()
    marker_reproj_error_px: float | None = None


@dataclass
class CollectionStats:
    total_frames_seen: int = 0
    frames_kept: int = 0
    glove_frames_valid: int = 0
    skip_counts: dict[str, int] = field(default_factory=dict)

    def record_skip(self, reason: str) -> None:
        self.skip_counts[reason] = int(self.skip_counts.get(reason, 0)) + 1


@dataclass(frozen=True)
class FrameCollectionResult:
    detection: tuple[np.ndarray, np.ndarray] | None
    marker_pose: MarkerPoseObservation | None
    deprojection: KeypointDeprojection | None
    q_encoder_rad20: np.ndarray | None
    candidate_frame: AlignmentFrame | None
    skip_reason: str | None
    glove_timestamp: float | None = None
    glove_sample_age_sec: float | None = None

    @property
    def kept(self) -> bool:
        return self.candidate_frame is not None and self.skip_reason is None


def _apply_manual_capture_gate(
    *,
    frame_result: FrameCollectionResult,
    capture_enabled: bool,
    kept_frames: list[AlignmentFrame],
    stats: CollectionStats,
) -> str:
    if (
        frame_result.skip_reason is None
        and frame_result.q_encoder_rad20 is not None
        and np.isfinite(np.asarray(frame_result.q_encoder_rad20, dtype=np.float64)).all()
    ):
        stats.glove_frames_valid += 1
    if not bool(capture_enabled):
        stats.record_skip("capture_paused")
        if frame_result.skip_reason is None and frame_result.candidate_frame is not None:
            return "paused:ready"
        return f"paused:{frame_result.skip_reason}"

    if frame_result.skip_reason is not None:
        stats.record_skip(frame_result.skip_reason)
        return f"skip:{frame_result.skip_reason}"

    assert frame_result.candidate_frame is not None
    kept_frames.append(frame_result.candidate_frame)
    stats.frames_kept += 1
    return "kept"


def _evaluate_runtime_frame(
    *,
    runtime_frame: RuntimeFrame,
    detector: Any,
    marker_pose_estimator: Callable[[RuntimeFrame, np.ndarray | None], MarkerPoseObservation | None],
    glove_reader: Any | None,
    deproject_keypoints: Callable[[RuntimeFrame, np.ndarray], KeypointDeprojection],
    previous_kept_frame: AlignmentFrame | None,
    min_valid_keypoints: int,
    max_glove_sample_age_sec: float = 0.5,
) -> FrameCollectionResult:
    detection = detector.detect(runtime_frame.color_bgr)
    if detection is None:
        return FrameCollectionResult(
            detection=None,
            marker_pose=None,
            deprojection=None,
            q_encoder_rad20=None,
            candidate_frame=None,
            skip_reason="no_hand_detection",
        )

    marker_pose = marker_pose_estimator(
        runtime_frame,
        None if previous_kept_frame is None else previous_kept_frame.camera_T_marker,
    )
    if marker_pose is None:
        return FrameCollectionResult(
            detection=detection,
            marker_pose=None,
            deprojection=None,
            q_encoder_rad20=None,
            candidate_frame=None,
            skip_reason="marker_pose_unavailable",
        )

    keypoints_2d, confidence = detection
    deprojection = deproject_keypoints(runtime_frame, np.asarray(keypoints_2d, dtype=np.float64))
    valid_mask = np.asarray(deprojection.valid_mask, dtype=bool).reshape(NUM_KEYPOINTS)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < int(min_valid_keypoints):
        return FrameCollectionResult(
            detection=detection,
            marker_pose=marker_pose,
            deprojection=deprojection,
            q_encoder_rad20=None,
            candidate_frame=None,
            skip_reason="too_few_valid_keypoints",
        )
    if not np.all(valid_mask[list(REQUIRED_KEYPOINT_INDICES)]):
        return FrameCollectionResult(
            detection=detection,
            marker_pose=marker_pose,
            deprojection=deprojection,
            q_encoder_rad20=None,
            candidate_frame=None,
            skip_reason="missing_anchor_keypoints",
        )

    if glove_reader is None:
        q_encoder_rad20 = np.full(20, np.nan, dtype=np.float64)
        glove_timestamp = None
        glove_sample_age_sec = None
    else:
        try:
            q_encoder_rad20, glove_timestamp, raw_line = glove_reader.snapshot_rad20()
            glove_timestamp = float(glove_timestamp)
            raw_line = str(raw_line).strip()
            sample_age = getattr(glove_reader, "sample_age_sec", None)
            glove_sample_age_sec = (
                float(sample_age())
                if callable(sample_age)
                else (
                    float("inf")
                    if glove_timestamp <= 0.0
                    else max(0.0, time.time() - glove_timestamp)
                )
            )
        except Exception:
            return FrameCollectionResult(
                detection=detection,
                marker_pose=marker_pose,
                deprojection=deprojection,
                q_encoder_rad20=None,
                candidate_frame=None,
                skip_reason="glove_sample_unavailable",
            )
        q_encoder_rad20 = np.asarray(q_encoder_rad20, dtype=np.float64).reshape(20)
        if glove_timestamp <= 0.0 or not raw_line or not np.isfinite(q_encoder_rad20).all():
            return FrameCollectionResult(
                detection=detection,
                marker_pose=marker_pose,
                deprojection=deprojection,
                q_encoder_rad20=q_encoder_rad20,
                candidate_frame=None,
                skip_reason="glove_sample_unavailable",
                glove_timestamp=glove_timestamp,
                glove_sample_age_sec=glove_sample_age_sec,
            )
        if glove_sample_age_sec > float(max_glove_sample_age_sec):
            return FrameCollectionResult(
                detection=detection,
                marker_pose=marker_pose,
                deprojection=deprojection,
                q_encoder_rad20=q_encoder_rad20,
                candidate_frame=None,
                skip_reason="glove_sample_stale",
                glove_timestamp=glove_timestamp,
                glove_sample_age_sec=glove_sample_age_sec,
            )
    candidate_frame = AlignmentFrame(
        timestamp=float(runtime_frame.timestamp),
        camera_T_marker=np.asarray(marker_pose.camera_T_marker_mm, dtype=np.float64).reshape(4, 4),
        q_encoder_rad20=np.asarray(q_encoder_rad20, dtype=np.float64).reshape(20),
        keypoints_camera_mm=np.asarray(deprojection.keypoints_camera_mm, dtype=np.float64).reshape(NUM_KEYPOINTS, 3),
        keypoint_confidence=np.asarray(confidence, dtype=np.float64).reshape(NUM_KEYPOINTS),
        keypoint_valid_mask=valid_mask,
        keypoints_uv=np.asarray(keypoints_2d, dtype=np.float64).reshape(NUM_KEYPOINTS, 2),
        depth_mm=np.asarray(deprojection.depth_mm, dtype=np.float64).reshape(NUM_KEYPOINTS),
        marker_ids_used=tuple(int(marker_id) for marker_id in marker_pose.marker_ids_used),
        marker_reproj_error_px=marker_pose.marker_reproj_error_px,
    )
    return FrameCollectionResult(
        detection=detection,
        marker_pose=marker_pose,
        deprojection=deprojection,
        q_encoder_rad20=np.asarray(q_encoder_rad20, dtype=np.float64).reshape(20),
        candidate_frame=candidate_frame,
        skip_reason=None,
        glove_timestamp=glove_timestamp,
        glove_sample_age_sec=glove_sample_age_sec,
    )


def collect_alignment_dataset(
    *,
    frames: Iterable[RuntimeFrame],
    detector: Any,
    marker_pose_estimator: Callable[[RuntimeFrame, np.ndarray | None], MarkerPoseObservation | None],
    glove_reader: Any | None,
    deproject_keypoints: Callable[[RuntimeFrame, np.ndarray], KeypointDeprojection],
    hand: str,
    source_config_paths: dict[str, str] | None = None,
    capture_note: str = "",
    capture_kind: str = "s2",
    min_valid_keypoints: int = 12,
    max_glove_sample_age_sec: float = 0.5,
    max_kept_frames: int | None = None,
) -> tuple[AlignmentDataset, CollectionStats]:
    kept_frames: list[AlignmentFrame] = []
    stats = CollectionStats()

    for runtime_frame in frames:
        stats.total_frames_seen += 1
        frame_result = _evaluate_runtime_frame(
            runtime_frame=runtime_frame,
            detector=detector,
            marker_pose_estimator=marker_pose_estimator,
            glove_reader=glove_reader,
            deproject_keypoints=deproject_keypoints,
            previous_kept_frame=kept_frames[-1] if kept_frames else None,
            min_valid_keypoints=int(min_valid_keypoints),
            max_glove_sample_age_sec=float(max_glove_sample_age_sec),
        )
        if frame_result.skip_reason is not None:
            stats.record_skip(frame_result.skip_reason)
            continue

        assert frame_result.candidate_frame is not None
        kept_frames.append(frame_result.candidate_frame)
        stats.frames_kept += 1
        if np.isfinite(frame_result.candidate_frame.q_encoder_rad20).all():
            stats.glove_frames_valid += 1
        if max_kept_frames is not None and stats.frames_kept >= int(max_kept_frames):
            break

    dataset = AlignmentDataset(
        hand=str(hand).strip().lower(),
        frames=tuple(kept_frames),
        capture_kind=str(capture_kind).strip().lower(),
        source_config_paths=source_config_paths or {},
        capture_note=capture_note,
    )
    return dataset, stats


def _sample_depth_m(depth_frame: Any, u: int, v: int, *, window_radius: int) -> float | None:
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


def deproject_keypoints_with_realsense(
    frame: RuntimeFrame,
    keypoints_2d: np.ndarray,
    *,
    window_radius: int,
    rs_module: Any | None = None,
) -> KeypointDeprojection:
    if rs_module is None:
        rs_module = rs
    if rs_module is None:
        raise RuntimeError("pyrealsense2 is required for RealSense deprojection.")

    keypoints = np.asarray(keypoints_2d, dtype=np.float64).reshape(NUM_KEYPOINTS, 2)
    intr = rs_module.video_stream_profile(frame.depth_frame.profile).get_intrinsics()
    points_mm = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
    valid_mask = np.zeros(NUM_KEYPOINTS, dtype=bool)
    depth_mm = np.full(NUM_KEYPOINTS, np.nan, dtype=np.float64)
    for idx, uv in enumerate(keypoints):
        u = int(round(float(uv[0])))
        v = int(round(float(uv[1])))
        z_m = _sample_depth_m(frame.depth_frame, u, v, window_radius=int(window_radius))
        if z_m is None or z_m <= 0.0:
            continue
        xyz_m = rs_module.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))
        points_mm[idx] = 1000.0 * np.asarray(xyz_m, dtype=np.float64).reshape(3)
        depth_mm[idx] = 1000.0 * float(z_m)
        valid_mask[idx] = True
    return KeypointDeprojection(keypoints_camera_mm=points_mm, valid_mask=valid_mask, depth_mm=depth_mm)


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


def start_realsense_pipeline(
    *,
    width: int,
    height: int,
    fps: int,
    camera_serial: str | None = None,
) -> tuple[Any, Any, dict[str, np.ndarray]]:
    if rs is None:
        raise RuntimeError("pyrealsense2 is required. Install librealsense Python bindings first.")
    pipeline = rs.pipeline()
    config = rs.config()
    if camera_serial:
        config.enable_device(str(camera_serial))
    config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr_dict = _rs_intrinsics_to_opencv_dict(color_profile.get_intrinsics())
    return pipeline, align, intr_dict


def next_realsense_frame(pipeline: Any, align: Any) -> RuntimeFrame | None:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    color_bgr = np.asanyarray(color_frame.get_data())
    return RuntimeFrame(timestamp=time.time(), color_bgr=color_bgr, depth_frame=depth_frame)


def make_realsense_marker_pose_estimator(
    *,
    marker_body_config_path: str | Path,
    intr_dict: dict[str, np.ndarray],
    outlier_threshold_mm: float,
    reprojection_threshold_px: float,
    refine_subpix: bool = True,
    motion_tolerant: bool = True,
) -> Callable[[RuntimeFrame, np.ndarray | None], MarkerPoseObservation | None]:
    config_path = Path(marker_body_config_path).expanduser().resolve()
    overlay_cfg = HandCubeOverlayConfig.load(config_path)
    target_cfg = overlay_cfg.build_target_aruco_config()
    camera_matrix = np.asarray(intr_dict["K"], dtype=np.float64).reshape(3, 3)

    def estimator(frame: RuntimeFrame, previous_pose_mm: np.ndarray | None) -> MarkerPoseObservation | None:
        tag_dict = _detect_localize_aruco_tags(
            img_bgr=frame.color_bgr,
            aruco_dict=target_cfg["aruco_dict"],
            marker_size_map=target_cfg["marker_size_map"],
            fisheye_intr_dict=intr_dict,
            refine_subpix=bool(refine_subpix),
            motion_tolerant=bool(motion_tolerant),
            corner_refine_mode="apriltag" if refine_subpix else "none",
        )
        previous_pose_m = None
        if previous_pose_mm is not None:
            previous_pose_m = np.asarray(previous_pose_mm, dtype=np.float64).reshape(4, 4).copy()
            previous_pose_m[:3, 3] *= 0.001
        pose_estimate: BodyPoseEstimate | None = _estimate_body_pose_in_camera_from_tag_dict(
            tag_dict,
            overlay_cfg,
            camera_matrix,
            reference_camera_body=previous_pose_m,
            outlier_threshold_m=float(outlier_threshold_mm) * 0.001,
            reprojection_error_threshold_px=float(reprojection_threshold_px),
        )
        if pose_estimate is None:
            return None
        camera_T_marker_mm = np.asarray(pose_estimate.transform_camera_body, dtype=np.float64).reshape(4, 4).copy()
        camera_T_marker_mm[:3, 3] *= 1000.0
        return MarkerPoseObservation(
            camera_T_marker_mm=camera_T_marker_mm,
            marker_ids_used=tuple(int(marker_id) for marker_id in pose_estimate.source_marker_ids),
            marker_reproj_error_px=float(pose_estimate.mean_reprojection_error_px),
        )

    return estimator


def _build_parser() -> argparse.ArgumentParser:
    joint_communication = hand_joint_communication("left")
    camera = camera_communication("primary")
    camera_intrinsics = json.loads(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Collect an offline DexAlign dataset from RGB-D, marker pose, and glove encoders.")
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--capture-kind", choices=["s1", "s2"], default="s2")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output-dir", default=str(DEXALIGN_CALIBRATION_DIR))
    parser.add_argument("--marker-body-config", default=str(DEFAULT_LEFT_TAGS_TO_MARKER_FILE))
    parser.add_argument("--glove-port", default=resolve_joint_port("left"))
    parser.add_argument("--glove-baud", type=int, default=int(joint_communication["baud"]))
    parser.add_argument("--glove-mode", choices=["raw", "angles"], default=str(joint_communication["mode"]))
    parser.add_argument("--glove-calib-file", default=str(DEFAULT_GLOVE_CALIBRATION_FILE))
    parser.add_argument(
        "--glove-startup-timeout-sec",
        type=float,
        default=float(joint_communication["startup_timeout_sec"]),
    )
    parser.add_argument(
        "--glove-max-sample-age-sec",
        type=float,
        default=float(joint_communication["max_sample_age_sec"]),
    )
    parser.add_argument("--marker2hand-file", default=str(DEFAULT_LEFT_MARKER_TO_WRIST_FILE))
    parser.add_argument("--camera-serial", default=resolve_realsense_serial("primary"))
    parser.add_argument("--width", type=int, default=int(camera_intrinsics["image_width"]))
    parser.add_argument("--height", type=int, default=int(camera_intrinsics["image_height"]))
    parser.add_argument("--fps", type=int, default=int(camera_intrinsics["fps"]))
    parser.add_argument("--target-kept-frames", type=int, default=600)
    parser.add_argument("--depth-window-radius", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--body-outlier-threshold-mm", type=float, default=20.0)
    parser.add_argument("--body-reprojection-threshold-px", type=float, default=5.0)
    parser.add_argument("--min-valid-keypoints", type=int, default=12)
    parser.add_argument("--capture-note", default="")
    parser.add_argument("--strict-detector", action="store_true")
    parser.add_argument("--no-refine-subpix", action="store_true")
    parser.add_argument("--camera-window-name", default="DexAlign Collect Camera")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    session_id, session_dir = ensure_session_dir(args.session_id, base_dir=Path(args.output_dir))
    glove_reader = None
    if str(args.capture_kind).strip().lower() == "s2":
        glove_reader = live_listener(
            port=args.glove_port,
            baud=int(args.glove_baud),
            mode=str(args.glove_mode),
            calib_file=args.glove_calib_file,
            startup_timeout_sec=float(args.glove_startup_timeout_sec),
        )
        first_q, first_timestamp, first_raw_line = glove_reader.snapshot_rad20()
        if first_timestamp <= 0.0 or not str(first_raw_line).strip() or not np.isfinite(first_q).all():
            raise RuntimeError("DexSlide listener started without a valid first joint frame")
        print(
            "[dexalign-collect] glove="
            f"{glove_reader.port} baud={glove_reader.baud} mode={glove_reader.mode} "
            f"sample_age={glove_reader.sample_age_sec():.3f}s first_frame=OK"
        )
    preview = None
    detector = None
    try:
        preview = start_camera_preview_process(str(args.camera_window_name))
        detector = LandmarkDetector(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=float(args.min_detection_confidence),
            min_tracking_confidence=float(args.min_tracking_confidence),
        )
    except Exception:
        if preview is not None:
            stop_camera_preview_process(preview)
        if glove_reader is not None:
            glove_reader.stop()
        raise

    active_marker_m: np.ndarray | None = None
    try:
        _initial_marker, _result_marker, active_marker_mm = load_marker2hand_asset_mm(args.marker2hand_file)
        active_marker_m = np.asarray(active_marker_mm, dtype=np.float64).reshape(4, 4).copy()
        active_marker_m[:3, 3] *= 0.001
    except Exception as exc:
        print(f"[dexalign-collect] warning: failed to load marker2hand preview asset: {exc}")

    pipeline = None
    frame_idx = 0
    last_frame_time = time.perf_counter()
    fps_ema = 0.0
    last_glove_q: np.ndarray | None = None
    glove_motion_deg = float("nan")
    kept_frames: list[AlignmentFrame] = []
    stats = CollectionStats()
    try:
        pipeline, align, intr_dict = start_realsense_pipeline(
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            camera_serial=str(args.camera_serial),
        )
        print(
            f"[dexalign-collect] camera_serial={args.camera_serial} "
            f"stream={int(args.width)}x{int(args.height)}@{int(args.fps)}"
        )
        marker_pose_estimator = make_realsense_marker_pose_estimator(
            marker_body_config_path=args.marker_body_config,
            intr_dict=intr_dict,
            outlier_threshold_mm=float(args.body_outlier_threshold_mm),
            reprojection_threshold_px=float(args.body_reprojection_threshold_px),
            refine_subpix=not args.no_refine_subpix,
            motion_tolerant=not args.strict_detector,
        )

        plt.ion()
        fig = plt.figure("DexAlign Collect 3D", figsize=(8.5, 7.2))
        ax = fig.add_subplot(111, projection="3d")
        info_text = fig.text(0.02, 0.02, "", family="monospace", fontsize=9)
        capture_enabled = False

        def _set_capture_enabled(enabled: bool) -> None:
            nonlocal capture_enabled
            enabled = bool(enabled)
            if capture_enabled == enabled:
                return
            capture_enabled = enabled
            print(f"[dexalign-collect] capture={'ON' if capture_enabled else 'PAUSED'}")
            fig.canvas.draw_idle()

        def _on_key(event: Any) -> None:
            key = "" if event.key is None else str(event.key).lower()
            if key in {" ", "space"}:
                _set_capture_enabled(not capture_enabled)

        fig.canvas.mpl_connect("key_press_event", _on_key)
        print("[dexalign-collect] capture=PAUSED，聚焦相机或 3D 窗口后按 SPACE 切换。")

        while plt.fignum_exists(fig.number) and not preview.stop_event.is_set():
            for command in drain_camera_preview_commands(preview):
                if command == "space":
                    _set_capture_enabled(not capture_enabled)
            if stats.frames_kept >= int(args.target_kept_frames):
                break

            runtime_frame = next_realsense_frame(pipeline, align)
            if runtime_frame is None:
                continue

            ax.cla()
            stats.total_frames_seen += 1
            frame_result = _evaluate_runtime_frame(
                runtime_frame=runtime_frame,
                detector=detector,
                marker_pose_estimator=marker_pose_estimator,
                glove_reader=glove_reader,
                deproject_keypoints=lambda frame, keypoints_2d: deproject_keypoints_with_realsense(
                    frame,
                    keypoints_2d,
                    window_radius=int(args.depth_window_radius),
                ),
                previous_kept_frame=kept_frames[-1] if kept_frames else None,
                min_valid_keypoints=int(args.min_valid_keypoints),
                max_glove_sample_age_sec=float(args.glove_max_sample_age_sec),
            )
            decision_text = _apply_manual_capture_gate(
                frame_result=frame_result,
                capture_enabled=capture_enabled,
                kept_frames=kept_frames,
                stats=stats,
            )
            if (
                frame_result.q_encoder_rad20 is not None
                and np.isfinite(frame_result.q_encoder_rad20).all()
            ):
                current_q = np.asarray(frame_result.q_encoder_rad20, dtype=np.float64).reshape(20)
                if last_glove_q is not None:
                    glove_motion_deg = float(np.max(np.abs(np.rad2deg(current_q - last_glove_q))))
                last_glove_q = current_q.copy()

            marker_pose = frame_result.marker_pose
            deprojection = frame_result.deprojection
            if deprojection is None:
                keypoints_camera_mm = np.full((21, 3), np.nan, dtype=np.float64)
                valid_mask = np.zeros(21, dtype=bool)
                depth_mm = np.full(21, np.nan, dtype=np.float64)
            else:
                keypoints_camera_mm = deprojection.keypoints_camera_mm
                valid_mask = deprojection.valid_mask
                depth_mm = deprojection.depth_mm

            keypoints_camera_m = np.asarray(keypoints_camera_mm, dtype=np.float64) * 0.001
            _draw_axes(
                ax,
                np.eye(4, dtype=np.float64),
                label="camera",
                length_m=0.05,
                colors=("#dc2626", "#16a34a", "#2563eb"),
            )
            if marker_pose is not None:
                marker_pose_m = np.asarray(marker_pose.camera_T_marker_mm, dtype=np.float64).reshape(4, 4).copy()
                marker_pose_m[:3, 3] *= 0.001
                _draw_axes(ax, marker_pose_m, label="marker", length_m=0.04, colors=("#ef4444", "#22c55e", "#3b82f6"))
                if active_marker_m is not None:
                    wrist_pose_m = marker_pose_m @ active_marker_m
                    _draw_axes(ax, wrist_pose_m, label="wrist", length_m=0.035, colors=("#f97316", "#84cc16", "#0ea5e9"))
            _draw_hand(ax, keypoints_camera_m, valid_mask)

            reference_points = keypoints_camera_m[np.isfinite(keypoints_camera_m).all(axis=1)]
            if marker_pose is not None:
                marker_pose_m = np.asarray(marker_pose.camera_T_marker_mm, dtype=np.float64).reshape(4, 4).copy()
                marker_pose_m[:3, 3] *= 0.001
                reference_points = np.vstack([reference_points, marker_pose_m[:3, 3][None, :]])
            if reference_points.size == 0:
                reference_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
            _set_axes_equal(ax, reference_points)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(f"DexAlign Collect 3D [{'REC' if capture_enabled else 'PAUSED'}]")

            now = time.perf_counter()
            dt = max(now - last_frame_time, 1e-6)
            instant_fps = 1.0 / dt
            fps_ema = instant_fps if frame_idx == 0 else (0.85 * fps_ema + 0.15 * instant_fps)
            last_frame_time = now
            frame_idx += 1

            if marker_pose is not None:
                marker_translation = 0.001 * np.asarray(marker_pose.camera_T_marker_mm[:3, 3], dtype=np.float64).reshape(3)
            else:
                marker_translation = np.array([math.nan, math.nan, math.nan], dtype=np.float64)
            wrist_point = keypoints_camera_m[0] if valid_mask[0] else np.array([math.nan, math.nan, math.nan], dtype=np.float64)
            info_text.set_text(
                "\n".join(
                    [
                        f"capture = {'ON' if capture_enabled else 'PAUSED'}  key = SPACE",
                        f"decision = {decision_text}",
                        f"capture_kind = {str(args.capture_kind).lower()}",
                        f"kept = {stats.frames_kept}/{int(args.target_kept_frames)}  seen = {stats.total_frames_seen}",
                        (
                            f"glove = {glove_reader.port}  age_s = {frame_result.glove_sample_age_sec:.3f}  "
                            f"motion_deg = {glove_motion_deg:.2f}"
                            if glove_reader is not None
                            and frame_result.glove_sample_age_sec is not None
                            else "glove = not required for S1"
                        ),
                        (
                            f"marker_t_m = [{marker_translation[0]:+.3f}, {marker_translation[1]:+.3f}, {marker_translation[2]:+.3f}]"
                            if np.isfinite(marker_translation).all()
                            else "marker_t_m = [nan, nan, nan]"
                        ),
                        (
                            f"wrist_xyz_m = [{wrist_point[0]:+.3f}, {wrist_point[1]:+.3f}, {wrist_point[2]:+.3f}]"
                            if np.isfinite(wrist_point).all()
                            else "wrist_xyz_m = [nan, nan, nan]"
                        ),
                        f"valid_keypoints = {int(np.count_nonzero(valid_mask))}",
                    ]
                )
            )

            camera_overlay = _compose_camera_overlay(
                runtime_frame.color_bgr,
                detector=detector,
                detection=frame_result.detection,
                marker_pose=marker_pose,
                valid_mask=valid_mask,
                depth_mm=depth_mm,
                fps_ema=fps_ema,
                frame_idx=frame_idx,
                extra_status_lines=[
                    f"{'REC' if capture_enabled else 'PAUSED'} | SPACE",
                    f"{str(args.capture_kind).upper()} kept={stats.frames_kept}/{int(args.target_kept_frames)} {decision_text}",
                    (
                        f"glove={glove_reader.port} age={frame_result.glove_sample_age_sec:.3f}s"
                        if glove_reader is not None
                        and frame_result.glove_sample_age_sec is not None
                        else "glove=not-required"
                    ),
                ],
            )
            enqueue_camera_preview_frame(preview, camera_overlay)
            plt.pause(0.001)

        dataset = AlignmentDataset(
            hand=str(args.hand).strip().lower(),
            frames=tuple(kept_frames),
            capture_kind=str(args.capture_kind).strip().lower(),
            source_config_paths={
                "marker_body_config": str(Path(args.marker_body_config).expanduser().resolve()),
                "marker2hand_file": str(Path(args.marker2hand_file).expanduser().resolve()),
                "communications": str(DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE.resolve()),
            },
            capture_note=str(args.capture_note),
        )
        if glove_reader is not None:
            dataset.source_config_paths["glove_calibration"] = str(Path(args.glove_calib_file).expanduser().resolve())
        dataset_path, meta_path = save_alignment_dataset(
            session_dir,
            dataset,
            extra_meta={
                "session_id": session_id,
                "capture_summary": {
                    "total_frames_seen": stats.total_frames_seen,
                    "frames_kept": stats.frames_kept,
                    "glove_frames_valid": stats.glove_frames_valid,
                    "skip_counts": stats.skip_counts,
                },
                "communications": {
                    "camera_serial": str(args.camera_serial),
                    "camera_stream": [int(args.width), int(args.height), int(args.fps)],
                    "glove_port": None if glove_reader is None else str(glove_reader.port),
                    "glove_baud": None if glove_reader is None else int(glove_reader.baud),
                    "glove_mode": None if glove_reader is None else str(glove_reader.mode),
                    "glove_max_sample_age_sec": float(args.glove_max_sample_age_sec),
                },
            },
        )
    finally:
        stop_camera_preview_process(preview)
        detector.close()
        if pipeline is not None:
            pipeline.stop()
        if glove_reader is not None:
            glove_reader.stop()
        plt.ioff()

    print(
        f"[dexalign-collect] session={session_id} kept={dataset.num_frames} "
        f"seen={stats.total_frames_seen} skip={json.dumps(stats.skip_counts, ensure_ascii=False)}"
    )
    print(f"[dexalign-collect] dataset={dataset_path}")
    print(f"[dexalign-collect] meta={meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
