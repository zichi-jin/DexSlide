from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from dexslide.calibration.landmark_detector import LandmarkDetector
from dexslide.communications import camera_communication, resolve_realsense_serial
from dexslide.paths import DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - runtime dependency
    rs = None


VIEW_CHOICES = ("color", "depth", "split")


@dataclass(frozen=True)
class HandObservation:
    keypoints_2d: np.ndarray
    confidence: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    median_depth_mm: float | None


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


def _sample_landmark_depth_mm(
    depth_mm: np.ndarray,
    keypoints_2d: np.ndarray,
    window_radius: int = 2,
) -> float | None:
    depth = np.asarray(depth_mm, dtype=np.uint16)
    pts = np.asarray(keypoints_2d, dtype=np.float32).reshape(-1, 2)
    if depth.ndim != 2 or pts.size == 0:
        return None

    height, width = depth.shape
    valid_samples: list[float] = []
    for u_f, v_f in pts:
        u = int(round(float(u_f)))
        v = int(round(float(v_f)))
        x0 = max(0, u - int(window_radius))
        x1 = min(width, u + int(window_radius) + 1)
        y0 = max(0, v - int(window_radius))
        y1 = min(height, v + int(window_radius) + 1)
        patch = depth[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        valid = patch[patch > 0]
        if valid.size == 0:
            continue
        valid_samples.append(float(np.median(valid)))

    if not valid_samples:
        return None
    return float(np.median(np.asarray(valid_samples, dtype=np.float64)))


def _draw_hand_observation(
    image_bgr: np.ndarray,
    observation: HandObservation,
    *,
    label: str,
) -> np.ndarray:
    out = image_bgr.copy()
    x0, y0, x1, y1 = observation.bbox_xyxy
    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 220, 255), 2)

    text = label
    if observation.median_depth_mm is not None:
        text += f"  z={observation.median_depth_mm / 1000.0:.3f}m"
    cv2.putText(
        out,
        text,
        (x0, max(22, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _compose_display_frame(
    color_bgr: np.ndarray,
    depth_bgr: np.ndarray,
    *,
    view: str,
) -> np.ndarray:
    if view == "color":
        return color_bgr
    if view == "depth":
        return depth_bgr
    if view != "split":
        raise ValueError(f"Unsupported view: {view}")
    return np.hstack([color_bgr, depth_bgr])


def _draw_status_lines(
    image_bgr: np.ndarray,
    lines: list[str],
) -> np.ndarray:
    out = image_bgr.copy()
    line_height = 24
    box_height = max(36, 12 + line_height * len(lines))
    cv2.rectangle(out, (8, 8), (560, box_height), (15, 15, 15), -1)
    cv2.rectangle(out, (8, 8), (560, box_height), (90, 90, 90), 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (18, 32 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return out


def _make_observation(
    keypoints_2d: np.ndarray,
    confidence: np.ndarray,
    color_shape: tuple[int, ...],
    depth_mm: np.ndarray,
) -> HandObservation:
    return HandObservation(
        keypoints_2d=np.asarray(keypoints_2d, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        bbox_xyxy=_estimate_bbox_xyxy(keypoints_2d, color_shape),
        median_depth_mm=_sample_landmark_depth_mm(depth_mm, keypoints_2d),
    )


def _build_parser() -> argparse.ArgumentParser:
    camera = camera_communication("primary")
    camera_intrinsics = json.loads(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Realtime RealSense D435i + MediaPipe hand landmark preview."
    )
    parser.add_argument("--camera-serial", default=resolve_realsense_serial("primary"))
    parser.add_argument("--width", type=int, default=int(camera_intrinsics["image_width"]), help="Color/depth stream width.")
    parser.add_argument("--height", type=int, default=int(camera_intrinsics["image_height"]), help="Color/depth stream height.")
    parser.add_argument("--fps", type=int, default=int(camera_intrinsics["fps"]), help="Requested camera FPS.")
    parser.add_argument(
        "--view",
        choices=VIEW_CHOICES,
        default="split",
        help="Display color, depth, or a side-by-side split view.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence.",
    )
    parser.add_argument(
        "--max-num-hands",
        type=int,
        default=1,
        help="Maximum number of hands passed to MediaPipe Hands.",
    )
    parser.add_argument(
        "--depth-alpha",
        type=float,
        default=0.55,
        help="Blend ratio when drawing landmarks over colorized depth.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if rs is None:
        raise SystemExit("pyrealsense2 is required. Install librealsense Python bindings first.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(args.camera_serial))
    config.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, int(args.fps))
    config.enable_stream(rs.stream.depth, int(args.width), int(args.height), rs.format.z16, int(args.fps))

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    detector = LandmarkDetector(
        static_image_mode=False,
        max_num_hands=int(args.max_num_hands),
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    )

    window_name = "DexSlide RealSense Hand Preview"
    last_frame_time = time.perf_counter()
    smoothed_fps = 0.0
    frame_idx = 0
    pipeline_started = False

    try:
        pipeline.start(config)
        pipeline_started = True
        print(
            "[preview] controls: q / ESC quit, c=color, d=depth, s=split. "
            "MediaPipe runs on RGB, depth only for visualization and landmark depth readout."
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
            depth_mm = np.asanyarray(depth_frame.get_data())
            depth_bgr = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            detection = detector.detect(color_bgr)
            color_overlay = color_bgr.copy()
            depth_overlay = cv2.addWeighted(color_bgr, 1.0 - float(args.depth_alpha), depth_bgr, float(args.depth_alpha), 0.0)

            hand_found = detection is not None
            median_depth_mm: float | None = None
            mean_conf: float | None = None
            if detection is not None:
                keypoints_2d, confidence = detection
                observation = _make_observation(
                    keypoints_2d=keypoints_2d,
                    confidence=confidence,
                    color_shape=color_bgr.shape,
                    depth_mm=depth_mm,
                )
                median_depth_mm = observation.median_depth_mm
                mean_conf = float(np.mean(observation.confidence)) if observation.confidence.size else None
                color_overlay = detector.draw_landmarks(color_overlay, observation.keypoints_2d, observation.confidence)
                depth_overlay = detector.draw_landmarks(depth_overlay, observation.keypoints_2d, observation.confidence)
                color_overlay = _draw_hand_observation(color_overlay, observation, label="hand")
                depth_overlay = _draw_hand_observation(depth_overlay, observation, label="hand")

            now = time.perf_counter()
            dt = max(now - last_frame_time, 1e-6)
            instant_fps = 1.0 / dt
            smoothed_fps = instant_fps if frame_idx == 0 else (0.85 * smoothed_fps + 0.15 * instant_fps)
            last_frame_time = now
            frame_idx += 1

            depth_valid_ratio = float(np.count_nonzero(depth_mm)) / float(max(1, depth_mm.size))
            status_lines = [
                f"view={args.view}  fps={smoothed_fps:.1f}  frame={frame_idx}",
                f"hand_detected={hand_found}  mean_conf={mean_conf if mean_conf is not None else float('nan'):.3f}",
                f"landmark_depth_m={median_depth_mm / 1000.0 if median_depth_mm is not None else float('nan'):.3f}  depth_valid_ratio={depth_valid_ratio:.3f}",
                f"resolution={color_bgr.shape[1]}x{color_bgr.shape[0]}",
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
    finally:
        detector.close()
        if pipeline_started:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
