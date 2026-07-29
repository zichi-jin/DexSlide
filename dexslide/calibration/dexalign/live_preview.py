from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


@dataclass(frozen=True)
class CameraPreviewProcess:
    frame_queue: mp.Queue
    command_queue: mp.Queue
    stop_event: mp.Event
    process: mp.Process


def _estimate_bbox_xyxy(
    keypoints_2d: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    padding_px: int = 24,
) -> tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    points = np.asarray(keypoints_2d, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return (0, 0, max(0, width - 1), max(0, height - 1))

    x0 = int(np.floor(float(np.min(points[:, 0])))) - int(padding_px)
    y0 = int(np.floor(float(np.min(points[:, 1])))) - int(padding_px)
    x1 = int(np.ceil(float(np.max(points[:, 0])))) + int(padding_px)
    y1 = int(np.ceil(float(np.max(points[:, 1])))) + int(padding_px)
    x0 = int(np.clip(x0, 0, max(0, width - 1)))
    y0 = int(np.clip(y0, 0, max(0, height - 1)))
    x1 = int(np.clip(x1, 0, max(0, width - 1)))
    y1 = int(np.clip(y1, 0, max(0, height - 1)))
    return (x0, y0, x1, y1)


def _draw_status_lines(
    image_bgr: np.ndarray,
    lines: list[str],
    *,
    max_box_width: int = 620,
) -> np.ndarray:
    output = image_bgr.copy()
    image_height, image_width = output.shape[:2]
    line_height = 21
    estimated_width = max((len(line) for line in lines), default=1) * 8 + 20
    box_width = min(max(220, estimated_width), int(max_box_width), max(40, image_width - 16))
    box_height = min(max(34, 10 + line_height * len(lines)), max(34, image_height - 16))
    x0 = max(8, image_width - box_width - 8)
    y0 = 8
    x1 = min(image_width - 8, x0 + box_width)
    y1 = min(image_height - 8, y0 + box_height)
    cv2.rectangle(output, (x0, y0), (x1, y1), (15, 15, 15), -1)
    cv2.rectangle(output, (x0, y0), (x1, y1), (90, 90, 90), 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (x0 + 10, y0 + 23 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return output


def _compose_camera_overlay(
    color_bgr: np.ndarray,
    *,
    detector: Any,
    detection: tuple[np.ndarray, np.ndarray] | None,
    marker_pose: Any | None,
    valid_mask: np.ndarray,
    depth_mm: np.ndarray,
    fps_ema: float,
    frame_idx: int,
    extra_status_lines: list[str] | None = None,
) -> np.ndarray:
    overlay = color_bgr.copy()
    hand_conf = float("nan")
    if detection is not None:
        keypoints_2d, confidence = detection
        overlay = detector.draw_landmarks(overlay, keypoints_2d, confidence)
        bbox = _estimate_bbox_xyxy(keypoints_2d, overlay.shape)
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (80, 220, 255), 2)
        hand_conf = float(np.mean(confidence)) if np.asarray(confidence).size else float("nan")

    if marker_pose is None:
        marker_line = "marker=NA"
    else:
        marker_t_m = 0.001 * np.asarray(marker_pose.camera_T_marker_mm[:3, 3], dtype=np.float64).reshape(3)
        marker_line = (
            f"marker=OK ids={list(marker_pose.marker_ids_used)} "
            f"t_m=[{marker_t_m[0]:+.3f}, {marker_t_m[1]:+.3f}, {marker_t_m[2]:+.3f}] "
            f"reproj={0.0 if marker_pose.marker_reproj_error_px is None else float(marker_pose.marker_reproj_error_px):.2f}px"
        )

    status_lines = [
        (
            f"frame={frame_idx} fps={fps_ema:.1f} hand_conf={hand_conf:.3f}"
            if np.isfinite(hand_conf)
            else f"frame={frame_idx} fps={fps_ema:.1f} hand_conf=nan"
        ),
        marker_line,
    ]
    if extra_status_lines:
        status_lines.extend(str(line) for line in extra_status_lines)
    status_lines.append("SPACE action | q/ESC quit")
    return _draw_status_lines(overlay, status_lines)


def _camera_preview_worker(
    frame_queue: mp.Queue,
    command_queue: mp.Queue,
    stop_event: mp.Event,
    *,
    window_name: str,
) -> None:
    latest_frame: np.ndarray | None = None
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while not stop_event.is_set():
            try:
                while True:
                    item = frame_queue.get_nowait()
                    if item is None:
                        stop_event.set()
                        return
                    latest_frame = np.asarray(item, dtype=np.uint8)
            except queue.Empty:
                pass

            if latest_frame is not None:
                cv2.imshow(window_name, latest_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop_event.set()
                return
            if key == ord(" "):
                try:
                    command_queue.put_nowait("space")
                except queue.Full:
                    pass
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                stop_event.set()
                return
            time.sleep(0.005)
    finally:
        cv2.destroyAllWindows()


def start_camera_preview_process(window_name: str) -> CameraPreviewProcess:
    frame_queue: mp.Queue = mp.Queue(maxsize=1)
    command_queue: mp.Queue = mp.Queue(maxsize=8)
    stop_event = mp.Event()
    process = mp.Process(
        target=_camera_preview_worker,
        args=(frame_queue, command_queue, stop_event),
        kwargs={"window_name": str(window_name)},
        daemon=True,
    )
    process.start()
    return CameraPreviewProcess(
        frame_queue=frame_queue,
        command_queue=command_queue,
        stop_event=stop_event,
        process=process,
    )


def enqueue_camera_preview_frame(preview: CameraPreviewProcess, image_bgr: np.ndarray) -> None:
    frame = np.ascontiguousarray(np.asarray(image_bgr, dtype=np.uint8))
    try:
        while True:
            preview.frame_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        preview.frame_queue.put_nowait(frame)
    except queue.Full:
        pass


def drain_camera_preview_commands(preview: CameraPreviewProcess) -> list[str]:
    commands: list[str] = []
    try:
        while True:
            commands.append(str(preview.command_queue.get_nowait()))
    except queue.Empty:
        return commands


def stop_camera_preview_process(preview: CameraPreviewProcess) -> None:
    preview.stop_event.set()
    try:
        preview.frame_queue.put_nowait(None)
    except queue.Full:
        try:
            preview.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            preview.frame_queue.put_nowait(None)
        except queue.Full:
            pass
    if preview.process.is_alive():
        preview.process.join(timeout=1.5)
    if preview.process.is_alive():
        preview.process.terminate()
        preview.process.join(timeout=1.0)


def _draw_axes(ax: Any, transform_m: np.ndarray, *, label: str, length_m: float, colors: tuple[str, str, str]) -> None:
    transform = np.asarray(transform_m, dtype=np.float64).reshape(4, 4)
    origin = transform[:3, 3]
    basis = transform[:3, :3]
    for axis_idx, color in enumerate(colors):
        endpoint = origin + length_m * basis[:, axis_idx]
        ax.plot([origin[0], endpoint[0]], [origin[1], endpoint[1]], [origin[2], endpoint[2]], color=color, linewidth=2.0)
    ax.text(origin[0], origin[1], origin[2], label, fontsize=10)


def _draw_hand(ax: Any, keypoints_m: np.ndarray, valid_mask: np.ndarray) -> None:
    points = np.asarray(keypoints_m, dtype=np.float64).reshape(21, 3)
    valid = np.asarray(valid_mask, dtype=bool).reshape(21)
    visible = np.where(valid[:, None], points, np.nan)
    ax.scatter(visible[:, 0], visible[:, 1], visible[:, 2], color="#2563eb", s=18)
    for parent_idx, child_idx in HAND_CONNECTIONS:
        segment = visible[[parent_idx, child_idx]]
        if not np.isfinite(segment).all():
            continue
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#2563eb", linewidth=1.5, alpha=0.8)


def _set_axes_equal(ax: Any, reference_points: np.ndarray) -> None:
    points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.65, 0.18)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
