"""OpenCV capture backend selection and probing."""

from __future__ import annotations

import time
from typing import Any

import cv2

from dexslide.vision.aruco_pose_tracker import _parse_capture_source


def capture_backend_attempts(source: int | str) -> list[tuple[str, int | None]]:
    if isinstance(source, str) and source.startswith("/dev/"):
        return [("v4l2", cv2.CAP_V4L2), ("default", None)]
    if isinstance(source, int):
        return [("default", None), ("v4l2", cv2.CAP_V4L2)]
    return [("default", None)]


def configure_capture_device(cap: cv2.VideoCapture, *, width: int | None, height: int | None, fps: float | None, buffer_size: int) -> None:
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))


def open_capture_with_fallback(*, source: str | int, width: int | None, height: int | None, fps: float | None, buffer_size: int, purpose: str) -> tuple[cv2.VideoCapture, int | str]:
    requested = _parse_capture_source(source)
    tried: list[str] = []
    for candidate in [requested]:
        for backend_label, backend in capture_backend_attempts(candidate):
            tried.append(f"{candidate}[{backend_label}]")
            cap = cv2.VideoCapture(candidate) if backend is None else cv2.VideoCapture(candidate, backend)
            configure_capture_device(cap, width=width, height=height, fps=fps, buffer_size=buffer_size)
            if not cap.isOpened():
                cap.release()
                continue
            frame_shape: tuple[int, ...] | None = None
            for _ in range(5):
                ok, frame_bgr = cap.read()
                if ok and frame_bgr is not None:
                    frame_shape = tuple(int(value) for value in frame_bgr.shape)
                    break
                time.sleep(0.03)
            if frame_shape is None:
                cap.release()
                continue
            shape_desc = "x".join(str(value) for value in frame_shape)
            print(f"[camera] requested={requested} selected={candidate} backend={backend_label} frame_shape={shape_desc}")
            return cap, candidate
    tried_desc = ", ".join(tried) if tried else "<none>"
    raise RuntimeError(f"Failed to open capture source for {purpose}: requested={requested}, tried=[{tried_desc}]")

