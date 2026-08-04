"""Camera-device adapter used by the vision pipeline.

All device-specific behavior lives here: OpenCV backend selection, requested
mode, and FOURCC.  Frames are returned exactly as supplied by the device.
"""

from __future__ import annotations

import time
from typing import Any

import cv2

from dexslide.vision.camera.capture_backend import open_capture_with_fallback


class CameraStream:
    """Open one camera device and expose its configured image stream."""

    def __init__(
        self,
        *,
        source: str | int,
        width: int | None,
        height: int | None,
        fps: float | None,
        buffer_size: int = 2,
        fourcc: str | None = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = int(buffer_size)
        self.fourcc = fourcc
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> "CameraStream":
        if self._capture is None:
            self._capture, _selected = open_capture_with_fallback(
                source=self.source,
                width=self.width,
                height=self.height,
                fps=self.fps,
                buffer_size=self.buffer_size,
                purpose="DexSlide camera stream",
                fourcc=self.fourcc,
            )
        return self

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self) -> tuple[float, Any]:
        if self._capture is None:
            raise RuntimeError("CameraStream has not been started")
        for _attempt in range(5):
            ok, frame_bgr = self._capture.read()
            timestamp = time.time()
            if ok and frame_bgr is not None:
                return timestamp, frame_bgr
            time.sleep(0.01)
        raise RuntimeError("Failed to read a camera frame")

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
