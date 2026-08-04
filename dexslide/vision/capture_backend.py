"""Backward-compatible imports for the camera capture backend.

New code should import from :mod:`dexslide.vision.camera.capture_backend`.
"""

from dexslide.vision.camera.capture_backend import (
    capture_backend_attempts,
    configure_capture_device,
    open_capture_with_fallback,
)

__all__ = [
    "capture_backend_attempts",
    "configure_capture_device",
    "open_capture_with_fallback",
]
