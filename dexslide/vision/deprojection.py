"""Depth and pixel deprojection helpers shared by calibration workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_depth_m(depth_image: np.ndarray, u: float, v: float, *, window: int = 2) -> float | None:
    image = np.asarray(depth_image)
    height, width = image.shape[:2]
    center_u = int(round(float(u)))
    center_v = int(round(float(v)))
    if not (0 <= center_u < width and 0 <= center_v < height):
        return None
    radius = max(0, int(window))
    patch = image[
        max(0, center_v - radius): min(height, center_v + radius + 1),
        max(0, center_u - radius): min(width, center_u + radius + 1),
    ]
    values = np.asarray(patch, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def deproject_pixel_realsense(intrinsics: Any, u: float, v: float, depth_m: float, *, rs_module: Any) -> np.ndarray:
    point = rs_module.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(u), float(v)],
        float(depth_m),
    )
    return np.asarray(point, dtype=np.float64).reshape(3)

