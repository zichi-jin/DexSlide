"""Geometric disambiguation for camera-facing ArUco pose candidates.

For this project a candidate is admissible when the visible marker face is
camera-facing under OpenCV camera_T_aruco convention::

    dot(R_camera_aruco[:, 2], t_camera_aruco) < 0

Here ``t_camera_aruco`` is the camera-origin-to-marker-origin vector.  A
different pose convention can override the threshold with ``max_cosine``.

This module deliberately does not choose between multiple admissible IPPE
solutions.  It only applies the physical camera-facing constraint; temporal or
multi-marker selection belongs to the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dexslide.kinematics.transforms import rvec_tvec_to_transform


def aruco_pose_transform(pose: Any) -> np.ndarray:
    """Normalize a pose candidate to a finite 4x4 ``camera_T_aruco`` matrix."""

    if isinstance(pose, Mapping):
        if "transform" in pose:
            pose = pose["transform"]
        elif "matrix" in pose:
            pose = pose["matrix"]
        elif "rvec" in pose and "tvec" in pose:
            return rvec_tvec_to_transform(pose["rvec"], pose["tvec"])
    transform = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(transform).all():
        raise ValueError("ArUco pose candidate contains non-finite values")
    return transform


def camera_facing_cosine(camera_T_aruco: Any) -> float:
    """Return the direction cosine between ArUco ``+Z`` and ``t_camera_aruco``."""

    transform = aruco_pose_transform(camera_T_aruco)
    z_axis_camera = np.asarray(transform[:3, 2], dtype=np.float64).reshape(3)
    marker_position_camera = np.asarray(transform[:3, 3], dtype=np.float64).reshape(3)
    z_norm = float(np.linalg.norm(z_axis_camera))
    position_norm = float(np.linalg.norm(marker_position_camera))
    if z_norm <= 1e-12 or position_norm <= 1e-12:
        raise ValueError("Cannot compute ArUco camera-facing cosine for a zero-length vector")
    return float(np.dot(z_axis_camera, marker_position_camera) / (z_norm * position_norm))


def filter_camera_facing_aruco_pose_candidates(
    candidates: Sequence[Any],
    *,
    max_cosine: float = 0.0,
) -> list[Any]:
    """Keep only candidates satisfying ``dot(R[:, 2], t) < max_cosine``.

    The default uses a strict camera-facing test: ``cosine < 0``.  Candidates
    failing to parse or failing the geometric constraint are rejected.  An
    empty result is intentional and must not silently fall back to another
    PnP branch.
    """

    threshold = float(max_cosine)
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("max_cosine must satisfy -1 <= max_cosine <= 1")
    accepted: list[Any] = []
    for candidate in candidates:
        try:
            cosine = camera_facing_cosine(candidate)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue
        if cosine < threshold:
            accepted.append(candidate)
    return accepted


__all__ = [
    "aruco_pose_transform",
    "camera_facing_cosine",
    "filter_camera_facing_aruco_pose_candidates",
]
