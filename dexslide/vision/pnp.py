"""Shared OpenCV PnP and reprojection helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dexslide.kinematics.transforms import rvec_tvec_to_transform, transform_to_rvec_tvec
from dexslide.vision.marker_geometry import marker_square_object_points


def solve_pnp_transform(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
    initial_transform: np.ndarray | None = None,
    flags: int | None = None,
) -> np.ndarray | None:
    object_array = np.ascontiguousarray(np.asarray(object_points, dtype=np.float64).reshape(-1, 3))
    image_array = np.ascontiguousarray(np.asarray(image_points, dtype=np.float64).reshape(-1, 2))
    camera = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3))
    dist = np.zeros((1, 5), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    rvec = tvec = None
    if initial_transform is not None:
        rvec, tvec = transform_to_rvec_tvec(initial_transform)
        rvec = rvec.reshape(3, 1)
        tvec = tvec.reshape(3, 1)
    solve_flags = cv2.SOLVEPNP_ITERATIVE if flags is None else int(flags)
    try:
        success, solved_rvec, solved_tvec = cv2.solvePnP(
            object_array,
            image_array,
            camera,
            dist,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=initial_transform is not None,
            flags=solve_flags,
        )
    except cv2.error:
        return None
    if not success:
        return None
    return rvec_tvec_to_transform(solved_rvec, solved_tvec)


def reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    transform_camera_object: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    rvec, tvec = transform_to_rvec_tvec(transform_camera_object)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.zeros((1, 5), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64),
    )
    return np.linalg.norm(projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2), axis=1)


__all__ = ["marker_square_object_points", "reprojection_errors", "solve_pnp_transform"]

