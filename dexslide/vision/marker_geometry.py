"""Shared geometric primitives for mounted ArUco marker bodies."""

from __future__ import annotations

import numpy as np


def marker_square_object_points(marker_size_m: float) -> np.ndarray:
    half_size = 0.5 * float(marker_size_m)
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    )


def normalize_marker_axes_rows(rows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64).reshape(3, 3)
    out = matrix.copy()
    for idx in range(3):
        norm = float(np.linalg.norm(out[idx]))
        if norm < 1e-9:
            raise ValueError("Marker axes matrix contains a near-zero row.")
        out[idx] /= norm
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if abs(float(np.dot(out[first], out[second]))) > 5e-3:
            raise ValueError("Marker axes must be mutually orthogonal.")
    if float(np.dot(np.cross(out[0], out[1]), out[2])) < 0.995:
        raise ValueError("Marker axes must be right-handed and satisfy x × y = z.")
    return out

