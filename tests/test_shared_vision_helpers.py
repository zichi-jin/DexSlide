from __future__ import annotations

import numpy as np
import pytest

from dexslide.vision.deprojection import sample_depth_m
from dexslide.vision.marker_geometry import marker_square_object_points, normalize_marker_axes_rows
from dexslide.vision.pnp import reprojection_errors


def test_marker_geometry_is_canonical_and_right_handed() -> None:
    points = marker_square_object_points(0.04)
    assert points.shape == (4, 3)
    axes = normalize_marker_axes_rows(np.eye(3))
    np.testing.assert_allclose(axes, np.eye(3))


def test_sample_depth_uses_valid_median() -> None:
    depth = np.array([[0.0, 1.0, 1.2], [0.9, 1.1, 0.0]], dtype=np.float32)
    assert sample_depth_m(depth, 1, 0, window=1) == pytest.approx(1.05)
    assert sample_depth_m(depth, -1, 0) is None


def test_reprojection_errors_returns_per_point_residuals() -> None:
    points = marker_square_object_points(0.04)
    transform = np.eye(4)
    camera = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    points[:, 2] = 1.0
    projected = np.column_stack((500.0 * points[:, 0] + 320.0, 500.0 * points[:, 1] + 240.0))
    errors = reprojection_errors(points, projected, transform, camera)
    np.testing.assert_allclose(errors, 0.0, atol=1e-8)
