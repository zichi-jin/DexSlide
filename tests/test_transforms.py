from __future__ import annotations

import math

import numpy as np

from dexslide.kinematics.transforms import (
    average_transforms,
    clip_norm,
    invert_transform,
    make_transform,
    quaternion_xyzw_to_rotmat,
    rotmat_to_quaternion_xyzw,
    rotvec_to_rotmat,
    rvec_tvec_to_transform,
    slerp_rotation_matrices,
    transform_points,
    transform_to_rvec_tvec,
)


def _rot_z(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_transform_roundtrip_and_point_application() -> None:
    transform = make_transform(
        rotvec_to_rotmat(np.array([0.2, -0.3, 0.4], dtype=np.float64)),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )
    inverse = invert_transform(transform)

    np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1e-9)
    np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-9)

    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    expected = (points @ transform[:3, :3].T) + transform[:3, 3][None, :]
    np.testing.assert_allclose(transform_points(transform, points), expected, atol=1e-9)


def test_rvec_tvec_transform_conversion_roundtrip() -> None:
    rvec = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    tvec = np.array([0.4, -0.5, 0.6], dtype=np.float64)
    transform = rvec_tvec_to_transform(rvec, tvec)
    recovered_rvec, recovered_tvec = transform_to_rvec_tvec(transform)

    np.testing.assert_allclose(recovered_rvec, rvec, atol=1e-9)
    np.testing.assert_allclose(recovered_tvec, tvec, atol=1e-9)


def test_quaternion_conversion_and_slerp_rotation() -> None:
    rot = _rot_z(math.pi / 2.0)
    quat = rotmat_to_quaternion_xyzw(rot)

    np.testing.assert_allclose(quaternion_xyzw_to_rotmat(quat), rot, atol=1e-9)
    np.testing.assert_allclose(
        slerp_rotation_matrices(np.eye(3, dtype=np.float64), rot, 0.5),
        _rot_z(math.pi / 4.0),
        atol=1e-9,
    )


def test_average_transforms_and_clip_norm() -> None:
    first = make_transform(np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 0.0]))
    second = make_transform(np.eye(3, dtype=np.float64), np.array([2.0, 4.0, 6.0]))
    averaged = average_transforms([first, second], [0.25, 0.75])

    assert averaged is not None
    np.testing.assert_allclose(averaged[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(averaged[:3, 3], np.array([1.5, 3.0, 4.5]), atol=1e-9)
    np.testing.assert_allclose(clip_norm(np.array([3.0, 4.0]), 2.0), np.array([1.2, 1.6]), atol=1e-9)
