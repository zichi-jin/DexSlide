"""Rigid transform helpers shared by calibration, vision, and teleop code."""

from __future__ import annotations

import cv2
import numpy as np


def normalize_quaternion_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quaternion_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quaternion_xyzw(rot: np.ndarray) -> np.ndarray:
    mat = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (mat[2, 1] - mat[1, 2]) / s
        qy = (mat[0, 2] - mat[2, 0]) / s
        qz = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        qw = (mat[2, 1] - mat[1, 2]) / s
        qx = 0.25 * s
        qy = (mat[0, 1] + mat[1, 0]) / s
        qz = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        qw = (mat[0, 2] - mat[2, 0]) / s
        qx = (mat[0, 1] + mat[1, 0]) / s
        qy = 0.25 * s
        qz = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        qw = (mat[1, 0] - mat[0, 1]) / s
        qx = (mat[0, 2] + mat[2, 0]) / s
        qy = (mat[1, 2] + mat[2, 1]) / s
        qz = 0.25 * s
    return normalize_quaternion_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))


def slerp_quaternion_xyzw(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray, alpha: float) -> np.ndarray:
    qa = normalize_quaternion_xyzw(q1_xyzw)
    qb = normalize_quaternion_xyzw(q2_xyzw)

    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot

    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        blended = (1.0 - alpha_clamped) * qa + alpha_clamped * qb
        return normalize_quaternion_xyzw(blended)

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha_clamped
    scale_a = float(np.sin(theta_0 - theta) / max(sin_theta_0, 1e-12))
    scale_b = float(np.sin(theta) / max(sin_theta_0, 1e-12))
    return normalize_quaternion_xyzw(scale_a * qa + scale_b * qb)


def slerp_rotation_matrices(rot_a: np.ndarray, rot_b: np.ndarray, alpha: float) -> np.ndarray:
    return quaternion_xyzw_to_rotmat(
        slerp_quaternion_xyzw(
            rotmat_to_quaternion_xyzw(rot_a),
            rotmat_to_quaternion_xyzw(rot_b),
            alpha,
        )
    )


def make_transform(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(trans, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ trans)
    return inv


def transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return (pts @ mat[:3, :3].T) + mat[:3, 3][None, :]


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rot = rotvec_to_rotmat(rvec)
    return make_transform(rot, np.asarray(tvec, dtype=np.float64).reshape(3))


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return np.asarray(rot, dtype=np.float64).reshape(3, 3)


def transform_to_rvec_tvec(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(mat[:3, :3])
    return rvec.reshape(3), mat[:3, 3].reshape(3)


def relative_rotvec(rot_from: np.ndarray, rot_to: np.ndarray) -> np.ndarray:
    relative_rot = np.asarray(rot_from, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        rot_to,
        dtype=np.float64,
    ).reshape(3, 3)
    rotvec, _ = cv2.Rodrigues(relative_rot)
    return np.asarray(rotvec, dtype=np.float64).reshape(3)


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= max(float(max_norm), 1e-12):
        return vec
    return vec * (float(max_norm) / norm)


def project_rotation_to_so3(rot: np.ndarray) -> np.ndarray:
    u, _s, vh = np.linalg.svd(np.asarray(rot, dtype=np.float64).reshape(3, 3))
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return np.asarray(projected, dtype=np.float64).reshape(3, 3)


def average_rotation_matrices(
    rotations: list[np.ndarray] | tuple[np.ndarray, ...],
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    if not rotations:
        return np.eye(3, dtype=np.float64)
    if weights is None:
        weights_arr = np.ones(len(rotations), dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights_arr.shape[0] != len(rotations):
            raise ValueError(
                f"Expected {len(rotations)} rotation weights, got {weights_arr.shape[0]}"
            )
        weights_arr = np.maximum(weights_arr, 0.0)
        if float(np.sum(weights_arr)) <= 1e-12:
            weights_arr = np.ones(len(rotations), dtype=np.float64)
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for rot, weight in zip(rotations, weights_arr):
        accumulator += float(weight) * np.asarray(rot, dtype=np.float64).reshape(3, 3)
    return project_rotation_to_so3(accumulator)


def average_transforms(
    transforms: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray | None:
    arr = np.asarray(transforms, dtype=np.float64)
    if arr.size == 0:
        return None
    arr = arr.reshape(-1, 4, 4)
    if weights is None:
        weights_arr = np.ones(arr.shape[0], dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights_arr.shape[0] != arr.shape[0]:
            raise ValueError(
                f"Expected {arr.shape[0]} transform weights, got {weights_arr.shape[0]}"
            )
    weights_arr = np.maximum(weights_arr, 0.0)
    if float(np.sum(weights_arr)) <= 1e-12:
        weights_arr = np.ones(arr.shape[0], dtype=np.float64)
    weights_arr = weights_arr / float(np.sum(weights_arr))
    mean_translation = np.sum(arr[:, :3, 3] * weights_arr[:, None], axis=0)
    mean_rotation = average_rotation_matrices(list(arr[:, :3, :3]), weights_arr)
    return make_transform(mean_rotation, mean_translation)
