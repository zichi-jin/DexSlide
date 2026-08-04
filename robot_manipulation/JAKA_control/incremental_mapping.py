"""Deterministic table-frame to JAKA incremental motion mathematics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from dexslide.kinematics.transforms import make_transform

from .workspace_mapping import WorkspaceAxisMapping


@dataclass(frozen=True)
class TeleopAnchorState:
    glove_anchor_translation_m: np.ndarray
    robot_anchor_translation_mm: np.ndarray
    previous_desired_robot_transform: np.ndarray
    anchor_frame_idx: int


def rotation_y_deg(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotation_z_deg(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def shortest_angle_delta_rad(current: float, target: float) -> float:
    return float((current - target + math.pi) % (2.0 * math.pi) - math.pi)


def rpy_to_rotation_matrix(rpy_rad: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(value) for value in np.asarray(rpy_rad, dtype=np.float64).reshape(3)]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = max(-1.0, min(1.0, float(-rot[2, 0])))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        rz = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        rx = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        rz = 0.0
    return rx, ry, rz


def pose_mmrad_to_transform(pose: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return make_transform(rpy_to_rotation_matrix(pose[3:6]), np.asarray(pose[:3], dtype=np.float64))


def transform_to_pose_mmrad(transform: np.ndarray) -> tuple[float, float, float, float, float, float]:
    mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = rotation_matrix_to_rpy(mat[:3, :3])
    translation = tuple(float(value) for value in mat[:3, 3])
    return (*translation, *rotation)


def pose_is_close(
    current_pose: tuple[float, float, float, float, float, float],
    target_pose: tuple[float, float, float, float, float, float],
    *,
    position_tol_mm: float,
    angle_tol_deg: float,
) -> bool:
    angle_tol_rad = math.radians(float(angle_tol_deg))
    position_ok = all(abs(current_pose[idx] - target_pose[idx]) <= position_tol_mm for idx in range(3))
    rotation_ok = all(
        abs(shortest_angle_delta_rad(current_pose[idx], target_pose[idx])) <= angle_tol_rad
        for idx in range(3, 6)
    )
    return position_ok and rotation_ok


def make_safe_start_pose_sdk(mmdeg_pose: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
    return (*[float(value) for value in mmdeg_pose[:3]], *[math.radians(float(value)) for value in mmdeg_pose[3:]])


def clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= max(float(max_norm), 1e-12):
        return vec
    return vec * (float(max_norm) / norm)


def compute_transform_delta(previous_transform: np.ndarray, current_transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous = np.asarray(previous_transform, dtype=np.float64).reshape(4, 4)
    current = np.asarray(current_transform, dtype=np.float64).reshape(4, 4)
    delta_translation = current[:3, 3] - previous[:3, 3]
    delta_rotation = current[:3, :3] @ previous[:3, :3].T
    delta_rotvec, _ = cv2.Rodrigues(delta_rotation)
    return delta_translation, np.asarray(delta_rotvec, dtype=np.float64).reshape(3)


def reflection_matrix_from_mapping(mapping: WorkspaceAxisMapping) -> np.ndarray | None:
    signs = np.asarray(mapping.mirror_translation_sign, dtype=np.float64).reshape(3)
    if not np.allclose(np.abs(signs), np.ones(3), atol=1e-9):
        return None
    reflection = np.diag(signs)
    expected = float(np.linalg.det(reflection)) * signs
    if not np.allclose(mapping.mirror_rotation_sign, expected, atol=1e-9):
        return None
    return reflection


def map_translation_delta_to_robot_mm(delta_translation_table_m: np.ndarray, mapping: WorkspaceAxisMapping) -> np.ndarray:
    delta = np.asarray(delta_translation_table_m, dtype=np.float64).reshape(3)
    mapped = mapping.mirror_translation_sign * (mapping.translation_scale * delta)
    return 1000.0 * (mapping.robot_from_table_rotation @ mapped)


def map_rotation_delta_to_robot_rad(delta_rotation_table_rad: np.ndarray, mapping: WorkspaceAxisMapping) -> np.ndarray:
    delta = np.asarray(delta_rotation_table_rad, dtype=np.float64).reshape(3)
    mapped = mapping.mirror_rotation_sign * (mapping.rotation_scale * delta)
    return mapping.robot_from_table_rotation @ mapped


def map_rotation_delta_matrix_to_robot(delta_rotation_table: np.ndarray, mapping: WorkspaceAxisMapping) -> np.ndarray:
    reflection = reflection_matrix_from_mapping(mapping)
    delta = np.asarray(delta_rotation_table, dtype=np.float64).reshape(3, 3)
    if reflection is not None and np.allclose(mapping.rotation_scale, np.ones(3), atol=1e-9):
        mirrored = reflection @ delta @ reflection
        return mapping.robot_from_table_rotation @ mirrored @ mapping.robot_from_table_rotation.T
    delta_rotvec, _ = cv2.Rodrigues(delta)
    mapped_rotvec = map_rotation_delta_to_robot_rad(delta_rotvec.reshape(3), mapping)
    mapped_rotation, _ = cv2.Rodrigues(mapped_rotvec.reshape(3, 1))
    return np.asarray(mapped_rotation, dtype=np.float64).reshape(3, 3)


def clip_translation_to_workspace_mm(
    translation_mm: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> tuple[np.ndarray, bool]:
    translation = np.asarray(translation_mm, dtype=np.float64).reshape(3)
    if not np.isfinite(translation).all():
        raise ValueError("TCP translation contains non-finite values")
    clipped = np.clip(
        translation,
        mapping.teleop_workspace_min_mm,
        mapping.teleop_workspace_max_mm,
    )
    return clipped, not np.allclose(clipped, translation, atol=1e-9)


def build_servo_increment_from_robot_delta(
    delta_translation_robot_mm: np.ndarray,
    delta_rotation_robot_rad: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    translation = np.asarray(delta_translation_robot_mm, dtype=np.float64).reshape(3)
    rotation = np.asarray(delta_rotation_robot_rad, dtype=np.float64).reshape(3)
    translation = (
        np.zeros(3, dtype=np.float64)
        if float(np.linalg.norm(translation)) < mapping.translation_deadband_mm
        else clamp_norm(translation, mapping.max_translation_step_mm)
    )
    rotation = (
        np.zeros(3, dtype=np.float64)
        if math.degrees(float(np.linalg.norm(rotation))) < mapping.rotation_deadband_deg
        else clamp_norm(rotation, math.radians(mapping.max_rotation_step_deg))
    )
    return np.concatenate([translation, rotation])


def build_servo_increment(
    delta_translation_table_m: np.ndarray,
    delta_rotation_table_rad: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    return build_servo_increment_from_robot_delta(
        map_translation_delta_to_robot_mm(delta_translation_table_m, mapping),
        map_rotation_delta_to_robot_rad(delta_rotation_table_rad, mapping),
        mapping,
    )


def build_desired_robot_transform(
    current_glove_transform: np.ndarray,
    anchor_state: TeleopAnchorState,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    current_glove = np.asarray(current_glove_transform, dtype=np.float64).reshape(4, 4)
    glove_anchor_translation = np.asarray(
        anchor_state.glove_anchor_translation_m,
        dtype=np.float64,
    ).reshape(3)
    robot_anchor_translation = np.asarray(
        anchor_state.robot_anchor_translation_mm,
        dtype=np.float64,
    ).reshape(3)
    translation = robot_anchor_translation + map_translation_delta_to_robot_mm(
        current_glove[:3, 3] - glove_anchor_translation,
        mapping,
    )
    rotation = map_absolute_rotation_matrix_to_robot(current_glove[:3, :3], mapping)
    return make_transform(rotation, translation)


def map_absolute_rotation_matrix_to_robot(
    rotation_table: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    rotation = np.asarray(rotation_table, dtype=np.float64).reshape(3, 3)
    reflection = reflection_matrix_from_mapping(mapping)
    if reflection is not None and np.allclose(mapping.rotation_scale, np.ones(3), atol=1e-9):
        mirrored = reflection @ rotation @ reflection
        return mapping.robot_from_table_rotation @ mirrored @ mapping.robot_from_table_rotation.T
    return map_rotation_delta_matrix_to_robot(rotation, mapping)


def build_servo_increment_from_desired_transforms(
    previous_desired_robot_transform: np.ndarray,
    current_desired_robot_transform: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    translation, rotation = compute_transform_delta(
        previous_desired_robot_transform, current_desired_robot_transform
    )
    return build_servo_increment_from_robot_delta(translation, rotation, mapping)


__all__ = [
    "TeleopAnchorState",
    "build_desired_robot_transform",
    "build_servo_increment",
    "build_servo_increment_from_desired_transforms",
    "build_servo_increment_from_robot_delta",
    "clamp_norm",
    "clip_translation_to_workspace_mm",
    "compute_transform_delta",
    "make_safe_start_pose_sdk",
    "map_absolute_rotation_matrix_to_robot",
    "map_rotation_delta_matrix_to_robot",
    "map_rotation_delta_to_robot_rad",
    "map_translation_delta_to_robot_mm",
    "pose_is_close",
    "pose_mmrad_to_transform",
    "reflection_matrix_from_mapping",
    "rotation_matrix_to_rpy",
    "rotation_y_deg",
    "rotation_z_deg",
    "rpy_to_rotation_matrix",
    "shortest_angle_delta_rad",
    "transform_to_pose_mmrad",
]
