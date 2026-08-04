"""JAKA table-to-base workspace mapping and configuration loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WorkspaceAxisMapping:
    robot_from_table_transform: np.ndarray
    same_pose_fixed_rotation: np.ndarray
    mirror_translation_sign: np.ndarray
    mirror_rotation_sign: np.ndarray
    translation_scale: np.ndarray
    rotation_scale: np.ndarray
    safe_start_pose_mmdeg: tuple[float, float, float, float, float, float]
    safe_start_speed_mm_s: float
    task_space_zero_pose_mmdeg: tuple[float, float, float, float, float, float]
    task_space_zero_speed_mm_s: float
    teleop_workspace_min_mm: np.ndarray
    teleop_workspace_max_mm: np.ndarray
    translation_deadband_mm: float
    rotation_deadband_deg: float
    max_translation_step_mm: float
    max_rotation_step_deg: float
    frame_jump_reject_mm: float
    frame_jump_reject_deg: float
    anchor_stable_frames: int

    @property
    def robot_from_table_rotation(self) -> np.ndarray:
        return np.asarray(self.robot_from_table_transform[:3, :3], dtype=np.float64)


def load_workspace_axis_mapping(path: str | Path) -> WorkspaceAxisMapping:
    mapping_path = Path(path).expanduser().resolve()
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported workspace axis mapping schema_version in {mapping_path}")
    table_to_robot = payload.get("table_to_robot", {})
    robot_from_table = np.eye(4, dtype=np.float64)
    robot_from_table[:3, :3] = np.asarray(table_to_robot.get("rotation_matrix"), dtype=np.float64).reshape(3, 3)
    robot_from_table[:3, 3] = np.asarray(table_to_robot.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    fixed = payload.get("glove_to_orca_same_pose_fixed_rotation", {})
    safe_start = tuple(float(x) for x in payload.get("safe_start_pose_mmdeg", []))
    if len(safe_start) != 6:
        raise ValueError(f"`safe_start_pose_mmdeg` must contain 6 values in {mapping_path}")
    safe_start_speed_mm_s = float(payload.get("safe_start_speed_mm_s", 0.0))
    if safe_start_speed_mm_s <= 0.0:
        raise ValueError(f"`safe_start_speed_mm_s` must be positive in {mapping_path}")
    task_space_zero = tuple(float(x) for x in payload.get("task_space_zero_pose_mmdeg", []))
    if len(task_space_zero) != 6:
        raise ValueError(f"`task_space_zero_pose_mmdeg` must contain 6 values in {mapping_path}")
    task_space_zero_speed_mm_s = float(payload.get("task_space_zero_speed_mm_s", 0.0))
    if task_space_zero_speed_mm_s <= 0.0:
        raise ValueError(f"`task_space_zero_speed_mm_s` must be positive in {mapping_path}")
    mirror = payload.get("mirror", {})
    raw_workspace = payload.get("teleop_workspace_xyz_mm", {})
    if not isinstance(raw_workspace, dict):
        raise ValueError(f"teleop_workspace_xyz_mm must be an object in {mapping_path}")
    try:
        workspace_bounds = np.asarray(
            [raw_workspace["x"], raw_workspace["y"], raw_workspace["z"]],
            dtype=np.float64,
        ).reshape(3, 2)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"teleop_workspace_xyz_mm must define x/y/z [min, max] bounds in {mapping_path}"
        ) from exc
    if not np.isfinite(workspace_bounds).all() or np.any(workspace_bounds[:, 0] >= workspace_bounds[:, 1]):
        raise ValueError(f"teleop_workspace_xyz_mm has invalid bounds in {mapping_path}")
    return WorkspaceAxisMapping(
        robot_from_table_transform=robot_from_table,
        same_pose_fixed_rotation=np.asarray(fixed.get("rotation_matrix"), dtype=np.float64).reshape(3, 3),
        mirror_translation_sign=np.asarray(mirror.get("translation_sign"), dtype=np.float64).reshape(3),
        mirror_rotation_sign=np.asarray(mirror.get("rotation_vector_sign"), dtype=np.float64).reshape(3),
        translation_scale=np.asarray(payload.get("translation_scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3),
        rotation_scale=np.asarray(payload.get("rotation_scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3),
        safe_start_pose_mmdeg=safe_start,  # type: ignore[arg-type]
        safe_start_speed_mm_s=safe_start_speed_mm_s,
        task_space_zero_pose_mmdeg=task_space_zero,  # type: ignore[arg-type]
        task_space_zero_speed_mm_s=task_space_zero_speed_mm_s,
        teleop_workspace_min_mm=workspace_bounds[:, 0].copy(),
        teleop_workspace_max_mm=workspace_bounds[:, 1].copy(),
        translation_deadband_mm=float(payload.get("translation_deadband_mm", 0.8)),
        rotation_deadband_deg=float(payload.get("rotation_deadband_deg", 0.6)),
        max_translation_step_mm=float(payload.get("max_translation_step_mm", 3.0)),
        max_rotation_step_deg=float(payload.get("max_rotation_step_deg", 2.0)),
        frame_jump_reject_mm=float(payload.get("frame_jump_reject_mm", 25.0)),
        frame_jump_reject_deg=float(payload.get("frame_jump_reject_deg", 18.0)),
        anchor_stable_frames=max(1, int(payload.get("anchor_stable_frames", 5))),
    )
