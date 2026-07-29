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
    mirror = payload.get("mirror", {})
    return WorkspaceAxisMapping(
        robot_from_table_transform=robot_from_table,
        same_pose_fixed_rotation=np.asarray(fixed.get("rotation_matrix"), dtype=np.float64).reshape(3, 3),
        mirror_translation_sign=np.asarray(mirror.get("translation_sign"), dtype=np.float64).reshape(3),
        mirror_rotation_sign=np.asarray(mirror.get("rotation_vector_sign"), dtype=np.float64).reshape(3),
        translation_scale=np.asarray(payload.get("translation_scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3),
        rotation_scale=np.asarray(payload.get("rotation_scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3),
        safe_start_pose_mmdeg=safe_start,  # type: ignore[arg-type]
        translation_deadband_mm=float(payload.get("translation_deadband_mm", 0.8)),
        rotation_deadband_deg=float(payload.get("rotation_deadband_deg", 0.6)),
        max_translation_step_mm=float(payload.get("max_translation_step_mm", 3.0)),
        max_rotation_step_deg=float(payload.get("max_rotation_step_deg", 2.0)),
        frame_jump_reject_mm=float(payload.get("frame_jump_reject_mm", 25.0)),
        frame_jump_reject_deg=float(payload.get("frame_jump_reject_deg", 18.0)),
        anchor_stable_frames=max(1, int(payload.get("anchor_stable_frames", 5))),
    )

