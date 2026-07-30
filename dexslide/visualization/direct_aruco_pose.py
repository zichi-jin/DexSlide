"""Pose and joint transforms used by the direct ArUco overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexslide.calibration.palm_triangle_alignment import apply_body_to_wrist_transform, average_body_to_wrist_transforms
from dexslide.kinematics.hand_overlay import compose_overlay_joint_angles
from dexslide.vision.hand_cube_overlay import CubePoseEstimate, HandCubeOverlayConfig

def _compute_overlay_joint_angles(
    raw_joint_angles: np.ndarray,
    cfg: HandCubeOverlayConfig,
) -> np.ndarray:
    return compose_overlay_joint_angles(
        raw_joint_angles=np.asarray(raw_joint_angles, dtype=np.float64),
        joint_zero_rad=np.asarray(cfg.joint_zero_rad, dtype=np.float64),
        joint_base_render_rad=np.asarray(cfg.joint_base_render_rad, dtype=np.float64),
    )


def _optional_path_arg(raw: str | None) -> Path | None:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _resolve_camera_body_transform(
    *,
    transform_camera_table: np.ndarray | None,
    cube_pose: CubePoseEstimate | None,
    camera_body_pose: object | None,
) -> tuple[np.ndarray | None, str]:
    if cube_pose is not None and transform_camera_table is not None:
        return (
            np.asarray(transform_camera_table, dtype=np.float64).reshape(4, 4)
            @ np.asarray(cube_pose.transform_table_cube, dtype=np.float64).reshape(4, 4),
            "table_body",
        )
    if camera_body_pose is not None:
        return (
            np.asarray(camera_body_pose.transform_camera_body, dtype=np.float64).reshape(4, 4).copy(),
            "camera_body",
        )
    return None, "none"


def _apply_runtime_body_to_wrist_transform(
    cfg: HandCubeOverlayConfig,
    samples_body_to_wrist: list[np.ndarray],
) -> np.ndarray | None:
    mean_transform = average_body_to_wrist_transforms(samples_body_to_wrist)
    if mean_transform is None:
        return None
    return apply_body_to_wrist_transform(cfg, mean_transform)


