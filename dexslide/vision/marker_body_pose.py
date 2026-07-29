"""Marker-body pose estimation and diagnostics."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dexslide.kinematics.transforms import (
    average_rotation_matrices,
    invert_transform,
    make_transform,
    transform_points,
    transform_to_rvec_tvec,
)
from dexslide.vision.marker_body_model_impl import (
    CubePoseEstimate,
    HandCubeOverlayConfig,
    MarkerBodyConsistencyItem,
    MarkerBodyConsistencyReport,
    MarkerMount,
    _MarkerObservation,
    _MarkerPoseBranchCandidate,
    _marker_pose_weight,
    _marker_square_object_points,
    _sanitize_weights,
    compose_overlay_joint_angles,
    marker_to_wrist_asset_transforms,
    marker_to_wrist_entry_from_transform,
    resolve_hand_overlay_asset_paths,
    try_load_hand_cube_overlay_config,
)
from dexslide.vision.marker_body_solver import *  # noqa: F401,F403
from dexslide.vision.marker_body_diagnostics import *  # noqa: F401,F403
