"""Shared contracts between vision producers and streaming consumers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VisionHandPose:
    transform_table_hand: np.ndarray
    valid: bool
    marker_ids: tuple[int, ...] = ()
    reprojection_error_px: float | None = None
    # Marker-body (ball/cube centroid) pose in table coordinates.  The public
    # hand pose is derived from this pose and the configured body-to-wrist
    # transform; consumers should draw axes at the body, not at the wrist.
    transform_table_body: np.ndarray | None = None
    marker_corners_px: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class VisionSceneFrame:
    timestamp: float
    frame_bgr: np.ndarray | None
    image_size: tuple[int, int]
    camera_T_table: np.ndarray
    table_valid: bool
    table_marker_corners_px: np.ndarray | None
    hands: dict[str, VisionHandPose]
