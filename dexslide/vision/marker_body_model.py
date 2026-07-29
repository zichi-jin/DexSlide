"""Marker-body model and configuration API.

The implementation is currently shared with the pose solver for backward
compatibility; new callers should import configuration/model symbols here.
"""

from dexslide.vision.marker_body_pose import (
    HandCubeOverlayConfig,
    load_marker_to_wrist_asset,
    MarkerBodyConsistencyItem,
    MarkerBodyConsistencyReport,
    MarkerMount,
    marker_to_wrist_asset_transforms,
    marker_to_wrist_entry_from_transform,
    resolve_hand_overlay_asset_paths,
    try_load_hand_cube_overlay_config,
)

__all__ = [
    "HandCubeOverlayConfig",
    "load_marker_to_wrist_asset",
    "MarkerBodyConsistencyItem",
    "MarkerBodyConsistencyReport",
    "MarkerMount",
    "marker_to_wrist_asset_transforms",
    "marker_to_wrist_entry_from_transform",
    "resolve_hand_overlay_asset_paths",
    "try_load_hand_cube_overlay_config",
]
