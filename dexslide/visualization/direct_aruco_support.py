"""Small runtime policy helpers for direct ArUco capture."""

from __future__ import annotations

from dexslide.vision.aruco_pose_tracker import _parse_capture_source

def _use_realsense_backend(*, camera_backend: str, wrist_align_enabled: bool, use_realsense_rgb: bool) -> bool:
    return bool(str(camera_backend).strip().lower() == "realsense" or wrist_align_enabled or use_realsense_rgb)


def _iter_capture_candidates(source: str | int) -> list[int | str]:
    return [_parse_capture_source(source)]

