#!/usr/bin/env python3
"""CLI entry point for the direct ArUco overlay runtime."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from dexslide.visualization.direct_aruco_runtime import main
from dexslide.visualization.direct_aruco_runtime import (
    CubePoseEstimate, _apply_overlay_joint_calibration, _apply_runtime_body_to_wrist_transform,
    _compute_overlay_joint_angles, _load_overlay_joint_calibration, _resolve_camera_body_transform,
    _use_realsense_backend, _open_capture_with_fallback, _iter_capture_candidates,
)
from dexslide.visualization.direct_aruco_config import (
    _default_hand_overlay_config_path, _default_overlay_joint_calibration_file,
    _default_overlay_marker2hand_file, _default_overlay_skeleton_file,
)

# 兼容旧测试和外部脚本对历史 helper 的导入。
DEXALIGN_CALIBRATION_DIR = __import__("dexslide.paths", fromlist=["DEXALIGN_CALIBRATION_DIR"]).DEXALIGN_CALIBRATION_DIR

def _default_overlay_skeleton_file() -> Path:
    from dexslide.visualization import direct_aruco_config as cfg
    original = cfg.DEXALIGN_CALIBRATION_DIR
    cfg.DEXALIGN_CALIBRATION_DIR = DEXALIGN_CALIBRATION_DIR
    try: return cfg._default_overlay_skeleton_file()
    finally: cfg.DEXALIGN_CALIBRATION_DIR = original

def _default_overlay_marker2hand_file() -> Path:
    from dexslide.visualization import direct_aruco_config as cfg
    original = cfg.DEXALIGN_CALIBRATION_DIR
    cfg.DEXALIGN_CALIBRATION_DIR = DEXALIGN_CALIBRATION_DIR
    try: return cfg._default_overlay_marker2hand_file()
    finally: cfg.DEXALIGN_CALIBRATION_DIR = original

def _default_overlay_joint_calibration_file() -> str:
    from dexslide.visualization import direct_aruco_config as cfg
    original = cfg.DEXALIGN_CALIBRATION_DIR
    cfg.DEXALIGN_CALIBRATION_DIR = DEXALIGN_CALIBRATION_DIR
    try: return cfg._default_overlay_joint_calibration_file()
    finally: cfg.DEXALIGN_CALIBRATION_DIR = original

if __name__ == "__main__":
    main()
