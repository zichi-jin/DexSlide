"""Shared filesystem paths for the DexSlide PC-side tools."""

from __future__ import annotations

from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PYTHON_ROOT / "assets"
SKELETONS_DIR = ASSETS_DIR / "skeletons"
CALIBRATION_DIR = ASSETS_DIR / "calibration"
DIRECT_ARUCO_CALIBRATION_DIR = CALIBRATION_DIR / "direct_aruco"
ROBOT_HANDS_DIR = ASSETS_DIR / "robot_hands"
ORCAHAND_DESCRIPTION_DIR = ROBOT_HANDS_DIR / "orcahand_description"
ORCAHAND_URDF_DIR = ORCAHAND_DESCRIPTION_DIR / "models" / "urdf"
ORCAHAND_RETARGETING_DIR = ORCAHAND_DESCRIPTION_DIR / "retargeting"
DEFAULT_SKELETON_FILE = SKELETONS_DIR / "skeleton.json"
DEFAULT_RESULTS_FILE = SKELETONS_DIR / "offline_bone_mm_results.json"
DEFAULT_GLOVE_CALIBRATION_FILE = CALIBRATION_DIR / "glove_calibration.json"
DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "d435i_960_540.json"
)
DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "table_aruco_4x4_120mm.yaml"
)
DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "target_aruco_4x4_50mm.yaml"
)
DEFAULT_ORCAHAND_RIGHT_URDF_FILE = ORCAHAND_URDF_DIR / "orcahand_right.urdf"
DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE = (
    ORCAHAND_RETARGETING_DIR / "orcahand_v1_right_vector_21d.json"
)
DEFAULT_ORCAHAND_RIGHT_CONFIG_FILE = (
    PYTHON_ROOT
    / "robot_manipulation"
    / "orca_control"
    / "orca_dependencies"
    / "orca_core"
    / "models"
    / "v1"
    / "orcahand_right"
    / "config.yaml"
)

# Backward-compatible alias for older entry points.
DEFAULT_FIRMWARE_CALIBRATION_FILE = DEFAULT_GLOVE_CALIBRATION_FILE
