"""Shared filesystem paths for the DexSlide PC-side tools."""

from __future__ import annotations

from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PYTHON_ROOT / "assets"
SKELETONS_DIR = ASSETS_DIR / "skeletons"
CALIBRATION_DIR = ASSETS_DIR / "calibration"
DIRECT_ARUCO_CALIBRATION_DIR = CALIBRATION_DIR / "direct_aruco"
DEXALIGN_CALIBRATION_DIR = CALIBRATION_DIR / "dexalign"
GLOVE_TO_ROBOT_HAND_DIR = ASSETS_DIR / "glove_to_robot_hand"
ROBOT_HANDS_DIR = ASSETS_DIR / "robot_hands"
ORCAHAND_DESCRIPTION_DIR = ROBOT_HANDS_DIR / "orcahand_description"
ORCAHAND_URDF_DIR = ORCAHAND_DESCRIPTION_DIR / "models" / "urdf"
ORCAHAND_RETARGETING_DIR = ORCAHAND_DESCRIPTION_DIR / "retargeting"
DEFAULT_SKELETON_FILE = SKELETONS_DIR / "skeleton.json"
DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE = ASSETS_DIR / "dexslide_communications.json"
DEFAULT_RESULTS_FILE = SKELETONS_DIR / "photos2skeletons_dataset.json"
DEFAULT_GLOVE_CALIBRATION_FILE = CALIBRATION_DIR / "glove_calibration.json"
DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "d435i_intrinsic.json"
)
DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "table_aruco.yaml"
)
DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE = DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE
DEFAULT_LEFT_TAGS_TO_MARKER_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "left_tags2marker.json"
)
DEFAULT_LEFT_MARKER_TO_WRIST_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "left_marker2wrist.json"
)
DEFAULT_LEFT_MARKER_TO_WRIST_DATASET_FILE = (
    DIRECT_ARUCO_CALIBRATION_DIR / "left_marker2wrist_dataset.json"
)

# Backward-compatible alias for older entry points that still talk about a single
# "hand marker body config" path. The new asset layout resolves the paired
# marker->wrist file from this tags->marker file at load time.
DEFAULT_HAND_MARKER_BODY_LEFT_CONFIG_FILE = DEFAULT_LEFT_TAGS_TO_MARKER_FILE
DEFAULT_ORCAHAND_RIGHT_URDF_FILE = ORCAHAND_URDF_DIR / "orcahand_right.urdf"
DEFAULT_ORCAHAND_RIGHT_RETARGET_V1_FILE = (
    ORCAHAND_RETARGETING_DIR / "orcahand_v1_right_vector_21d.json"
)
DEFAULT_ORCAHAND_RIGHT_RETARGET_V2_FILE = (
    ORCAHAND_RETARGETING_DIR / "orcahand_v2_right_vector_semantic_12d.json"
)
DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE = DEFAULT_ORCAHAND_RIGHT_RETARGET_V2_FILE
DEFAULT_DEX_TO_ORCA_DIRECT_JOINT_MAP_FILE = (
    GLOVE_TO_ROBOT_HAND_DIR / "joint_map_dexslide_to_orcahand.json"
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
