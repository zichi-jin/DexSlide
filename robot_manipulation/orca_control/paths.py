"""Filesystem paths for the OrcaHand integration."""

from __future__ import annotations

from pathlib import Path

ROBOT_MANIPULATION_ROOT = Path(__file__).resolve().parents[1]
ORCA_ASSETS_DIR = ROBOT_MANIPULATION_ROOT / "assets" / "orca_hand"
ORCA_CONFIG_DIR = ORCA_ASSETS_DIR / "configs"
ORCA_DESCRIPTION_DIR = ORCA_ASSETS_DIR / "description"
ORCA_RETARGETING_DIR = ORCA_DESCRIPTION_DIR / "retargeting"
DEFAULT_ORCAHAND_RIGHT_URDF_FILE = ORCA_DESCRIPTION_DIR / "models" / "urdf" / "orcahand_right.urdf"
DEFAULT_ORCAHAND_RIGHT_RETARGET_V1_FILE = (
    ORCA_RETARGETING_DIR / "orcahand_v1_right_vector_21d.json"
)
DEFAULT_ORCAHAND_RIGHT_RETARGET_V2_FILE = (
    ORCA_RETARGETING_DIR / "orcahand_v2_right_vector_semantic_12d.json"
)
DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE = DEFAULT_ORCAHAND_RIGHT_RETARGET_V2_FILE
DEFAULT_DEX_TO_ORCA_DIRECT_JOINT_MAP_FILE = ORCA_CONFIG_DIR / "joint_map_dexslide_to_orcahand.json"

