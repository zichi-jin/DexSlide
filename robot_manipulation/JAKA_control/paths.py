"""Filesystem paths for the JAKA integration."""

from __future__ import annotations

from pathlib import Path

ROBOT_MANIPULATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROBOT_MANIPULATION_ROOT.parent
JAKA_CONTROL_DIR = ROBOT_MANIPULATION_ROOT / "JAKA_control"
JAKA_SDK_DIR = JAKA_CONTROL_DIR / "JAKA_dependecies" / "x86_64-linux-gnu"
JAKA_ASSETS_DIR = ROBOT_MANIPULATION_ROOT / "assets" / "jaka"
JAKA_CONFIG_DIR = JAKA_ASSETS_DIR / "configs"
DEFAULT_WORKSPACE_MAPPING_FILE = JAKA_CONFIG_DIR / "workspace_axis_mapping.json"
DEFAULT_PAYLOAD_CONFIG_FILE = JAKA_CONFIG_DIR / "jaka_s5_orcahand_payload.json"

