"""Shared filesystem paths for the DexSlide PC-side tools."""

from __future__ import annotations

from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PYTHON_ROOT.parents[1]
ASSETS_DIR = PYTHON_ROOT / "assets"
SKELETONS_DIR = ASSETS_DIR / "skeletons"
DEFAULT_SKELETON_FILE = SKELETONS_DIR / "skeleton.json"
DEFAULT_RESULTS_FILE = SKELETONS_DIR / "offline_bone_mm_results.json"
DEFAULT_FIRMWARE_CALIBRATION_FILE = (
    PROJECT_ROOT / "firmware" / "dexslide_stm32" / "scripts" / "glove_calibration.json"
)
