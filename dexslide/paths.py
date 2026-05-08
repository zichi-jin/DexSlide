"""Shared filesystem paths for the DexSlide PC-side tools."""

from __future__ import annotations

from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PYTHON_ROOT / "assets"
SKELETONS_DIR = ASSETS_DIR / "skeletons"
CALIBRATION_DIR = ASSETS_DIR / "calibration"
DEFAULT_SKELETON_FILE = SKELETONS_DIR / "skeleton.json"
DEFAULT_RESULTS_FILE = SKELETONS_DIR / "offline_bone_mm_results.json"
DEFAULT_GLOVE_CALIBRATION_FILE = CALIBRATION_DIR / "glove_calibration.json"

# Backward-compatible alias for older entry points.
DEFAULT_FIRMWARE_CALIBRATION_FILE = DEFAULT_GLOVE_CALIBRATION_FILE
