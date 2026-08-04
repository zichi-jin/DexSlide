from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "jaka_dexslide_incremental_teleop.py"
SPEC = importlib.util.spec_from_file_location("jaka_dexslide_incremental_teleop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fixed_rotation_cancels_in_relative_rotation() -> None:
    fixed = MODULE.rotation_y_deg(90.0) @ MODULE.rotation_z_deg(-45.0)
    glove0 = MODULE.rotation_y_deg(10.0)
    glove1 = MODULE.rotation_z_deg(15.0) @ MODULE.rotation_y_deg(10.0)

    relative_glove = glove1 @ glove0.T
    relative_orca = (glove1 @ fixed) @ (glove0 @ fixed).T
    np.testing.assert_allclose(relative_orca, relative_glove, atol=1e-9)


def test_mapping_mirrors_translation_and_rotation_vectors() -> None:
    mapping = MODULE.load_workspace_axis_mapping(MODULE.DEFAULT_MAPPING_FILE)
    delta_translation_m = np.array([0.01, 0.02, -0.03], dtype=np.float64)
    delta_rotation_rad = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    translation_mm = MODULE.map_translation_delta_to_robot_mm(delta_translation_m, mapping)
    rotation_rad = MODULE.map_rotation_delta_to_robot_rad(delta_rotation_rad, mapping)

    np.testing.assert_allclose(translation_mm, np.array([-10.0, 20.0, -30.0]), atol=1e-9)
    np.testing.assert_allclose(rotation_rad, np.array([0.1, -0.2, -0.3]), atol=1e-9)


def test_desired_robot_transform_respects_anchor_and_reflection() -> None:
    mapping = MODULE.load_workspace_axis_mapping(MODULE.DEFAULT_MAPPING_FILE)
    anchor = MODULE.TeleopAnchorState(
        glove_anchor_translation_m=np.zeros(3, dtype=np.float64),
        robot_anchor_translation_mm=np.zeros(3, dtype=np.float64),
        previous_desired_robot_transform=np.eye(4, dtype=np.float64),
        anchor_frame_idx=0,
    )
    current_glove = MODULE.make_transform(
        MODULE.rotation_z_deg(20.0),
        np.array([0.01, 0.02, -0.03], dtype=np.float64),
    )

    desired_robot = MODULE.build_desired_robot_transform(current_glove, anchor, mapping)
    reflection = np.diag([-1.0, 1.0, 1.0])
    expected_rotation = reflection @ current_glove[:3, :3] @ reflection

    np.testing.assert_allclose(desired_robot[:3, 3], np.array([-10.0, 20.0, -30.0]), atol=1e-9)
    np.testing.assert_allclose(desired_robot[:3, :3], expected_rotation, atol=1e-9)


def test_servo_increment_applies_deadband_and_norm_limit() -> None:
    mapping = MODULE.load_workspace_axis_mapping(MODULE.DEFAULT_MAPPING_FILE)

    tiny_increment = MODULE.build_servo_increment(
        np.array([1e-5, 1e-5, 1e-5], dtype=np.float64),
        np.array([1e-4, 0.0, 0.0], dtype=np.float64),
        mapping,
    )
    np.testing.assert_allclose(tiny_increment, np.zeros(6), atol=1e-12)

    large_increment = MODULE.build_servo_increment(
        np.array([0.05, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, np.deg2rad(10.0)], dtype=np.float64),
        mapping,
    )
    assert np.linalg.norm(large_increment[:3]) <= mapping.max_translation_step_mm + 1e-9
    assert np.rad2deg(np.linalg.norm(large_increment[3:])) <= mapping.max_rotation_step_deg + 1e-9


def test_increment_from_desired_transforms_tracks_previous_desired() -> None:
    mapping = MODULE.load_workspace_axis_mapping(MODULE.DEFAULT_MAPPING_FILE)
    previous = np.eye(4, dtype=np.float64)
    current = MODULE.make_transform(
        MODULE.rotation_z_deg(1.0),
        np.array([2.0, 0.0, 0.0], dtype=np.float64),
    )
    increment = MODULE.build_servo_increment_from_desired_transforms(previous, current, mapping)

    np.testing.assert_allclose(increment[:3], np.array([2.0, 0.0, 0.0]), atol=1e-9)
    assert abs(np.rad2deg(increment[5]) - 1.0) < 1e-6
