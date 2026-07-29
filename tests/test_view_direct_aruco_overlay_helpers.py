from __future__ import annotations

from pathlib import Path

import numpy as np

from dexslide.calibration.palm_triangle_alignment import capture_body_to_wrist_transform_sample
from dexslide.paths import DIRECT_ARUCO_CALIBRATION_DIR
from dexslide.world_pose.hand_cube_overlay import HandCubeOverlayConfig, MarkerMount, make_transform
from scripts.view_direct_aruco_overlay import CubePoseEstimate
from scripts.view_direct_aruco_overlay import (
    _apply_overlay_joint_calibration,
    _apply_runtime_body_to_wrist_transform,
    _compute_overlay_joint_angles,
    _default_hand_overlay_config_path,
    _load_overlay_joint_calibration,
    _resolve_camera_body_transform,
)


def _rot_z(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _base_cfg() -> HandCubeOverlayConfig:
    return HandCubeOverlayConfig(
        hand="left",
        aruco_bound_size_m=0.03,
        marker_square_size_m=0.04,
        markers={
            1: MarkerMount(
                marker_id=1,
                axes_rows_body=np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        },
    )


def test_default_hand_overlay_config_path_auto_falls_back_to_existing_config() -> None:
    path = _default_hand_overlay_config_path("auto")
    assert path == DIRECT_ARUCO_CALIBRATION_DIR / "left_tags2marker.json"
    assert Path(path).exists()


def test_compute_overlay_joint_angles_uses_joint_zero_and_base_offsets() -> None:
    cfg = _base_cfg()
    raw_joint = np.linspace(-0.3, 0.2, 20, dtype=np.float64)
    joint_zero = np.linspace(-0.1, 0.1, 20, dtype=np.float64)
    joint_base = np.linspace(0.4, 0.8, 20, dtype=np.float64)
    cfg.set_joint_zero(joint_zero)
    cfg.set_joint_base_render(joint_base)

    adjusted = _compute_overlay_joint_angles(raw_joint, cfg=cfg)

    np.testing.assert_allclose(adjusted, (raw_joint - joint_zero) + joint_base, atol=1e-9)


def test_load_overlay_joint_calibration_prefers_top_level_joint_arrays(tmp_path) -> None:
    payload = {
        "hand": "left",
        "joint_scale": [1.0 + 0.01 * idx for idx in range(20)],
        "joint_bias_rad": [0.001 * idx for idx in range(20)],
        "summary": {
            "joint_scale": [0.0] * 20,
            "joint_bias_rad": [0.0] * 20,
        },
    }
    path = tmp_path / "optimized_joint_calibration.json"
    path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scale, bias, loaded_path = _load_overlay_joint_calibration(path)

    np.testing.assert_allclose(scale, np.asarray(payload["joint_scale"], dtype=np.float64), atol=1e-12)
    np.testing.assert_allclose(bias, np.asarray(payload["joint_bias_rad"], dtype=np.float64), atol=1e-12)
    assert loaded_path == path.resolve()


def test_apply_overlay_joint_calibration_matches_affine_rule() -> None:
    raw_joint = np.linspace(-0.2, 0.3, 20, dtype=np.float64)
    scale = np.linspace(0.8, 1.2, 20, dtype=np.float64)
    bias = np.linspace(-0.05, 0.04, 20, dtype=np.float64)

    adjusted = _apply_overlay_joint_calibration(raw_joint, joint_scale=scale, joint_bias=bias)

    np.testing.assert_allclose(adjusted, (scale * raw_joint) + bias, atol=1e-12)


def test_resolve_camera_body_transform_prefers_table_pose_when_available() -> None:
    transform_camera_table = make_transform(np.eye(3, dtype=np.float64), np.array([0.1, 0.2, 0.3], dtype=np.float64))
    cube_pose = CubePoseEstimate(
        transform_table_cube=make_transform(np.eye(3, dtype=np.float64), np.array([0.4, -0.1, 0.5], dtype=np.float64)),
        source_marker_ids=[1, 2],
        max_position_deviation_m=0.001,
    )

    transform_camera_body, source = _resolve_camera_body_transform(
        transform_camera_table=transform_camera_table,
        cube_pose=cube_pose,
        camera_body_pose=None,
    )

    assert source == "table_body"
    np.testing.assert_allclose(
        transform_camera_body,
        transform_camera_table @ cube_pose.transform_table_cube,
        atol=1e-9,
    )


def test_apply_runtime_body_to_wrist_transform_updates_full_pose() -> None:
    cfg = _base_cfg()
    transform_a = make_transform(_rot_z(0.0), np.array([0.0, 0.0, 0.0], dtype=np.float64))
    transform_b = make_transform(_rot_z(np.pi / 2.0), np.array([0.2, 0.4, 0.6], dtype=np.float64))

    applied = _apply_runtime_body_to_wrist_transform(cfg, [transform_a, transform_b])

    assert applied is not None
    expected = make_transform(_rot_z(np.pi / 4.0), np.array([0.1, 0.2, 0.3], dtype=np.float64))
    np.testing.assert_allclose(applied, expected, atol=1e-9)
    np.testing.assert_allclose(cfg.cube_to_wrist_transform(), expected, atol=1e-9)


def test_runtime_wrist_align_replace_sample_applies_current_instant_pose_only() -> None:
    cfg = _base_cfg()
    samples = [
        make_transform(_rot_z(0.0), np.array([0.01, 0.02, 0.03], dtype=np.float64)),
        make_transform(_rot_z(0.4), np.array([0.04, 0.05, 0.06], dtype=np.float64)),
    ]
    current = make_transform(_rot_z(-0.2), np.array([0.3, 0.2, 0.1], dtype=np.float64))

    capture_body_to_wrist_transform_sample(
        samples,
        current,
        replace_existing=True,
    )
    applied = _apply_runtime_body_to_wrist_transform(cfg, samples)

    assert applied is not None
    assert len(samples) == 1
    np.testing.assert_allclose(applied, current, atol=1e-9)
    np.testing.assert_allclose(cfg.cube_to_wrist_transform(), current, atol=1e-9)


def test_runtime_wrist_align_append_sample_applies_running_mean_pose() -> None:
    cfg = _base_cfg()
    transform_a = make_transform(_rot_z(0.0), np.array([0.01, 0.02, 0.03], dtype=np.float64))
    transform_b = make_transform(_rot_z(np.pi / 2.0), np.array([0.05, 0.06, 0.07], dtype=np.float64))
    samples = [transform_a]

    capture_body_to_wrist_transform_sample(
        samples,
        transform_b,
        replace_existing=False,
    )
    applied = _apply_runtime_body_to_wrist_transform(cfg, samples)

    assert applied is not None
    expected = make_transform(_rot_z(np.pi / 4.0), np.array([0.03, 0.04, 0.05], dtype=np.float64))
    assert len(samples) == 2
    np.testing.assert_allclose(applied, expected, atol=1e-9)
    np.testing.assert_allclose(cfg.cube_to_wrist_transform(), expected, atol=1e-9)
