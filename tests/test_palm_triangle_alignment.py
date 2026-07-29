from __future__ import annotations

import json

import numpy as np

from dexslide.calibration.palm_triangle_alignment import (
    PALM_TRIANGLE_LANDMARK_INDICES,
    apply_body_to_wrist_transform,
    average_body_to_wrist_transforms,
    capture_body_to_wrist_transform_sample,
    estimate_body_to_wrist_transform_from_triangles,
    save_body_to_wrist_alignment_outputs,
    select_palm_triangle_points,
)
from dexslide.kinematics.transforms import make_transform
from dexslide.vision.hand_cube_overlay import HandCubeOverlayConfig, MarkerMount


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


def _sample_cfg() -> HandCubeOverlayConfig:
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


def test_select_palm_triangle_points_handles_full_landmark_array() -> None:
    landmarks = np.arange(21 * 3, dtype=np.float64).reshape(21, 3)

    selected = select_palm_triangle_points(landmarks)

    np.testing.assert_allclose(selected, landmarks[np.asarray(PALM_TRIANGLE_LANDMARK_INDICES), :], atol=1e-9)


def test_estimate_body_to_wrist_transform_from_triangles_recovers_known_pose() -> None:
    source_triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.04, 0.01, 0.0],
            [0.02, 0.05, 0.0],
        ],
        dtype=np.float64,
    )
    expected_transform = make_transform(
        _rot_z(0.37),
        np.array([0.12, -0.08, 0.31], dtype=np.float64),
    )
    observed_triangle = (
        source_triangle @ expected_transform[:3, :3].T
        + expected_transform[:3, 3][None, :]
    )

    estimated = estimate_body_to_wrist_transform_from_triangles(source_triangle, observed_triangle)

    assert estimated is not None
    np.testing.assert_allclose(estimated, expected_transform, atol=1e-9)


def test_average_body_to_wrist_transforms_averages_translation_and_rotation() -> None:
    transform_a = make_transform(_rot_z(0.0), np.array([0.0, 0.0, 0.0], dtype=np.float64))
    transform_b = make_transform(_rot_z(np.pi / 2.0), np.array([0.2, 0.4, 0.6], dtype=np.float64))

    mean_transform = average_body_to_wrist_transforms([transform_a, transform_b])

    assert mean_transform is not None
    expected = make_transform(_rot_z(np.pi / 4.0), np.array([0.1, 0.2, 0.3], dtype=np.float64))
    np.testing.assert_allclose(mean_transform, expected, atol=1e-9)


def test_capture_body_to_wrist_transform_sample_replace_and_append_semantics() -> None:
    first = make_transform(_rot_z(0.0), np.array([0.01, 0.02, 0.03], dtype=np.float64))
    second = make_transform(_rot_z(np.pi / 2.0), np.array([0.05, 0.06, 0.07], dtype=np.float64))
    third = make_transform(_rot_z(-0.2), np.array([0.3, 0.2, 0.1], dtype=np.float64))
    samples = [first]

    mean_after_append = capture_body_to_wrist_transform_sample(
        samples,
        second,
        replace_existing=False,
    )
    mean_after_replace = capture_body_to_wrist_transform_sample(
        samples,
        third,
        replace_existing=True,
    )

    np.testing.assert_allclose(
        mean_after_append,
        make_transform(_rot_z(np.pi / 4.0), np.array([0.03, 0.04, 0.05], dtype=np.float64)),
        atol=1e-9,
    )
    assert len(samples) == 1
    np.testing.assert_allclose(samples[0], third, atol=1e-9)
    np.testing.assert_allclose(mean_after_replace, third, atol=1e-9)


def test_apply_body_to_wrist_transform_updates_config() -> None:
    cfg = _sample_cfg()
    transform = make_transform(_rot_z(0.2), np.array([0.02, -0.03, 0.04], dtype=np.float64))

    applied = apply_body_to_wrist_transform(cfg, transform)

    np.testing.assert_allclose(applied, transform, atol=1e-9)
    np.testing.assert_allclose(cfg.cube_to_wrist_transform(), transform, atol=1e-9)


def test_save_body_to_wrist_alignment_outputs_writes_full_pose_report(tmp_path) -> None:
    cfg = _sample_cfg()
    input_config_path = tmp_path / "input.yaml"
    output_config_path = tmp_path / "output_marker2wrist.json"
    output_report_path = tmp_path / "output_marker2wrist_dataset.json"
    cfg.save(input_config_path)

    transform_a = make_transform(_rot_z(0.0), np.array([0.0, 0.0, 0.0], dtype=np.float64))
    transform_b = make_transform(_rot_z(np.pi / 2.0), np.array([0.2, 0.4, 0.6], dtype=np.float64))

    mean_transform, std_translation = save_body_to_wrist_alignment_outputs(
        input_config_path=input_config_path,
        output_config_path=output_config_path,
        output_report_path=output_report_path,
        cfg=cfg,
        samples_body_to_wrist=[transform_a, transform_b],
    )

    expected = make_transform(_rot_z(np.pi / 4.0), np.array([0.1, 0.2, 0.3], dtype=np.float64))
    np.testing.assert_allclose(mean_transform, expected, atol=1e-9)
    np.testing.assert_allclose(std_translation, np.array([0.1, 0.2, 0.3], dtype=np.float64), atol=1e-9)

    saved_cfg = HandCubeOverlayConfig.load(output_config_path)
    np.testing.assert_allclose(saved_cfg.cube_to_wrist_transform(), expected, atol=1e-9)

    report = json.loads(output_report_path.read_text(encoding="utf-8"))
    assert report["sample_count"] == 2
    assert report["input_tags2marker_path"] == input_config_path.name
    assert report["output_marker2wrist_path"] == output_config_path.name

    marker_to_wrist = json.loads(output_config_path.read_text(encoding="utf-8"))
    assert marker_to_wrist["tags2marker_path"] == input_config_path.name
    assert marker_to_wrist["result"]["data_source"] == f"数据取自：{output_report_path.name}"
    np.testing.assert_allclose(
        np.asarray(marker_to_wrist["result"]["rot"], dtype=np.float64),
        expected[:3, :3],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(marker_to_wrist["result"]["trans"], dtype=np.float64),
        expected[:3, 3],
        atol=1e-9,
    )
