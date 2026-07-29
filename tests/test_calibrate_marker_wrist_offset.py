from __future__ import annotations

import json

import cv2
import numpy as np

from dexslide.calibration.calibrate_marker_wrist_offset import (
    _camera_points_to_body_points,
    _estimate_body_pose_in_camera_from_tag_dict,
    _make_camera_frame_result_from_tag_dict,
    _save_alignment_outputs,
    _wrist_body_point,
)
from dexslide.world_pose.hand_cube_overlay import (
    HandCubeOverlayConfig,
    MarkerMount,
    make_transform,
    transform_points,
)


SQRT2_INV = float(1.0 / np.sqrt(2.0))


def _rot_x(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rot_y(rad: float) -> np.ndarray:
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
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


def _sample_cfg() -> HandCubeOverlayConfig:
    return HandCubeOverlayConfig(
        hand="left",
        aruco_bound_size_m=0.03,
        marker_square_size_m=0.04,
        markers={
            1: MarkerMount(
                marker_id=1,
                axes_rows_body=np.asarray(
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
            ),
            2: MarkerMount(
                marker_id=2,
                axes_rows_body=np.asarray(
                    [
                        [-SQRT2_INV, SQRT2_INV, 0.0],
                        [0.0, 0.0, 1.0],
                        [SQRT2_INV, SQRT2_INV, 0.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        },
    )


def _project_marker_corners(
    transform_camera_marker: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    half = 0.5 * float(marker_size_m)
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(transform_camera_marker[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        rvec,
        transform_camera_marker[:3, 3].reshape(3, 1),
        camera_matrix,
        np.zeros((1, 5), dtype=np.float64),
    )
    return projected.reshape(4, 2)


def test_make_camera_frame_result_from_tag_dict_preserves_marker_ids() -> None:
    tag_dict = {
        7: {
            "rvec": np.array([0.1, -0.2, 0.3], dtype=np.float64),
            "tvec": np.array([0.4, 0.5, 0.6], dtype=np.float64),
            "corners": np.zeros((4, 2), dtype=np.float64),
        }
    }

    frame_result = _make_camera_frame_result_from_tag_dict(tag_dict)

    assert sorted(frame_result["targets"].keys()) == ["7"]
    assert frame_result["targets"]["7"]["detected"] is True
    assert np.asarray(frame_result["targets"]["7"]["target_in_camera"]["matrix"]).shape == (4, 4)


def test_estimate_body_pose_in_camera_recovers_single_marker_pose() -> None:
    cfg = _sample_cfg()
    cfg.markers = {1: cfg.markers[1]}
    camera_matrix = np.array(
        [
            [620.0, 0.0, 320.0],
            [0.0, 618.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform_camera_body = make_transform(
        _rot_z(0.12) @ _rot_y(-0.08) @ _rot_x(0.04),
        np.array([0.03, -0.015, 0.42], dtype=np.float64),
    )
    transform_camera_marker = transform_camera_body @ cfg.markers[1].body_to_marker_transform(
        cfg.marker_center_radius_m
    )
    corners = _project_marker_corners(
        transform_camera_marker,
        cfg.aruco_bound_size_m,
        camera_matrix,
    )
    rvec, _ = cv2.Rodrigues(transform_camera_marker[:3, :3])
    tag_dict = {
        1: {
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
            "tvec": np.asarray(transform_camera_marker[:3, 3], dtype=np.float64).reshape(3),
            "corners": corners.copy(),
            "undistorted_corners": corners.copy(),
        }
    }

    estimate = _estimate_body_pose_in_camera_from_tag_dict(
        tag_dict,
        cfg,
        camera_matrix,
        reference_camera_body=None,
        outlier_threshold_m=0.02,
        reprojection_error_threshold_px=5.0,
    )

    assert estimate is not None
    np.testing.assert_allclose(estimate.transform_camera_body, transform_camera_body, atol=1e-7)
    assert estimate.source_marker_ids == (1,)


def test_camera_points_to_body_points_inverts_pose_for_batches() -> None:
    transform_camera_body = make_transform(
        _rot_z(0.5),
        np.array([0.2, -0.1, 0.5], dtype=np.float64),
    )
    body_points = np.array(
        [
            [0.05, 0.02, 0.12],
            [-0.03, 0.08, -0.04],
        ],
        dtype=np.float64,
    )

    camera_points = transform_points(transform_camera_body, body_points)
    recovered = _camera_points_to_body_points(transform_camera_body, camera_points)

    np.testing.assert_allclose(recovered, body_points, atol=1e-9)


def test_wrist_body_point_inverts_camera_body_transform() -> None:
    transform_camera_body = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.2, -0.1, 0.5], dtype=np.float64),
    )
    wrist_camera_xyz = np.array([0.25, -0.08, 0.62], dtype=np.float64)

    wrist_body_xyz = _wrist_body_point(transform_camera_body, wrist_camera_xyz)

    np.testing.assert_allclose(wrist_body_xyz, np.array([0.05, 0.02, 0.12], dtype=np.float64), atol=1e-9)


def test_save_alignment_outputs_reports_full_pose(tmp_path) -> None:
    cfg = _sample_cfg()
    expected_rotation = _rot_z(-0.31) @ _rot_y(0.14) @ _rot_x(0.09)
    initial_guess = make_transform(
        _rot_z(0.23) @ _rot_y(-0.17) @ _rot_x(0.11),
        np.array([0.11, -0.12, 0.13], dtype=np.float64),
    )
    cfg.set_body_to_wrist_transform(
        initial_guess
    )
    input_config_path = tmp_path / "input.yaml"
    output_config_path = tmp_path / "output_marker2wrist.json"
    output_report_path = tmp_path / "output_marker2wrist_dataset.json"
    cfg.save(input_config_path)

    transform_a = make_transform(expected_rotation, np.array([0.01, 0.02, 0.03], dtype=np.float64))
    transform_b = make_transform(expected_rotation, np.array([0.05, 0.06, 0.07], dtype=np.float64))

    mean_transform, std_translation = _save_alignment_outputs(
        input_config_path=input_config_path,
        output_config_path=output_config_path,
        output_report_path=output_report_path,
        cfg=cfg,
        samples_body_to_wrist=np.stack([transform_a, transform_b], axis=0),
        initial_guess_transform=initial_guess,
    )

    np.testing.assert_allclose(mean_transform[:3, 3], np.array([0.03, 0.04, 0.05], dtype=np.float64), atol=1e-9)
    np.testing.assert_allclose(mean_transform[:3, :3], expected_rotation, atol=1e-9)
    np.testing.assert_allclose(std_translation, np.array([0.02, 0.02, 0.02], dtype=np.float64), atol=1e-9)

    saved_cfg = HandCubeOverlayConfig.load(output_config_path)
    np.testing.assert_allclose(saved_cfg.cube_to_wrist_transform(), mean_transform, atol=1e-9)

    report = json.loads(output_report_path.read_text(encoding="utf-8"))
    assert report["sample_count"] == 2
    assert report["note"].startswith("This calibration uses the MediaPipe palm triangle")
    assert report["input_tags2marker_path"] == input_config_path.name
    assert report["output_marker2wrist_path"] == output_config_path.name

    marker_to_wrist = json.loads(output_config_path.read_text(encoding="utf-8"))
    assert marker_to_wrist["tags2marker_path"] == input_config_path.name
    assert marker_to_wrist["result"]["data_source"] == f"数据取自：{output_report_path.name}"
    np.testing.assert_allclose(
        np.asarray(marker_to_wrist["initial_guess"]["rot"], dtype=np.float64),
        initial_guess[:3, :3],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(marker_to_wrist["initial_guess"]["trans"], dtype=np.float64),
        initial_guess[:3, 3],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(marker_to_wrist["result"]["rot"], dtype=np.float64),
        mean_transform[:3, :3],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(marker_to_wrist["result"]["trans"], dtype=np.float64),
        mean_transform[:3, 3],
        atol=1e-9,
    )
