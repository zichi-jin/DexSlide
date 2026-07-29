from __future__ import annotations

import cv2
import numpy as np

from dexslide.world_pose.hand_cube_overlay import make_transform, transform_points
from dexslide.world_pose.marker_body_pose_tracker import MarkerBodyPoseTracker
from tests.test_hand_cube_overlay import _marker_object_points, _rot_x, _rot_y, _rot_z, _sample_cfg


def _project_points(
    transform_camera_object: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(transform_camera_object[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        rvec,
        transform_camera_object[:3, 3].reshape(3, 1),
        camera_matrix,
        np.zeros((1, 5), dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def _build_frame_result(cfg, transform_camera_table: np.ndarray, transform_table_body: np.ndarray, camera_matrix: np.ndarray):
    transform_camera_body = transform_camera_table @ transform_table_body
    targets: dict[str, dict[str, object]] = {}
    for marker_id, mount in cfg.markers.items():
        transform_body_marker = mount.body_to_marker_transform(cfg.marker_center_radius_m)
        transform_camera_marker = transform_camera_body @ transform_body_marker
        transform_table_marker = transform_table_body @ transform_body_marker
        object_points_body = transform_points(
            transform_body_marker,
            _marker_object_points(cfg.aruco_bound_size_m),
        )
        projected = _project_points(transform_camera_body, object_points_body, camera_matrix)
        targets[str(marker_id)] = {
            "detected": True,
            "target_in_camera": {"matrix": transform_camera_marker.tolist()},
            "target_in_table": {"matrix": transform_table_marker.tolist()},
            "undistorted_corners": projected.tolist(),
            "marker_size_m": cfg.aruco_bound_size_m,
        }
    return {
        "table_in_camera": {"matrix": transform_camera_table.tolist()},
        "targets": targets,
    }


def test_marker_body_pose_tracker_update_returns_pose_and_diagnostics() -> None:
    cfg = _sample_cfg()
    camera_matrix = np.array(
        [
            [820.0, 0.0, 320.0],
            [0.0, 815.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform_camera_table = make_transform(
        _rot_z(0.12) @ _rot_y(-0.08),
        np.array([0.03, -0.04, 0.72], dtype=np.float64),
    )
    transform_table_body = make_transform(
        _rot_z(-0.28) @ _rot_y(0.18) @ _rot_x(-0.11),
        np.array([0.16, -0.07, 0.02], dtype=np.float64),
    )
    frame_result = _build_frame_result(cfg, transform_camera_table, transform_table_body, camera_matrix)

    tracker = MarkerBodyPoseTracker(
        cfg,
        pose_solver="joint_pnp",
        smoothing_alpha=0.35,
        outlier_threshold_m=0.02,
        reprojection_error_threshold_px=3.0,
        enable_diagnostics=True,
    )
    result = tracker.update(frame_result=frame_result, camera_matrix=camera_matrix)

    assert result.raw_pose is not None
    assert result.smoothed_pose is not None
    assert result.consistency_report is not None
    np.testing.assert_allclose(result.raw_pose.transform_table_cube, transform_table_body, atol=1e-6)
    np.testing.assert_allclose(result.smoothed_pose.transform_table_cube, transform_table_body, atol=1e-6)
    assert result.consistency_report.marker_ids == [1, 2, 3]


def test_marker_body_pose_tracker_reset_drops_previous_smoothing_state() -> None:
    cfg = _sample_cfg()
    camera_matrix = np.array(
        [
            [820.0, 0.0, 320.0],
            [0.0, 815.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform_camera_table = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 0.7], dtype=np.float64),
    )
    pose_a = make_transform(np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 0.0], dtype=np.float64))
    pose_b = make_transform(_rot_z(0.3), np.array([0.2, -0.1, 0.05], dtype=np.float64))

    tracker = MarkerBodyPoseTracker(
        cfg,
        pose_solver="joint_pnp",
        smoothing_alpha=0.2,
        outlier_threshold_m=0.02,
        reprojection_error_threshold_px=3.0,
    )
    tracker.update(
        frame_result=_build_frame_result(cfg, transform_camera_table, pose_a, camera_matrix),
        camera_matrix=camera_matrix,
    )
    result_b = tracker.update(
        frame_result=_build_frame_result(cfg, transform_camera_table, pose_b, camera_matrix),
        camera_matrix=camera_matrix,
    )
    assert result_b.raw_pose is not None
    assert result_b.smoothed_pose is not None
    assert not np.allclose(result_b.smoothed_pose.transform_table_cube, result_b.raw_pose.transform_table_cube)

    tracker.reset()
    result_after_reset = tracker.update(
        frame_result=_build_frame_result(cfg, transform_camera_table, pose_b, camera_matrix),
        camera_matrix=camera_matrix,
    )
    assert result_after_reset.raw_pose is not None
    assert result_after_reset.smoothed_pose is not None
    np.testing.assert_allclose(
        result_after_reset.smoothed_pose.transform_table_cube,
        result_after_reset.raw_pose.transform_table_cube,
        atol=1e-9,
    )


def test_marker_body_pose_tracker_supports_marker_average_solver() -> None:
    cfg = _sample_cfg()
    camera_matrix = np.array(
        [
            [820.0, 0.0, 320.0],
            [0.0, 815.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform_camera_table = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 0.7], dtype=np.float64),
    )
    transform_table_body = make_transform(
        _rot_z(0.12) @ _rot_x(-0.08),
        np.array([0.06, -0.03, 0.02], dtype=np.float64),
    )
    frame_result = _build_frame_result(cfg, transform_camera_table, transform_table_body, camera_matrix)

    tracker = MarkerBodyPoseTracker(
        cfg,
        pose_solver="marker_average",
        smoothing_alpha=1.0,
        outlier_threshold_m=0.02,
        reprojection_error_threshold_px=3.0,
    )
    result = tracker.update(frame_result=frame_result, camera_matrix=camera_matrix)

    assert result.raw_pose is not None
    assert result.smoothed_pose is not None
    assert result.raw_pose.solver_mode == "marker_average"
    assert result.smoothed_pose.solver_mode == "marker_average"
    np.testing.assert_allclose(result.raw_pose.transform_table_cube, transform_table_body, atol=1e-7)


def test_marker_body_pose_tracker_drops_pose_when_detection_disappears() -> None:
    cfg = _sample_cfg()
    camera_matrix = np.array(
        [
            [820.0, 0.0, 320.0],
            [0.0, 815.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform_camera_table = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.0, 0.0, 0.7], dtype=np.float64),
    )
    transform_table_body = make_transform(
        _rot_z(0.18) @ _rot_x(-0.06),
        np.array([0.04, -0.02, 0.03], dtype=np.float64),
    )
    tracker = MarkerBodyPoseTracker(
        cfg,
        pose_solver="marker_average",
        smoothing_alpha=1.0,
        outlier_threshold_m=0.02,
        reprojection_error_threshold_px=3.0,
    )

    detected = tracker.update(
        frame_result=_build_frame_result(cfg, transform_camera_table, transform_table_body, camera_matrix),
        camera_matrix=camera_matrix,
    )
    assert detected.raw_pose is not None
    assert detected.smoothed_pose is not None

    lost = tracker.update(
        frame_result={
            "time_wall": 1.0 / 30.0,
            "table_in_camera": {"matrix": transform_camera_table.tolist()},
            "targets": {},
        },
        camera_matrix=camera_matrix,
    )
    assert lost.raw_pose is None
    assert lost.smoothed_pose is None
