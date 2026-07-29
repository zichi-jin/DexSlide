import math

import cv2
import numpy as np
import yaml

from dexslide.kinematics.transforms import invert_transform, rvec_tvec_to_transform
from dexslide import paths
from dexslide.vision import aruco_pose_tracker as apt
from dexslide.vision import direct_aruco_tracker as dat


def test_normalize_target_marker_ids_excludes_duplicates_and_table() -> None:
    marker_ids = dat._normalize_target_marker_ids([5, 0, 5, 7, 6, 7], table_marker_id=0)
    assert marker_ids == [5, 6, 7]


def test_invert_transform_roundtrip() -> None:
    transform = rvec_tvec_to_transform(
        np.array([0.2, -0.3, 0.4], dtype=np.float64),
        np.array([0.5, -0.6, 0.7], dtype=np.float64),
    )
    inv = invert_transform(transform)
    np.testing.assert_allclose(transform @ inv, np.eye(4), atol=1e-7)
    np.testing.assert_allclose(inv @ transform, np.eye(4), atol=1e-7)


def test_relative_transform_from_camera_poses_recovers_known_target_pose() -> None:
    t_camera_table = rvec_tvec_to_transform(
        np.array([0.0, 0.0, math.pi / 3.0], dtype=np.float64),
        np.array([0.4, -0.2, 1.1], dtype=np.float64),
    )
    t_table_target = rvec_tvec_to_transform(
        np.array([0.1, -0.2, 0.05], dtype=np.float64),
        np.array([0.15, 0.25, -0.05], dtype=np.float64),
    )
    t_camera_target = t_camera_table @ t_table_target

    recovered = dat._relative_transform_from_camera_poses(t_camera_table, t_camera_target)
    np.testing.assert_allclose(recovered, t_table_target, atol=1e-7)


def test_pose_dict_from_transform_contains_expected_position_and_matrix() -> None:
    transform = rvec_tvec_to_transform(
        np.zeros(3, dtype=np.float64),
        np.array([1.2, -3.4, 5.6], dtype=np.float64),
    )
    pose_dict = dat._pose_dict_from_transform(transform)

    np.testing.assert_allclose(pose_dict["position_m"], np.array([1.2, -3.4, 5.6]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(pose_dict["matrix"], dtype=np.float64), transform, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(pose_dict["quaternion_xyzw"], dtype=np.float64),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        atol=1e-7,
    )


def test_direct_aruco_default_assets_exist_and_match_expected_sizes() -> None:
    assert paths.DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE.is_file()
    assert paths.DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE.is_file()
    assert paths.DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE.is_file()

    with paths.DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE.open("r", encoding="utf-8") as handle:
        table_cfg = yaml.safe_load(handle)
    with paths.DEFAULT_DIRECT_ARUCO_TARGET_CONFIG_FILE.open("r", encoding="utf-8") as handle:
        target_cfg = yaml.safe_load(handle)

    assert table_cfg["aruco_dict"]["predefined"] == "DICT_4X4_50"
    assert table_cfg["marker_size_map"]["0"] == 0.195
    assert target_cfg["aruco_dict"]["predefined"] == "DICT_4X4_50"
    assert target_cfg["marker_size_map"]["default"] == 0.02


def test_build_direct_aruco_frame_result_emits_relative_target_pose() -> None:
    t_camera_table = rvec_tvec_to_transform(
        np.array([0.0, 0.0, 0.2], dtype=np.float64),
        np.array([0.1, -0.3, 0.8], dtype=np.float64),
    )
    t_table_target = rvec_tvec_to_transform(
        np.array([0.05, -0.1, 0.15], dtype=np.float64),
        np.array([0.2, 0.05, -0.02], dtype=np.float64),
    )
    t_camera_target = t_camera_table @ t_table_target

    table_rvec, _ = cv2.Rodrigues(t_camera_table[:3, :3])
    target_rvec, _ = cv2.Rodrigues(t_camera_target[:3, :3])
    tag_dict = {
        0: {
            "rvec": table_rvec.reshape(3),
            "tvec": t_camera_table[:3, 3],
            "corners": np.zeros((4, 2), dtype=np.float64),
        },
        5: {
            "rvec": target_rvec.reshape(3),
            "tvec": t_camera_target[:3, 3],
            "corners": np.zeros((4, 2), dtype=np.float64),
        },
    }

    frame_result = dat._build_direct_aruco_frame_result(
        frame_idx=7,
        image_size=[1280, 720],
        # image_size=[960, 540],
        table_marker_id=0,
        target_marker_ids=[5],
        tag_dict=tag_dict,
        time_wall=123.0,
    )

    assert frame_result["frame_idx"] == 7
    assert frame_result["detected_ids"] == [0, 5]
    assert frame_result["table_detected"] is True
    assert frame_result["n_world_targets"] == 1
    recovered = np.asarray(frame_result["targets"]["5"]["target_in_table"]["matrix"], dtype=np.float64)
    np.testing.assert_allclose(recovered, t_table_target, atol=1e-7)
    assert frame_result["targets"]["5"]["corners"] == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert frame_result["targets"]["5"]["undistorted_corners"] == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert frame_result["targets"]["5"]["marker_size_m"] == 0.0


def test_motion_tolerant_detector_parameters_expand_detection_range() -> None:
    param = apt._configure_aruco_detector_parameters(
        apt._aruco_detector_parameters(),
        refine_subpix=True,
        motion_tolerant=True,
    )

    assert param.adaptiveThreshWinSizeMin == 3
    assert param.adaptiveThreshWinSizeStep == 4
    assert param.minMarkerPerimeterRate <= 0.01
    assert param.maxMarkerPerimeterRate >= 4.0
    assert param.polygonalApproxAccuracyRate >= 0.06
    assert param.errorCorrectionRate >= 0.75


def test_corner_refine_mode_can_switch_to_apriltag() -> None:
    param = apt._configure_aruco_detector_parameters(
        apt._aruco_detector_parameters(),
        refine_subpix=True,
        motion_tolerant=False,
        corner_refine_mode="apriltag",
    )

    assert param.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_APRILTAG


def test_estimate_single_marker_pose_candidates_exposes_ippe_branches() -> None:
    marker_size_m = 0.03
    object_points = apt._marker_square_object_points(marker_size_m)
    camera_matrix = np.array(
        [
            [520.0, 0.0, 320.0],
            [0.0, 520.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi * 0.95), -np.sin(np.pi * 0.95)],
            [0.0, np.sin(np.pi * 0.95), np.cos(np.pi * 0.95)],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [np.cos(0.22), 0.0, np.sin(0.22)],
            [0.0, 1.0, 0.0],
            [-np.sin(0.22), 0.0, np.cos(0.22)],
        ],
        dtype=np.float64,
    )
    transform_camera_marker = dat._transform_from_rvec_tvec(
        cv2.Rodrigues(rot_y @ rot_x)[0].reshape(3),
        np.array([0.03, -0.01, 0.45], dtype=np.float64),
    )
    rvec_true, _ = cv2.Rodrigues(transform_camera_marker[:3, :3])
    corners, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        rvec_true,
        transform_camera_marker[:3, 3].reshape(3, 1),
        camera_matrix,
        np.zeros((1, 5), dtype=np.float64),
    )
    candidates = apt._estimate_single_marker_pose_candidates(
        corners.reshape(1, 4, 2),
        marker_size_m,
        camera_matrix,
    )

    assert len(candidates) >= 2
    assert float(candidates[0]["reprojection_error_px"]) <= float(candidates[1]["reprojection_error_px"])
