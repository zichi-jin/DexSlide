from __future__ import annotations

import cv2
import numpy as np
import yaml

from dexslide.kinematics.transforms import (
    invert_transform,
    make_transform,
    transform_points,
    transform_to_rvec_tvec,
)
from dexslide.paths import DIRECT_ARUCO_CALIBRATION_DIR
from dexslide.vision.hand_cube_overlay import (
    HandCubeOverlayConfig,
    MarkerMount,
    compose_overlay_joint_angles,
    diagnose_marker_body_consistency,
    estimate_cube_pose_in_table,
    resolve_marker_body_tag_pose_branches,
    try_load_hand_cube_overlay_config,
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


def _marker_object_points(marker_size_m: float) -> np.ndarray:
    half = 0.5 * float(marker_size_m)
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


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


def _mount(rows: list[list[float]], marker_id: int) -> MarkerMount:
    return MarkerMount(marker_id=marker_id, axes_rows_body=np.asarray(rows, dtype=np.float64))


def _tag_entry_from_candidates(candidates: list[np.ndarray], *, reproj_errors: list[float]) -> dict[str, object]:
    if len(candidates) != len(reproj_errors):
        raise ValueError("candidates and reproj_errors must have the same length")
    pose_candidates = []
    for transform_camera_marker, reproj_error in zip(candidates, reproj_errors):
        rvec, tvec = transform_to_rvec_tvec(transform_camera_marker)
        pose_candidates.append(
            {
                "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
                "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
                "reprojection_error_px": float(reproj_error),
            }
        )
    primary = pose_candidates[0]
    return {
        "rvec": np.asarray(primary["rvec"], dtype=np.float64).reshape(3),
        "tvec": np.asarray(primary["tvec"], dtype=np.float64).reshape(3),
        "pose_candidates": pose_candidates,
    }


def _sample_cfg() -> HandCubeOverlayConfig:
    return HandCubeOverlayConfig(
        hand="left",
        aruco_bound_size_m=0.03,
        marker_square_size_m=0.04,
        markers={
            1: _mount([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 1),
            2: _mount(
                [
                    [-SQRT2_INV, SQRT2_INV, 0.0],
                    [0.0, 0.0, 1.0],
                    [SQRT2_INV, SQRT2_INV, 0.0],
                ],
                2,
            ),
            3: _mount([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], 3),
        },
    )


def test_repo_left_tags2marker_json_loads() -> None:
    cfg = HandCubeOverlayConfig.load(DIRECT_ARUCO_CALIBRATION_DIR / "left_tags2marker.json")
    assert cfg.hand == "left"
    assert len(cfg.marker_ids()) == 18


def test_marker_mount_rejects_non_right_handed_axes() -> None:
    try:
        MarkerMount(
            marker_id=1,
            axes_rows_body=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=np.float64,
            ),
        )
    except ValueError as exc:
        assert "marker_id=1" in str(exc)
        assert "right-handed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid marker axes to raise ValueError")


def test_load_invalid_yaml_reports_marker_face_id(tmp_path) -> None:
    path = tmp_path / "bad_marker_body.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "hand": "left",
                "aruco_dict": {"predefined": "DICT_4X4_50"},
                "aruco_bound_size_mm": 30.0,
                "marker_square_size_mm": 40.0,
                "marker_face_id": {
                    13: [
                        [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    try:
        HandCubeOverlayConfig.load(path)
    except ValueError as exc:
        message = str(exc)
        assert "marker_face_id[13]" in message
        assert str(path) in message
        assert "right-handed" in message
    else:  # pragma: no cover
        raise AssertionError("Expected invalid YAML to raise ValueError")


def test_estimate_cube_pose_in_table_recovers_single_marker_pose() -> None:
    cfg = _sample_cfg()
    cfg.markers = {1: cfg.markers[1]}
    transform_table_body = make_transform(
        _rot_z(0.25) @ _rot_y(-0.15) @ _rot_x(0.1),
        np.array([0.12, -0.34, 0.56], dtype=np.float64),
    )
    transform_table_marker = transform_table_body @ cfg.markers[1].body_to_marker_transform(
        cfg.marker_center_radius_m
    )
    frame_result = {
        "targets": {
            "1": {
                "target_in_table": {
                    "matrix": transform_table_marker.tolist(),
                }
            }
        }
    }

    estimate = estimate_cube_pose_in_table(frame_result, cfg)
    assert estimate is not None
    np.testing.assert_allclose(estimate.transform_table_cube, transform_table_body, atol=1e-7)
    assert estimate.source_marker_ids == [1]
    assert estimate.max_position_deviation_m == 0.0


def test_estimate_cube_pose_in_table_averages_multiple_marker_views() -> None:
    cfg = _sample_cfg()
    transform_table_body = make_transform(
        _rot_z(-0.35) @ _rot_y(0.22) @ _rot_x(-0.18),
        np.array([-0.18, 0.09, 0.42], dtype=np.float64),
    )

    frame_targets = {}
    for marker_id, marker_mount in cfg.markers.items():
        transform_body_marker = marker_mount.body_to_marker_transform(cfg.marker_center_radius_m)
        transform_table_marker = transform_table_body @ transform_body_marker
        frame_targets[str(marker_id)] = {
            "target_in_table": {
                "matrix": transform_table_marker.tolist(),
            }
        }

    estimate = estimate_cube_pose_in_table({"targets": frame_targets}, cfg)
    assert estimate is not None
    np.testing.assert_allclose(estimate.transform_table_cube, transform_table_body, atol=1e-7)


def test_resolve_marker_body_tag_pose_branches_prefers_multi_marker_consensus() -> None:
    cfg = _sample_cfg()
    transform_camera_body_true = make_transform(
        _rot_y(0.24) @ _rot_x(np.pi * 0.94) @ _rot_z(-0.08),
        np.array([0.02, -0.01, 0.42], dtype=np.float64),
    )
    transform_camera_body_wrong = make_transform(
        _rot_y(-0.18) @ _rot_x(np.pi * 0.88) @ _rot_z(0.31),
        np.array([0.055, 0.024, 0.455], dtype=np.float64),
    )

    tag_dict: dict[int, dict[str, object]] = {}
    for marker_id, marker_mount in cfg.markers.items():
        transform_body_marker = marker_mount.body_to_marker_transform(cfg.marker_center_radius_m)
        transform_camera_marker_true = transform_camera_body_true @ transform_body_marker
        if marker_id == 2:
            transform_camera_marker_wrong = transform_camera_body_wrong @ transform_body_marker
            tag_dict[marker_id] = _tag_entry_from_candidates(
                [transform_camera_marker_wrong, transform_camera_marker_true],
                reproj_errors=[0.02, 0.08],
            )
        else:
            tag_dict[marker_id] = _tag_entry_from_candidates(
                [transform_camera_marker_true],
                reproj_errors=[0.03],
            )

    report = resolve_marker_body_tag_pose_branches(tag_dict, cfg)

    assert report["resolved_marker_ids"] == [1, 2, 3]
    selected_transform_camera_marker = make_transform(
        cv2.Rodrigues(np.asarray(tag_dict[2]["rvec"], dtype=np.float64).reshape(3, 1))[0],
        np.asarray(tag_dict[2]["tvec"], dtype=np.float64).reshape(3),
    )
    expected_transform_camera_marker = (
        transform_camera_body_true @ cfg.markers[2].body_to_marker_transform(cfg.marker_center_radius_m)
    )
    np.testing.assert_allclose(
        selected_transform_camera_marker,
        expected_transform_camera_marker,
        atol=1e-7,
    )
    assert tag_dict[2]["selected_pose_candidate_index"] == 1


def test_resolve_marker_body_tag_pose_branches_uses_reference_for_single_marker() -> None:
    cfg = _sample_cfg()
    cfg.markers = {1: cfg.markers[1]}
    transform_camera_body_true = make_transform(
        _rot_y(0.18) @ _rot_x(np.pi * 0.93) @ _rot_z(0.04),
        np.array([0.01, -0.015, 0.40], dtype=np.float64),
    )
    transform_camera_body_wrong = make_transform(
        _rot_y(-0.21) @ _rot_x(np.pi * 0.89) @ _rot_z(-0.28),
        np.array([0.05, 0.03, 0.455], dtype=np.float64),
    )
    transform_body_marker = cfg.markers[1].body_to_marker_transform(cfg.marker_center_radius_m)
    tag_dict = {
        1: _tag_entry_from_candidates(
            [
                transform_camera_body_wrong @ transform_body_marker,
                transform_camera_body_true @ transform_body_marker,
            ],
            reproj_errors=[0.01, 0.09],
        )
    }

    report = resolve_marker_body_tag_pose_branches(
        tag_dict,
        cfg,
        reference_camera_body=transform_camera_body_true,
    )

    assert report["resolved_marker_ids"] == [1]
    assert tag_dict[1]["selected_pose_candidate_index"] == 1


def test_estimate_cube_pose_in_table_marker_average_mode_uses_per_marker_average() -> None:
    cfg = _sample_cfg()
    transform_table_body = make_transform(
        _rot_z(-0.35) @ _rot_y(0.22) @ _rot_x(-0.18),
        np.array([-0.18, 0.09, 0.42], dtype=np.float64),
    )

    frame_targets = {}
    for marker_id, marker_mount in cfg.markers.items():
        transform_body_marker = marker_mount.body_to_marker_transform(cfg.marker_center_radius_m)
        transform_table_marker = transform_table_body @ transform_body_marker
        frame_targets[str(marker_id)] = {
            "target_in_table": {
                "matrix": transform_table_marker.tolist(),
            }
        }

    estimate = estimate_cube_pose_in_table(
        {"targets": frame_targets},
        cfg,
        pose_solver="marker_average",
    )
    assert estimate is not None
    assert estimate.solver_mode == "marker_average"
    np.testing.assert_allclose(estimate.transform_table_cube, transform_table_body, atol=1e-7)
    assert estimate.source_marker_ids == [1, 2, 3]


def test_estimate_cube_pose_in_table_rejects_position_outlier_when_three_markers_exist() -> None:
    cfg = _sample_cfg()
    transform_table_body = make_transform(
        _rot_z(0.1) @ _rot_y(-0.2),
        np.array([0.05, -0.12, 0.48], dtype=np.float64),
    )

    frame_targets = {}
    for marker_id, marker_mount in cfg.markers.items():
        transform_table_marker = transform_table_body @ marker_mount.body_to_marker_transform(
            cfg.marker_center_radius_m
        )
        if marker_id == 3:
            transform_table_marker = transform_table_marker.copy()
            transform_table_marker[:3, 3] += np.array([0.06, -0.04, 0.03], dtype=np.float64)
        frame_targets[str(marker_id)] = {
            "target_in_camera": {"matrix": np.eye(4, dtype=np.float64).tolist()},
            "target_in_table": {"matrix": transform_table_marker.tolist()},
        }

    estimate = estimate_cube_pose_in_table(
        {"targets": frame_targets},
        cfg,
        outlier_threshold_m=0.02,
    )
    assert estimate is not None
    assert estimate.source_marker_ids == [1, 2]
    np.testing.assert_allclose(estimate.transform_table_cube[:3, 3], transform_table_body[:3, 3], atol=1e-7)


def test_estimate_cube_pose_in_table_joint_pnp_recovers_body_pose() -> None:
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

    estimate = estimate_cube_pose_in_table(
        {
            "table_in_camera": {"matrix": transform_camera_table.tolist()},
            "targets": targets,
        },
        cfg,
        camera_matrix=camera_matrix,
        reprojection_error_threshold_px=3.0,
    )

    assert estimate is not None
    assert estimate.solver_mode == "joint_pnp"
    np.testing.assert_allclose(estimate.transform_table_cube, transform_table_body, atol=1e-6)
    assert estimate.source_marker_ids == [1, 2, 3]
    assert estimate.mean_reprojection_error_px <= 1e-6


def test_estimate_cube_pose_in_table_joint_pnp_drops_bad_marker() -> None:
    cfg = _sample_cfg()
    camera_matrix = np.array(
        [
            [900.0, 0.0, 320.0],
            [0.0, 900.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform_camera_table = make_transform(
        _rot_y(0.07),
        np.array([0.0, -0.02, 0.68], dtype=np.float64),
    )
    transform_table_body = make_transform(
        _rot_z(0.14) @ _rot_x(-0.09),
        np.array([0.08, 0.05, -0.01], dtype=np.float64),
    )
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
        if marker_id == 3:
            projected = projected + np.array([[18.0, -15.0]], dtype=np.float64)
        targets[str(marker_id)] = {
            "detected": True,
            "target_in_camera": {"matrix": transform_camera_marker.tolist()},
            "target_in_table": {"matrix": transform_table_marker.tolist()},
            "undistorted_corners": projected.tolist(),
            "marker_size_m": cfg.aruco_bound_size_m,
        }

    estimate = estimate_cube_pose_in_table(
        {
            "table_in_camera": {"matrix": transform_camera_table.tolist()},
            "targets": targets,
        },
        cfg,
        camera_matrix=camera_matrix,
        reprojection_error_threshold_px=2.5,
    )

    assert estimate is not None
    assert estimate.solver_mode == "joint_pnp"
    assert estimate.source_marker_ids == [1, 2]
    np.testing.assert_allclose(estimate.transform_table_cube, transform_table_body, atol=2e-4)
    assert estimate.max_reprojection_error_px < 2.5


def test_diagnose_marker_body_consistency_flags_yaml_orientation_mismatch() -> None:
    cfg_true = _sample_cfg()
    cfg_bad = _sample_cfg()
    cfg_bad.markers[3] = _mount([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], 3)
    transform_table_body = make_transform(
        _rot_z(-0.21) @ _rot_y(0.13) @ _rot_x(-0.05),
        np.array([0.11, -0.06, 0.38], dtype=np.float64),
    )

    targets: dict[str, dict[str, object]] = {}
    for marker_id, mount in cfg_true.markers.items():
        transform_table_marker = transform_table_body @ mount.body_to_marker_transform(
            cfg_true.marker_center_radius_m
        )
        targets[str(marker_id)] = {
            "target_in_table": {"matrix": transform_table_marker.tolist()},
        }

    report = diagnose_marker_body_consistency({"targets": targets}, cfg_bad)
    assert report is not None
    assert report.marker_ids == [1, 2, 3]
    worst = report.items[0]
    assert worst.marker_id == 3
    assert worst.peer_rotation_error_deg > 20.0


def test_hand_cube_overlay_config_roundtrip_preserves_mount_transform(tmp_path) -> None:
    cfg = HandCubeOverlayConfig(
        hand="left",
        aruco_bound_size_m=0.03,
        marker_square_size_m=0.04,
        markers={
            5: _mount([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], 5),
        },
    )
    transform_body_wrist = make_transform(
        _rot_z(0.4) @ _rot_x(-0.2),
        np.array([-0.00265, 0.09, -0.0701], dtype=np.float64),
    )
    cfg.set_cube_to_wrist_transform(transform_body_wrist)
    cfg.set_joint_zero(np.linspace(-0.2, 0.3, 20, dtype=np.float64))
    cfg.set_joint_base_render(np.linspace(0.1, 0.6, 20, dtype=np.float64))
    path = tmp_path / "hand_marker_body_left.yaml"
    cfg.save(path)

    loaded = HandCubeOverlayConfig.load(path)
    np.testing.assert_allclose(loaded.cube_to_wrist_transform(), transform_body_wrist, atol=1e-7)
    np.testing.assert_allclose(np.asarray(loaded.joint_zero_rad, dtype=np.float64), np.linspace(-0.2, 0.3, 20))
    np.testing.assert_allclose(
        np.asarray(loaded.joint_base_render_rad, dtype=np.float64),
        np.linspace(0.1, 0.6, 20),
    )
    assert loaded.marker_ids() == [5]
    assert loaded.build_target_aruco_config()["marker_size_map"][5] == 0.03


def test_invert_transform_roundtrip_for_hand_cube_helper() -> None:
    transform = make_transform(
        _rot_y(0.3) @ _rot_z(-0.5),
        np.array([0.3, -0.1, 0.8], dtype=np.float64),
    )
    inv = invert_transform(transform)
    np.testing.assert_allclose(transform @ inv, np.eye(4), atol=1e-7)
    np.testing.assert_allclose(inv @ transform, np.eye(4), atol=1e-7)


def test_compose_overlay_joint_angles_uses_frozen_render_pose_as_baseline() -> None:
    joint_zero = np.linspace(-0.3, 0.2, 20, dtype=np.float64)
    joint_base = np.linspace(0.1, 0.6, 20, dtype=np.float64)
    raw_same_as_zero = joint_zero.copy()
    raw_offset = joint_zero + 0.05

    np.testing.assert_allclose(
        compose_overlay_joint_angles(raw_same_as_zero, joint_zero, joint_base),
        joint_base,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        compose_overlay_joint_angles(raw_offset, joint_zero, joint_base),
        joint_base + 0.05,
        atol=1e-9,
    )


def test_try_load_hand_cube_overlay_config_returns_none_for_invalid_yaml(tmp_path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("{not valid yaml", encoding="utf-8")
    assert try_load_hand_cube_overlay_config(path) is None
