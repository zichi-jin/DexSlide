from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np

import scripts.view_dexalign_capture_3d as preview
from dexslide.kinematics.transforms import make_transform


class _FakeDetector:
    def __init__(self) -> None:
        self.called = False

    def draw_landmarks(self, image_bgr: np.ndarray, keypoints_2d: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        self.called = True
        output = image_bgr.copy()
        point = tuple(np.round(np.asarray(keypoints_2d[0], dtype=np.float64)).astype(int))
        output[point[1], point[0]] = np.array([255, 255, 255], dtype=np.uint8)
        return output


def test_estimate_bbox_xyxy_clamps_to_image_extent() -> None:
    keypoints = np.array(
        [
            [-10.0, 15.0],
            [30.0, 25.0],
            [70.0, 55.0],
        ],
        dtype=np.float32,
    )

    bbox = preview._estimate_bbox_xyxy(keypoints, (60, 80, 3), padding_px=8)

    assert bbox == (0, 7, 78, 59)


def test_compose_camera_overlay_draws_keypoints_and_status() -> None:
    detector = _FakeDetector()
    color = np.zeros((120, 160, 3), dtype=np.uint8)
    keypoints = np.stack([np.array([40.0 + idx, 30.0 + idx], dtype=np.float64) for idx in range(21)], axis=0)
    confidence = np.ones(21, dtype=np.float64)
    valid_mask = np.zeros(21, dtype=bool)
    valid_mask[:12] = True
    depth_mm = np.full(21, np.nan, dtype=np.float64)
    depth_mm[:12] = 450.0
    marker_pose = preview.MarkerPoseObservation(
        camera_T_marker_mm=make_transform(np.eye(3, dtype=np.float64), np.array([10.0, 20.0, 500.0], dtype=np.float64)),
        marker_ids_used=(1, 2),
        marker_reproj_error_px=1.5,
    )

    overlay = preview._compose_camera_overlay(
        color,
        detector=detector,
        detection=(keypoints, confidence),
        marker_pose=marker_pose,
        valid_mask=valid_mask,
        depth_mm=depth_mm,
        fps_ema=27.5,
        frame_idx=8,
    )

    assert detector.called is True
    assert overlay.shape == color.shape
    assert np.count_nonzero(overlay) > 0


def test_drain_camera_preview_commands_returns_space_events() -> None:
    command_queue: queue.Queue[str] = queue.Queue()
    command_queue.put("space")
    command_queue.put("space")

    commands = preview.drain_camera_preview_commands(
        SimpleNamespace(command_queue=command_queue)
    )

    assert commands == ["space", "space"]
    assert command_queue.empty()


def test_preview_parser_defaults_to_intrinsics_file() -> None:
    parser = preview._build_parser()

    args = parser.parse_args([])

    assert args.width == 960
    assert args.height == 540
    assert args.enable_table_frame is False


def test_format_marker2hand_pose_text_includes_translation_and_rotation() -> None:
    transform = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.061, 0.093, -0.130], dtype=np.float64),
    )

    text = preview._format_marker2hand_pose_text(transform)

    assert "marker2hand_6d" in text
    assert "t_m=[+0.061, +0.093, -0.130]" in text
    assert "rvec_deg=[+0.0, +0.0, +0.0]" in text


def test_build_keypoint_text_lines_lists_all_21_points() -> None:
    keypoints_camera_mm = np.full((21, 3), np.nan, dtype=np.float64)
    keypoints_camera_mm[0] = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    valid_mask = np.zeros(21, dtype=bool)
    valid_mask[0] = True

    lines = preview._build_keypoint_text_lines(keypoints_camera_mm, valid_mask, frame_name="table")

    assert len(lines) == 22
    assert lines[0] == "mediapipe_keypoints_xyz_table_mm:"
    assert lines[1].startswith("00 wrist")
    assert "ok" in lines[1]
    assert "[  +10.0,   +20.0,   +30.0]" in lines[1]
    assert any(line.startswith("20 pinky_tip") for line in lines)


def test_transform_keypoints_to_frame_applies_table_transform_and_preserves_invalid_points() -> None:
    transform_camera_table = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.1, -0.2, 0.3], dtype=np.float64),
    )
    keypoints_camera_m = np.full((21, 3), np.nan, dtype=np.float64)
    keypoints_camera_m[0] = np.array([0.4, 0.5, 0.6], dtype=np.float64)

    transformed = preview._transform_keypoints_to_frame(keypoints_camera_m, transform_camera_table)

    np.testing.assert_allclose(transformed[0], np.array([0.5, 0.3, 0.9], dtype=np.float64), atol=1e-9)
    assert np.isnan(transformed[1]).all()


def test_thumb_base_freeze_tracks_palm_motion_instead_of_staying_in_camera_frame() -> None:
    palm_pose_a = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.10, -0.02, 0.60], dtype=np.float64),
    )
    palm_pose_b = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.16, 0.01, 0.72], dtype=np.float64),
    )
    keypoints_camera_m = np.full((21, 3), np.nan, dtype=np.float64)
    keypoints_camera_m[preview.THUMB_BASE_INDEX] = np.array([0.12, -0.01, 0.62], dtype=np.float64)
    valid_mask = np.zeros(21, dtype=bool)
    valid_mask[preview.THUMB_BASE_INDEX] = True

    frozen_local = preview._capture_thumb_base_local_point(palm_pose_a, keypoints_camera_m, valid_mask)
    assert frozen_local is not None

    transformed_keypoints, transformed_valid, frozen_camera_m, mode = preview._apply_thumb_base_freeze(
        keypoints_camera_m,
        valid_mask,
        palm_pose_b,
        frozen_local,
    )

    expected = preview.transform_points(palm_pose_b, np.asarray(frozen_local, dtype=np.float64).reshape(1, 3))[0]
    assert mode == "frozen_tracking"
    assert transformed_valid[preview.THUMB_BASE_INDEX]
    np.testing.assert_allclose(frozen_camera_m, expected, atol=1e-9)
    np.testing.assert_allclose(transformed_keypoints[preview.THUMB_BASE_INDEX], expected, atol=1e-9)


def test_estimate_runtime_hand_pose_from_keypoints_recovers_known_pose() -> None:
    hand_pose = make_transform(
        np.eye(3, dtype=np.float64),
        np.array([0.120, -0.030, 0.650], dtype=np.float64),
    )
    palm_local = np.full((21, 3), np.nan, dtype=np.float64)
    palm_local[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    palm_local[5] = np.array([0.030, -0.030, 0.0], dtype=np.float64)
    palm_local[9] = np.array([0.030, -0.010, 0.0], dtype=np.float64)
    palm_local[13] = np.array([0.030, 0.010, 0.0], dtype=np.float64)
    palm_local[17] = np.array([0.030, 0.030, 0.0], dtype=np.float64)
    keypoints_camera_m = palm_local.copy()
    keypoints_camera_m[:, :3] = np.where(
        np.isfinite(keypoints_camera_m[:, :3]),
        keypoints_camera_m[:, :3] + hand_pose[:3, 3][None, :],
        np.nan,
    )
    valid_mask = np.isfinite(keypoints_camera_m).all(axis=1)

    estimated = preview._estimate_runtime_hand_pose_from_keypoints(keypoints_camera_m, valid_mask)

    assert estimated is not None
    np.testing.assert_allclose(estimated[:3, 3], hand_pose[:3, 3], atol=1e-9)
    np.testing.assert_allclose(estimated[:3, :3], hand_pose[:3, :3], atol=1e-9)
