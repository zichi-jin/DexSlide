from __future__ import annotations

import numpy as np

from dexslide.calibration.realsense_hand_preview import (
    _compose_display_frame,
    _estimate_bbox_xyxy,
    _sample_landmark_depth_mm,
)


def test_estimate_bbox_xyxy_clamps_to_image_extent() -> None:
    keypoints = np.array(
        [
            [-10.0, 15.0],
            [30.0, 25.0],
            [70.0, 55.0],
        ],
        dtype=np.float32,
    )

    bbox = _estimate_bbox_xyxy(keypoints, (60, 80, 3), padding_px=8)

    assert bbox == (0, 7, 78, 59)


def test_sample_landmark_depth_mm_ignores_zero_depth_and_uses_median() -> None:
    depth_mm = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 410, 420, 430, 0],
            [0, 415, 0, 440, 0],
            [0, 405, 425, 435, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint16,
    )
    keypoints = np.array([[2.0, 2.0]], dtype=np.float32)

    depth_value = _sample_landmark_depth_mm(depth_mm, keypoints, window_radius=1)

    assert depth_value == 422.5


def test_compose_display_frame_split_concatenates_horizontally() -> None:
    color = np.zeros((4, 6, 3), dtype=np.uint8)
    depth = np.ones((4, 6, 3), dtype=np.uint8) * 255

    merged = _compose_display_frame(color, depth, view="split")

    assert merged.shape == (4, 12, 3)
    np.testing.assert_array_equal(merged[:, :6], color)
    np.testing.assert_array_equal(merged[:, 6:], depth)
