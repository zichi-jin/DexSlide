from __future__ import annotations

import numpy as np

from dexslide.calibration.dexalign.objective import (
    KEYPOINT_CLASS_WEIGHT_MAP,
    KEYPOINT_CLASS_WEIGHTS,
    alignment_residual_vector,
    compose_parameter_vector,
    pack_marker2hand_transform,
)
from dexslide.calibration.dexalign.skeleton_param import flatten_skeleton
from dexslide.calibration.dexalign.types import AlignmentDataset, AlignmentFrame
from dexslide.paths import DEFAULT_SKELETON_FILE
from dexslide.retargeting.human_model import HUMAN_LANDMARK_NAMES, DexSlideHumanModel
from dexslide.world_pose.hand_cube_overlay import make_transform


def _base_dataset() -> tuple[dict, np.ndarray, np.ndarray]:
    model = DexSlideHumanModel(DEFAULT_SKELETON_FILE, hand="left", unit_scale=1.0)
    observed = model.landmarks_from_angles(np.zeros(20, dtype=np.float64))
    transform = make_transform(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
    return model.skeleton, transform, observed


def test_alignment_residual_vector_is_zero_when_all_weights_zero() -> None:
    skeleton, marker_transform, observed = _base_dataset()
    frame = AlignmentFrame(
        timestamp=0.0,
        camera_T_marker=np.eye(4, dtype=np.float64),
        q_encoder_rad20=np.zeros(20, dtype=np.float64),
        keypoints_camera_mm=observed,
        keypoint_confidence=np.zeros(21, dtype=np.float64),
        keypoint_valid_mask=np.zeros(21, dtype=bool),
    )
    dataset = AlignmentDataset(hand="left", frames=(frame,))
    x = compose_parameter_vector(flatten_skeleton(skeleton), pack_marker2hand_transform(marker_transform))

    residual = alignment_residual_vector(x, dataset, skeleton)

    assert np.allclose(residual, 0.0)


def test_alignment_residual_vector_matches_manual_single_point_case() -> None:
    skeleton, marker_transform, observed = _base_dataset()
    observed_shifted = observed.copy()
    delta = np.array([1.5, -2.0, 3.0], dtype=np.float64)
    target_idx = 8
    observed_shifted[target_idx] = observed_shifted[target_idx] + delta
    confidence = np.zeros(21, dtype=np.float64)
    confidence[[0, 5, target_idx]] = 1.0
    valid_mask = np.zeros(21, dtype=bool)
    valid_mask[[0, 5, target_idx]] = True

    frame = AlignmentFrame(
        timestamp=0.0,
        camera_T_marker=np.eye(4, dtype=np.float64),
        q_encoder_rad20=np.zeros(20, dtype=np.float64),
        keypoints_camera_mm=observed_shifted,
        keypoint_confidence=confidence,
        keypoint_valid_mask=valid_mask,
    )
    dataset = AlignmentDataset(hand="left", frames=(frame,))
    x = compose_parameter_vector(flatten_skeleton(skeleton), pack_marker2hand_transform(marker_transform))

    residual = alignment_residual_vector(x, dataset, skeleton).reshape(1, 21, 3)

    expected = -KEYPOINT_CLASS_WEIGHTS[target_idx] * delta
    np.testing.assert_allclose(residual[0, target_idx], expected, atol=1e-9)
    np.testing.assert_allclose(residual[0, 0], np.zeros(3, dtype=np.float64), atol=1e-9)
    np.testing.assert_allclose(residual[0, 5], np.zeros(3, dtype=np.float64), atol=1e-9)


def test_alignment_residual_vector_ignores_invalid_nan_points() -> None:
    skeleton, marker_transform, observed = _base_dataset()
    observed_invalid = observed.copy()
    observed_invalid[12] = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    confidence = np.ones(21, dtype=np.float64)
    valid_mask = np.ones(21, dtype=bool)
    valid_mask[12] = False

    frame = AlignmentFrame(
        timestamp=0.0,
        camera_T_marker=np.eye(4, dtype=np.float64),
        q_encoder_rad20=np.zeros(20, dtype=np.float64),
        keypoints_camera_mm=observed_invalid,
        keypoint_confidence=confidence,
        keypoint_valid_mask=valid_mask,
    )
    dataset = AlignmentDataset(hand="left", frames=(frame,))
    x = compose_parameter_vector(flatten_skeleton(skeleton), pack_marker2hand_transform(marker_transform))

    residual = alignment_residual_vector(x, dataset, skeleton)

    assert np.all(np.isfinite(residual))
    reshaped = residual.reshape(1, 21, 3)
    np.testing.assert_allclose(reshaped[0, 12], np.zeros(3, dtype=np.float64), atol=1e-9)


def test_keypoint_class_weight_map_explicitly_covers_all_landmarks() -> None:
    assert tuple(KEYPOINT_CLASS_WEIGHT_MAP.keys()) == tuple(HUMAN_LANDMARK_NAMES)
    expected = np.asarray([KEYPOINT_CLASS_WEIGHT_MAP[name] for name in HUMAN_LANDMARK_NAMES], dtype=np.float64)
    np.testing.assert_allclose(KEYPOINT_CLASS_WEIGHTS, expected, atol=0.0)
