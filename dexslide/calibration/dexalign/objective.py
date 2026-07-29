from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dexslide.retargeting.human_model import HUMAN_LANDMARK_NAMES, DexSlideHumanModel
from dexslide.kinematics.transforms import (
    rvec_tvec_to_transform,
    transform_points,
    transform_to_rvec_tvec,
)

from .skeleton_param import SKELETON_PARAM_SIZE, unflatten_skeleton
from .types import AlignmentDataset
from .weights import KEYPOINT_CLASS_WEIGHT_MAP


def _build_keypoint_class_weights() -> np.ndarray:
    missing = [name for name in HUMAN_LANDMARK_NAMES if name not in KEYPOINT_CLASS_WEIGHT_MAP]
    extras = [name for name in KEYPOINT_CLASS_WEIGHT_MAP if name not in HUMAN_LANDMARK_NAMES]
    if missing or extras:
        raise ValueError(
            "KEYPOINT_CLASS_WEIGHT_MAP keys must exactly match HUMAN_LANDMARK_NAMES. "
            f"missing={missing}, extras={extras}"
        )
    weights = np.asarray(
        [float(KEYPOINT_CLASS_WEIGHT_MAP[name]) for name in HUMAN_LANDMARK_NAMES],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all():
        raise ValueError("KEYPOINT_CLASS_WEIGHT_MAP must contain only finite values.")
    return weights


KEYPOINT_CLASS_WEIGHTS = _build_keypoint_class_weights()


@dataclass(frozen=True)
class AlignmentEvaluation:
    predicted_keypoints_camera_mm: np.ndarray
    observed_keypoints_camera_mm: np.ndarray
    raw_residual_mm: np.ndarray
    weighted_residual_vector: np.ndarray
    effective_weights: np.ndarray
    metric_mask: np.ndarray
    used_frame_mask: np.ndarray
    per_keypoint_error_mm: np.ndarray
    frame_mean_error_mm: np.ndarray
    keypoint_mean_error_mm: np.ndarray


def pack_marker2hand_transform(transform_marker2hand_mm: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform_marker2hand_mm, dtype=np.float64).reshape(4, 4)
    rotvec, trans = transform_to_rvec_tvec(transform)
    return np.concatenate([trans, rotvec], axis=0)


def unpack_marker2hand_params(params: np.ndarray) -> np.ndarray:
    values = np.asarray(params, dtype=np.float64).reshape(6)
    return rvec_tvec_to_transform(values[3:], values[:3])


def compose_parameter_vector(theta_skeleton: np.ndarray, marker2hand_params: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta_skeleton, dtype=np.float64).reshape(-1)
    marker = np.asarray(marker2hand_params, dtype=np.float64).reshape(6)
    return np.concatenate([theta, marker], axis=0)


def _mean_over_mask(values: np.ndarray, mask: np.ndarray, *, axis: int) -> np.ndarray:
    masked = np.where(mask, values, 0.0)
    count = np.sum(mask, axis=axis)
    total = np.sum(masked, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / np.maximum(count, 1)
    return np.where(count > 0, mean, np.nan)


def evaluate_alignment(
    dataset: AlignmentDataset,
    skeleton: dict[str, Any],
    marker2hand_transform_mm: np.ndarray,
) -> AlignmentEvaluation:
    num_frames = dataset.num_frames
    if num_frames == 0:
        return AlignmentEvaluation(
            predicted_keypoints_camera_mm=np.zeros((0, 21, 3), dtype=np.float64),
            observed_keypoints_camera_mm=np.zeros((0, 21, 3), dtype=np.float64),
            raw_residual_mm=np.zeros((0, 21, 3), dtype=np.float64),
            weighted_residual_vector=np.zeros(0, dtype=np.float64),
            effective_weights=np.zeros((0, 21), dtype=np.float64),
            metric_mask=np.zeros((0, 21), dtype=bool),
            used_frame_mask=np.zeros(0, dtype=bool),
            per_keypoint_error_mm=np.zeros((0, 21), dtype=np.float64),
            frame_mean_error_mm=np.zeros(0, dtype=np.float64),
            keypoint_mean_error_mm=np.zeros(21, dtype=np.float64),
        )

    model = DexSlideHumanModel(skeleton, hand=dataset.hand, unit_scale=1.0)
    q_encoder = dataset.q_encoder_array()
    camera_T_marker = dataset.camera_T_marker_array()
    observed = dataset.keypoints_camera_array()
    confidence = dataset.keypoint_confidence_array()
    valid_mask = dataset.keypoint_valid_mask_array()

    predicted = np.zeros_like(observed, dtype=np.float64)
    for frame_idx in range(num_frames):
        hand_keypoints = model.landmarks_from_angles(q_encoder[frame_idx])
        camera_T_hand = camera_T_marker[frame_idx] @ marker2hand_transform_mm
        predicted[frame_idx] = transform_points(camera_T_hand, hand_keypoints)

    finite_mask = np.isfinite(predicted).all(axis=-1) & np.isfinite(observed).all(axis=-1)
    effective_weights = (
        confidence
        * valid_mask.astype(np.float64)
        * KEYPOINT_CLASS_WEIGHTS[None, :]
    )
    effective_weights = np.where(finite_mask, effective_weights, 0.0)
    used_frame_mask = np.count_nonzero(effective_weights > 0.0, axis=1) >= 3
    effective_weights = np.where(used_frame_mask[:, None], effective_weights, 0.0)
    metric_mask = effective_weights > 0.0

    raw_delta = np.where(
        finite_mask[..., None],
        predicted - observed,
        0.0,
    )
    per_keypoint_error = np.linalg.norm(raw_delta, axis=-1)
    per_keypoint_error = np.where(metric_mask, per_keypoint_error, np.nan)

    weighted_residual = raw_delta * effective_weights[..., None]
    weighted_vector = weighted_residual.reshape(-1)
    if not np.isfinite(weighted_vector).all():
        weighted_vector = np.full_like(weighted_vector, 1e6, dtype=np.float64)

    frame_mean_error = _mean_over_mask(
        np.where(np.isfinite(per_keypoint_error), per_keypoint_error, 0.0),
        metric_mask,
        axis=1,
    )
    keypoint_mean_error = _mean_over_mask(
        np.where(np.isfinite(per_keypoint_error), per_keypoint_error, 0.0),
        metric_mask,
        axis=0,
    )
    return AlignmentEvaluation(
        predicted_keypoints_camera_mm=predicted,
        observed_keypoints_camera_mm=observed,
        raw_residual_mm=raw_delta,
        weighted_residual_vector=weighted_vector,
        effective_weights=effective_weights,
        metric_mask=metric_mask,
        used_frame_mask=used_frame_mask,
        per_keypoint_error_mm=per_keypoint_error,
        frame_mean_error_mm=frame_mean_error,
        keypoint_mean_error_mm=keypoint_mean_error,
    )


def alignment_residual_vector(
    x: np.ndarray,
    dataset: AlignmentDataset,
    template_skeleton: dict[str, Any],
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.shape[0] != SKELETON_PARAM_SIZE + 6:
        raise ValueError(f"Expected parameter vector length {SKELETON_PARAM_SIZE + 6}, got {values.shape[0]}")
    skeleton = unflatten_skeleton(values[:SKELETON_PARAM_SIZE], template_skeleton)
    marker2hand = unpack_marker2hand_params(values[SKELETON_PARAM_SIZE:])
    evaluation = evaluate_alignment(dataset, skeleton, marker2hand)
    return evaluation.weighted_residual_vector
