"""Runtime alignment feature extraction and evaluation helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from dexslide.kinematics.live_hand import (
    FINGERS,
    FINGER_OFFSET,
    RUNTIME_PALM_COORDINATE_MODE,
    THUMB_CHAIN_RX_FIELD,
    finger_points,
    rot_x,
    runtime_palm_points,
    thumb_chain_rx_rad,
)
from dexslide.retargeting.human_model import HUMAN_LANDMARK_NAMES
from dexslide.kinematics.transforms import invert_transform, transform_points

from .objective import AlignmentEvaluation, KEYPOINT_CLASS_WEIGHTS
from .skeleton_param import FINGER_BONE_LAYOUT
from .types import AlignmentDataset, AlignmentFrame
from .frame_pool import FramePool
from .progress import SolverProgressLogger
from .weights import (
    KEYPOINT_CLASS_WEIGHT_MAP,
    STEP2_JOINT_BIAS_BOUND_DEG,
    STEP2_JOINT_BIAS_BOUND_RAD,
    STEP2_JOINT_SCALE_LOWER_BOUND,
    STEP2_JOINT_SCALE_UPPER_BOUND,
    STEP2_HUBER_F_SCALE,
    STEP2_BIAS_REG_WEIGHTS,
    STEP2_SCALE_REG_WEIGHTS,
    STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG,
    STEP3_HUBER_F_SCALE,
    STEP3_THUMB_BASE_RX_BOUND_DEG,
    STEP3_THUMB_BASE_RX_BOUND_RAD,
    STEP3_TRANSLATION_PRIOR_WEIGHT,
    STEP3_TRANSLATION_SAMPLE_WEIGHT,
)


PALM_BASE_NAMES: tuple[str, ...] = ("thumb_base", "index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp")
PALM_BASE_KEYS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
PALM_BASE_INDICES: np.ndarray = np.array([1, 5, 9, 13, 17], dtype=np.int64)
PALM_VERTEX_INDICES: np.ndarray = np.array([1, 2, 3, 4, 5], dtype=np.int64)
SEGMENT_INDEX_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
)
SEGMENT_NAMES: tuple[str, ...] = (
    "thumb.metacarpal",
    "thumb.proximal",
    "thumb.distal",
    "index.proximal",
    "index.middle",
    "index.distal",
    "middle.proximal",
    "middle.middle",
    "middle.distal",
    "ring.proximal",
    "ring.middle",
    "ring.distal",
    "pinky.proximal",
    "pinky.middle",
    "pinky.distal",
)
NUM_SEGMENTS = len(SEGMENT_INDEX_PAIRS)
PALM_RADIUS_UPPER_BOUND_MM = 150.0
BONE_LENGTH_UPPER_BOUND_MM = 130.0
MARKER_TRANSLATION_BOUND_MM = np.array([250.0, 250.0, 250.0], dtype=np.float64)



def _build_frame_pool(
    dataset_s1: AlignmentDataset | None,
    dataset_s2: AlignmentDataset,
) -> FramePool:
    hand = dataset_s2.hand
    frames: list[AlignmentFrame] = []
    source_labels: list[str] = []

    if dataset_s1 is not None:
        if dataset_s1.hand != hand:
            raise ValueError(f"dataset_s1.hand={dataset_s1.hand!r} does not match dataset_s2.hand={hand!r}")
        frames.extend(dataset_s1.frames)
        source_labels.extend(["s1"] * dataset_s1.num_frames)

    frames.extend(dataset_s2.frames)
    source_labels.extend(["s2"] * dataset_s2.num_frames)
    return FramePool(hand=hand, frames=tuple(frames), source_labels=tuple(source_labels))


def _frame_pool_summary(frame_pool: FramePool) -> dict[str, Any]:
    finite_q_mask = frame_pool.finite_q_mask()
    wrist_valid_count = 0
    palm_frame_count = 0
    translation_frame_count = 0
    source_counts: dict[str, int] = {}
    for frame, source_label in zip(frame_pool.frames, frame_pool.source_labels):
        source_counts[source_label] = int(source_counts.get(source_label, 0)) + 1
        finite_points = _finite_point_mask(frame.keypoints_camera_mm, frame.keypoint_valid_mask)
        if finite_points[0]:
            wrist_valid_count += 1
            translation_frame_count += 1
        if finite_points[0] and bool(np.any(finite_points[PALM_BASE_INDICES])):
            palm_frame_count += 1
    return {
        "total_frames": int(frame_pool.num_frames),
        "source_counts": source_counts,
        "frames_with_finite_q": int(np.count_nonzero(finite_q_mask)),
        "frames_without_finite_q": int(frame_pool.num_frames - np.count_nonzero(finite_q_mask)),
        "frames_with_wrist_observation": int(wrist_valid_count),
        "frames_with_any_palm_base_observation": int(palm_frame_count),
        "frames_with_translation_sample": int(translation_frame_count),
    }


def _finite_q_frame_mask(frame_pool: FramePool) -> np.ndarray:
    return frame_pool.finite_q_mask()


def _runtime_palm(skeleton: dict[str, Any], hand: str) -> dict[str, np.ndarray]:
    return runtime_palm_points(skeleton, hand)


def _runtime_landmarks(skeleton: dict[str, Any], q_encoder_rad20: np.ndarray, hand: str) -> np.ndarray:
    palm = _runtime_palm(skeleton, hand)
    q = np.asarray(q_encoder_rad20, dtype=np.float64).reshape(20)
    landmarks = np.zeros((len(HUMAN_LANDMARK_NAMES), 3), dtype=np.float64)
    landmarks[0] = palm["wrist"]
    cursor = 1
    for finger in FINGERS:
        start = FINGER_OFFSET[finger]
        landmarks[cursor : cursor + 4] = finger_points(
            finger,
            q[start : start + 4],
            skeleton,
            palm,
            hand,
        )
        cursor += 4
    return landmarks


def _mean_over_mask(values: np.ndarray, mask: np.ndarray, *, axis: int) -> np.ndarray:
    masked = np.where(mask, values, 0.0)
    count = np.sum(mask, axis=axis)
    total = np.sum(masked, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / np.maximum(count, 1)
    return np.where(count > 0, mean, np.nan)


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return value / norm


def _clamp_direction_delta(
    initial_direction: np.ndarray,
    candidate_direction: np.ndarray,
    *,
    max_delta_deg: float | None,
) -> np.ndarray | None:
    initial = _normalize_vector(initial_direction)
    candidate = _normalize_vector(candidate_direction)
    if initial is None:
        return candidate
    if candidate is None:
        return None
    if max_delta_deg is None or max_delta_deg <= 0.0:
        return candidate
    max_delta = float(np.deg2rad(max_delta_deg))
    dot = float(np.clip(np.dot(initial, candidate), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if not np.isfinite(angle) or angle <= max_delta:
        return candidate
    if angle < 1e-8:
        return candidate
    sin_angle = float(np.sin(angle))
    if abs(sin_angle) < 1e-8:
        return initial
    ratio = max_delta / angle
    blended = (
        (np.sin((1.0 - ratio) * angle) / sin_angle) * initial
        + (np.sin(ratio * angle) / sin_angle) * candidate
    )
    normalized = _normalize_vector(blended)
    return initial if normalized is None else normalized


def _finite_point_mask(points_xyz: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return np.asarray(valid_mask, dtype=bool) & np.isfinite(np.asarray(points_xyz, dtype=np.float64)).all(axis=1)


def _points_in_hand_frame(
    frame_keypoints_camera_mm: np.ndarray,
    frame_camera_T_marker: np.ndarray,
    marker2hand_mm: np.ndarray,
) -> np.ndarray:
    camera_T_hand = np.asarray(frame_camera_T_marker, dtype=np.float64).reshape(4, 4) @ np.asarray(
        marker2hand_mm,
        dtype=np.float64,
    ).reshape(4, 4)
    hand_T_camera = invert_transform(camera_T_hand)
    return transform_points(hand_T_camera, frame_keypoints_camera_mm)


def _segment_weights(
    confidence: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    conf = np.asarray(confidence, dtype=np.float64).reshape(21)
    valid = np.asarray(valid_mask, dtype=bool).reshape(21)
    weights = np.zeros(NUM_SEGMENTS, dtype=np.float64)
    for idx, (parent_idx, child_idx) in enumerate(SEGMENT_INDEX_PAIRS):
        if not (valid[parent_idx] and valid[child_idx]):
            continue
        weights[idx] = float(
            np.sqrt(
                max(0.0, conf[parent_idx])
                * max(0.0, conf[child_idx])
                * KEYPOINT_CLASS_WEIGHTS[parent_idx]
                * KEYPOINT_CLASS_WEIGHTS[child_idx]
            )
        )
    return weights


def _segment_directions(points_xyz: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(21, 3)
    valid = np.asarray(valid_mask, dtype=bool).reshape(21)
    directions = np.full((NUM_SEGMENTS, 3), np.nan, dtype=np.float64)
    segment_valid = np.zeros(NUM_SEGMENTS, dtype=bool)
    for idx, (parent_idx, child_idx) in enumerate(SEGMENT_INDEX_PAIRS):
        if not (valid[parent_idx] and valid[child_idx]):
            continue
        direction = _normalize_vector(points[child_idx] - points[parent_idx])
        if direction is None:
            continue
        directions[idx] = direction
        segment_valid[idx] = True
    return directions, segment_valid


def _joint_affine_apply(
    q_encoder_rad20: np.ndarray,
    joint_scale: np.ndarray,
    joint_bias: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_encoder_rad20, dtype=np.float64).reshape(20)
    scale = np.asarray(joint_scale, dtype=np.float64).reshape(20)
    bias = np.asarray(joint_bias, dtype=np.float64).reshape(20)
    return scale * q + bias


def _point_error_dict(values: np.ndarray) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name, error in zip(HUMAN_LANDMARK_NAMES, np.asarray(values, dtype=np.float64).reshape(-1)):
        output[name] = None if not np.isfinite(error) else float(error)
    return output


def _segment_error_dict(values: np.ndarray) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name, error in zip(SEGMENT_NAMES, np.asarray(values, dtype=np.float64).reshape(-1)):
        output[name] = None if not np.isfinite(error) else float(error)
    return output


def _empty_alignment_evaluation() -> AlignmentEvaluation:
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


def evaluate_runtime_alignment(
    frame_pool: FramePool,
    skeleton: dict[str, Any],
    marker2hand_mm: np.ndarray,
    joint_scale: np.ndarray,
    joint_bias: np.ndarray,
) -> AlignmentEvaluation:
    if frame_pool.num_frames == 0:
        return _empty_alignment_evaluation()

    predicted = np.zeros((frame_pool.num_frames, 21, 3), dtype=np.float64)
    observed = frame_pool.keypoints_camera_array()
    confidence = frame_pool.keypoint_confidence_array()
    valid_mask = frame_pool.keypoint_valid_mask_array()
    q_encoder = frame_pool.q_encoder_array()
    camera_T_marker = frame_pool.camera_T_marker_array()

    for frame_idx in range(frame_pool.num_frames):
        if not np.isfinite(q_encoder[frame_idx]).all():
            predicted[frame_idx] = np.nan
            continue
        q_hat = _joint_affine_apply(q_encoder[frame_idx], joint_scale, joint_bias)
        hand_keypoints = _runtime_landmarks(skeleton, q_hat, frame_pool.hand)
        predicted[frame_idx] = transform_points(camera_T_marker[frame_idx] @ marker2hand_mm, hand_keypoints)

    finite_mask = np.isfinite(predicted).all(axis=-1) & np.isfinite(observed).all(axis=-1)
    effective_weights = confidence * valid_mask.astype(np.float64) * KEYPOINT_CLASS_WEIGHTS[None, :]
    effective_weights = np.where(finite_mask, effective_weights, 0.0)
    used_frame_mask = np.count_nonzero(effective_weights > 0.0, axis=1) >= 3
    effective_weights = np.where(used_frame_mask[:, None], effective_weights, 0.0)
    metric_mask = effective_weights > 0.0

    raw_delta = np.where(finite_mask[..., None], predicted - observed, 0.0)
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


def _palm_radii_and_directions(skeleton: dict[str, Any], hand: str) -> tuple[np.ndarray, np.ndarray]:
    palm = _runtime_palm(skeleton, hand)
    radii = np.zeros(len(PALM_BASE_KEYS), dtype=np.float64)
    directions = np.zeros((len(PALM_BASE_KEYS), 3), dtype=np.float64)
    for idx, key in enumerate(PALM_BASE_KEYS):
        point = np.asarray(palm[key], dtype=np.float64).reshape(3)
        radii[idx] = float(np.linalg.norm(point))
        direction = _normalize_vector(point)
        directions[idx] = point if direction is None else direction
    return radii, directions


def _make_runtime_shaped_skeleton(
    template_skeleton: dict[str, Any],
    *,
    hand: str,
    base_radii_mm: np.ndarray,
    base_directions: np.ndarray,
    finger_lengths_mm: np.ndarray | None = None,
) -> dict[str, Any]:
    skeleton = copy.deepcopy(template_skeleton)
    vertices = np.zeros((6, 3), dtype=np.float64)
    vertices[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    for idx in range(len(PALM_BASE_KEYS)):
        direction = _normalize_vector(base_directions[idx])
        if direction is None:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        vertices[int(PALM_VERTEX_INDICES[idx])] = float(max(0.0, base_radii_mm[idx])) * direction
    skeleton.setdefault("palm", {})
    skeleton["palm"]["vertices"] = vertices.tolist()
    skeleton["palm"]["coordinate_mode"] = RUNTIME_PALM_COORDINATE_MODE

    if finger_lengths_mm is not None:
        lengths = np.asarray(finger_lengths_mm, dtype=np.float64).reshape(len(FINGER_BONE_LAYOUT))
        for idx, (finger, bone_name) in enumerate(FINGER_BONE_LAYOUT):
            skeleton.setdefault(finger, {})
            skeleton[finger][bone_name] = float(max(0.0, lengths[idx]))

    return skeleton


def _mean_angle_error_deg(
    predicted_dirs: np.ndarray,
    observed_dirs: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(predicted_dirs, dtype=np.float64)
    obs = np.asarray(observed_dirs, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    dots = np.sum(pred * obs, axis=-1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    angles = np.where(valid, angles, np.nan)
    return np.nanmean(angles, axis=0)


def _precompute_step2_observations(
    frame_pool: FramePool,
    marker2hand_initial_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed_dirs = np.full((frame_pool.num_frames, NUM_SEGMENTS, 3), np.nan, dtype=np.float64)
    segment_valid = np.zeros((frame_pool.num_frames, NUM_SEGMENTS), dtype=bool)
    segment_weights = np.zeros((frame_pool.num_frames, NUM_SEGMENTS), dtype=np.float64)
    keypoints_camera = frame_pool.keypoints_camera_array()
    confidence = frame_pool.keypoint_confidence_array()
    valid_mask = frame_pool.keypoint_valid_mask_array()
    camera_T_marker = frame_pool.camera_T_marker_array()

    for frame_idx in range(frame_pool.num_frames):
        frame_valid_points = _finite_point_mask(keypoints_camera[frame_idx], valid_mask[frame_idx])
        hand_points = _points_in_hand_frame(
            keypoints_camera[frame_idx],
            camera_T_marker[frame_idx],
            marker2hand_initial_mm,
        )
        observed_dirs[frame_idx], segment_valid[frame_idx] = _segment_directions(hand_points, frame_valid_points)
        segment_weights[frame_idx] = _segment_weights(confidence[frame_idx], frame_valid_points)
        segment_weights[frame_idx] = np.where(segment_valid[frame_idx], segment_weights[frame_idx], 0.0)
    return observed_dirs, segment_valid, segment_weights



