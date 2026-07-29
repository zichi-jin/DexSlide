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


@dataclass(frozen=True)
class DexAlignStep1Result:
    skeleton_l1: dict[str, Any]
    base_directions: np.ndarray
    used_counts: np.ndarray
    mean_angular_delta_deg: np.ndarray


@dataclass(frozen=True)
class DexAlignStep2Result:
    joint_scale: np.ndarray
    joint_bias: np.ndarray
    solver_result: OptimizeResult
    mean_segment_error_deg_before: np.ndarray
    mean_segment_error_deg_after: np.ndarray
    segment_valid_counts: np.ndarray


@dataclass(frozen=True)
class DexAlignStep3Result:
    optimized_skeleton: dict[str, Any]
    optimized_marker2hand: np.ndarray
    solver_result: OptimizeResult
    initial_eval: AlignmentEvaluation
    final_eval: AlignmentEvaluation
    point_residual_vector_initial: np.ndarray
    point_residual_vector_final: np.ndarray
    translation_sample_residual_initial: np.ndarray
    translation_sample_residual_final: np.ndarray
    translation_sample_count: int
    optimize_thumb_base_rx: bool
    thumb_base_rx_delta_rad: float


@dataclass(frozen=True)
class DexAlignV2RunResult:
    step1: DexAlignStep1Result
    step2: DexAlignStep2Result
    step3: DexAlignStep3Result
    summary: dict[str, Any]


@dataclass(frozen=True)
class FramePool:
    hand: str
    frames: tuple[AlignmentFrame, ...]
    source_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.source_labels):
            raise ValueError("frames and source_labels must have identical length")

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def subset(self, keep_mask: np.ndarray) -> FramePool:
        mask = np.asarray(keep_mask, dtype=bool).reshape(self.num_frames)
        return FramePool(
            hand=self.hand,
            frames=tuple(frame for frame, keep in zip(self.frames, mask) if keep),
            source_labels=tuple(label for label, keep in zip(self.source_labels, mask) if keep),
        )

    def timestamps(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(0, dtype=np.float64)
        return np.asarray([frame.timestamp for frame in self.frames], dtype=np.float64)

    def camera_T_marker_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 4, 4), dtype=np.float64)
        return np.stack([frame.camera_T_marker for frame in self.frames], axis=0)

    def q_encoder_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 20), dtype=np.float64)
        return np.stack([frame.q_encoder_rad20 for frame in self.frames], axis=0)

    def keypoints_camera_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 21, 3), dtype=np.float64)
        return np.stack([frame.keypoints_camera_mm for frame in self.frames], axis=0)

    def keypoint_confidence_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 21), dtype=np.float64)
        return np.stack([frame.keypoint_confidence for frame in self.frames], axis=0)

    def keypoint_valid_mask_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 21), dtype=bool)
        return np.stack([frame.keypoint_valid_mask for frame in self.frames], axis=0)

    def finite_q_mask(self) -> np.ndarray:
        q = self.q_encoder_array()
        if q.size == 0:
            return np.zeros(self.num_frames, dtype=bool)
        return np.isfinite(q).all(axis=1)


class SolverProgressLogger:
    def __init__(
        self,
        stage_name: str,
        *,
        log_fn: Callable[[str], None] | None,
        min_interval_sec: float = 1.0,
        metric_label: str | None = None,
        metric_getter: Callable[[], float | None] | None = None,
    ) -> None:
        self.stage_name = str(stage_name)
        self.log_fn = log_fn
        self.min_interval_sec = float(min_interval_sec)
        self.metric_label = None if metric_label is None else str(metric_label)
        self.metric_getter = metric_getter
        self.start_time = time.monotonic()
        self.last_log_time = self.start_time
        self.eval_count = 0
        self.best_cost = float("inf")

    def wrap(self, residual_fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        def wrapped(params: np.ndarray) -> np.ndarray:
            residual = np.asarray(residual_fn(params), dtype=np.float64).reshape(-1)
            self.eval_count += 1
            finite = residual[np.isfinite(residual)]
            cost = float("inf") if finite.size == 0 else float(0.5 * np.dot(finite, finite))
            if cost < self.best_cost:
                self.best_cost = cost
            now = time.monotonic()
            if self.log_fn is not None and (now - self.last_log_time) >= self.min_interval_sec:
                rms = float("nan") if finite.size == 0 else float(np.sqrt(np.mean(np.square(finite))))
                metric_suffix = ""
                if self.metric_label is not None and self.metric_getter is not None:
                    metric_value = self.metric_getter()
                    if metric_value is not None and np.isfinite(float(metric_value)):
                        metric_suffix = f" {self.metric_label}={float(metric_value):.6g}"
                self.log_fn(
                    f"[dexalign2] {self.stage_name}: evals={self.eval_count} "
                    f"cost={cost:.6g} best={self.best_cost:.6g} weighted_rms={rms:.6g}"
                    f"{metric_suffix} "
                    f"elapsed={now - self.start_time:.1f}s"
                )
                self.last_log_time = now
            return residual

        return wrapped

    def finish(self, result: OptimizeResult) -> None:
        if self.log_fn is None:
            return
        elapsed = time.monotonic() - self.start_time
        self.log_fn(
            f"[dexalign2] {self.stage_name}: done success={bool(result.success)} "
            f"status={int(result.status)} nfev={int(result.nfev)} "
            f"cost={float(result.cost):.6g} elapsed={elapsed:.1f}s"
        )


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


def run_step1_palm_shape(
    frame_pool: FramePool,
    initial_skeleton: dict[str, Any],
    marker2hand_initial_mm: np.ndarray,
) -> DexAlignStep1Result:
    radii_mm, initial_dirs = _palm_radii_and_directions(initial_skeleton, frame_pool.hand)
    direction_sums = np.zeros((len(PALM_BASE_KEYS), 3), dtype=np.float64)
    weight_sums = np.zeros(len(PALM_BASE_KEYS), dtype=np.float64)
    angle_samples: list[list[float]] = [[] for _ in PALM_BASE_KEYS]

    for frame in frame_pool.frames:
        finite_points = _finite_point_mask(frame.keypoints_camera_mm, frame.keypoint_valid_mask)
        if not finite_points[0]:
            continue
        hand_points = _points_in_hand_frame(
            frame.keypoints_camera_mm,
            frame.camera_T_marker,
            marker2hand_initial_mm,
        )
        wrist = np.asarray(hand_points[0], dtype=np.float64).reshape(3)
        for base_slot, keypoint_idx in enumerate(PALM_BASE_INDICES.tolist()):
            if not finite_points[keypoint_idx]:
                continue
            direction = _normalize_vector(hand_points[keypoint_idx] - wrist)
            if direction is None:
                continue
            weight = float(
                np.sqrt(
                    max(0.0, frame.keypoint_confidence[0])
                    * max(0.0, frame.keypoint_confidence[keypoint_idx])
                    * KEYPOINT_CLASS_WEIGHTS[0]
                    * KEYPOINT_CLASS_WEIGHTS[keypoint_idx]
                )
            )
            if weight <= 0.0:
                continue
            effective_direction = _clamp_direction_delta(
                initial_dirs[base_slot],
                direction,
                max_delta_deg=None if base_slot == 0 else STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG,
            )
            if effective_direction is None:
                continue
            direction_sums[base_slot] += weight * effective_direction
            weight_sums[base_slot] += weight
            dot = float(np.clip(np.dot(effective_direction, initial_dirs[base_slot]), -1.0, 1.0))
            angle_samples[base_slot].append(float(np.degrees(np.arccos(dot))))

    base_directions = initial_dirs.copy()
    used_counts = np.zeros(len(PALM_BASE_KEYS), dtype=np.int64)
    mean_angular_delta_deg = np.full(len(PALM_BASE_KEYS), np.nan, dtype=np.float64)
    for idx in range(len(PALM_BASE_KEYS)):
        direction = _normalize_vector(direction_sums[idx])
        if direction is not None and weight_sums[idx] > 0.0:
            base_directions[idx] = direction
        used_counts[idx] = int(len(angle_samples[idx]))
        if angle_samples[idx]:
            mean_angular_delta_deg[idx] = float(np.mean(np.asarray(angle_samples[idx], dtype=np.float64)))

    skeleton_l1 = _make_runtime_shaped_skeleton(
        initial_skeleton,
        hand=frame_pool.hand,
        base_radii_mm=radii_mm,
        base_directions=base_directions,
        finger_lengths_mm=np.asarray(
            [float(initial_skeleton[finger][bone_name]) for finger, bone_name in FINGER_BONE_LAYOUT],
            dtype=np.float64,
        ),
    )
    return DexAlignStep1Result(
        skeleton_l1=skeleton_l1,
        base_directions=base_directions,
        used_counts=used_counts,
        mean_angular_delta_deg=mean_angular_delta_deg,
    )


def run_step2_joint_calibration(
    frame_pool: FramePool,
    skeleton_l1: dict[str, Any],
    marker2hand_initial_mm: np.ndarray,
    *,
    max_nfev: int,
    log_fn: Callable[[str], None] | None = None,
    show_progress: bool = False,
) -> DexAlignStep2Result:
    q_mask = _finite_q_frame_mask(frame_pool)
    eligible_pool = frame_pool.subset(q_mask)
    observed_dirs, observed_valid, observed_weights = _precompute_step2_observations(eligible_pool, marker2hand_initial_mm)
    q_encoder = eligible_pool.q_encoder_array()
    if q_encoder.size == 0:
        raise ValueError("Step 2 requires at least one frame with finite glove angles.")

    def _predict_segment_dirs(joint_scale: np.ndarray, joint_bias: np.ndarray) -> np.ndarray:
        predicted = np.full_like(observed_dirs, np.nan)
        for frame_idx in range(eligible_pool.num_frames):
            q_hat = _joint_affine_apply(q_encoder[frame_idx], joint_scale, joint_bias)
            hand_points = _runtime_landmarks(skeleton_l1, q_hat, eligible_pool.hand)
            predicted[frame_idx], _ = _segment_directions(
                hand_points,
                np.ones(21, dtype=bool),
            )
        return predicted

    def residual_fn(params: np.ndarray) -> np.ndarray:
        values = np.asarray(params, dtype=np.float64).reshape(40)
        joint_scale = values[:20]
        joint_bias = values[20:]
        predicted_dirs = _predict_segment_dirs(joint_scale, joint_bias)
        progress_metric["value"] = float(np.nanmean(_mean_angle_error_deg(predicted_dirs, observed_dirs, observed_valid)))
        residual = np.where(
            observed_valid[..., None],
            (predicted_dirs - observed_dirs) * observed_weights[..., None],
            0.0,
        ).reshape(-1)
        reg_scale = STEP2_SCALE_REG_WEIGHTS * (joint_scale - 1.0)
        reg_bias = STEP2_BIAS_REG_WEIGHTS * joint_bias
        return np.concatenate([residual, reg_scale, reg_bias], axis=0)

    progress_metric: dict[str, float | None] = {"value": None}
    x0 = np.concatenate([np.ones(20, dtype=np.float64), np.zeros(20, dtype=np.float64)], axis=0)
    lower = np.concatenate(
        [
            np.full(20, STEP2_JOINT_SCALE_LOWER_BOUND, dtype=np.float64),
            np.full(20, -STEP2_JOINT_BIAS_BOUND_RAD, dtype=np.float64),
        ],
        axis=0,
    )
    upper = np.concatenate(
        [
            np.full(20, STEP2_JOINT_SCALE_UPPER_BOUND, dtype=np.float64),
            np.full(20, STEP2_JOINT_BIAS_BOUND_RAD, dtype=np.float64),
        ],
        axis=0,
    )
    progress = SolverProgressLogger(
        "step2",
        log_fn=log_fn if show_progress else None,
        metric_label="mean-segment-error-deg",
        metric_getter=lambda: progress_metric["value"],
    )
    solver_result = least_squares(
        progress.wrap(residual_fn),
        x0,
        method="trf",
        loss="huber",
        f_scale=STEP2_HUBER_F_SCALE,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
    )
    progress.finish(solver_result)
    joint_scale = np.asarray(solver_result.x[:20], dtype=np.float64)
    joint_bias = np.asarray(solver_result.x[20:], dtype=np.float64)

    predicted_before = _predict_segment_dirs(np.ones(20, dtype=np.float64), np.zeros(20, dtype=np.float64))
    predicted_after = _predict_segment_dirs(joint_scale, joint_bias)
    return DexAlignStep2Result(
        joint_scale=joint_scale,
        joint_bias=joint_bias,
        solver_result=solver_result,
        mean_segment_error_deg_before=_mean_angle_error_deg(predicted_before, observed_dirs, observed_valid),
        mean_segment_error_deg_after=_mean_angle_error_deg(predicted_after, observed_dirs, observed_valid),
        segment_valid_counts=np.sum(observed_valid, axis=0).astype(np.int64),
    )


def _extract_length_params(skeleton: dict[str, Any], hand: str) -> tuple[np.ndarray, np.ndarray]:
    palm_radii_mm, _base_dirs = _palm_radii_and_directions(skeleton, hand)
    finger_lengths = np.asarray(
        [float(skeleton[finger][bone_name]) for finger, bone_name in FINGER_BONE_LAYOUT],
        dtype=np.float64,
    )
    return palm_radii_mm, finger_lengths


def _translation_samples_mm(dataset: AlignmentDataset) -> np.ndarray:
    return _translation_samples_from_frames(dataset.frames)


def _translation_samples_from_frames(frames: tuple[AlignmentFrame, ...] | list[AlignmentFrame]) -> np.ndarray:
    samples: list[np.ndarray] = []
    for frame in frames:
        finite_points = _finite_point_mask(frame.keypoints_camera_mm, frame.keypoint_valid_mask)
        if not finite_points[0]:
            continue
        camera_T_marker = np.asarray(frame.camera_T_marker, dtype=np.float64).reshape(4, 4)
        camera_R_marker = camera_T_marker[:3, :3]
        camera_t_marker = camera_T_marker[:3, 3]
        wrist_camera = np.asarray(frame.keypoints_camera_mm[0], dtype=np.float64).reshape(3)
        sample = camera_R_marker.T @ (wrist_camera - camera_t_marker)
        if np.isfinite(sample).all():
            samples.append(sample.astype(np.float64))
    if not samples:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(samples, axis=0)


def run_step3_lengths_and_translation(
    frame_pool: FramePool,
    initial_skeleton: dict[str, Any],
    skeleton_l1: dict[str, Any],
    marker2hand_initial_mm: np.ndarray,
    joint_scale: np.ndarray,
    joint_bias: np.ndarray,
    *,
    max_nfev: int,
    optimize_thumb_base_rx: bool = False,
    log_fn: Callable[[str], None] | None = None,
    show_progress: bool = False,
) -> DexAlignStep3Result:
    base_radii0, finger_lengths0 = _extract_length_params(skeleton_l1, frame_pool.hand)
    _unused_radii, base_directions = _palm_radii_and_directions(skeleton_l1, frame_pool.hand)
    initial_thumb_chain_rx_rad = thumb_chain_rx_rad(skeleton_l1)
    translation0 = np.asarray(marker2hand_initial_mm[:3, 3], dtype=np.float64).reshape(3)
    translation_samples = _translation_samples_from_frames(frame_pool.frames)
    parameter_count = 24 if optimize_thumb_base_rx else 23

    def _thumb_base_rx_delta(params: np.ndarray) -> float:
        if not optimize_thumb_base_rx:
            return 0.0
        return float(np.asarray(params, dtype=np.float64).reshape(parameter_count)[23])

    def _build_skeleton(params: np.ndarray) -> dict[str, Any]:
        values = np.asarray(params, dtype=np.float64).reshape(parameter_count)
        palm_radii = values[:5]
        finger_lengths = values[5:20]
        effective_base_directions = base_directions.copy()
        thumb_base_rx_delta = _thumb_base_rx_delta(values)
        effective_base_directions[0] = rot_x(thumb_base_rx_delta) @ effective_base_directions[0]
        skeleton = _make_runtime_shaped_skeleton(
            skeleton_l1,
            hand=frame_pool.hand,
            base_radii_mm=palm_radii,
            base_directions=effective_base_directions,
            finger_lengths_mm=finger_lengths,
        )
        if optimize_thumb_base_rx or THUMB_CHAIN_RX_FIELD in skeleton.get("palm", {}):
            skeleton["palm"][THUMB_CHAIN_RX_FIELD] = float(initial_thumb_chain_rx_rad + thumb_base_rx_delta)
        return skeleton

    def _build_marker2hand(params: np.ndarray) -> np.ndarray:
        transform = np.asarray(marker2hand_initial_mm, dtype=np.float64).reshape(4, 4).copy()
        transform[:3, 3] = np.asarray(params[20:23], dtype=np.float64)
        return transform

    def _point_evaluation(skeleton: dict[str, Any], marker2hand_mm: np.ndarray) -> AlignmentEvaluation:
        return evaluate_runtime_alignment(
            frame_pool,
            skeleton,
            marker2hand_mm,
            joint_scale,
            joint_bias,
        )

    def _translation_residual_vector(translation_mm: np.ndarray) -> np.ndarray:
        translation = np.asarray(translation_mm, dtype=np.float64).reshape(3)
        if translation_samples.shape[0] == 0:
            sample_residual = np.zeros(0, dtype=np.float64)
        else:
            sample_residual = (
                STEP3_TRANSLATION_SAMPLE_WEIGHT
                * (translation_samples - translation[None, :])
            ).reshape(-1)
        prior_residual = STEP3_TRANSLATION_PRIOR_WEIGHT * (translation - translation0)
        return np.concatenate([sample_residual, prior_residual], axis=0)

    def residual_fn(params: np.ndarray) -> np.ndarray:
        skeleton = _build_skeleton(params)
        marker2hand_mm = _build_marker2hand(params)
        evaluation = _point_evaluation(skeleton, marker2hand_mm)
        progress_metric["value"] = float(np.nanmean(evaluation.frame_mean_error_mm))
        point_residual = evaluation.weighted_residual_vector
        translation_residual = _translation_residual_vector(params[20:23])
        return np.concatenate([point_residual, translation_residual], axis=0)

    x0_parts = [base_radii0, finger_lengths0, translation0]
    lower_parts = [
        np.full(5, 1.0, dtype=np.float64),
        np.full(15, 1.0, dtype=np.float64),
        translation0 - MARKER_TRANSLATION_BOUND_MM,
    ]
    upper_parts = [
        np.full(5, PALM_RADIUS_UPPER_BOUND_MM, dtype=np.float64),
        np.full(15, BONE_LENGTH_UPPER_BOUND_MM, dtype=np.float64),
        translation0 + MARKER_TRANSLATION_BOUND_MM,
    ]
    if optimize_thumb_base_rx:
        x0_parts.append(np.zeros(1, dtype=np.float64))
        lower_parts.append(np.array([-STEP3_THUMB_BASE_RX_BOUND_RAD], dtype=np.float64))
        upper_parts.append(np.array([STEP3_THUMB_BASE_RX_BOUND_RAD], dtype=np.float64))
    x0 = np.concatenate(x0_parts, axis=0)
    lower = np.concatenate(lower_parts, axis=0)
    upper = np.concatenate(upper_parts, axis=0)
    progress_metric: dict[str, float | None] = {"value": None}
    progress = SolverProgressLogger(
        "step3",
        log_fn=log_fn if show_progress else None,
        metric_label="mean-keypoint-error-mm",
        metric_getter=lambda: progress_metric["value"],
    )
    solver_result = least_squares(
        progress.wrap(residual_fn),
        x0,
        method="trf",
        loss="huber",
        f_scale=STEP3_HUBER_F_SCALE,
        bounds=(lower, upper),
        max_nfev=int(max_nfev),
    )
    progress.finish(solver_result)

    initial_params = x0
    optimized_params = np.asarray(solver_result.x, dtype=np.float64).reshape(parameter_count)
    initial_skeleton_eval = _build_skeleton(initial_params)
    initial_marker_eval = _build_marker2hand(initial_params)
    optimized_skeleton = _build_skeleton(optimized_params)
    optimized_marker2hand = _build_marker2hand(optimized_params)
    initial_eval = evaluate_runtime_alignment(
        frame_pool,
        initial_skeleton_eval,
        initial_marker_eval,
        joint_scale,
        joint_bias,
    )
    final_eval = evaluate_runtime_alignment(
        frame_pool,
        optimized_skeleton,
        optimized_marker2hand,
        joint_scale,
        joint_bias,
    )
    return DexAlignStep3Result(
        optimized_skeleton=optimized_skeleton,
        optimized_marker2hand=optimized_marker2hand,
        solver_result=solver_result,
        initial_eval=initial_eval,
        final_eval=final_eval,
        point_residual_vector_initial=initial_eval.weighted_residual_vector,
        point_residual_vector_final=final_eval.weighted_residual_vector,
        translation_sample_residual_initial=_translation_residual_vector(initial_params[20:23]),
        translation_sample_residual_final=_translation_residual_vector(optimized_params[20:23]),
        translation_sample_count=int(translation_samples.shape[0]),
        optimize_thumb_base_rx=bool(optimize_thumb_base_rx),
        thumb_base_rx_delta_rad=_thumb_base_rx_delta(optimized_params),
    )


def _l2_cost(residual_vector: np.ndarray) -> float:
    residual = np.asarray(residual_vector, dtype=np.float64).reshape(-1)
    return float(0.5 * np.dot(residual, residual))


def run_dexalign_v2(
    *,
    dataset_s1: AlignmentDataset | None,
    dataset_s2: AlignmentDataset,
    initial_skeleton: dict[str, Any],
    marker2hand_initial_mm: np.ndarray,
    max_nfev_step2: int = 200,
    max_nfev_step3: int = 250,
    optimize_thumb_base_rx: bool = False,
    log_fn: Callable[[str], None] | None = None,
    show_progress: bool = False,
) -> DexAlignV2RunResult:
    frame_pool = _build_frame_pool(dataset_s1, dataset_s2)
    pool_summary = _frame_pool_summary(frame_pool)
    if pool_summary["frames_with_finite_q"] <= 0:
        raise ValueError("DexAlign 2.0 requires at least one frame with finite glove angles in the merged frame pool.")

    if show_progress and log_fn is not None:
        log_fn(
            "[dexalign2] frame-pool: "
            f"total={pool_summary['total_frames']} "
            f"s1={pool_summary['source_counts'].get('s1', 0)} "
            f"s2={pool_summary['source_counts'].get('s2', 0)} "
            f"finite_q={pool_summary['frames_with_finite_q']} "
            f"translation={pool_summary['frames_with_translation_sample']}"
        )
        log_fn("[dexalign2] step1: start palm-base direction estimation")

    step1 = run_step1_palm_shape(frame_pool, initial_skeleton, marker2hand_initial_mm)
    if show_progress and log_fn is not None:
        log_fn(
            "[dexalign2] step1: done "
            + " ".join(
                f"{name}={int(count)}"
                for name, count in zip(PALM_BASE_NAMES, step1.used_counts)
            )
        )

    if show_progress and log_fn is not None:
        log_fn("[dexalign2] step2: start joint affine calibration from all finite-q frames")
    step2 = run_step2_joint_calibration(
        frame_pool,
        step1.skeleton_l1,
        marker2hand_initial_mm,
        max_nfev=int(max_nfev_step2),
        log_fn=log_fn,
        show_progress=show_progress,
    )
    if show_progress and log_fn is not None:
        before_deg = np.asarray(step2.mean_segment_error_deg_before, dtype=np.float64)
        after_deg = np.asarray(step2.mean_segment_error_deg_after, dtype=np.float64)
        log_fn(
            "[dexalign2] step2: mean-segment-error-deg "
            f"before={float(np.nanmean(before_deg)):.3f} after={float(np.nanmean(after_deg)):.3f}"
        )

    if show_progress and log_fn is not None:
        thumb_rx_suffix = " + thumb-base-rx" if optimize_thumb_base_rx else ""
        log_fn(
            "[dexalign2] step3: start length + marker translation optimization"
            f"{thumb_rx_suffix} on merged frame pool"
        )
    step3 = run_step3_lengths_and_translation(
        frame_pool,
        initial_skeleton,
        step1.skeleton_l1,
        marker2hand_initial_mm,
        step2.joint_scale,
        step2.joint_bias,
        max_nfev=int(max_nfev_step3),
        optimize_thumb_base_rx=bool(optimize_thumb_base_rx),
        log_fn=log_fn,
        show_progress=show_progress,
    )
    if show_progress and log_fn is not None:
        thumb_rx_suffix = (
            f" thumb-base-rx-deg={np.degrees(step3.thumb_base_rx_delta_rad):.3f}"
            if step3.optimize_thumb_base_rx
            else ""
        )
        log_fn(
            "[dexalign2] step3: mean-keypoint-error-mm "
            f"before={float(np.nanmean(step3.initial_eval.frame_mean_error_mm)):.3f} "
            f"after={float(np.nanmean(step3.final_eval.frame_mean_error_mm)):.3f}"
            f"{thumb_rx_suffix}"
        )

    summary = {
        "algorithm": "DexAlign 2.0",
        "frame_pool_mode": "merged_frames_gated_by_modalities",
        "frame_pool_summary": pool_summary,
        "capture_fallback": None if dataset_s1 is not None else "dataset_s1_missing_reused_s2",
        "step1": {
            "used_counts": {name: int(value) for name, value in zip(PALM_BASE_NAMES, step1.used_counts)},
            "mean_angular_delta_deg": {
                name: None if not np.isfinite(value) else float(value)
                for name, value in zip(PALM_BASE_NAMES, step1.mean_angular_delta_deg)
            },
            "thumb_max_base_direction_delta_deg": None,
            "non_thumb_max_base_direction_delta_deg": float(STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG),
        },
        "loss_weights": {
            "keypoint_class_weights": {name: float(value) for name, value in KEYPOINT_CLASS_WEIGHT_MAP.items()},
            "step1": {
                "thumb_max_base_direction_delta_deg": None,
                "non_thumb_max_base_direction_delta_deg": float(STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG),
            },
            "step2": {
                "scale_reg_weights": STEP2_SCALE_REG_WEIGHTS.tolist(),
                "bias_reg_weights_rad": STEP2_BIAS_REG_WEIGHTS.tolist(),
                "thumb_scale_reg_weight": float(STEP2_SCALE_REG_WEIGHTS[0]),
                "thumb_bias_reg_weight_rad": float(STEP2_BIAS_REG_WEIGHTS[0]),
                "non_thumb_scale_reg_weight": float(STEP2_SCALE_REG_WEIGHTS[4]),
                "non_thumb_bias_reg_weight_rad": float(STEP2_BIAS_REG_WEIGHTS[4]),
                "joint_scale_lower_bound": float(STEP2_JOINT_SCALE_LOWER_BOUND),
                "joint_scale_upper_bound": float(STEP2_JOINT_SCALE_UPPER_BOUND),
                "joint_bias_bound_deg": float(STEP2_JOINT_BIAS_BOUND_DEG),
                "joint_bias_bound_rad": float(STEP2_JOINT_BIAS_BOUND_RAD),
                "huber_f_scale": float(STEP2_HUBER_F_SCALE),
            },
            "step3": {
                "translation_sample_weight": float(STEP3_TRANSLATION_SAMPLE_WEIGHT),
                "translation_prior_weight": float(STEP3_TRANSLATION_PRIOR_WEIGHT),
                "huber_f_scale": float(STEP3_HUBER_F_SCALE),
                "thumb_base_rx_bound_deg": float(STEP3_THUMB_BASE_RX_BOUND_DEG),
            },
        },
        "step2": {
            "joint_scale": step2.joint_scale.tolist(),
            "joint_bias_rad": step2.joint_bias.tolist(),
            "segment_valid_counts": {
                name: int(value) for name, value in zip(SEGMENT_NAMES, step2.segment_valid_counts)
            },
            "mean_segment_error_deg_before": _segment_error_dict(step2.mean_segment_error_deg_before),
            "mean_segment_error_deg_after": _segment_error_dict(step2.mean_segment_error_deg_after),
            "solver_status": {
                "success": bool(step2.solver_result.success),
                "status": int(step2.solver_result.status),
                "message": str(step2.solver_result.message),
                "nfev": int(step2.solver_result.nfev),
                "cost": float(step2.solver_result.cost),
            },
        },
        "step3": {
            "initial_cost_l2": _l2_cost(
                np.concatenate(
                    [step3.point_residual_vector_initial, step3.translation_sample_residual_initial],
                    axis=0,
                )
            ),
            "final_cost_l2": _l2_cost(
                np.concatenate(
                    [step3.point_residual_vector_final, step3.translation_sample_residual_final],
                    axis=0,
                )
            ),
            "initial_mean_error_mm": float(np.nanmean(step3.initial_eval.frame_mean_error_mm)),
            "final_mean_error_mm": float(np.nanmean(step3.final_eval.frame_mean_error_mm)),
            "per_keypoint_mean_error_mm_before": _point_error_dict(step3.initial_eval.keypoint_mean_error_mm),
            "per_keypoint_mean_error_mm_after": _point_error_dict(step3.final_eval.keypoint_mean_error_mm),
            "per_frame_mean_error_mm_before": [
                None if not np.isfinite(value) else float(value) for value in step3.initial_eval.frame_mean_error_mm
            ],
            "per_frame_mean_error_mm_after": [
                None if not np.isfinite(value) else float(value) for value in step3.final_eval.frame_mean_error_mm
            ],
            "translation_sample_count": int(step3.translation_sample_count),
            "optimized_marker2hand_translation_mm": step3.optimized_marker2hand[:3, 3].tolist(),
            "optimize_thumb_base_rx": bool(step3.optimize_thumb_base_rx),
            "thumb_base_rx_delta_rad": float(step3.thumb_base_rx_delta_rad),
            "thumb_base_rx_delta_deg": float(np.degrees(step3.thumb_base_rx_delta_rad)),
            "thumb_chain_rx_rad": float(thumb_chain_rx_rad(step3.optimized_skeleton)),
            "solver_status": {
                "success": bool(step3.solver_result.success),
                "status": int(step3.solver_result.status),
                "message": str(step3.solver_result.message),
                "nfev": int(step3.solver_result.nfev),
                "cost": float(step3.solver_result.cost),
            },
        },
    }
    return DexAlignV2RunResult(step1=step1, step2=step2, step3=step3, summary=summary)
