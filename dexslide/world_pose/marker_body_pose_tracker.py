"""Stateful marker-body pose tracking built on top of the hand marker-body geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dexslide.world_pose.hand_cube_overlay import (
    CubePoseEstimate,
    HandCubeOverlayConfig,
    MarkerBodyConsistencyReport,
    average_rotation_matrices,
    diagnose_marker_body_consistency,
    estimate_cube_pose_in_table,
    make_transform,
    quaternion_xyzw_to_rotmat,
    rotmat_to_quaternion_xyzw,
)


def _smooth_marker_body_pose(
    previous_pose: CubePoseEstimate | None,
    current_pose: CubePoseEstimate | None,
    *,
    alpha: float,
) -> CubePoseEstimate | None:
    if current_pose is None:
        return None
    if previous_pose is None or alpha >= 0.999:
        return current_pose

    alpha_clamped = float(np.clip(alpha, 1e-3, 0.999))
    prev_transform = previous_pose.transform_table_cube
    curr_transform = current_pose.transform_table_cube
    smoothed_position = (
        (1.0 - alpha_clamped) * np.asarray(prev_transform[:3, 3], dtype=np.float64)
        + alpha_clamped * np.asarray(curr_transform[:3, 3], dtype=np.float64)
    )
    smoothed_rotation = average_rotation_matrices(
        [prev_transform[:3, :3], curr_transform[:3, :3]],
        [1.0 - alpha_clamped, alpha_clamped],
    )
    return CubePoseEstimate(
        transform_table_cube=make_transform(smoothed_rotation, smoothed_position),
        source_marker_ids=list(current_pose.source_marker_ids),
        max_position_deviation_m=float(current_pose.max_position_deviation_m),
        solver_mode=str(current_pose.solver_mode),
        mean_reprojection_error_px=float(current_pose.mean_reprojection_error_px),
        max_reprojection_error_px=float(current_pose.max_reprojection_error_px),
    )


@dataclass(frozen=True)
class MarkerBodyPoseTrackerResult:
    raw_pose: CubePoseEstimate | None
    smoothed_pose: CubePoseEstimate | None
    consistency_report: MarkerBodyConsistencyReport | None


@dataclass
class _OneEuroVectorState:
    raw_value: np.ndarray | None = None
    filtered_value: np.ndarray | None = None
    filtered_derivative: np.ndarray | None = None


def _alpha_from_cutoff(dt: float, cutoff_hz: float) -> float:
    cutoff = max(float(cutoff_hz), 1e-4)
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return float(1.0 / (1.0 + tau / max(float(dt), 1e-6)))


def _lowpass_vector(previous: np.ndarray | None, current: np.ndarray, alpha: float) -> np.ndarray:
    current_arr = np.asarray(current, dtype=np.float64).reshape(-1)
    if previous is None:
        return current_arr.copy()
    previous_arr = np.asarray(previous, dtype=np.float64).reshape(-1)
    return (1.0 - float(alpha)) * previous_arr + float(alpha) * current_arr


def _relative_rotvec(rot_from: np.ndarray, rot_to: np.ndarray) -> np.ndarray:
    relative_rot = np.asarray(rot_from, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        rot_to,
        dtype=np.float64,
    ).reshape(3, 3)
    rotvec, _ = cv2.Rodrigues(relative_rot)
    return np.asarray(rotvec, dtype=np.float64).reshape(3)


def _slerp_quaternion(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray, alpha: float) -> np.ndarray:
    qa = np.asarray(q1_xyzw, dtype=np.float64).reshape(4)
    qb = np.asarray(q2_xyzw, dtype=np.float64).reshape(4)
    qa /= max(float(np.linalg.norm(qa)), 1e-12)
    qb /= max(float(np.linalg.norm(qb)), 1e-12)

    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot

    alpha_clamped = float(np.clip(alpha, 0.0, 1.0))
    if dot > 0.9995:
        blended = (1.0 - alpha_clamped) * qa + alpha_clamped * qb
        return blended / max(float(np.linalg.norm(blended)), 1e-12)

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha_clamped
    sin_theta = float(np.sin(theta))
    scale_a = float(np.sin(theta_0 - theta) / max(sin_theta_0, 1e-12))
    scale_b = float(sin_theta / max(sin_theta_0, 1e-12))
    blended = scale_a * qa + scale_b * qb
    return blended / max(float(np.linalg.norm(blended)), 1e-12)


def _copy_pose_with_transform(pose: CubePoseEstimate, transform_table_cube: np.ndarray) -> CubePoseEstimate:
    return CubePoseEstimate(
        transform_table_cube=np.asarray(transform_table_cube, dtype=np.float64).reshape(4, 4).copy(),
        source_marker_ids=list(pose.source_marker_ids),
        max_position_deviation_m=float(pose.max_position_deviation_m),
        solver_mode=str(pose.solver_mode),
        mean_reprojection_error_px=float(pose.mean_reprojection_error_px),
        max_reprojection_error_px=float(pose.max_reprojection_error_px),
    )


class MarkerBodyPoseTracker:
    def __init__(
        self,
        config: HandCubeOverlayConfig,
        *,
        pose_solver: str,
        smoothing_alpha: float,
        outlier_threshold_m: float,
        reprojection_error_threshold_px: float,
        enable_diagnostics: bool = False,
    ) -> None:
        self._config = config
        self._pose_solver = str(pose_solver).strip().lower()
        self._smoothing_alpha = float(smoothing_alpha)
        self._outlier_threshold_m = float(outlier_threshold_m)
        self._reprojection_error_threshold_px = float(reprojection_error_threshold_px)
        self._enable_diagnostics = bool(enable_diagnostics)
        self._smoothed_pose: CubePoseEstimate | None = None
        self._position_state = _OneEuroVectorState()
        self._rotation_state = _OneEuroVectorState()
        self._last_timestamp: float | None = None

        responsiveness = float(np.clip(self._smoothing_alpha, 0.0, 1.0))
        self._position_min_cutoff_hz = 0.35 + 2.85 * responsiveness
        self._position_beta = 0.02 + 0.24 * responsiveness
        self._rotation_min_cutoff_hz = 0.30 + 2.40 * responsiveness
        self._rotation_beta = 0.02 + 0.16 * responsiveness
        self._derivative_cutoff_hz = 1.5

    def reset(self) -> None:
        self._smoothed_pose = None
        self._position_state = _OneEuroVectorState()
        self._rotation_state = _OneEuroVectorState()
        self._last_timestamp = None

    def _compute_dt(self, frame_result: dict[str, object]) -> float:
        time_wall = frame_result.get("time_wall")
        if time_wall is None:
            return 1.0 / 30.0
        current_time = float(time_wall)
        if self._last_timestamp is None:
            self._last_timestamp = current_time
            return 1.0 / 30.0
        dt = max(current_time - self._last_timestamp, 1e-3)
        self._last_timestamp = current_time
        return float(min(dt, 0.25))

    def _filter_position(self, measured_position: np.ndarray, dt: float) -> np.ndarray:
        measured = np.asarray(measured_position, dtype=np.float64).reshape(3)
        if self._position_state.filtered_value is None or self._position_state.raw_value is None:
            self._position_state.raw_value = measured.copy()
            self._position_state.filtered_value = measured.copy()
            self._position_state.filtered_derivative = np.zeros(3, dtype=np.float64)
            return measured.copy()

        raw_derivative = (measured - self._position_state.raw_value) / max(float(dt), 1e-6)
        derivative_alpha = _alpha_from_cutoff(dt, self._derivative_cutoff_hz)
        derivative_hat = _lowpass_vector(
            self._position_state.filtered_derivative,
            raw_derivative,
            derivative_alpha,
        )
        if self._smoothing_alpha >= 0.999:
            filtered = measured.copy()
        else:
            cutoff_hz = self._position_min_cutoff_hz + self._position_beta * float(np.linalg.norm(derivative_hat))
            value_alpha = _alpha_from_cutoff(dt, cutoff_hz)
            filtered = _lowpass_vector(self._position_state.filtered_value, measured, value_alpha)

        self._position_state.raw_value = measured.copy()
        self._position_state.filtered_value = filtered.copy()
        self._position_state.filtered_derivative = derivative_hat.copy()
        return filtered

    def _filter_rotation(self, measured_rotation: np.ndarray, dt: float) -> np.ndarray:
        measured = np.asarray(measured_rotation, dtype=np.float64).reshape(3, 3)
        if self._rotation_state.filtered_value is None or self._rotation_state.raw_value is None:
            self._rotation_state.raw_value = measured.copy()
            self._rotation_state.filtered_value = measured.copy()
            self._rotation_state.filtered_derivative = np.zeros(3, dtype=np.float64)
            return measured.copy()

        raw_derivative = _relative_rotvec(self._rotation_state.raw_value, measured) / max(float(dt), 1e-6)
        derivative_alpha = _alpha_from_cutoff(dt, self._derivative_cutoff_hz)
        derivative_hat = _lowpass_vector(
            self._rotation_state.filtered_derivative,
            raw_derivative,
            derivative_alpha,
        )
        if self._smoothing_alpha >= 0.999:
            filtered = measured.copy()
        else:
            cutoff_hz = self._rotation_min_cutoff_hz + self._rotation_beta * float(np.linalg.norm(derivative_hat))
            value_alpha = _alpha_from_cutoff(dt, cutoff_hz)
            q_prev = rotmat_to_quaternion_xyzw(self._rotation_state.filtered_value)
            q_curr = rotmat_to_quaternion_xyzw(measured)
            filtered = quaternion_xyzw_to_rotmat(_slerp_quaternion(q_prev, q_curr, value_alpha))

        self._rotation_state.raw_value = measured.copy()
        self._rotation_state.filtered_value = filtered.copy()
        self._rotation_state.filtered_derivative = derivative_hat.copy()
        return filtered

    def _filter_pose(self, accepted_pose: CubePoseEstimate, dt: float) -> CubePoseEstimate:
        transform = np.asarray(accepted_pose.transform_table_cube, dtype=np.float64).reshape(4, 4)
        filtered_position = self._filter_position(transform[:3, 3], dt)
        filtered_rotation = self._filter_rotation(transform[:3, :3], dt)
        filtered_transform = make_transform(filtered_rotation, filtered_position)
        return _copy_pose_with_transform(accepted_pose, filtered_transform)

    def update(
        self,
        *,
        frame_result: dict[str, object],
        camera_matrix: np.ndarray,
    ) -> MarkerBodyPoseTrackerResult:
        camera = np.asarray(camera_matrix, dtype=np.float64)
        dt = self._compute_dt(frame_result)
        raw_pose = estimate_cube_pose_in_table(
            frame_result,
            self._config,
            outlier_threshold_m=self._outlier_threshold_m,
            camera_matrix=camera,
            reprojection_error_threshold_px=self._reprojection_error_threshold_px,
            pose_solver=self._pose_solver,
        )

        report = None
        if raw_pose is not None:
            report = diagnose_marker_body_consistency(
                frame_result,
                self._config,
                fused_pose=raw_pose,
                camera_matrix=camera,
            )
        if raw_pose is None:
            self._smoothed_pose = None
            self._position_state = _OneEuroVectorState()
            self._rotation_state = _OneEuroVectorState()
            smoothed_pose = None
        else:
            smoothed_pose = self._filter_pose(raw_pose, dt)
            self._smoothed_pose = smoothed_pose

        return MarkerBodyPoseTrackerResult(
            raw_pose=raw_pose,
            smoothed_pose=smoothed_pose,
            consistency_report=report if self._enable_diagnostics else None,
        )
