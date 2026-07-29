"""SE(3) pose smoothing for DexSlide glove wrist poses."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from dexslide.world_pose.hand_cube_overlay import (
    make_transform,
    quaternion_xyzw_to_rotmat,
    rotmat_to_quaternion_xyzw,
)


@dataclass(frozen=True)
class GlovePoseFilterConfig:
    position_time_constant_s: float = 0.18
    rotation_time_constant_s: float = 0.20
    max_position_step_mm: float = 18.0
    max_rotation_step_deg: float = 18.0
    max_dt_s: float = 0.12


@dataclass(frozen=True)
class GlovePoseFilterResult:
    transform_table_hand: np.ndarray | None
    initialized: bool
    fresh_observation: bool
    used_hold: bool
    dt_s: float | None


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= max(float(max_norm), 1e-12):
        return vec
    return vec * (float(max_norm) / norm)


def _alpha_from_tau(dt_s: float, tau_s: float) -> float:
    tau = max(float(tau_s), 1e-4)
    dt = max(float(dt_s), 1e-6)
    return float(1.0 - math.exp(-dt / tau))


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
    scale_a = float(np.sin(theta_0 - theta) / max(sin_theta_0, 1e-12))
    scale_b = float(np.sin(theta) / max(sin_theta_0, 1e-12))
    blended = scale_a * qa + scale_b * qb
    return blended / max(float(np.linalg.norm(blended)), 1e-12)


class GlovePoseFilter:
    def __init__(self, config: GlovePoseFilterConfig | None = None) -> None:
        self._config = config or GlovePoseFilterConfig()
        self._last_transform: np.ndarray | None = None
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._last_transform = None
        self._last_timestamp = None

    def update(
        self,
        transform_table_hand: np.ndarray | None,
        timestamp_s: float | None = None,
    ) -> GlovePoseFilterResult:
        if transform_table_hand is None:
            if self._last_transform is None:
                return GlovePoseFilterResult(None, False, False, False, None)
            return GlovePoseFilterResult(self._last_transform.copy(), True, False, True, None)

        observed = np.asarray(transform_table_hand, dtype=np.float64).reshape(4, 4)
        if self._last_transform is None:
            self._last_transform = observed.copy()
            self._last_timestamp = float(timestamp_s) if timestamp_s is not None else None
            return GlovePoseFilterResult(self._last_transform.copy(), True, True, False, None)

        if timestamp_s is None or self._last_timestamp is None:
            dt_s = 1.0 / 30.0
        else:
            dt_s = float(timestamp_s) - float(self._last_timestamp)
        dt_s = float(np.clip(dt_s, 1e-3, self._config.max_dt_s))

        prev = self._last_transform
        prev_translation = np.asarray(prev[:3, 3], dtype=np.float64)
        obs_translation = np.asarray(observed[:3, 3], dtype=np.float64)
        pos_alpha = _alpha_from_tau(dt_s, self._config.position_time_constant_s)
        filtered_translation = prev_translation + pos_alpha * (obs_translation - prev_translation)
        filtered_translation = prev_translation + _clip_norm(
            filtered_translation - prev_translation,
            0.001 * float(self._config.max_position_step_mm),
        )

        prev_rotation = np.asarray(prev[:3, :3], dtype=np.float64)
        obs_rotation = np.asarray(observed[:3, :3], dtype=np.float64)
        rot_alpha = _alpha_from_tau(dt_s, self._config.rotation_time_constant_s)
        filtered_rotation = quaternion_xyzw_to_rotmat(
            _slerp_quaternion(
                rotmat_to_quaternion_xyzw(prev_rotation),
                rotmat_to_quaternion_xyzw(obs_rotation),
                rot_alpha,
            )
        )

        delta_rotation = filtered_rotation @ prev_rotation.T
        delta_rotvec, _ = cv2.Rodrigues(delta_rotation)
        max_rotation_step_rad = math.radians(float(self._config.max_rotation_step_deg))
        delta_rotvec = _clip_norm(delta_rotvec.reshape(3), max_rotation_step_rad)
        filtered_rotation, _ = cv2.Rodrigues(delta_rotvec.reshape(3, 1))
        filtered_rotation = np.asarray(filtered_rotation, dtype=np.float64).reshape(3, 3) @ prev_rotation

        self._last_transform = make_transform(filtered_rotation, filtered_translation)
        self._last_timestamp = float(timestamp_s) if timestamp_s is not None else self._last_timestamp
        return GlovePoseFilterResult(self._last_transform.copy(), True, True, False, dt_s)
