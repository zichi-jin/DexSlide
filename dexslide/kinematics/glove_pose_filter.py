"""SE(3) pose smoothing for DexSlide glove wrist poses."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from dexslide.kinematics.transforms import (
    clip_norm,
    make_transform,
    quaternion_xyzw_to_rotmat,
    rotmat_to_quaternion_xyzw,
    slerp_quaternion_xyzw,
)


@dataclass(frozen=True)
class GlovePoseFilterConfig:
    position_time_constant_s: float = 0.18
    rotation_time_constant_s: float = 0.20
    max_position_step_mm: float = 18.0
    max_rotation_step_deg: float = 18.0
    max_dt_s: float = 0.12
    robust_window_size: int = 3


@dataclass(frozen=True)
class GlovePoseFilterResult:
    transform_table_hand: np.ndarray | None
    initialized: bool
    fresh_observation: bool
    used_hold: bool
    dt_s: float | None


def _alpha_from_tau(dt_s: float, tau_s: float) -> float:
    tau = max(float(tau_s), 1e-4)
    dt = max(float(dt_s), 1e-6)
    return float(1.0 - math.exp(-dt / tau))


def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        second,
        dtype=np.float64,
    ).reshape(3, 3)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _robust_pose_observation(observations: list[np.ndarray]) -> np.ndarray:
    """Return a coordinate median plus rotation medoid for a short pose window."""

    if len(observations) < 3:
        return observations[-1].copy()
    translations = np.stack(
        [np.asarray(transform[:3, 3], dtype=np.float64) for transform in observations]
    )
    filtered_translation = np.median(translations, axis=0)
    rotations = [
        np.asarray(transform[:3, :3], dtype=np.float64).reshape(3, 3)
        for transform in observations
    ]
    rotation_scores = [
        sum(_rotation_distance_rad(rotation, other) for other in rotations)
        for rotation in rotations
    ]
    filtered_rotation = rotations[int(np.argmin(rotation_scores))]
    return make_transform(filtered_rotation, filtered_translation)


class GlovePoseFilter:
    def __init__(self, config: GlovePoseFilterConfig | None = None) -> None:
        self._config = config or GlovePoseFilterConfig()
        window_size = int(self._config.robust_window_size)
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("robust_window_size must be a positive odd integer")
        self._last_transform: np.ndarray | None = None
        self._last_timestamp: float | None = None
        self._observations: deque[np.ndarray] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._last_transform = None
        self._last_timestamp = None
        self._observations.clear()

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
        if not np.isfinite(observed).all():
            raise ValueError("transform_table_hand must contain only finite values")
        self._observations.append(observed.copy())
        observed = _robust_pose_observation(list(self._observations))
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
        filtered_translation = prev_translation + clip_norm(
            filtered_translation - prev_translation,
            0.001 * float(self._config.max_position_step_mm),
        )

        prev_rotation = np.asarray(prev[:3, :3], dtype=np.float64)
        obs_rotation = np.asarray(observed[:3, :3], dtype=np.float64)
        rot_alpha = _alpha_from_tau(dt_s, self._config.rotation_time_constant_s)
        filtered_rotation = quaternion_xyzw_to_rotmat(
            slerp_quaternion_xyzw(
                rotmat_to_quaternion_xyzw(prev_rotation),
                rotmat_to_quaternion_xyzw(obs_rotation),
                rot_alpha,
            )
        )

        delta_rotation = filtered_rotation @ prev_rotation.T
        delta_rotvec, _ = cv2.Rodrigues(delta_rotation)
        max_rotation_step_rad = math.radians(float(self._config.max_rotation_step_deg))
        delta_rotvec = clip_norm(delta_rotvec.reshape(3), max_rotation_step_rad)
        filtered_rotation, _ = cv2.Rodrigues(delta_rotvec.reshape(3, 1))
        filtered_rotation = np.asarray(filtered_rotation, dtype=np.float64).reshape(3, 3) @ prev_rotation

        self._last_transform = make_transform(filtered_rotation, filtered_translation)
        self._last_timestamp = float(timestamp_s) if timestamp_s is not None else self._last_timestamp
        return GlovePoseFilterResult(self._last_transform.copy(), True, True, False, dt_s)
