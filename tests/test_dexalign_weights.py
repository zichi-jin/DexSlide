from __future__ import annotations

import numpy as np

from dexslide.calibration.dexalign.pipeline_v2 import _clamp_direction_delta
from dexslide.calibration.dexalign.weights import (
    STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG,
    STEP2_BIAS_REG_WEIGHTS,
    STEP2_FOUR_FINGERS_BIAS_REG_WEIGHT,
    STEP2_FOUR_FINGERS_SCALE_REG_WEIGHT,
    STEP2_SCALE_REG_WEIGHTS,
    STEP2_THUMB_BIAS_REG_WEIGHT,
    STEP2_THUMB_SCALE_REG_WEIGHT,
)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    return value / np.linalg.norm(value)


def test_step2_joint_regularization_weights_split_thumb_and_four_fingers() -> None:
    assert STEP2_SCALE_REG_WEIGHTS.shape == (20,)
    assert STEP2_BIAS_REG_WEIGHTS.shape == (20,)
    np.testing.assert_allclose(STEP2_SCALE_REG_WEIGHTS[:4], STEP2_THUMB_SCALE_REG_WEIGHT, atol=0.0)
    np.testing.assert_allclose(STEP2_BIAS_REG_WEIGHTS[:4], STEP2_THUMB_BIAS_REG_WEIGHT, atol=0.0)
    np.testing.assert_allclose(STEP2_SCALE_REG_WEIGHTS[4:], STEP2_FOUR_FINGERS_SCALE_REG_WEIGHT, atol=0.0)
    np.testing.assert_allclose(STEP2_BIAS_REG_WEIGHTS[4:], STEP2_FOUR_FINGERS_BIAS_REG_WEIGHT, atol=0.0)


def test_clamp_direction_delta_limits_non_thumb_base_update() -> None:
    initial = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    candidate = _unit(np.array([0.0, 1.0, 0.0], dtype=np.float64))

    clamped = _clamp_direction_delta(
        initial,
        candidate,
        max_delta_deg=STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG,
    )

    assert clamped is not None
    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(initial, clamped), -1.0, 1.0))))
    assert abs(angle_deg - STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG) < 1e-6


def test_clamp_direction_delta_keeps_candidate_when_within_limit() -> None:
    initial = _unit(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    candidate = _unit(np.array([1.0, 0.05, 0.0], dtype=np.float64))

    clamped = _clamp_direction_delta(
        initial,
        candidate,
        max_delta_deg=STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG,
    )

    assert clamped is not None
    np.testing.assert_allclose(clamped, candidate, atol=1e-9)
