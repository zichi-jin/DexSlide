from __future__ import annotations

import numpy as np
import pytest

from dexslide.kinematics.glove_pose_filter import GlovePoseFilter, GlovePoseFilterConfig
from dexslide.kinematics.transforms import make_transform


def _pose(x_m: float, angle_rad: float = 0.0) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return make_transform(rotation, np.array([x_m, 0.0, 0.0], dtype=np.float64))


def test_robust_window_rejects_single_frame_pose_spike() -> None:
    pose_filter = GlovePoseFilter(
        GlovePoseFilterConfig(
            position_time_constant_s=0.001,
            rotation_time_constant_s=0.001,
            max_position_step_mm=1000.0,
            max_rotation_step_deg=180.0,
            robust_window_size=3,
        )
    )
    pose_filter.update(_pose(0.0), 0.00)
    pose_filter.update(_pose(0.10, 1.0), 0.02)
    result = pose_filter.update(_pose(0.0), 0.04)
    assert result.transform_table_hand is not None
    assert abs(float(result.transform_table_hand[0, 3])) < 1e-6
    np.testing.assert_allclose(result.transform_table_hand[:3, :3], np.eye(3), atol=1e-6)


def test_filter_reset_clears_robust_observation_history() -> None:
    pose_filter = GlovePoseFilter(GlovePoseFilterConfig(robust_window_size=3))
    pose_filter.update(_pose(0.0), 0.00)
    pose_filter.update(_pose(0.10), 0.02)
    pose_filter.reset()
    result = pose_filter.update(_pose(0.25), 1.00)
    assert result.transform_table_hand is not None
    np.testing.assert_allclose(result.transform_table_hand, _pose(0.25), atol=1e-9)


def test_filter_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError, match="positive odd integer"):
        GlovePoseFilter(GlovePoseFilterConfig(robust_window_size=2))
