from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexslide.kinematics.glove_pose_filter import GlovePoseFilter, GlovePoseFilterConfig


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "jaka_dexslide_incremental_teleop.py"
SPEC = importlib.util.spec_from_file_location("jaka_dexslide_incremental_teleop", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _translate_x(distance_m: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = float(distance_m)
    return transform


def test_glove_pose_filter_holds_last_pose_when_observation_missing() -> None:
    pose_filter = GlovePoseFilter()
    observed = _translate_x(0.012)

    first = pose_filter.update(observed, timestamp_s=0.0)
    held = pose_filter.update(None, timestamp_s=0.2)

    assert first.initialized is True
    assert first.fresh_observation is True
    assert first.used_hold is False
    assert held.initialized is True
    assert held.fresh_observation is False
    assert held.used_hold is True
    np.testing.assert_allclose(held.transform_table_hand, observed, atol=1e-12)


def test_glove_pose_filter_recovers_from_tracking_loss_with_limited_step() -> None:
    pose_filter = GlovePoseFilter(
        GlovePoseFilterConfig(
            position_time_constant_s=0.01,
            rotation_time_constant_s=0.01,
            max_position_step_mm=5.0,
            max_rotation_step_deg=5.0,
            max_dt_s=0.05,
        )
    )
    pose_filter.update(_translate_x(0.0), timestamp_s=0.0)
    pose_filter.update(None, timestamp_s=0.5)

    recovered = pose_filter.update(_translate_x(0.100), timestamp_s=1.0)

    assert recovered.fresh_observation is True
    assert recovered.used_hold is False
    assert recovered.transform_table_hand is not None
    assert abs(float(recovered.transform_table_hand[0, 3]) - 0.005) < 1e-6


def test_smooth_wrist_pose_sample_preserves_metadata_and_applies_filter() -> None:
    pose_filter = GlovePoseFilter(
        GlovePoseFilterConfig(
            position_time_constant_s=0.01,
            rotation_time_constant_s=0.01,
            max_position_step_mm=2.0,
            max_rotation_step_deg=5.0,
            max_dt_s=0.05,
        )
    )
    first = MODULE.WristPoseSample(
        transform_table_hand=_translate_x(0.0),
        frame_idx=7,
        time_wall=0.0,
        source_marker_ids=(1, 2),
    )
    second = MODULE.WristPoseSample(
        transform_table_hand=_translate_x(0.100),
        frame_idx=8,
        time_wall=1.0,
        source_marker_ids=(2, 3),
    )

    MODULE.smooth_wrist_pose_sample(pose_filter, first)
    filtered = MODULE.smooth_wrist_pose_sample(pose_filter, second)

    assert filtered.frame_idx == 8
    assert filtered.time_wall == 1.0
    assert filtered.source_marker_ids == (2, 3)
    assert 0.0 < float(filtered.transform_table_hand[0, 3]) < 0.100
    assert abs(float(filtered.transform_table_hand[0, 3]) - 0.002) < 1e-6
