from __future__ import annotations

"""Centralized DexAlign loss weights for manual tuning.

修改这些值时，尽量只动 step2 / step3 的正则和鲁棒损失，不改 thumb 的运动学逻辑。
"""

import numpy as np


KEYPOINT_CLASS_WEIGHT_MAP = {
    "wrist": 1.0,
    "thumb_base": 0.05,
    "thumb_knuckle": 2.0,
    "thumb_ip": 1.0,
    "thumb_tip": 1.0,
    "index_mcp": 2.0,
    "index_pip": 1.0,
    "index_dip": 1.0,
    "index_tip": 2.0,
    "middle_mcp": 2.0,
    "middle_pip": 1.0,
    "middle_dip": 1.0,
    "middle_tip": 2.0,
    "ring_mcp": 1.0,
    "ring_pip": 1.0,
    "ring_dip": 1.0,
    "ring_tip": 1.0,
    "pinky_mcp": 1.0,
    "pinky_pip": 1.0,
    "pinky_dip": 1.0,
    "pinky_tip": 1.0,
}


# Step 2: joint affine calibration.
# Thumb stays on the original regularization strength.
STEP2_THUMB_SCALE_REG_WEIGHT = 0.08
STEP2_THUMB_BIAS_REG_WEIGHT = 0.5
# The four non-thumb fingers are constrained more tightly.
STEP2_FOUR_FINGERS_SCALE_REG_WEIGHT = 50   # 0.45
STEP2_FOUR_FINGERS_BIAS_REG_WEIGHT = 50  # 0.12
STEP2_JOINT_SCALE_LOWER_BOUND = 0.5
STEP2_JOINT_SCALE_UPPER_BOUND = 5
STEP2_JOINT_BIAS_BOUND_DEG = 35
STEP2_JOINT_BIAS_BOUND_RAD = np.deg2rad(STEP2_JOINT_BIAS_BOUND_DEG)
STEP2_HUBER_F_SCALE = 0.08


def build_step2_joint_reg_weights() -> tuple[np.ndarray, np.ndarray]:
    scale = np.concatenate(
        [
            np.full(4, STEP2_THUMB_SCALE_REG_WEIGHT, dtype=np.float64),
            np.full(16, STEP2_FOUR_FINGERS_SCALE_REG_WEIGHT, dtype=np.float64),
        ]
    )
    bias = np.concatenate(
        [
            np.full(4, STEP2_THUMB_BIAS_REG_WEIGHT, dtype=np.float64),
            np.full(16, STEP2_FOUR_FINGERS_BIAS_REG_WEIGHT, dtype=np.float64),
        ]
    )
    return scale, bias


STEP2_SCALE_REG_WEIGHTS, STEP2_BIAS_REG_WEIGHTS = build_step2_joint_reg_weights()


# Step 1: palm base direction trust region.
# Thumb remains fully data-driven. The four non-thumb base directions are
# prevented from drifting too far away from the initial guess.
STEP1_NON_THUMB_MAX_BASE_DIRECTION_DELTA_DEG = 90


# Step 3: lengths + marker translation.
STEP3_TRANSLATION_SAMPLE_WEIGHT = 0.06
STEP3_TRANSLATION_PRIOR_WEIGHT = 0.03
STEP3_HUBER_F_SCALE = 15.0
STEP3_THUMB_BASE_RX_BOUND_DEG = 45.0
STEP3_THUMB_BASE_RX_BOUND_RAD = np.deg2rad(STEP3_THUMB_BASE_RX_BOUND_DEG)
