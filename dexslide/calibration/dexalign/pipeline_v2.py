"""Compatibility facade for the DexAlign v2 pipeline.

The implementation lives in :mod:`dexslide.calibration.dexalign.steps`; this
module remains as the stable import surface for existing tools and tests.
"""

from . import steps as _impl
from .steps import (
    DexAlignStep1Result,
    DexAlignStep2Result,
    DexAlignStep3Result,
    DexAlignV2RunResult,
    FramePool,
    run_dexalign_v2,
    run_step1_palm_shape,
    run_step2_joint_calibration,
    run_step3_lengths_and_translation,
)

__all__ = [
    "DexAlignStep1Result",
    "DexAlignStep2Result",
    "DexAlignStep3Result",
    "DexAlignV2RunResult",
    "FramePool",
    "run_dexalign_v2",
    "run_step1_palm_shape",
    "run_step2_joint_calibration",
    "run_step3_lengths_and_translation",
]


def __getattr__(name: str):
    return getattr(_impl, name)

