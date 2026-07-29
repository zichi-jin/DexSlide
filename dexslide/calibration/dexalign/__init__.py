"""DexAlign offline alignment helpers."""

from .io_utils import load_alignment_dataset, save_alignment_dataset
from .objective import (
    AlignmentEvaluation,
    alignment_residual_vector,
    compose_parameter_vector,
    evaluate_alignment,
    pack_marker2hand_transform,
    unpack_marker2hand_params,
)
from .skeleton_param import (
    BONE_PARAM_COUNT,
    PALM_PARAM_COUNT,
    PALM_PARAM_START,
    SKELETON_PARAM_SIZE,
    flatten_skeleton,
    make_bounds,
    unflatten_skeleton,
)
from .types import AlignmentDataset, AlignmentFrame, OptimizationResult


def run_alignment_optimization(*args, **kwargs):
    from .optimize_alignment import run_alignment_optimization as _run_alignment_optimization

    return _run_alignment_optimization(*args, **kwargs)

__all__ = [
    "AlignmentDataset",
    "AlignmentEvaluation",
    "AlignmentFrame",
    "BONE_PARAM_COUNT",
    "OptimizationResult",
    "PALM_PARAM_COUNT",
    "PALM_PARAM_START",
    "SKELETON_PARAM_SIZE",
    "alignment_residual_vector",
    "compose_parameter_vector",
    "evaluate_alignment",
    "flatten_skeleton",
    "load_alignment_dataset",
    "make_bounds",
    "pack_marker2hand_transform",
    "run_alignment_optimization",
    "save_alignment_dataset",
    "unflatten_skeleton",
    "unpack_marker2hand_params",
]
