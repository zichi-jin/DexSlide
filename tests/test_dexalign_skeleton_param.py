from __future__ import annotations

import numpy as np

from dexslide.calibration.dexalign.io_utils import load_runtime_skeleton
from dexslide.calibration.dexalign.skeleton_param import (
    BONE_PARAM_COUNT,
    SKELETON_PARAM_SIZE,
    flatten_skeleton,
    make_bounds,
    unflatten_skeleton,
)
from dexslide.paths import DEFAULT_SKELETON_FILE


def test_flatten_unflatten_roundtrip_preserves_runtime_structure() -> None:
    skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)

    theta = flatten_skeleton(skeleton)
    recovered = unflatten_skeleton(theta, skeleton)

    assert theta.shape == (SKELETON_PARAM_SIZE,)
    assert recovered["note"] == skeleton["note"]
    np.testing.assert_allclose(
        np.asarray(recovered["palm"]["vertices"], dtype=np.float64),
        np.asarray(skeleton["palm"]["vertices"], dtype=np.float64),
        atol=1e-9,
    )
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        for key, value in skeleton[finger].items():
            assert float(recovered[finger][key]) == float(value)


def test_unflatten_keeps_wrist_fixed_at_origin() -> None:
    skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)
    theta = flatten_skeleton(skeleton)
    theta[-3:] += np.array([5.0, -7.0, 2.0], dtype=np.float64)

    recovered = unflatten_skeleton(theta, skeleton)

    np.testing.assert_allclose(
        np.asarray(recovered["palm"]["vertices"][0], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        atol=1e-9,
    )


def test_make_bounds_keeps_bone_lengths_in_open_positive_range() -> None:
    skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)

    lower, upper = make_bounds(skeleton)

    assert lower.shape == (SKELETON_PARAM_SIZE,)
    assert upper.shape == (SKELETON_PARAM_SIZE,)
    assert np.all(lower[:BONE_PARAM_COUNT] > 0.0)
    assert np.all(upper[:BONE_PARAM_COUNT] <= 130.0)
