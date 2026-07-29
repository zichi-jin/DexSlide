from __future__ import annotations

import copy

import numpy as np

from dexslide.calibration.dexalign.io_utils import load_runtime_skeleton
from dexslide.kinematics.live_hand import finger_points, rot_x, runtime_palm_points
from dexslide.paths import DEFAULT_SKELETON_FILE


def test_four_finger_zero_pose_points_along_hand_x_axis() -> None:
    skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)
    palm = runtime_palm_points(skeleton, "left")

    for finger in ("index", "middle", "ring", "pinky"):
        pts = finger_points(finger, np.zeros(4, dtype=np.float64), skeleton, palm, "left")
        deltas = np.diff(pts, axis=0)
        np.testing.assert_allclose(deltas[:, 1:], 0.0, atol=1e-9)
        assert np.all(deltas[:, 0] > 0.0)


def test_thumb_chain_rx_rotates_the_downstream_thumb_frame() -> None:
    skeleton = load_runtime_skeleton(DEFAULT_SKELETON_FILE)
    palm = runtime_palm_points(skeleton, "left")
    q = np.zeros(4, dtype=np.float64)
    baseline = finger_points("thumb", q, skeleton, palm, "left")

    angle = np.deg2rad(18.0)
    rotated_skeleton = copy.deepcopy(skeleton)
    rotated_skeleton.setdefault("palm", {})["thumb_chain_rx_rad"] = float(angle)
    rotated = finger_points("thumb", q, rotated_skeleton, palm, "left")

    np.testing.assert_allclose(rotated[0], baseline[0], atol=1e-9)
    np.testing.assert_allclose(rotated[1] - rotated[0], rot_x(angle) @ (baseline[1] - baseline[0]), atol=1e-9)
