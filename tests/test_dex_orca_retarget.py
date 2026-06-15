from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from dexslide.paths import DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE, DEFAULT_SKELETON_FILE
from dexslide.retargeting.engine import _build_ref_value, _load_retarget_document, create_dex_retargeter
from dexslide.retargeting.human_model import DexSlideHumanModel


def _choose_retarget_python() -> str | None:
    override = os.environ.get("DEX_RETARGETING_PYTHON")
    if override:
        return override
    for candidate in ("python3.10", "python3.11", "python3.12"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def test_human_model_outputs_21_landmarks() -> None:
    model = DexSlideHumanModel(DEFAULT_SKELETON_FILE, hand="right")
    landmarks = model.landmarks_from_angles(np.zeros(20, dtype=np.float64))
    assert landmarks.shape == (21, 3)
    assert np.all(np.isfinite(landmarks))
    assert np.linalg.norm(landmarks[4] - landmarks[0]) > 1e-6
    assert np.linalg.norm(landmarks[20] - landmarks[0]) > 1e-6


def test_human_model_left_right_and_mirror_are_consistent() -> None:
    zeros = np.zeros(20, dtype=np.float64)
    left = DexSlideHumanModel(DEFAULT_SKELETON_FILE, hand="left").landmarks_from_angles(zeros)
    right = DexSlideHumanModel(DEFAULT_SKELETON_FILE, hand="right").landmarks_from_angles(zeros)
    mirrored_left = DexSlideHumanModel(
        DEFAULT_SKELETON_FILE,
        hand="left",
        mirror_reconstruction=True,
    ).landmarks_from_angles(zeros)

    assert np.allclose(left[:, 0], right[:, 0])
    assert np.allclose(left[:, 2], right[:, 2])
    assert np.allclose(left[:, 1], -right[:, 1])
    assert np.allclose(mirrored_left, right)


def test_orca_retarget_config_is_well_formed() -> None:
    document = _load_retarget_document(DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE)
    assert len(document["human_joint_names"]) == 20
    assert len(document["orcahand_joint_ids"]) == 17
    assert len(document["orcahand_urdf_joint_names"]) == 17
    assert document["fixed_qpos"]["right_wrist"] == 0.0
    ref_value = _build_ref_value(
        DexSlideHumanModel(DEFAULT_SKELETON_FILE).landmarks_from_angles(np.zeros(20, dtype=np.float64)),
        document,
    )
    assert ref_value.shape == (10, 3)
    assert np.all(np.isfinite(ref_value))


def test_retarget_worker_roundtrip_if_dependency_available() -> None:
    python_bin = _choose_retarget_python()
    if python_bin is None:
        pytest.skip("retarget worker smoke test requires DEX_RETARGETING_PYTHON or python3.10-3.12")

    probe = subprocess.run(
        [python_bin, "-c", "from dex_retargeting.retargeting_config import RetargetingConfig; print('ok')"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("selected interpreter does not provide dex_retargeting")

    retargeter = create_dex_retargeter(
        config_file=DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE,
        skeleton_file=DEFAULT_SKELETON_FILE,
        use_subprocess=True,
        dex_retargeting_python=python_bin,
    )
    try:
        qpos = retargeter.retarget(np.zeros(20, dtype=np.float64))
    finally:
        retargeter.close()

    assert qpos.shape == (17,)
    assert np.all(np.isfinite(qpos))
