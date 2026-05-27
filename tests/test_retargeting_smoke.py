from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "retarget_urdf_to_urdf.py"
SOURCE_JOINT_NAMES = [
    "thumb_joint",
    "index_joint",
    "middle_joint",
    "ring_joint",
    "pinky_joint",
]
TIP_LINK_NAMES = [
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]
TIP_INDEX_MAP = {
    4: "thumb_tip",
    8: "index_tip",
    12: "middle_tip",
    16: "ring_tip",
    20: "pinky_tip",
}


def _choose_retarget_python() -> str | None:
    override = os.environ.get("DEX_RETARGETING_PYTHON")
    if override:
        return override
    for candidate in ("python3.10", "python3.11", "python3.12"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def _build_pythonpath(extra_paths: list[str]) -> str:
    existing = os.environ.get("PYTHONPATH")
    parts: list[str] = []

    def _append_with_cmeel(path_str: str) -> None:
        if not path_str:
            return
        parts.append(path_str)
        root = Path(path_str)
        candidates = list(root.glob("cmeel.prefix/lib/python*/site-packages"))
        candidates += list(root.glob("cmeel.prefix/lib64/python*/site-packages"))
        candidates += list(root.glob("lib/python*/site-packages"))
        candidates += list(root.glob("lib64/python*/site-packages"))
        for candidate in candidates:
            if candidate.is_dir():
                parts.append(str(candidate))

    for path_str in extra_paths:
        _append_with_cmeel(path_str)

    extra = os.environ.get("DEX_RETARGETING_EXTRA_PYTHONPATH")
    if extra:
        for path_str in extra.split(os.pathsep):
            _append_with_cmeel(path_str)
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _can_import_retargeting(python_bin: str, pythonpath: str) -> tuple[bool, str]:
    probe = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "from dex_retargeting.retargeting_config import RetargetingConfig; "
                "from dex_retargeting.robot_wrapper import RobotWrapper; "
                "print('ok')"
            ),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    return probe.returncode == 0, (probe.stdout + probe.stderr).strip()


def _compute_tip_positions(
    python_bin: str,
    pythonpath: str,
    urdf_path: Path,
    qpos_path: Path,
    link_names: list[str],
) -> np.ndarray:
    code = """
import json
import sys
import numpy as np
from dex_retargeting.robot_wrapper import RobotWrapper

urdf_path = sys.argv[1]
qpos_path = sys.argv[2]
link_names = json.loads(sys.argv[3])
robot = RobotWrapper(urdf_path)
link_ids = [robot.get_link_index(name) for name in link_names]
qpos_seq = np.load(qpos_path)
out = []
for qpos in qpos_seq:
    robot.compute_forward_kinematics(qpos)
    out.append([robot.get_link_pose(link_id)[:3, 3].tolist() for link_id in link_ids])
print(json.dumps(out))
"""
    run = subprocess.run(
        [python_bin, "-c", code, str(urdf_path), str(qpos_path), json.dumps(link_names)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )
    if run.returncode != 0:
        pytest.fail(
            "failed to compute tip positions for retargeting smoke test.\n"
            f"stdout:\n{run.stdout}\n\nstderr:\n{run.stderr}"
        )
    return np.asarray(json.loads(run.stdout), dtype=np.float64)


def _finger_block(name: str, base_y: float, length: float) -> str:
    return f"""
  <joint name="{name}_joint" type="revolute">
    <parent link="palm"/>
    <child link="{name}_link"/>
    <origin xyz="0 {base_y:.3f} 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="1.2" effort="1" velocity="1"/>
  </joint>
  <link name="{name}_link">
    <inertial>
      <origin xyz="{length / 2:.3f} 0 0" rpy="0 0 0"/>
      <mass value="0.02"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
  </link>
  <joint name="{name}_tip_joint" type="fixed">
    <parent link="{name}_link"/>
    <child link="{name}_tip"/>
    <origin xyz="{length:.3f} 0 0" rpy="0 0 0"/>
  </joint>
  <link name="{name}_tip">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>"""


def _write_hand_urdf(path: Path) -> None:
    fingers = [
        ("thumb", -0.040, 0.060),
        ("index", -0.020, 0.085),
        ("middle", 0.000, 0.095),
        ("ring", 0.020, 0.090),
        ("pinky", 0.040, 0.075),
    ]
    blocks = "\n".join(_finger_block(name, base_y, length) for name, base_y, length in fingers)
    urdf = f"""<?xml version="1.0"?>
<robot name="smoke_hand">
  <link name="palm">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>
{blocks}
</robot>
"""
    path.write_text(urdf, encoding="utf-8")


def _write_target_config(path: Path, urdf_name: str) -> None:
    config = {
        "retargeting": {
            "type": "position",
            "urdf_path": urdf_name,
            "target_joint_names": SOURCE_JOINT_NAMES,
            "target_link_names": TIP_LINK_NAMES,
            "target_link_human_indices": [4, 8, 12, 16, 20],
            "low_pass_alpha": 1.0,
        }
    }
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def test_retarget_urdf_to_urdf_smoke(tmp_path: Path) -> None:
    python_bin = _choose_retarget_python()
    if python_bin is None:
        pytest.skip("retargeting smoke test requires Python 3.10-3.12 or DEX_RETARGETING_PYTHON")

    pythonpath = _build_pythonpath([])
    ok, import_output = _can_import_retargeting(python_bin, pythonpath)
    if not ok:
        pytest.skip(
            "retargeting smoke test requires dex-retargeting in the selected interpreter. "
            "Set DEX_RETARGETING_PYTHON and optionally DEX_RETARGETING_EXTRA_PYTHONPATH. "
            f"Probe output: {import_output}"
        )

    source_urdf = tmp_path / "source_hand.urdf"
    target_urdf = tmp_path / "target_hand.urdf"
    _write_hand_urdf(source_urdf)
    _write_hand_urdf(target_urdf)

    target_config = tmp_path / "target_config.json"
    _write_target_config(target_config, target_urdf.name)

    source_qpos = np.asarray(
        [
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.15, 0.30, 0.45, 0.60, 0.75],
            [0.75, 0.60, 0.45, 0.30, 0.15],
        ],
        dtype=np.float64,
    )
    source_qpos_path = tmp_path / "source_qpos.npz"
    np.savez_compressed(
        source_qpos_path,
        qpos=source_qpos,
        joint_names=np.asarray(SOURCE_JOINT_NAMES, dtype=object),
    )

    index_map_path = tmp_path / "source_index_link_map.json"
    index_map_path.write_text(json.dumps(TIP_INDEX_MAP, indent=2), encoding="utf-8")

    output_path = tmp_path / "retarget_output.npz"
    env = {**os.environ, "PYTHONPATH": pythonpath}
    run = subprocess.run(
        [
            python_bin,
            str(SCRIPT_PATH),
            "--target-config",
            str(target_config),
            "--source-urdf",
            str(source_urdf),
            "--source-qpos",
            str(source_qpos_path),
            "--source-index-link-map",
            str(index_map_path),
            "--output",
            str(output_path),
            "--print-every",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if run.returncode != 0:
        pytest.fail(
            "retargeting smoke test subprocess failed.\n"
            f"stdout:\n{run.stdout}\n\nstderr:\n{run.stderr}"
        )

    output = np.load(output_path, allow_pickle=True)
    target_qpos = np.asarray(output["qpos"], dtype=np.float64)
    target_joint_names = [str(x) for x in output["joint_names"].tolist()]
    source_joint_names = [str(x) for x in output["source_joint_names"].tolist()]
    source_index_by_name = {name: i for i, name in enumerate(SOURCE_JOINT_NAMES)}
    reordered_source_qpos = source_qpos[:, [source_index_by_name[name] for name in target_joint_names]]
    reordered_source_path = tmp_path / "expected_source_qpos.npy"
    target_qpos_path = tmp_path / "retargeted_qpos.npy"
    np.save(reordered_source_path, reordered_source_qpos)
    np.save(target_qpos_path, target_qpos)

    assert target_qpos.shape == source_qpos.shape
    assert sorted(target_joint_names) == sorted(SOURCE_JOINT_NAMES)
    assert sorted(source_joint_names) == sorted(SOURCE_JOINT_NAMES)
    assert np.all(np.isfinite(target_qpos))

    source_tip_positions = _compute_tip_positions(
        python_bin,
        pythonpath,
        source_urdf,
        reordered_source_path,
        TIP_LINK_NAMES,
    )
    target_tip_positions = _compute_tip_positions(
        python_bin,
        pythonpath,
        target_urdf,
        target_qpos_path,
        TIP_LINK_NAMES,
    )
    tip_errors = np.linalg.norm(source_tip_positions - target_tip_positions, axis=2)
    assert float(tip_errors.max()) < 0.03
    assert float(tip_errors.mean()) < 0.015

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["num_frames"] == 3
    assert meta["target_dof"] == 5
