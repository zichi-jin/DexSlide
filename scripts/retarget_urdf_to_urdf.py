#!/usr/bin/env python3
from __future__ import annotations

"""
Retarget source-hand joint trajectory to target dexterous-hand trajectory using dex-retargeting.

INPUT
1) --target-config
   dex-retargeting config YAML file (contains "retargeting" block).
   Example fields:
     retargeting.type, retargeting.urdf_path, retargeting.target_link_human_indices, ...
2) --source-urdf
   Source hand URDF path (the hand that provides joint trajectory).
3) --source-qpos
   Source joint trajectory file. Supported formats:
   - .npy: shape [T, D]
   - .npz: expects key "qpos" (preferred) and optional "joint_names"
   - .json: expects {"qpos": [[...], ...], "joint_names": [...]}
   - .pkl: supports dict with key "data" (list of qpos) or raw array-like
4) --source-index-link-map
   JSON/YAML mapping from "human index" (integer used by dex-retargeting config) to source URDF link name.
   Example:
     {
       "0": "wrist_link",
       "4": "thumb_tip_link",
       "8": "index_tip_link",
       "12": "middle_tip_link",
       "16": "ring_tip_link",
       "20": "pinky_tip_link"
     }

OUTPUT
1) --output
   Output .npz file containing:
   - qpos: target retargeted trajectory, shape [T, D_target]
   - joint_names: target joint names in output order
   - source_joint_names: source robot dof joint names used in computation
2) sidecar metadata json:
   "<output>.meta.json"

OTHER REQUIRED FILES / PARAMETERS
- dex-retargeting python package must be installed:
    pip install dex_retargeting
- source URDF and target URDF must use meter units consistently.
- If source trajectory joint order differs from source URDF dof order,
  provide --source-joint-names-file OR include joint_names in --source-qpos.

NOTES
- This script computes source-link 3D positions from source URDF + qpos, then feeds those positions/vectors
  to dex-retargeting optimizer according to target config indices.
- It supports position / vector / dexpilot retargeting as long as index mapping is provided for required indices.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        try:
            import yaml
        except ImportError as ex:
            raise RuntimeError(
                f"PyYAML is required to read {path}. Install with: pip install pyyaml"
            ) from ex
        with path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict content in {path}, got {type(obj)}")
    return obj


def _load_source_qpos(path: Path) -> tuple[np.ndarray, list[str] | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        qpos = np.asarray(arr, dtype=np.float64)
        names = None
    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
        else:
            first_key = list(data.keys())[0]
            qpos = np.asarray(data[first_key], dtype=np.float64)
        names = None
        if "joint_names" in data:
            names = [str(x) for x in data["joint_names"].tolist()]
    elif suffix == ".json":
        obj = _load_yaml_or_json(path)
        if "qpos" not in obj:
            raise ValueError(f"JSON source trajectory must contain key 'qpos': {path}")
        qpos = np.asarray(obj["qpos"], dtype=np.float64)
        names = [str(x) for x in obj.get("joint_names", [])] or None
    elif suffix == ".pkl":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            if "qpos" in obj:
                qpos = np.asarray(obj["qpos"], dtype=np.float64)
            elif "data" in obj:
                qpos = np.asarray(obj["data"], dtype=np.float64)
            else:
                raise ValueError(f"Unsupported pkl dict format in {path}, expected key 'qpos' or 'data'.")
            names = obj.get("joint_names", None)
            if names is not None:
                names = [str(x) for x in names]
        else:
            qpos = np.asarray(obj, dtype=np.float64)
            names = None
    else:
        raise ValueError(f"Unsupported --source-qpos format: {path.suffix}")

    if qpos.ndim != 2:
        raise ValueError(f"Source qpos must be 2D [T, D], got shape {qpos.shape}")
    return qpos, names


def _load_joint_names(path: Path) -> list[str]:
    obj = _load_yaml_or_json(path)
    if "joint_names" in obj:
        raw = obj["joint_names"]
    else:
        raw = obj
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"joint names file must be list or contain key 'joint_names': {path}")
    return [str(x) for x in raw]


def _reorder_source_qpos_to_robot(
    source_qpos: np.ndarray,
    input_joint_names: list[str] | None,
    source_robot_joint_names: list[str],
) -> np.ndarray:
    dof = len(source_robot_joint_names)
    if input_joint_names is None:
        if source_qpos.shape[1] != dof:
            raise ValueError(
                "No source joint names provided. "
                f"Expected source_qpos dim={dof} (source robot dof), got {source_qpos.shape[1]}."
            )
        return source_qpos

    name_to_idx = {name: i for i, name in enumerate(input_joint_names)}
    missing = [name for name in source_robot_joint_names if name not in name_to_idx]
    if missing:
        raise ValueError(
            f"Source trajectory joint names missing {len(missing)} source robot joints. "
            f"Examples: {missing[:8]}"
        )
    reorder_idx = np.array([name_to_idx[name] for name in source_robot_joint_names], dtype=int)
    return source_qpos[:, reorder_idx]


def _parse_index_to_link_map(path: Path) -> dict[int, str]:
    obj = _load_yaml_or_json(path)
    if "index_to_link" in obj and isinstance(obj["index_to_link"], dict):
        raw = obj["index_to_link"]
    else:
        raw = obj
    if not isinstance(raw, dict):
        raise ValueError(f"Index-link map must be dict or contain dict key 'index_to_link': {path}")

    out: dict[int, str] = {}
    for k, v in raw.items():
        idx = int(k)
        out[idx] = str(v)
    return out


def _parse_target_fixed_qpos(
    path: Path | None,
    fixed_joint_names: list[str],
) -> np.ndarray:
    if path is None:
        return np.zeros(len(fixed_joint_names), dtype=np.float64)

    obj = _load_yaml_or_json(path)
    if isinstance(obj, dict) and ("fixed_qpos" in obj):
        raw = obj["fixed_qpos"]
    else:
        raw = obj

    if isinstance(raw, (list, tuple)):
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.shape[0] != len(fixed_joint_names):
            raise ValueError(
                f"fixed_qpos list length mismatch, expected {len(fixed_joint_names)} got {arr.shape[0]}"
            )
        return arr

    if isinstance(raw, dict):
        out = np.zeros(len(fixed_joint_names), dtype=np.float64)
        for i, name in enumerate(fixed_joint_names):
            if name in raw:
                out[i] = float(raw[name])
        return out

    raise ValueError(
        "Unsupported fixed qpos format. Provide list[float] or dict[joint_name->value]."
    )


def _build_ref_value(
    human_points: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if indices.ndim == 1:
        return human_points[indices, :]
    if indices.ndim == 2 and indices.shape[0] == 2:
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        return human_points[task_indices, :] - human_points[origin_indices, :]
    raise ValueError(f"Unsupported target_link_human_indices shape: {indices.shape}")


def _try_import_dex_retargeting():
    try:
        from dex_retargeting.retargeting_config import RetargetingConfig
        from dex_retargeting.robot_wrapper import RobotWrapper
    except ImportError as ex:
        raise RuntimeError(
            "dex_retargeting is not installed. Install with: pip install dex_retargeting"
        ) from ex
    return RetargetingConfig, RobotWrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget source-hand URDF + qpos trajectory to target hand qpos via dex-retargeting."
    )
    parser.add_argument("--target-config", required=True, help="dex-retargeting target config yaml path.")
    parser.add_argument("--target-urdf-dir", default=None, help="Optional base dir for relative URDF in target config.")
    parser.add_argument("--source-urdf", required=True, help="Source hand URDF path.")
    parser.add_argument("--source-qpos", required=True, help="Source trajectory file (.npy/.npz/.json/.pkl).")
    parser.add_argument(
        "--source-joint-names-file",
        default=None,
        help="Optional joint names file (json/yaml list or {joint_names:[...]}).",
    )
    parser.add_argument(
        "--source-index-link-map",
        required=True,
        help="JSON/YAML mapping from dex-retargeting human index to source URDF link name.",
    )
    parser.add_argument(
        "--target-fixed-qpos-file",
        default=None,
        help="Optional json/yaml for fixed (non-optimized) target joints.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames. 0 means all.")
    parser.add_argument("--print-every", type=int, default=100, help="Progress print interval in frames.")
    parser.add_argument("--output", required=True, help="Output .npz path for retargeted qpos trajectory.")
    args = parser.parse_args()

    target_config_path = Path(args.target_config).expanduser().resolve()
    source_urdf_path = Path(args.source_urdf).expanduser().resolve()
    source_qpos_path = Path(args.source_qpos).expanduser().resolve()
    source_index_link_map_path = Path(args.source_index_link_map).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_config_path.is_file():
        raise SystemExit(f"--target-config not found: {target_config_path}")
    if not source_urdf_path.is_file():
        raise SystemExit(f"--source-urdf not found: {source_urdf_path}")
    if not source_qpos_path.is_file():
        raise SystemExit(f"--source-qpos not found: {source_qpos_path}")
    if not source_index_link_map_path.is_file():
        raise SystemExit(f"--source-index-link-map not found: {source_index_link_map_path}")

    RetargetingConfig, RobotWrapper = _try_import_dex_retargeting()

    target_urdf_dir = (
        Path(args.target_urdf_dir).expanduser().resolve()
        if args.target_urdf_dir
        else target_config_path.parent
    )
    RetargetingConfig.set_default_urdf_dir(str(target_urdf_dir))

    retargeting = RetargetingConfig.load_from_file(target_config_path).build()
    source_robot = RobotWrapper(str(source_urdf_path))

    source_qpos_raw, source_names_from_traj = _load_source_qpos(source_qpos_path)
    source_names_from_file = None
    if args.source_joint_names_file is not None:
        source_names_from_file = _load_joint_names(Path(args.source_joint_names_file).expanduser().resolve())
    source_input_names = source_names_from_file if source_names_from_file is not None else source_names_from_traj
    source_qpos = _reorder_source_qpos_to_robot(
        source_qpos_raw,
        source_input_names,
        source_robot.dof_joint_names,
    )

    if args.max_frames > 0:
        source_qpos = source_qpos[: args.max_frames]
    num_frames = source_qpos.shape[0]
    print(f"[retarget] source frames: {num_frames}, source dof: {source_qpos.shape[1]}")

    indices = np.asarray(retargeting.optimizer.target_link_human_indices, dtype=int)
    needed_indices = sorted({int(x) for x in indices.reshape(-1).tolist()})
    max_index = max(needed_indices)

    index_to_link = _parse_index_to_link_map(source_index_link_map_path)
    missing = [idx for idx in needed_indices if idx not in index_to_link]
    if missing:
        raise SystemExit(
            f"source index-link map missing required human indices used by target config: {missing}"
        )

    index_to_link_id: dict[int, int] = {}
    for idx in needed_indices:
        link_name = index_to_link[idx]
        try:
            link_id = source_robot.get_link_index(link_name)
        except ValueError as ex:
            raise SystemExit(
                f"Invalid mapped link for human index {idx}: '{link_name}' is not in source URDF links."
            ) from ex
        index_to_link_id[idx] = link_id

    fixed_qpos = _parse_target_fixed_qpos(
        Path(args.target_fixed_qpos_file).expanduser().resolve() if args.target_fixed_qpos_file else None,
        retargeting.optimizer.fixed_joint_names,
    )

    target_qpos_seq = []
    for i in range(num_frames):
        q_src = source_qpos[i]
        source_robot.compute_forward_kinematics(q_src)
        human_points = np.full((max_index + 1, 3), np.nan, dtype=np.float64)
        for idx, link_id in index_to_link_id.items():
            pose = source_robot.get_link_pose(link_id)
            human_points[idx] = pose[:3, 3]

        ref_value = _build_ref_value(human_points=human_points, indices=indices)
        q_tgt = retargeting.retarget(ref_value=ref_value, fixed_qpos=fixed_qpos)
        target_qpos_seq.append(q_tgt.astype(np.float64))

        if args.print_every > 0 and (i % args.print_every == 0):
            print(f"[retarget] frame {i}/{num_frames}")

    target_qpos = np.stack(target_qpos_seq, axis=0)

    np.savez_compressed(
        output_path,
        qpos=target_qpos,
        joint_names=np.asarray(retargeting.joint_names, dtype=object),
        source_joint_names=np.asarray(source_robot.dof_joint_names, dtype=object),
    )
    meta = {
        "target_config": str(target_config_path),
        "target_urdf_dir": str(target_urdf_dir),
        "source_urdf": str(source_urdf_path),
        "source_qpos": str(source_qpos_path),
        "source_index_link_map": str(source_index_link_map_path),
        "num_frames": int(num_frames),
        "target_dof": int(target_qpos.shape[1]),
        "target_joint_names": [str(x) for x in retargeting.joint_names],
        "source_joint_names": [str(x) for x in source_robot.dof_joint_names],
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[retarget] done. output: {output_path}")
    print(f"[retarget] meta:   {meta_path}")


if __name__ == "__main__":
    main()

