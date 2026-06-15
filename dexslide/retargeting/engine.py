"""Cached DexSlide-to-OrcaHand retargeting engine."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from dexslide.paths import DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE, DEFAULT_SKELETON_FILE
from dexslide.retargeting.human_model import (
    DEFAULT_HUMAN_JOINT_NAMES,
    DexSlideHumanModel,
)

_DEFAULT_RETARGETER_LOCK = threading.Lock()
_DEFAULT_RETARGETER: DexOrcaRetargeter | None = None
_DEFAULT_RETARGETER_SPEC: tuple[Any, ...] | None = None
_EXTRA_PYTHONPATH_BOOTSTRAPPED = False


def _load_retarget_document(config_file: str | Path) -> dict[str, Any]:
    path = Path(config_file).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or "retargeting" not in document:
        raise ValueError(f"Invalid retarget config: {path}")
    return document


def _can_import_dex_retargeting() -> bool:
    _ensure_extra_pythonpath_loaded()
    try:
        from dex_retargeting.retargeting_config import RetargetingConfig  # noqa: F401
    except Exception:
        return False
    return True


def _ensure_extra_pythonpath_loaded() -> None:
    global _EXTRA_PYTHONPATH_BOOTSTRAPPED
    if _EXTRA_PYTHONPATH_BOOTSTRAPPED:
        return

    raw_paths = os.environ.get("DEX_RETARGETING_EXTRA_PYTHONPATH", "")
    if not raw_paths:
        _EXTRA_PYTHONPATH_BOOTSTRAPPED = True
        return

    candidates: list[str] = []
    for raw in raw_paths.split(os.pathsep):
        if not raw:
            continue
        root = Path(raw).expanduser()
        if root.is_dir():
            candidates.append(str(root))
            candidates.extend(str(path) for path in root.glob("cmeel.prefix/lib/python*/site-packages") if path.is_dir())
            candidates.extend(str(path) for path in root.glob("cmeel.prefix/lib64/python*/site-packages") if path.is_dir())
            candidates.extend(str(path) for path in root.glob("lib/python*/site-packages") if path.is_dir())
            candidates.extend(str(path) for path in root.glob("lib64/python*/site-packages") if path.is_dir())

    for candidate in reversed(candidates):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    _EXTRA_PYTHONPATH_BOOTSTRAPPED = True


def _build_ref_value(landmarks: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    retarget_cfg = config["retargeting"]
    indices = np.asarray(retarget_cfg["target_link_human_indices"], dtype=int)
    retarget_type = str(retarget_cfg["type"]).lower()
    if retarget_type == "vector":
        if indices.shape[0] != 2:
            raise ValueError(f"Vector retarget indices must have shape (2, N), got {indices.shape}")
        return landmarks[indices[1], :] - landmarks[indices[0], :]
    if retarget_type == "position":
        return landmarks[indices, :]
    raise ValueError(f"Unsupported retarget type for DexSlide wrapper: {retarget_type}")


class _DirectBackend:
    def __init__(self, config_path: Path, document: dict[str, Any], scaling_factor: float | None) -> None:
        _ensure_extra_pythonpath_loaded()
        from dex_retargeting.retargeting_config import RetargetingConfig

        override = {}
        if scaling_factor is not None:
            override["scaling_factor"] = float(scaling_factor)

        RetargetingConfig.set_default_urdf_dir(str(config_path.parent))
        self._retargeting = RetargetingConfig.from_dict(
            document["retargeting"].copy(),
            override=override,
        ).build()
        self.joint_names = list(self._retargeting.joint_names)
        self.fixed_joint_names = list(self._retargeting.optimizer.fixed_joint_names)

    def retarget(self, ref_value: np.ndarray, fixed_qpos: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._retargeting.retarget(ref_value=ref_value, fixed_qpos=fixed_qpos),
            dtype=np.float64,
        )

    def reset(self) -> None:
        self._retargeting.reset()

    def close(self) -> None:
        return None


class _WorkerBackend:
    def __init__(self, config_path: Path, scaling_factor: float | None, python_bin: str) -> None:
        worker_path = Path(__file__).resolve().with_name("_subprocess_worker.py")
        cmd = [python_bin, str(worker_path), "--config", str(config_path)]
        if scaling_factor is not None:
            cmd.extend(["--scaling-factor", str(float(scaling_factor))])

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        ready = self._read_response()
        if not ready.get("ok"):
            raise RuntimeError(str(ready.get("error", "retarget worker startup failed")))
        self.joint_names = [str(name) for name in ready["joint_names"]]
        self.fixed_joint_names = [str(name) for name in ready["fixed_joint_names"]]

    def _read_response(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("retarget worker stdout is unavailable")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("retarget worker terminated before responding")
        return json.loads(line)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process.stdin is None:
            raise RuntimeError("retarget worker stdin is unavailable")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = self._read_response()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "retarget worker request failed")))
        return response

    def retarget(self, ref_value: np.ndarray, fixed_qpos: np.ndarray) -> np.ndarray:
        response = self._request(
            {
                "cmd": "retarget",
                "ref_value": np.asarray(ref_value, dtype=np.float64).tolist(),
                "fixed_qpos": np.asarray(fixed_qpos, dtype=np.float64).tolist(),
            }
        )
        return np.asarray(response["qpos"], dtype=np.float64)

    def reset(self) -> None:
        self._request({"cmd": "reset"})

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._request({"cmd": "close"})
            except Exception:
                self._process.terminate()
        self._process.wait(timeout=5)


class DexOrcaRetargeter:
    """Retarget DexSlide human joint angles to OrcaHand target joints."""

    def __init__(
        self,
        *,
        config_file: str | Path = DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE,
        skeleton_file: str | Path = DEFAULT_SKELETON_FILE,
        hand: str = "right",
        mirror_reconstruction: bool = False,
        human_joint_names: list[str] | tuple[str, ...] | None = None,
        dex_retargeting_python: str | None = None,
        use_subprocess: bool | None = None,
        scaling_factor: float | None = None,
    ) -> None:
        self.config_path = Path(config_file).expanduser().resolve()
        self.document = _load_retarget_document(self.config_path)
        self.human_model = DexSlideHumanModel(
            skeleton_file=skeleton_file,
            hand=hand,
            mirror_reconstruction=mirror_reconstruction,
            joint_names=list(human_joint_names or self.document.get("human_joint_names", DEFAULT_HUMAN_JOINT_NAMES)),
        )
        self.joint_ids = [str(name) for name in self.document["orcahand_joint_ids"]]
        self._output_urdf_joint_names = [str(name) for name in self.document["orcahand_urdf_joint_names"]]
        fixed_qpos_map = {
            str(key): float(value) for key, value in self.document.get("fixed_qpos", {}).items()
        }

        resolved_scale = scaling_factor
        if resolved_scale is None and "scaling_factor" in self.document["retargeting"]:
            resolved_scale = float(self.document["retargeting"]["scaling_factor"])

        if use_subprocess is None:
            use_subprocess = not _can_import_dex_retargeting()

        if use_subprocess:
            python_bin = dex_retargeting_python or os.environ.get("DEX_RETARGETING_PYTHON")
            if not python_bin:
                raise RuntimeError(
                    "Bundled dex_retargeting is not available in the current interpreter. "
                    "Install the retarget runtime dependencies with `pip install -r requirements-retargeting.txt`. "
                    "If you still see `_ARRAY_API not found` or `numpy.core.multiarray failed to import`, "
                    "your current environment likely has NumPy 2.x with a NumPy 1.x-built nlopt wheel, so "
                    "reinstall the retarget stack with `numpy<2`. "
                    "or set DEX_RETARGETING_PYTHON to another compatible interpreter as a temporary fallback."
                )
            self._backend = _WorkerBackend(self.config_path, resolved_scale, python_bin)
        else:
            self._backend = _DirectBackend(self.config_path, self.document, resolved_scale)

        self._joint_index = {name: idx for idx, name in enumerate(self._backend.joint_names)}
        self._orca_indices = np.asarray(
            [self._joint_index[name] for name in self._output_urdf_joint_names],
            dtype=int,
        )
        self._fixed_qpos = np.asarray(
            [fixed_qpos_map.get(name, 0.0) for name in self._backend.fixed_joint_names],
            dtype=np.float64,
        )

    @property
    def backend_joint_names(self) -> list[str]:
        return list(self._backend.joint_names)

    @property
    def orca_indices(self) -> np.ndarray:
        return self._orca_indices.copy()

    def human_landmarks(self, human_joint_angles: np.ndarray | list[float] | dict[str, float]) -> np.ndarray:
        return self.human_model.landmarks_from_angles(human_joint_angles)

    def retarget_full_qpos(self, human_joint_angles: np.ndarray | list[float] | dict[str, float]) -> np.ndarray:
        landmarks = self.human_landmarks(human_joint_angles)
        ref_value = _build_ref_value(landmarks, self.document)
        return self._backend.retarget(ref_value=ref_value, fixed_qpos=self._fixed_qpos)

    def retarget(self, human_joint_angles: np.ndarray | list[float] | dict[str, float]) -> np.ndarray:
        robot_qpos = self.retarget_full_qpos(human_joint_angles)
        return robot_qpos[self._orca_indices].astype(np.float64, copy=False)

    def reset(self) -> None:
        self._backend.reset()

    def close(self) -> None:
        self._backend.close()

    __call__ = retarget


def create_dex_retargeter(
    *,
    config_file: str | Path = DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE,
    skeleton_file: str | Path = DEFAULT_SKELETON_FILE,
    hand: str = "right",
    mirror_reconstruction: bool = False,
    human_joint_names: list[str] | tuple[str, ...] | None = None,
    dex_retargeting_python: str | None = None,
    use_subprocess: bool | None = None,
    scaling_factor: float | None = None,
) -> DexOrcaRetargeter:
    return DexOrcaRetargeter(
        config_file=config_file,
        skeleton_file=skeleton_file,
        hand=hand,
        mirror_reconstruction=mirror_reconstruction,
        human_joint_names=human_joint_names,
        dex_retargeting_python=dex_retargeting_python,
        use_subprocess=use_subprocess,
        scaling_factor=scaling_factor,
    )


def _default_retargeter_spec(
    config_file: str | Path,
    skeleton_file: str | Path,
    hand: str,
    mirror_reconstruction: bool,
    human_joint_names: list[str] | tuple[str, ...] | None,
    dex_retargeting_python: str | None,
    use_subprocess: bool | None,
    scaling_factor: float | None,
) -> tuple[Any, ...]:
    return (
        str(Path(config_file).expanduser().resolve()),
        str(Path(skeleton_file).expanduser().resolve()),
        hand,
        mirror_reconstruction,
        tuple(human_joint_names or ()),
        dex_retargeting_python or os.environ.get("DEX_RETARGETING_PYTHON", ""),
        use_subprocess,
        scaling_factor,
    )


def dex_retarget(
    human_joint_angles: np.ndarray | list[float] | dict[str, float],
    *,
    config_file: str | Path = DEFAULT_ORCAHAND_RIGHT_RETARGET_FILE,
    skeleton_file: str | Path = DEFAULT_SKELETON_FILE,
    hand: str = "right",
    mirror_reconstruction: bool = False,
    human_joint_names: list[str] | tuple[str, ...] | None = None,
    dex_retargeting_python: str | None = None,
    use_subprocess: bool | None = None,
    scaling_factor: float | None = None,
) -> np.ndarray:
    global _DEFAULT_RETARGETER
    global _DEFAULT_RETARGETER_SPEC

    spec = _default_retargeter_spec(
        config_file=config_file,
        skeleton_file=skeleton_file,
        hand=hand,
        mirror_reconstruction=mirror_reconstruction,
        human_joint_names=human_joint_names,
        dex_retargeting_python=dex_retargeting_python,
        use_subprocess=use_subprocess,
        scaling_factor=scaling_factor,
    )
    with _DEFAULT_RETARGETER_LOCK:
        if _DEFAULT_RETARGETER is None or _DEFAULT_RETARGETER_SPEC != spec:
            if _DEFAULT_RETARGETER is not None:
                _DEFAULT_RETARGETER.close()
            _DEFAULT_RETARGETER = create_dex_retargeter(
                config_file=config_file,
                skeleton_file=skeleton_file,
                hand=hand,
                mirror_reconstruction=mirror_reconstruction,
                human_joint_names=human_joint_names,
                dex_retargeting_python=dex_retargeting_python,
                use_subprocess=use_subprocess,
                scaling_factor=scaling_factor,
            )
            _DEFAULT_RETARGETER_SPEC = spec
        return _DEFAULT_RETARGETER.retarget(human_joint_angles)


def shutdown_default_retargeter() -> None:
    global _DEFAULT_RETARGETER
    global _DEFAULT_RETARGETER_SPEC
    with _DEFAULT_RETARGETER_LOCK:
        if _DEFAULT_RETARGETER is not None:
            _DEFAULT_RETARGETER.close()
            _DEFAULT_RETARGETER = None
            _DEFAULT_RETARGETER_SPEC = None


atexit.register(shutdown_default_retargeter)
