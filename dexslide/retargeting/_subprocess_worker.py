"""Persistent dex-retargeting worker for environments without local deps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or "retargeting" not in config:
        raise ValueError(f"Invalid retarget config: {path}")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="DexSlide retarget worker")
    parser.add_argument("--config", required=True, help="Retarget config json path")
    parser.add_argument("--scaling-factor", type=float, default=None)
    args = parser.parse_args()

    try:
        from dex_retargeting.retargeting_config import RetargetingConfig

        config_path = Path(args.config).expanduser().resolve()
        document = _load_config(config_path)
        override = {}
        if args.scaling_factor is not None:
            override["scaling_factor"] = float(args.scaling_factor)

        RetargetingConfig.set_default_urdf_dir(str(config_path.parent))
        retargeting = RetargetingConfig.from_dict(document["retargeting"].copy(), override=override).build()
        _emit(
            {
                "ok": True,
                "phase": "ready",
                "joint_names": list(retargeting.joint_names),
                "fixed_joint_names": list(retargeting.optimizer.fixed_joint_names),
            }
        )
    except Exception as exc:  # pragma: no cover - worker startup error path
        _emit({"ok": False, "phase": "startup", "error": str(exc)})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            cmd = request.get("cmd")
            if cmd == "close":
                _emit({"ok": True, "phase": "close"})
                return 0
            if cmd == "reset":
                retargeting.reset()
                _emit({"ok": True, "phase": "reset"})
                continue
            if cmd != "retarget":
                raise ValueError(f"Unsupported command: {cmd}")

            ref_value = np.asarray(request["ref_value"], dtype=np.float64)
            fixed_qpos = np.asarray(request.get("fixed_qpos", []), dtype=np.float64)
            qpos = retargeting.retarget(ref_value=ref_value, fixed_qpos=fixed_qpos)
            _emit({"ok": True, "phase": "retarget", "qpos": qpos.tolist()})
        except Exception as exc:  # pragma: no cover - worker runtime error path
            _emit({"ok": False, "phase": "runtime", "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
