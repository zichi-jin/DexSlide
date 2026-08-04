"""逐指两端点 DexSlide→OrcaHand 映射标定。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .direct_joint_mapping import SWING_JOINTS, TARGET_JOINTS, load_static_joint_map


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _lookup(mapping: Mapping[object, object], key: int, default: object = None) -> object:
    return mapping.get(key, mapping.get(str(key), default))


def build_fingerwise_calibration(
    *,
    source_joint_order: Sequence[str],
    finger_endpoints_deg: Mapping[str, Mapping[str, Mapping[str, float]]],
    static_map_file: str | Path,
    joint_roms_deg: Mapping[str, Sequence[float]],
    joint_to_motor_map: Mapping[str, int],
    joint_inversion: Mapping[str, bool],
    motor_calibration: Mapping[str, object],
) -> dict[str, object]:
    """Build a complete 16-DOF teleop mapping from five glove endpoint pairs."""
    static_map = load_static_joint_map(static_map_file)
    source_order = tuple(str(value) for value in source_joint_order)
    raw_joint_map = static_map["joint_map"]
    if not isinstance(raw_joint_map, Mapping):
        raise ValueError("Static joint map is missing joint_map")

    motor_limits = motor_calibration.get("motor_limits", {})
    ratios = motor_calibration.get("joint_to_motor_ratios", {})
    if not isinstance(motor_limits, Mapping) or not isinstance(ratios, Mapping):
        raise ValueError("Manual Orca calibration is missing motor limits or ratios")

    rules: dict[str, dict[str, object]] = {}
    for target in TARGET_JOINTS:
        item = raw_joint_map.get(target)
        if not isinstance(item, Mapping):
            raise ValueError(f"Static joint map is missing {target}")
        source = str(item["source"])
        finger = source.split(".", 1)[0]
        capture = finger_endpoints_deg.get(finger)
        if capture is None:
            raise ValueError(f"Missing endpoint capture for {finger}")
        left = capture.get("left")
        right = capture.get("right")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"{finger} must contain left and right endpoint captures")
        if source not in left or source not in right:
            raise ValueError(f"Missing DexSlide endpoint for {target} / {source}")

        source_left = float(left[source])
        source_right = float(right[source])
        if np.isclose(source_left, source_right, atol=1e-9):
            raise ValueError(f"DexSlide endpoint range is zero for {target} / {source}")
        target_left, target_right = map(float, joint_roms_deg[target])
        motor_id = abs(int(joint_to_motor_map[target]))
        limits = _lookup(motor_limits, motor_id)
        ratio = _lookup(ratios, motor_id, 0.0)
        if not isinstance(limits, Sequence) or len(limits) != 2 or any(value is None for value in limits):
            raise ValueError(f"Manual Orca calibration is missing motor limits for {target}")
        ratio = float(ratio)
        if ratio <= 0.0:
            raise ValueError(f"Manual Orca calibration has an invalid ratio for {target}")

        scale = (target_right - target_left) / (source_right - source_left)
        bias = target_left - scale * source_left
        motor_min, motor_max = map(float, limits)
        inverted = bool(joint_inversion.get(target, False))
        motor_scale = -ratio if inverted else ratio
        motor_bias = (
            motor_min + target_right * ratio if inverted else motor_min - target_left * ratio
        )
        rules[target] = {
            "source": source,
            "kind": "swing" if target in SWING_JOINTS else "bend",
            "scale": scale,
            "bias_deg": bias,
            "target_min_deg": min(target_left, target_right),
            "target_max_deg": max(target_left, target_right),
            "motor_id": motor_id,
            "motor_min_rad": min(motor_min, motor_max),
            "motor_max_rad": max(motor_min, motor_max),
            "motor_scale_rad_per_deg": motor_scale,
            "motor_bias_rad": motor_bias,
        }

    return {
        "schema_version": 2,
        "status": "complete",
        "source_unit": "deg",
        "target_unit": "deg",
        "source_joint_order": list(source_order),
        "target_joint_order": list(TARGET_JOINTS),
        "static_joint_map_name": static_map.get("name", "joint_map_dexslide_to_orcahand"),
        "joint_map": raw_joint_map,
        "finger_endpoints_deg": _json_value(finger_endpoints_deg),
        "joints": rules,
        "created_at_unix": time.time(),
    }


def save_fingerwise_calibration(path: str | Path, payload: Mapping[str, object]) -> None:
    """Persist the complete mapping as the sole DexSlide→Orca calibration file."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = ["FINGERS", "build_fingerwise_calibration", "save_fingerwise_calibration"]
