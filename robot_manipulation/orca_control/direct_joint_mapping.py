"""Pure degree-space mapping from DexSlide joints to OrcaHand targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SWING_JOINTS = (
    "thumb_mcp",
    "index_abd",
    "middle_abd",
    "ring_abd",
    "pinky_abd",
)
BEND_JOINTS = (
    "thumb_abd",
    "thumb_pip",
    "thumb_dip",
    "index_mcp",
    "index_pip",
    "middle_mcp",
    "middle_pip",
    "ring_mcp",
    "ring_pip",
    "pinky_mcp",
    "pinky_pip",
)
TARGET_JOINTS = (
    "thumb_mcp",
    "thumb_abd",
    "thumb_pip",
    "thumb_dip",
    "index_abd",
    "index_mcp",
    "index_pip",
    "middle_abd",
    "middle_mcp",
    "middle_pip",
    "ring_abd",
    "ring_mcp",
    "ring_pip",
    "pinky_abd",
    "pinky_mcp",
    "pinky_pip",
)


@dataclass(frozen=True)
class DirectJointRule:
    target: str
    source: str
    kind: str
    scale: float
    bias_deg: float
    target_min_deg: float
    target_max_deg: float
    motor_id: int
    motor_min_rad: float
    motor_max_rad: float
    motor_scale_rad_per_deg: float
    motor_bias_rad: float


@dataclass(frozen=True)
class DirectJointCalibration:
    source_joint_order: tuple[str, ...]
    target_joint_order: tuple[str, ...]
    rules: dict[str, DirectJointRule]
    payload: dict[str, object]

    @classmethod
    def load(cls, path: str | Path) -> "DirectJointCalibration":
        calibration_path = Path(path).expanduser().resolve()
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        return cls.from_payload(payload, source=str(calibration_path))

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        source: str = "payload",
    ) -> "DirectJointCalibration":
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError(f"Unsupported direct joint calibration schema: {source}")
        if payload.get("status") != "complete":
            raise ValueError(f"Direct joint calibration is not complete: {source}")
        if payload.get("source_unit") != "deg" or payload.get("target_unit") != "deg":
            raise ValueError("Direct joint calibration must use deg for source and target")
        source_order = tuple(str(value) for value in payload["source_joint_order"])
        target_order = tuple(str(value) for value in payload["target_joint_order"])
        if target_order != TARGET_JOINTS:
            raise ValueError(f"Unexpected Orca target joint order: {target_order}")
        raw_rules = payload.get("joints")
        if not isinstance(raw_rules, dict):
            raise ValueError("Direct joint calibration must contain a joints object")
        rules: dict[str, DirectJointRule] = {}
        for target in target_order:
            raw = raw_rules.get(target)
            if not isinstance(raw, dict):
                raise ValueError(f"Missing direct joint rule for {target}")
            rule = DirectJointRule(
                target=target,
                source=str(raw["source"]),
                kind=str(raw["kind"]),
                scale=float(raw["scale"]),
                bias_deg=float(raw["bias_deg"]),
                target_min_deg=float(raw["target_min_deg"]),
                target_max_deg=float(raw["target_max_deg"]),
                motor_id=int(raw["motor_id"]),
                motor_min_rad=float(raw["motor_min_rad"]),
                motor_max_rad=float(raw["motor_max_rad"]),
                motor_scale_rad_per_deg=float(raw["motor_scale_rad_per_deg"]),
                motor_bias_rad=float(raw["motor_bias_rad"]),
            )
            if rule.kind not in {"bend", "swing"}:
                raise ValueError(f"Unsupported joint kind for {target}: {rule.kind}")
            if rule.source not in source_order:
                raise ValueError(f"Unknown source joint for {target}: {rule.source}")
            if rule.target_max_deg <= rule.target_min_deg:
                raise ValueError(f"Invalid target limits for {target}")
            if rule.motor_max_rad <= rule.motor_min_rad:
                raise ValueError(f"Invalid motor limits for {target}")
            rules[target] = rule
        return cls(source_order, target_order, rules, payload)


@dataclass(frozen=True)
class DirectJointMappingResult:
    target_positions_deg: dict[str, float]
    predicted_motor_positions_rad: dict[int, float]
    clip_reasons: dict[str, tuple[str, ...]]

    @property
    def clipped(self) -> bool:
        return bool(self.clip_reasons)


class DirectJointMapper:
    """Map a 20-joint DexSlide sample without touching hardware."""

    def __init__(self, calibration: DirectJointCalibration) -> None:
        self.calibration = calibration

    @classmethod
    def from_file(cls, path: str | Path) -> "DirectJointMapper":
        return cls(DirectJointCalibration.load(path))

    def map(
        self,
        source_positions_deg: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        source_joint_order: Sequence[str] | None = None,
    ) -> DirectJointMappingResult:
        source = self._as_source_dict(source_positions_deg, source_joint_order)
        raw_targets = {
            target: self.calibration.rules[target].scale
            * float(source[self.calibration.rules[target].source])
            + self.calibration.rules[target].bias_deg
            for target in self.calibration.target_joint_order
        }
        return self.project_targets(raw_targets)

    def project_targets(
        self,
        raw_targets_deg: Mapping[str, float],
    ) -> DirectJointMappingResult:
        # Apply the configured joint and motor safety limits to target angles.
        missing = sorted(set(self.calibration.target_joint_order) - set(raw_targets_deg))
        if missing:
            raise ValueError(f"Missing Orca target joints: {missing}")
        targets: dict[str, float] = {}
        motor_positions: dict[int, float] = {}
        clip_reasons: dict[str, tuple[str, ...]] = {}
        for target in self.calibration.target_joint_order:
            rule = self.calibration.rules[target]
            raw_target = float(raw_targets_deg[target])
            if not np.isfinite(raw_target):
                raise ValueError(f"Orca target joint {target} is non-finite")
            reasons: list[str] = []
            clipped_target = float(np.clip(raw_target, rule.target_min_deg, rule.target_max_deg))
            if not np.isclose(clipped_target, raw_target, atol=1e-9):
                reasons.append("joint_limit")
            predicted_motor = (
                rule.motor_scale_rad_per_deg * clipped_target + rule.motor_bias_rad
            )
            clipped_motor = float(
                np.clip(predicted_motor, rule.motor_min_rad, rule.motor_max_rad)
            )
            if not np.isclose(clipped_motor, predicted_motor, atol=1e-9):
                reasons.append("motor_limit")
                if abs(rule.motor_scale_rad_per_deg) < 1e-12:
                    raise ValueError(f"Cannot enforce motor limit for {target}: zero motor scale")
                clipped_target = float(
                    np.clip(
                        (clipped_motor - rule.motor_bias_rad) / rule.motor_scale_rad_per_deg,
                        rule.target_min_deg,
                        rule.target_max_deg,
                    )
                )
                predicted_motor = (
                    rule.motor_scale_rad_per_deg * clipped_target + rule.motor_bias_rad
                )
            targets[target] = clipped_target
            motor_positions[rule.motor_id] = float(predicted_motor)
            if reasons:
                clip_reasons[target] = tuple(reasons)
        return DirectJointMappingResult(targets, motor_positions, clip_reasons)

    def source_vector(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray,
        source_joint_order: Sequence[str] | None = None,
    ) -> np.ndarray:
        # Return an input sample in the canonical source order.
        source = self._as_source_dict(values, source_joint_order)
        return np.asarray(
            [source[joint] for joint in self.calibration.source_joint_order],
            dtype=np.float64,
        )

    def _as_source_dict(
        self,
        values: Mapping[str, float] | Sequence[float] | np.ndarray,
        source_joint_order: Sequence[str] | None,
    ) -> dict[str, float]:
        if isinstance(values, Mapping):
            source = {str(key): float(value) for key, value in values.items()}
        else:
            order = tuple(source_joint_order or self.calibration.source_joint_order)
            vector = np.asarray(values, dtype=np.float64).reshape(-1)
            if not np.isfinite(vector).all():
                raise ValueError("DexSlide source joint vector contains non-finite values")
            if len(order) != vector.size:
                raise ValueError(
                    f"source vector length {vector.size} does not match joint order {len(order)}"
                )
            source = {joint: float(vector[idx]) for idx, joint in enumerate(order)}
        if not all(np.isfinite(value) for value in source.values()):
            raise ValueError("DexSlide source joint mapping contains non-finite values")
        missing = sorted(
            {rule.source for rule in self.calibration.rules.values()} - set(source)
        )
        if missing:
            raise ValueError(f"Missing DexSlide source joints: {missing}")
        return source


def load_static_joint_map(path: str | Path) -> dict[str, object]:
    mapping_path = Path(path).expanduser().resolve()
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if payload.get("unit") != "deg":
        raise ValueError(f"Direct joint map must declare unit=deg: {mapping_path}")
    targets = tuple(str(value) for value in payload.get("orcahand_joint_ids", []))
    if targets != TARGET_JOINTS:
        raise ValueError(f"Unexpected Orca target joint order in {mapping_path}")
    joint_map = payload.get("joint_map")
    if not isinstance(joint_map, dict) or set(joint_map) != set(TARGET_JOINTS):
        raise ValueError(f"Direct joint map must define exactly the 16 non-wrist joints: {mapping_path}")
    return payload


__all__ = [
    "BEND_JOINTS",
    "DirectJointCalibration",
    "DirectJointMapper",
    "DirectJointMappingResult",
    "DirectJointRule",
    "SWING_JOINTS",
    "TARGET_JOINTS",
    "load_static_joint_map",
]
