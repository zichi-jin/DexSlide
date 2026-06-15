"""Shared helpers for live DexSlide-to-OrcaHand retarget pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


# Temporary glove joint overrides for damaged sensors.
# After the hardware is repaired, set this dict back to {} to restore normal input.
FAULTY_GLOVE_JOINT_OVERRIDES_RAD: dict[str, float] = {
    "middle.DIP": 0.0,
    "index.MCP_back": 0.0,
}


def joint_vector_to_map(
    joint_angles: np.ndarray | Sequence[float],
    joint_names: Sequence[str],
) -> dict[str, float]:
    values = np.asarray(joint_angles, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joint values, got {values.shape[0]}")
    return {name: float(value) for name, value in zip(joint_names, values)}


def apply_glove_joint_overrides(
    human_joint_angles: np.ndarray | Sequence[float] | Mapping[str, float],
    joint_names: Sequence[str],
    overrides_rad: Mapping[str, float] | None = None,
) -> dict[str, float]:
    if isinstance(human_joint_angles, Mapping):
        joint_map = {str(name): float(value) for name, value in human_joint_angles.items()}
    else:
        joint_map = joint_vector_to_map(human_joint_angles, joint_names)

    for joint_name, override_value in (overrides_rad or FAULTY_GLOVE_JOINT_OVERRIDES_RAD).items():
        if joint_name in joint_map:
            joint_map[joint_name] = float(override_value)
    return joint_map


def format_named_values(
    joint_names: Sequence[str],
    joint_values: np.ndarray | Sequence[float],
    *,
    unit: str,
) -> str:
    values = np.asarray(joint_values, dtype=np.float64).reshape(-1)
    if values.shape[0] != len(joint_names):
        raise ValueError(f"Expected {len(joint_names)} joint values, got {values.shape[0]}")
    return ", ".join(
        f"{name}={float(value):7.2f} {unit}"
        for name, value in zip(joint_names, values)
    )

