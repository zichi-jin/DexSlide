from __future__ import annotations

import copy
from typing import Any

import numpy as np


FINGER_BONE_LAYOUT: tuple[tuple[str, str], ...] = (
    ("thumb", "metacarpal"),
    ("thumb", "proximal"),
    ("thumb", "distal"),
    ("index", "proximal"),
    ("index", "middle"),
    ("index", "distal"),
    ("middle", "proximal"),
    ("middle", "middle"),
    ("middle", "distal"),
    ("ring", "proximal"),
    ("ring", "middle"),
    ("ring", "distal"),
    ("pinky", "proximal"),
    ("pinky", "middle"),
    ("pinky", "distal"),
)
PALM_VERTEX_NAMES: tuple[str, ...] = (
    "thumb_base",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
)
PALM_VERTEX_INDICES: tuple[int, ...] = (1, 2, 3, 4, 5)
PALM_PARAM_DELTA_MM = np.array([45.0, 45.0, 30.0], dtype=np.float64)
BONE_PARAM_COUNT = len(FINGER_BONE_LAYOUT)
PALM_PARAM_START = BONE_PARAM_COUNT
PALM_PARAM_COUNT = len(PALM_VERTEX_NAMES) * 3
SKELETON_PARAM_SIZE = BONE_PARAM_COUNT + PALM_PARAM_COUNT


def _require_runtime_skeleton(skeleton: dict[str, Any]) -> None:
    for finger, bone_name in FINGER_BONE_LAYOUT:
        value = skeleton.get(finger, {}).get(bone_name)
        if value is None:
            raise ValueError(f"Missing skeleton field: {finger}.{bone_name}")
        if not np.isfinite(float(value)):
            raise ValueError(f"Non-finite skeleton field: {finger}.{bone_name}")
    vertices = np.asarray(skeleton.get("palm", {}).get("vertices", []), dtype=np.float64)
    if vertices.shape != (6, 3):
        raise ValueError(f"Expected palm.vertices shape (6, 3), got {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise ValueError("palm.vertices must be finite")


def flatten_skeleton(skeleton: dict[str, Any]) -> np.ndarray:
    _require_runtime_skeleton(skeleton)
    theta = np.zeros(SKELETON_PARAM_SIZE, dtype=np.float64)
    for idx, (finger, bone_name) in enumerate(FINGER_BONE_LAYOUT):
        theta[idx] = float(skeleton[finger][bone_name])
    palm_vertices = np.asarray(skeleton["palm"]["vertices"], dtype=np.float64).reshape(6, 3)
    for offset, vertex_index in enumerate(PALM_VERTEX_INDICES):
        start = PALM_PARAM_START + 3 * offset
        theta[start : start + 3] = palm_vertices[vertex_index]
    return theta


def validate_runtime_skeleton(skeleton: dict[str, Any]) -> None:
    _require_runtime_skeleton(skeleton)
    for finger, bone_name in FINGER_BONE_LAYOUT:
        value = float(skeleton[finger][bone_name])
        if value < 0.0:
            raise ValueError(f"Bone length must be non-negative: {finger}.{bone_name}")


def unflatten_skeleton(theta: np.ndarray, template_skeleton: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.shape[0] != SKELETON_PARAM_SIZE:
        raise ValueError(f"Expected theta length {SKELETON_PARAM_SIZE}, got {values.shape[0]}")
    if not np.isfinite(values).all():
        raise ValueError("theta must be finite")

    skeleton = copy.deepcopy(template_skeleton)
    for idx, (finger, bone_name) in enumerate(FINGER_BONE_LAYOUT):
        skeleton.setdefault(finger, {})
        skeleton[finger][bone_name] = float(values[idx])

    palm_vertices = np.asarray(skeleton.get("palm", {}).get("vertices", []), dtype=np.float64).reshape(6, 3).copy()
    palm_vertices[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    for offset, vertex_index in enumerate(PALM_VERTEX_INDICES):
        start = PALM_PARAM_START + 3 * offset
        palm_vertices[vertex_index] = values[start : start + 3]
    skeleton.setdefault("palm", {})
    skeleton["palm"]["vertices"] = palm_vertices.tolist()
    validate_runtime_skeleton(skeleton)
    return skeleton


def make_bounds(template_skeleton: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    theta0 = flatten_skeleton(template_skeleton)
    lower = np.zeros_like(theta0, dtype=np.float64)
    upper = np.zeros_like(theta0, dtype=np.float64)
    lower[:BONE_PARAM_COUNT] = 1.0
    upper[:BONE_PARAM_COUNT] = 130.0

    for offset in range(len(PALM_VERTEX_NAMES)):
        start = PALM_PARAM_START + 3 * offset
        base = theta0[start : start + 3]
        lower[start : start + 3] = base - PALM_PARAM_DELTA_MM
        upper[start : start + 3] = base + PALM_PARAM_DELTA_MM
    return lower, upper
