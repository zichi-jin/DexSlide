"""Human-hand landmark reconstruction for DexSlide retargeting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from dexslide.kinematics.live_hand import (
    FINGERS,
    apply_handedness,
    canonicalize_palm_xoy,
    finger_points,
)
from dexslide.paths import DEFAULT_SKELETON_FILE
from dexslide.serial_angles import JOINT_LABELS

DEFAULT_HUMAN_JOINT_NAMES = [f"{finger}.{joint}" for finger in FINGERS for joint in JOINT_LABELS]
HUMAN_LANDMARK_NAMES = [
    "wrist",
    "thumb_base",
    "thumb_knuckle",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


def _load_skeleton(skeleton_file: str | Path | Mapping[str, object]) -> dict:
    if isinstance(skeleton_file, Mapping):
        return dict(skeleton_file)
    path = Path(skeleton_file).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected skeleton json object, got {type(data)}")
    return data


class DexSlideHumanModel:
    """Convert DexSlide 20-DOF glove angles into 21 human landmarks."""

    def __init__(
        self,
        skeleton_file: str | Path | Mapping[str, object] = DEFAULT_SKELETON_FILE,
        *,
        hand: str = "right",
        mirror_reconstruction: bool = False,
        joint_names: list[str] | tuple[str, ...] | None = None,
        unit_scale: float = 0.001,
    ) -> None:
        self.skeleton = _load_skeleton(skeleton_file)
        self.hand = hand
        self.mirror_reconstruction = bool(mirror_reconstruction)
        self.joint_names = list(joint_names or DEFAULT_HUMAN_JOINT_NAMES)
        self.unit_scale = float(unit_scale)
        self.palm = apply_handedness(canonicalize_palm_xoy(self.skeleton), hand)

    def _coerce_joint_map(self, joint_angles: np.ndarray | list[float] | Mapping[str, float]) -> dict[str, float]:
        if isinstance(joint_angles, Mapping):
            return {str(key): float(value) for key, value in joint_angles.items()}

        vector = np.asarray(joint_angles, dtype=np.float64).reshape(-1)
        if vector.shape[0] != len(self.joint_names):
            raise ValueError(
                f"Expected {len(self.joint_names)} human joints, got {vector.shape[0]}. "
                "If the glove layout changed, pass explicit human_joint_names to create_dex_retargeter()."
            )
        return {name: float(value) for name, value in zip(self.joint_names, vector)}

    def _finger_raw4(self, joint_map: Mapping[str, float], finger: str) -> np.ndarray:
        return np.array(
            [
                float(joint_map.get(f"{finger}.DIP", 0.0)),
                float(joint_map.get(f"{finger}.PIP", 0.0)),
                float(joint_map.get(f"{finger}.MCP_front", joint_map.get(f"{finger}.MCP", 0.0))),
                float(joint_map.get(f"{finger}.MCP_back", 0.0)),
            ],
            dtype=np.float64,
        )

    def landmarks_from_angles(
        self,
        joint_angles: np.ndarray | list[float] | Mapping[str, float],
    ) -> np.ndarray:
        joint_map = self._coerce_joint_map(joint_angles)
        landmarks = np.zeros((len(HUMAN_LANDMARK_NAMES), 3), dtype=np.float64)
        landmarks[0] = self.palm["wrist"]

        base_index = 1
        for finger in FINGERS:
            points = finger_points(
                finger,
                self._finger_raw4(joint_map, finger),
                self.skeleton,
                self.palm,
                self.hand,
            )
            landmarks[base_index : base_index + 4] = points
            base_index += 4

        if self.mirror_reconstruction:
            landmarks[:, 1] *= -1.0

        return landmarks * self.unit_scale

    __call__ = landmarks_from_angles
