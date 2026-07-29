from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


NUM_KEYPOINTS = 21
NUM_GLOVE_JOINTS = 20
CAPTURE_KINDS: tuple[str, ...] = ("s1", "s2")


def _coerce_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    allow_nonfinite: bool = False,
    dtype: type[np.floating[Any]] | type[np.bool_] = np.float64,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} expected shape {shape}, got {array.shape}")
    if not allow_nonfinite and not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array.copy()


@dataclass(frozen=True)
class AlignmentFrame:
    timestamp: float
    camera_T_marker: np.ndarray
    q_encoder_rad20: np.ndarray
    keypoints_camera_mm: np.ndarray
    keypoint_confidence: np.ndarray
    keypoint_valid_mask: np.ndarray
    keypoints_uv: np.ndarray | None = None
    depth_mm: np.ndarray | None = None
    marker_ids_used: tuple[int, ...] = ()
    marker_reproj_error_px: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(
            self,
            "camera_T_marker",
            _coerce_array(self.camera_T_marker, name="camera_T_marker", shape=(4, 4)),
        )
        object.__setattr__(
            self,
            "q_encoder_rad20",
            _coerce_array(
                self.q_encoder_rad20,
                name="q_encoder_rad20",
                shape=(NUM_GLOVE_JOINTS,),
                allow_nonfinite=True,
            ),
        )
        object.__setattr__(
            self,
            "keypoints_camera_mm",
            _coerce_array(
                self.keypoints_camera_mm,
                name="keypoints_camera_mm",
                shape=(NUM_KEYPOINTS, 3),
                allow_nonfinite=True,
            ),
        )
        object.__setattr__(
            self,
            "keypoint_confidence",
            _coerce_array(
                self.keypoint_confidence,
                name="keypoint_confidence",
                shape=(NUM_KEYPOINTS,),
                allow_nonfinite=False,
            ),
        )
        object.__setattr__(
            self,
            "keypoint_valid_mask",
            _coerce_array(
                self.keypoint_valid_mask,
                name="keypoint_valid_mask",
                shape=(NUM_KEYPOINTS,),
                allow_nonfinite=False,
                dtype=np.bool_,
            ).astype(bool, copy=False),
        )
        if self.keypoints_uv is not None:
            object.__setattr__(
                self,
                "keypoints_uv",
                _coerce_array(
                    self.keypoints_uv,
                    name="keypoints_uv",
                    shape=(NUM_KEYPOINTS, 2),
                    allow_nonfinite=True,
                ),
            )
        if self.depth_mm is not None:
            object.__setattr__(
                self,
                "depth_mm",
                _coerce_array(
                    self.depth_mm,
                    name="depth_mm",
                    shape=(NUM_KEYPOINTS,),
                    allow_nonfinite=True,
                ),
            )
        object.__setattr__(
            self,
            "marker_ids_used",
            tuple(int(marker_id) for marker_id in self.marker_ids_used),
        )
        if self.marker_reproj_error_px is not None:
            object.__setattr__(self, "marker_reproj_error_px", float(self.marker_reproj_error_px))


@dataclass(frozen=True)
class AlignmentDataset:
    hand: str
    frames: tuple[AlignmentFrame, ...]
    capture_kind: str = "s2"
    source_config_paths: Mapping[str, str] | None = None
    capture_note: str = ""

    def __post_init__(self) -> None:
        hand = str(self.hand).strip().lower()
        if not hand:
            raise ValueError("hand must be non-empty")
        object.__setattr__(self, "hand", hand)
        object.__setattr__(self, "frames", tuple(self.frames))
        capture_kind = str(self.capture_kind).strip().lower()
        if capture_kind not in CAPTURE_KINDS:
            raise ValueError(f"capture_kind must be one of {CAPTURE_KINDS}, got {self.capture_kind!r}")
        object.__setattr__(self, "capture_kind", capture_kind)
        source_paths = dict(self.source_config_paths or {})
        object.__setattr__(
            self,
            "source_config_paths",
            {str(key): str(value) for key, value in source_paths.items()},
        )
        object.__setattr__(self, "capture_note", str(self.capture_note))

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def timestamps(self) -> np.ndarray:
        return np.asarray([frame.timestamp for frame in self.frames], dtype=np.float64)

    def camera_T_marker_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, 4, 4), dtype=np.float64)
        return np.stack([frame.camera_T_marker for frame in self.frames], axis=0)

    def q_encoder_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, NUM_GLOVE_JOINTS), dtype=np.float64)
        return np.stack([frame.q_encoder_rad20 for frame in self.frames], axis=0)

    def has_finite_glove_angles(self) -> bool:
        if not self.frames:
            return False
        q = self.q_encoder_array()
        return bool(np.isfinite(q).all())

    def keypoints_camera_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float64)
        return np.stack([frame.keypoints_camera_mm for frame in self.frames], axis=0)

    def keypoint_confidence_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, NUM_KEYPOINTS), dtype=np.float64)
        return np.stack([frame.keypoint_confidence for frame in self.frames], axis=0)

    def keypoint_valid_mask_array(self) -> np.ndarray:
        if not self.frames:
            return np.zeros((0, NUM_KEYPOINTS), dtype=bool)
        return np.stack([frame.keypoint_valid_mask for frame in self.frames], axis=0)


@dataclass(frozen=True)
class OptimizationResult:
    x_opt: np.ndarray
    optimized_skeleton: dict[str, Any]
    optimized_marker2hand: np.ndarray
    final_cost: float
    num_frames_used: int
    summary_metrics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_opt", np.asarray(self.x_opt, dtype=np.float64).reshape(-1).copy())
        object.__setattr__(
            self,
            "optimized_marker2hand",
            _coerce_array(
                self.optimized_marker2hand,
                name="optimized_marker2hand",
                shape=(4, 4),
            ),
        )
        object.__setattr__(self, "final_cost", float(self.final_cost))
        object.__setattr__(self, "num_frames_used", int(self.num_frames_used))
        object.__setattr__(self, "summary_metrics", dict(self.summary_metrics))
