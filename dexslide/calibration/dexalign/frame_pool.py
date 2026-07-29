"""Frame collection and array views for DexAlign optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexslide.calibration.dexalign.types import AlignmentFrame


@dataclass(frozen=True)
class FramePool:
    hand: str
    frames: tuple[AlignmentFrame, ...]
    source_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.source_labels):
            raise ValueError("frames and source_labels must have identical length")

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def subset(self, keep_mask: np.ndarray) -> "FramePool":
        mask = np.asarray(keep_mask, dtype=bool).reshape(self.num_frames)
        return FramePool(self.hand, tuple(frame for frame, keep in zip(self.frames, mask) if keep), tuple(label for label, keep in zip(self.source_labels, mask) if keep))

    def timestamps(self) -> np.ndarray:
        return np.asarray([frame.timestamp for frame in self.frames], dtype=np.float64) if self.frames else np.zeros(0, dtype=np.float64)

    def camera_T_marker_array(self) -> np.ndarray:
        return np.stack([frame.camera_T_marker for frame in self.frames], axis=0) if self.frames else np.zeros((0, 4, 4), dtype=np.float64)

    def q_encoder_array(self) -> np.ndarray:
        return np.stack([frame.q_encoder_rad20 for frame in self.frames], axis=0) if self.frames else np.zeros((0, 20), dtype=np.float64)

    def keypoints_camera_array(self) -> np.ndarray:
        return np.stack([frame.keypoints_camera_mm for frame in self.frames], axis=0) if self.frames else np.zeros((0, 21, 3), dtype=np.float64)

    def keypoint_confidence_array(self) -> np.ndarray:
        return np.stack([frame.keypoint_confidence for frame in self.frames], axis=0) if self.frames else np.zeros((0, 21), dtype=np.float64)

    def keypoint_valid_mask_array(self) -> np.ndarray:
        return np.stack([frame.keypoint_valid_mask for frame in self.frames], axis=0) if self.frames else np.zeros((0, 21), dtype=bool)

    def finite_q_mask(self) -> np.ndarray:
        q = self.q_encoder_array()
        return np.isfinite(q).all(axis=1) if q.size else np.zeros(self.num_frames, dtype=bool)

