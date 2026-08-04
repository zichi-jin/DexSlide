"""Pose-pair residual RBF mapper for DexSlide to OrcaHand teleoperation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .direct_joint_mapping import (
    DirectJointCalibration,
    DirectJointMapper,
    DirectJointMappingResult,
    TARGET_JOINTS,
)

CURVE_SCHEMA = "orca_teleop_pair_rbf_curve_v1"


class PairCurveMapper:
    """Map DexSlide joints through a stored residual RBF curve."""

    def __init__(
        self,
        calibration: DirectJointCalibration,
        source_samples_deg: np.ndarray,
        target_samples_deg: np.ndarray,
        center_deg: np.ndarray,
        scale_deg: np.ndarray,
        epsilon: float,
        weights_deg: np.ndarray,
        exact_match_tolerance_normalized: float = 1e-10,
    ) -> None:
        self.calibration = calibration
        self._baseline = DirectJointMapper(calibration)
        self._source_samples_deg = np.asarray(source_samples_deg, dtype=np.float64)
        self._target_samples_deg = np.asarray(target_samples_deg, dtype=np.float64)
        self._center_deg = np.asarray(center_deg, dtype=np.float64)
        self._scale_deg = np.asarray(scale_deg, dtype=np.float64)
        self._epsilon = float(epsilon)
        self._weights_deg = np.asarray(weights_deg, dtype=np.float64)
        self._exact_match_tolerance_normalized = float(exact_match_tolerance_normalized)
        self._validate()

    @classmethod
    def from_file(cls, path: str | Path) -> "PairCurveMapper":
        curve_path = Path(path).expanduser().resolve()
        payload = json.loads(curve_path.read_text(encoding="utf-8"))
        return cls.from_payload(payload, source=str(curve_path))

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        source: str = "payload",
    ) -> "PairCurveMapper":
        if payload.get("schema") != CURVE_SCHEMA:
            raise ValueError(f"Unsupported Orca pose-pair curve: {source}")
        if payload.get("source_unit") != "deg" or payload.get("target_unit") != "deg":
            raise ValueError("Pose-pair curve must use deg for source and target")
        baseline_payload = payload.get("baseline_calibration")
        if not isinstance(baseline_payload, Mapping):
            raise ValueError("Pose-pair curve is missing baseline_calibration")
        calibration = DirectJointCalibration.from_payload(
            dict(baseline_payload),
            source=f"{source}:baseline_calibration",
        )
        source_order = tuple(str(value) for value in payload.get("source_joint_order", ()))
        target_order = tuple(str(value) for value in payload.get("target_joint_order", ()))
        if source_order != calibration.source_joint_order:
            raise ValueError("Pose-pair curve source joint order differs from its baseline")
        if target_order != TARGET_JOINTS:
            raise ValueError(f"Unexpected pose-pair target joint order: {target_order}")
        normalization = payload.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("Pose-pair curve is missing normalization")
        return cls(
            calibration=calibration,
            source_samples_deg=np.asarray(payload.get("source_samples_deg"), dtype=np.float64),
            target_samples_deg=np.asarray(payload.get("target_samples_deg"), dtype=np.float64),
            center_deg=np.asarray(normalization.get("center_deg"), dtype=np.float64),
            scale_deg=np.asarray(normalization.get("scale_deg"), dtype=np.float64),
            epsilon=float(payload.get("epsilon", 0.0)),
            weights_deg=np.asarray(payload.get("weights_deg"), dtype=np.float64),
            exact_match_tolerance_normalized=float(
                payload.get("exact_match_tolerance_normalized", 1e-10)
            ),
        )

    def map(
        self,
        source_positions_deg: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        source_joint_order: Sequence[str] | None = None,
    ) -> DirectJointMappingResult:
        source = self._baseline.source_vector(source_positions_deg, source_joint_order)
        distances = self._normalized_distances(source)
        exact_index = int(np.argmin(distances))
        if distances[exact_index] <= self._exact_match_tolerance_normalized:
            target_vector = self._target_samples_deg[exact_index]
        else:
            baseline = self._baseline.map(source).target_positions_deg
            baseline_vector = np.asarray(
                [baseline[joint] for joint in TARGET_JOINTS],
                dtype=np.float64,
            )
            kernel = np.exp(-np.square(distances / self._epsilon))
            target_vector = baseline_vector + kernel @ self._weights_deg
        raw_targets = {
            joint: float(target_vector[index])
            for index, joint in enumerate(TARGET_JOINTS)
        }
        return self._baseline.project_targets(raw_targets)

    def _normalized_distances(self, source: np.ndarray) -> np.ndarray:
        normalized = (source - self._center_deg) / self._scale_deg
        samples = (self._source_samples_deg - self._center_deg) / self._scale_deg
        return np.linalg.norm(samples - normalized, axis=1)

    def _validate(self) -> None:
        source_size = len(self.calibration.source_joint_order)
        sample_count = self._source_samples_deg.shape[0] if self._source_samples_deg.ndim == 2 else 0
        if sample_count < 1 or self._source_samples_deg.shape != (sample_count, source_size):
            raise ValueError("Invalid pose-pair source sample matrix")
        if self._target_samples_deg.shape != (sample_count, len(TARGET_JOINTS)):
            raise ValueError("Invalid pose-pair target sample matrix")
        if self._weights_deg.shape != (sample_count, len(TARGET_JOINTS)):
            raise ValueError("Invalid pose-pair RBF weights")
        if self._center_deg.shape != (source_size,) or self._scale_deg.shape != (source_size,):
            raise ValueError("Invalid pose-pair normalization")
        if not np.isfinite(self._source_samples_deg).all() or not np.isfinite(self._target_samples_deg).all():
            raise ValueError("Pose-pair samples contain non-finite values")
        if not np.isfinite(self._weights_deg).all() or not np.isfinite(self._center_deg).all():
            raise ValueError("Pose-pair curve contains non-finite values")
        if not np.isfinite(self._scale_deg).all() or np.any(self._scale_deg <= 0.0):
            raise ValueError("Pose-pair normalization scale must be positive")
        if not np.isfinite(self._epsilon) or self._epsilon <= 0.0:
            raise ValueError("Pose-pair epsilon must be positive")


def load_orca_teleop_mapper(path: str | Path) -> DirectJointMapper | PairCurveMapper:
    """Load either the existing affine calibration or a pose-pair curve."""
    calibration_path = Path(path).expanduser().resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) == 2:
        return DirectJointMapper(
            DirectJointCalibration.from_payload(payload, source=str(calibration_path))
        )
    if payload.get("schema") == CURVE_SCHEMA:
        return PairCurveMapper.from_payload(payload, source=str(calibration_path))
    raise ValueError(f"Unsupported Orca teleop calibration: {calibration_path}")


__all__ = ["CURVE_SCHEMA", "PairCurveMapper", "load_orca_teleop_mapper"]
