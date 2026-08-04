#!/usr/bin/env python3
"""Fit an independent pose-pair residual RBF curve for DexSlide to OrcaHand."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

ROBOT_MANIPULATION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROBOT_MANIPULATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_manipulation.orca_control.direct_joint_mapping import (
    DirectJointCalibration,
    DirectJointMapper,
    TARGET_JOINTS,
)
from robot_manipulation.orca_control.pair_curve_mapping import CURVE_SCHEMA
from robot_manipulation.orca_control.paths import (
    DEFAULT_DIRECT_JOINT_CALIBRATION_FILE,
    ORCA_CONFIG_DIR,
)

DEFAULT_PAIRS_FILE = ORCA_CONFIG_DIR / "orca_teleop_pairs.json"
DEFAULT_OUTPUT_FILE = ORCA_CONFIG_DIR / "orca_teleop_pair_curve.json"


def _as_pose_vector(
    value: object,
    joint_order: tuple[str, ...],
    *,
    description: str,
) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    try:
        result = np.asarray([float(value[joint]) for joint in joint_order], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"{description} is missing {exc.args[0]}") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{description} contains non-finite values")
    return result


def build_pair_curve(
    pairs_payload: Mapping[str, object],
    baseline_payload: Mapping[str, object],
    *,
    epsilon: float = 1.0,
    ridge: float = 1e-8,
) -> dict[str, object]:
    """Fit residuals over the old affine mapper without changing that mapper."""
    if pairs_payload.get("schema_version") != 1:
        raise ValueError("Unsupported pose-pair data schema")
    if pairs_payload.get("source_unit") != "deg" or pairs_payload.get("target_unit") != "deg":
        raise ValueError("Pose-pair data must use deg for source and target")
    calibration = DirectJointCalibration.from_payload(dict(baseline_payload))
    source_order = tuple(str(value) for value in pairs_payload.get("source_joint_order", ()))
    target_order = tuple(str(value) for value in pairs_payload.get("target_joint_order", ()))
    if source_order != calibration.source_joint_order:
        raise ValueError("Pose-pair source joint order differs from the baseline calibration")
    if target_order != TARGET_JOINTS:
        raise ValueError(f"Unexpected pose-pair target joint order: {target_order}")
    raw_pairs = pairs_payload.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("At least one pose pair is required")

    source_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for index, pair in enumerate(raw_pairs, start=1):
        if not isinstance(pair, Mapping):
            raise ValueError(f"Pose pair {index} must be an object")
        dexslide = pair.get("dexslide")
        orca = pair.get("orca")
        if not isinstance(dexslide, Mapping) or not isinstance(orca, Mapping):
            raise ValueError(f"Pose pair {index} must contain dexslide and orca")
        source_rows.append(
            _as_pose_vector(
                dexslide.get("joint_angles_deg"),
                source_order,
                description=f"Pose pair {index} DexSlide",
            )
        )
        target_rows.append(
            _as_pose_vector(
                orca.get("joint_positions_deg"),
                TARGET_JOINTS,
                description=f"Pose pair {index} Orca",
            )
        )

    source_samples = np.vstack(source_rows)
    target_samples = np.vstack(target_rows)
    for first in range(source_samples.shape[0]):
        for second in range(first):
            if np.max(np.abs(source_samples[first] - source_samples[second])) <= 1e-8:
                if np.max(np.abs(target_samples[first] - target_samples[second])) > 1e-6:
                    raise ValueError(
                        "Two identical DexSlide poses have different Orca targets; "
                        "remove or correct the conflicting pair"
                    )

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be non-negative")

    baseline = DirectJointMapper(calibration)
    baseline_targets = np.vstack(
        [
            np.asarray(
                [
                    baseline.map(row).target_positions_deg[joint]
                    for joint in TARGET_JOINTS
                ],
                dtype=np.float64,
            )
            for row in source_samples
        ]
    )
    residuals = target_samples - baseline_targets
    center = np.mean(source_samples, axis=0)
    scale = np.maximum(np.ptp(source_samples, axis=0), 1.0)
    normalized = (source_samples - center) / scale
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    kernel = np.exp(-np.square(distances / float(epsilon)))
    try:
        weights = np.linalg.solve(
            kernel + float(ridge) * np.eye(kernel.shape[0]),
            residuals,
        )
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(kernel, residuals, rcond=None)[0]

    return {
        "schema": CURVE_SCHEMA,
        "source_unit": "deg",
        "target_unit": "deg",
        "source_joint_order": list(source_order),
        "target_joint_order": list(TARGET_JOINTS),
        "baseline_calibration": dict(baseline_payload),
        "source_samples_deg": source_samples.tolist(),
        "target_samples_deg": target_samples.tolist(),
        "normalization": {
            "center_deg": center.tolist(),
            "scale_deg": scale.tolist(),
        },
        "epsilon": float(epsilon),
        "ridge": float(ridge),
        "weights_deg": weights.tolist(),
        "exact_match_tolerance_normalized": 1e-10,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_FILE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_DIRECT_JOINT_CALIBRATION_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-8)
    args = parser.parse_args()

    pairs_path = args.pairs.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    if not pairs_path.exists():
        raise SystemExit(f"Pose-pair data file does not exist: {pairs_path}")
    if not baseline_path.exists():
        raise SystemExit(f"Baseline calibration file does not exist: {baseline_path}")
    pairs_payload = json.loads(pairs_path.read_text(encoding="utf-8"))
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    curve = build_pair_curve(
        pairs_payload,
        baseline_payload,
        epsilon=args.epsilon,
        ridge=args.ridge,
    )
    curve["created_at_unix"] = time.time()
    curve["pairs_file"] = str(pairs_path)
    curve["pairs_sha256"] = _file_sha256(pairs_path)
    args.output.expanduser().resolve().write_text(
        json.dumps(curve, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sample_count = len(curve["source_samples_deg"])
    print(f"已拟合 {sample_count} 个姿态对：{args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
