from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dexslide.kinematics.transforms import (
    average_transforms,
    make_transform,
    rotmat_to_quaternion_xyzw,
)
from dexslide.vision.marker_body_model import (
    HandCubeOverlayConfig,
    load_marker_to_wrist_asset,
    marker_to_wrist_asset_transforms,
    marker_to_wrist_entry_from_transform,
)


PALM_TRIANGLE_LANDMARK_INDICES: tuple[int, int, int] = (0, 5, 9)
PALM_TRIANGLE_LANDMARK_NAMES: tuple[str, str, str] = ("wrist", "index_mcp", "middle_mcp")
MEDIAPIPE_RGBD_NOTE = (
    "This calibration uses the MediaPipe palm triangle (wrist, index_mcp, middle_mcp) "
    "together with RGB-D depth and ArUco marker-body pose observations to solve the full body->wrist pose."
)


def _coerce_transform_samples(
    samples_body_to_wrist: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    if isinstance(samples_body_to_wrist, list):
        if not samples_body_to_wrist:
            return np.zeros((0, 4, 4), dtype=np.float64)
        arr = np.stack(samples_body_to_wrist, axis=0)
    else:
        arr = np.asarray(samples_body_to_wrist, dtype=np.float64)

    if arr.size == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Expected transform samples with shape (N, 4, 4), got {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def select_palm_triangle_points(
    landmark_points: np.ndarray,
    indices: tuple[int, int, int] = PALM_TRIANGLE_LANDMARK_INDICES,
) -> np.ndarray:
    points = np.asarray(landmark_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected landmark points with shape (N, 3), got {points.shape}")
    if points.shape[0] == len(indices):
        return points.reshape(len(indices), 3).copy()
    max_index = max(int(idx) for idx in indices)
    if points.shape[0] <= max_index:
        raise ValueError(
            f"Expected at least {max_index + 1} landmark points for indices {indices}, got {points.shape[0]}"
        )
    return points[np.asarray(indices, dtype=int), :].copy()


def estimate_body_to_wrist_transform_from_triangles(
    source_triangle_wrist_m: np.ndarray,
    observed_triangle_body_m: np.ndarray,
    *,
    min_triangle_area_m2: float = 1e-7,
) -> np.ndarray | None:
    source = select_palm_triangle_points(source_triangle_wrist_m)
    target = select_palm_triangle_points(observed_triangle_body_m)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        return None

    source_area = 0.5 * np.linalg.norm(np.cross(source[1] - source[0], source[2] - source[0]))
    target_area = 0.5 * np.linalg.norm(np.cross(target[1] - target[0], target[2] - target[0]))
    if source_area < float(min_triangle_area_m2) or target_area < float(min_triangle_area_m2):
        return None

    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid[None, :]
    target_centered = target - target_centroid[None, :]

    covariance = source_centered.T @ target_centered
    u, _s, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = target_centroid - rotation @ source_centroid
    return make_transform(rotation, translation)


def average_body_to_wrist_transforms(
    samples_body_to_wrist: list[np.ndarray] | np.ndarray,
) -> np.ndarray | None:
    samples = _coerce_transform_samples(samples_body_to_wrist)
    return average_transforms(samples)


def apply_body_to_wrist_transform(
    cfg: HandCubeOverlayConfig,
    transform_body_to_wrist: np.ndarray,
) -> np.ndarray:
    transform = np.asarray(transform_body_to_wrist, dtype=np.float64).reshape(4, 4).copy()
    cfg.set_body_to_wrist_transform(transform)
    return transform


def capture_body_to_wrist_transform_sample(
    samples_body_to_wrist: list[np.ndarray],
    sample_body_to_wrist: np.ndarray,
    *,
    replace_existing: bool,
) -> np.ndarray:
    sample = np.asarray(sample_body_to_wrist, dtype=np.float64).reshape(4, 4).copy()
    if replace_existing:
        samples_body_to_wrist.clear()
    samples_body_to_wrist.append(sample)
    mean_transform = average_body_to_wrist_transforms(samples_body_to_wrist)
    if mean_transform is None:
        raise ValueError("Expected at least one body->wrist transform sample.")
    return mean_transform


def save_body_to_wrist_alignment_outputs(
    *,
    input_config_path: Path,
    output_config_path: Path,
    output_report_path: Path,
    cfg: HandCubeOverlayConfig,
    samples_body_to_wrist: list[np.ndarray] | np.ndarray,
    initial_guess_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    samples = _coerce_transform_samples(samples_body_to_wrist)
    if samples.shape[0] == 0:
        raise ValueError("No body->wrist transform samples to save.")

    mean_transform = average_body_to_wrist_transforms(samples)
    if mean_transform is None:
        raise ValueError("Failed to average body->wrist transform samples.")
    std_translation = np.std(samples[:, :3, 3], axis=0)

    tags_to_marker_path = Path(input_config_path).expanduser().resolve()
    marker_to_wrist_path = Path(output_config_path).expanduser().resolve()
    dataset_path = Path(output_report_path).expanduser().resolve()

    existing_initial_guess = None
    if marker_to_wrist_path.exists():
        try:
            asset_doc = load_marker_to_wrist_asset(marker_to_wrist_path)
            existing_initial_guess, _existing_result, _active = marker_to_wrist_asset_transforms(
                asset_doc,
                asset_path=marker_to_wrist_path,
            )
        except Exception:
            existing_initial_guess = None

    if existing_initial_guess is not None:
        initial_guess_transform = existing_initial_guess
    if initial_guess_transform is None:
        initial_guess_transform = cfg.body_to_wrist_transform()

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "hand": str(cfg.hand),
        "input_tags2marker_path": tags_to_marker_path.name,
        "output_marker2wrist_path": marker_to_wrist_path.name,
        "saved_at_unix": float(time.time()),
        "sample_count": int(samples.shape[0]),
        "samples_body_to_wrist_transform_matrix": np.asarray(samples, dtype=np.float64).tolist(),
        "samples_body_to_wrist_translation_m": [
            [float(v) for v in row]
            for row in np.asarray(samples[:, :3, 3], dtype=np.float64)
        ],
        "samples_body_to_wrist_rotation_quaternion_xyzw": [
            [float(v) for v in rotmat_to_quaternion_xyzw(sample[:3, :3])]
            for sample in samples
        ],
        "note": MEDIAPIPE_RGBD_NOTE,
    }
    with dataset_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    marker_to_wrist_payload = {
        "hand": str(cfg.hand),
        "tags2marker_path": tags_to_marker_path.name,
        "note": MEDIAPIPE_RGBD_NOTE,
        "initial_guess": marker_to_wrist_entry_from_transform(initial_guess_transform),
        "result": marker_to_wrist_entry_from_transform(
            mean_transform,
            extra_fields={
                "data_source": f"数据取自：{dataset_path.name}",
                "translation_std_m": [float(x) for x in std_translation],
                "note": MEDIAPIPE_RGBD_NOTE,
            },
        ),
    }
    marker_to_wrist_path.parent.mkdir(parents=True, exist_ok=True)
    with marker_to_wrist_path.open("w", encoding="utf-8") as handle:
        json.dump(marker_to_wrist_payload, handle, ensure_ascii=False, indent=2)

    return np.asarray(mean_transform, dtype=np.float64).reshape(4, 4), np.asarray(std_translation, dtype=np.float64).reshape(3)
