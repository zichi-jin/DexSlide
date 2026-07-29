from __future__ import annotations

import json
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import numpy as np

from dexslide.paths import DEXALIGN_CALIBRATION_DIR
from dexslide.calibration.marker_assets import marker_to_wrist_asset_transforms

from .types import AlignmentDataset, AlignmentFrame, CAPTURE_KINDS, NUM_KEYPOINTS


def make_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def ensure_session_dir(session_id: str | None = None, *, base_dir: Path = DEXALIGN_CALIBRATION_DIR) -> tuple[str, Path]:
    resolved_session_id = str(session_id).strip() if session_id is not None else ""
    if not resolved_session_id:
        resolved_session_id = make_session_id()
    session_dir = Path(base_dir).expanduser().resolve() / resolved_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return resolved_session_id, session_dir


def load_runtime_skeleton(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected skeleton JSON object at {path}")
    return data


def save_runtime_skeleton(path: str | Path, skeleton: Mapping[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(skeleton), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _stack_optional_matrix(dataset: AlignmentDataset, attr_name: str, shape: tuple[int, ...]) -> np.ndarray:
    fill_value = np.nan
    if not dataset.frames:
        return np.zeros((0, *shape), dtype=np.float64)
    items = []
    for frame in dataset.frames:
        value = getattr(frame, attr_name)
        if value is None:
            items.append(np.full(shape, fill_value, dtype=np.float64))
        else:
            items.append(np.asarray(value, dtype=np.float64).reshape(shape))
    return np.stack(items, axis=0)


def dataset_stem_for_capture_kind(capture_kind: str) -> str:
    normalized = str(capture_kind).strip().lower()
    if normalized not in CAPTURE_KINDS:
        raise ValueError(f"capture_kind must be one of {CAPTURE_KINDS}, got {capture_kind!r}")
    return f"dataset_{normalized}"


def dataset_paths_for_capture_kind(session_dir: str | Path, capture_kind: str) -> tuple[Path, Path]:
    output_dir = Path(session_dir).expanduser().resolve()
    stem = dataset_stem_for_capture_kind(capture_kind)
    return output_dir / f"{stem}.npz", output_dir / f"{stem}_meta.json"


def _infer_capture_kind(npz_path: Path, meta: Mapping[str, Any]) -> str:
    meta_capture_kind = str(meta.get("capture_kind", "")).strip().lower()
    if meta_capture_kind in CAPTURE_KINDS:
        return meta_capture_kind
    stem = npz_path.stem.strip().lower()
    if stem in {"dataset_s1", "dataset_s2"}:
        return stem.split("_", 1)[1]
    return "s2"


def _dataset_arrays(dataset: AlignmentDataset) -> dict[str, np.ndarray]:
    return {
        "timestamps": dataset.timestamps(),
        "camera_T_marker": dataset.camera_T_marker_array(),
        "q_encoder_rad20": dataset.q_encoder_array(),
        "keypoints_camera_mm": dataset.keypoints_camera_array(),
        "keypoint_confidence": dataset.keypoint_confidence_array(),
        "keypoint_valid_mask": dataset.keypoint_valid_mask_array().astype(np.uint8),
        "keypoints_uv": _stack_optional_matrix(dataset, "keypoints_uv", (NUM_KEYPOINTS, 2)),
        "depth_mm": _stack_optional_matrix(dataset, "depth_mm", (NUM_KEYPOINTS,)),
    }


def _validate_dataset_arrays(
    *,
    npz_path: Path,
    arrays: Mapping[str, np.ndarray],
    expected_frame_count: int,
    meta: Mapping[str, Any] | None = None,
) -> None:
    frame_count = int(expected_frame_count)
    shape_expectations: dict[str, tuple[int, ...]] = {
        "timestamps": (frame_count,),
        "camera_T_marker": (frame_count, 4, 4),
        "q_encoder_rad20": (frame_count, 20),
        "keypoints_camera_mm": (frame_count, NUM_KEYPOINTS, 3),
        "keypoint_confidence": (frame_count, NUM_KEYPOINTS),
        "keypoint_valid_mask": (frame_count, NUM_KEYPOINTS),
        "keypoints_uv": (frame_count, NUM_KEYPOINTS, 2),
        "depth_mm": (frame_count, NUM_KEYPOINTS),
    }
    for key, expected_shape in shape_expectations.items():
        value = np.asarray(arrays[key])
        if value.shape != expected_shape:
            raise ValueError(
                f"Corrupted DexAlign dataset at {npz_path}: field {key} expected shape {expected_shape}, got {value.shape}"
            )

    if meta is None:
        return
    meta_frame_count = meta.get("frame_count")
    if meta_frame_count is not None and int(meta_frame_count) != frame_count:
        raise ValueError(
            f"Corrupted DexAlign dataset at {npz_path}: npz has {frame_count} frames but meta records {int(meta_frame_count)}"
        )

    marker_ids_used = meta.get("marker_ids_used")
    if marker_ids_used not in (None, []) and len(marker_ids_used) != frame_count:
        raise ValueError(
            f"Corrupted DexAlign dataset at {npz_path}: marker_ids_used length {len(marker_ids_used)} does not match frame_count {frame_count}"
        )

    marker_reproj_error_px = meta.get("marker_reproj_error_px")
    if marker_reproj_error_px not in (None, []) and len(marker_reproj_error_px) != frame_count:
        raise ValueError(
            f"Corrupted DexAlign dataset at {npz_path}: marker_reproj_error_px length {len(marker_reproj_error_px)} does not match frame_count {frame_count}"
        )


def save_alignment_dataset(
    session_dir: str | Path,
    dataset: AlignmentDataset,
    *,
    extra_meta: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(session_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, meta_path = dataset_paths_for_capture_kind(output_dir, dataset.capture_kind)
    arrays = _dataset_arrays(dataset)
    _validate_dataset_arrays(
        npz_path=dataset_path,
        arrays=arrays,
        expected_frame_count=dataset.num_frames,
        meta=None,
    )

    with NamedTemporaryFile(prefix=f"{dataset_path.stem}.", suffix=".npz", dir=output_dir, delete=False) as handle:
        temp_npz_path = Path(handle.name)
    try:
        np.savez_compressed(temp_npz_path, **arrays)
        with np.load(temp_npz_path) as payload:
            reloaded_arrays = {key: np.asarray(payload[key]) for key in payload.files}
        _validate_dataset_arrays(
            npz_path=temp_npz_path,
            arrays=reloaded_arrays,
            expected_frame_count=dataset.num_frames,
            meta=None,
        )
        temp_npz_path.replace(dataset_path)
    finally:
        if temp_npz_path.exists():
            temp_npz_path.unlink()

    meta: dict[str, Any] = {
        "hand": dataset.hand,
        "capture_kind": dataset.capture_kind,
        "frame_count": dataset.num_frames,
        "capture_note": dataset.capture_note,
        "source_config_paths": dict(dataset.source_config_paths or {}),
        "marker_ids_used": [list(frame.marker_ids_used) for frame in dataset.frames],
        "marker_reproj_error_px": [
            None if frame.marker_reproj_error_px is None else float(frame.marker_reproj_error_px)
            for frame in dataset.frames
        ],
        "saved_at_unix": time.time(),
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    with NamedTemporaryFile(prefix=f"{meta_path.stem}.", suffix=".json", dir=output_dir, delete=False, mode="w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_meta_path = Path(handle.name)
    temp_meta_path.replace(meta_path)
    return dataset_path, meta_path


def load_alignment_dataset(dataset_path: str | Path, meta_path: str | Path | None = None) -> AlignmentDataset:
    npz_path = Path(dataset_path).expanduser().resolve()
    resolved_meta_path = (
        Path(meta_path).expanduser().resolve()
        if meta_path is not None
        else npz_path.with_name(f"{npz_path.stem}_meta.json")
    )
    with np.load(npz_path) as payload:
        timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        camera_T_marker = np.asarray(payload["camera_T_marker"], dtype=np.float64)
        q_encoder_rad20 = np.asarray(payload["q_encoder_rad20"], dtype=np.float64)
        keypoints_camera_mm = np.asarray(payload["keypoints_camera_mm"], dtype=np.float64)
        keypoint_confidence = np.asarray(payload["keypoint_confidence"], dtype=np.float64)
        keypoint_valid_mask = np.asarray(payload["keypoint_valid_mask"], dtype=np.uint8).astype(bool)
        keypoints_uv = np.asarray(payload["keypoints_uv"], dtype=np.float64)
        depth_mm = np.asarray(payload["depth_mm"], dtype=np.float64)

    meta: dict[str, Any] = {}
    if resolved_meta_path.exists():
        with resolved_meta_path.open("r", encoding="utf-8") as handle:
            loaded_meta = json.load(handle)
        if isinstance(loaded_meta, dict):
            meta = loaded_meta
    if not meta and npz_path.stem == "dataset":
        legacy_meta_path = npz_path.with_name("dataset_meta.json")
        if legacy_meta_path.exists():
            with legacy_meta_path.open("r", encoding="utf-8") as handle:
                loaded_meta = json.load(handle)
            if isinstance(loaded_meta, dict):
                meta = loaded_meta

    _validate_dataset_arrays(
        npz_path=npz_path,
        arrays={
            "timestamps": timestamps,
            "camera_T_marker": camera_T_marker,
            "q_encoder_rad20": q_encoder_rad20,
            "keypoints_camera_mm": keypoints_camera_mm,
            "keypoint_confidence": keypoint_confidence,
            "keypoint_valid_mask": keypoint_valid_mask,
            "keypoints_uv": keypoints_uv,
            "depth_mm": depth_mm,
        },
        expected_frame_count=int(timestamps.shape[0]),
        meta=meta,
    )

    marker_ids_used = meta.get("marker_ids_used", [])
    marker_reproj_error_px = meta.get("marker_reproj_error_px", [])
    frames: list[AlignmentFrame] = []
    frame_count = int(timestamps.shape[0])
    for idx in range(frame_count):
        frame_uv = keypoints_uv[idx]
        frame_depth = depth_mm[idx]
        frames.append(
            AlignmentFrame(
                timestamp=float(timestamps[idx]),
                camera_T_marker=camera_T_marker[idx],
                q_encoder_rad20=q_encoder_rad20[idx],
                keypoints_camera_mm=keypoints_camera_mm[idx],
                keypoint_confidence=keypoint_confidence[idx],
                keypoint_valid_mask=keypoint_valid_mask[idx],
                keypoints_uv=None if np.isnan(frame_uv).all() else frame_uv,
                depth_mm=None if np.isnan(frame_depth).all() else frame_depth,
                marker_ids_used=tuple(marker_ids_used[idx]) if idx < len(marker_ids_used) else (),
                marker_reproj_error_px=(
                    None
                    if idx >= len(marker_reproj_error_px) or marker_reproj_error_px[idx] is None
                    else float(marker_reproj_error_px[idx])
                ),
            )
        )
    return AlignmentDataset(
        hand=str(meta.get("hand", "left")),
        frames=tuple(frames),
        capture_kind=_infer_capture_kind(npz_path, meta),
        source_config_paths=meta.get("source_config_paths", {}),
        capture_note=str(meta.get("capture_note", "")),
    )


def load_marker2hand_asset_mm(path: str | Path) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    initial_m, result_m, active_m = marker_to_wrist_asset_transforms(path)
    if active_m is None:
        raise ValueError(f"Marker2hand asset missing both initial_guess and result: {path}")

    def _to_mm(transform_m: np.ndarray | None) -> np.ndarray | None:
        if transform_m is None:
            return None
        transform_mm = np.asarray(transform_m, dtype=np.float64).reshape(4, 4).copy()
        transform_mm[:3, 3] *= 1000.0
        return transform_mm

    return _to_mm(initial_m), _to_mm(result_m), _to_mm(active_m)


def save_marker2hand_result(
    path: str | Path,
    *,
    hand: str,
    initial_transform_mm: np.ndarray | None,
    optimized_transform_mm: np.ndarray,
    source_dataset_path: str | None = None,
    note: str | None = None,
) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _entry(transform_mm: np.ndarray, *, extra_fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
        transform_m = np.asarray(transform_mm, dtype=np.float64).reshape(4, 4).copy()
        transform_m[:3, 3] *= 0.001
        payload = {
            "trans": [float(x) for x in transform_m[:3, 3]],
            "rot": transform_m[:3, :3].tolist(),
            "trans_unit": "m",
        }
        if extra_fields:
            payload.update(dict(extra_fields))
        return payload

    document: dict[str, Any] = {
        "hand": str(hand),
        "note": note or "DexAlign joint optimization of skeleton and marker2hand pose.",
    }
    if initial_transform_mm is not None:
        document["initial_guess"] = _entry(initial_transform_mm)
    document["result"] = _entry(
        optimized_transform_mm,
        extra_fields={
            "data_source": None if source_dataset_path is None else str(source_dataset_path),
        },
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def describe_dataset(dataset: AlignmentDataset) -> dict[str, Any]:
    if dataset.num_frames == 0:
        return {
            "hand": dataset.hand,
            "num_frames": 0,
            "valid_keypoints_mean": 0.0,
            "valid_keypoints_min": 0,
            "valid_keypoints_max": 0,
        }
    valid_counts = dataset.keypoint_valid_mask_array().sum(axis=1)
    return {
        "hand": dataset.hand,
        "num_frames": dataset.num_frames,
        "valid_keypoints_mean": float(np.mean(valid_counts)),
        "valid_keypoints_min": int(np.min(valid_counts)),
        "valid_keypoints_max": int(np.max(valid_counts)),
        "capture_note": dataset.capture_note,
        "source_config_paths": dict(dataset.source_config_paths or {}),
    }
