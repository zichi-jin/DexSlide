"""Chunked DexSlide scene recorder and dataset reader."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import cv2
import numpy as np

from dexslide.streaming import DexSlideHandSample, DexSlideScene, DexSlideSceneSample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f"{path.stem}.",
        suffix=".json",
        dir=path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _safe_role_name(role: str, source: Path) -> str:
    prefix = role.replace(".", "__").replace("/", "__")
    return f"{prefix}__{source.name}"


class DexSlideRecorder:
    """Write scene samples to bounded compressed NPZ chunks."""

    def __init__(
        self,
        save_dir: str | Path,
        scene: DexSlideScene,
        *,
        session_id: str | None = None,
        chunk_size: int = 1000,
    ) -> None:
        if int(chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        self.scene = scene
        self.chunk_size = int(chunk_size)
        root = Path(save_dir).expanduser().resolve()
        resolved_id = str(session_id or "").strip()
        if not resolved_id:
            resolved_id = time.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = root / resolved_id
        if self.session_dir.exists() and any(self.session_dir.iterdir()):
            raise FileExistsError(f"DexSlide session directory is not empty: {self.session_dir}")
        self.config_dir = self.session_dir / "configs"
        self.chunks_dir = self.session_dir / "chunks"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.session_dir / "session_meta.json"
        self._samples: list[DexSlideSceneSample] = []
        self._chunk_index = 0
        self._closed = False
        self._first_valid_saved = False
        self._meta = {
            "schema_version": 1,
            "session_id": resolved_id,
            "created_at_unix": time.time(),
            "closed_at_unix": None,
            "units": dict(scene.units),
            "joint_mode": scene.joint_mode,
            "hand_ids": list(scene.hand_ids),
            "table_marker_id": scene.table_marker_id,
            "configs": self._snapshot_configs(),
            "effective_calibration": scene.effective_calibration,
            "chunks": [],
            "sample_count": 0,
            "first_valid_frame": None,
        }
        _write_json_atomic(self.meta_path, self._meta)

    def _snapshot_configs(self) -> dict[str, dict[str, str]]:
        provenance: dict[str, dict[str, str]] = {}
        for role, source in self.scene.config_files.items():
            source_path = Path(source).expanduser().resolve()
            destination = self.config_dir / _safe_role_name(role, source_path)
            if role == "streaming_config":
                destination.write_text(
                    json.dumps(self.scene.config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            elif source_path.is_file():
                shutil.copy2(source_path, destination)
            else:
                raise FileNotFoundError(f"Cannot snapshot missing config {role}: {source_path}")
            provenance[role] = {
                "original_path": str(source_path),
                "session_copy": str(destination.relative_to(self.session_dir)),
                "sha256": _sha256(destination),
            }
        return provenance

    def __enter__(self) -> "DexSlideRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def write(self, sample: DexSlideSceneSample) -> None:
        if self._closed:
            raise RuntimeError("DexSlideRecorder is already closed")
        if sample.schema_version != self._meta["schema_version"]:
            raise ValueError("schema_version cannot change within a DexSlide session")
        if sample.units != self._meta["units"]:
            raise ValueError("units cannot change within a DexSlide session")
        if tuple(sample.hands.keys()) != tuple(self.scene.hand_ids):
            raise ValueError("configured hands cannot change within a DexSlide session")
        self._samples.append(sample)
        self._save_first_valid_frame(sample)
        if len(self._samples) >= self.chunk_size:
            self._flush_chunk()

    def _save_first_valid_frame(self, sample: DexSlideSceneSample) -> None:
        if self._first_valid_saved or not sample.table_valid:
            return
        valid_hands = [hand_id for hand_id, hand in sample.hands.items() if hand.valid]
        if not valid_hands:
            return
        frame = self.scene.latest_color_frame()
        if frame is None:
            return
        image_path = self.session_dir / "first_valid_frame.jpg"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Failed to write first valid DexSlide frame: {image_path}")
        self._first_valid_saved = True
        self._meta["first_valid_frame"] = {
            "timestamp": float(sample.timestamp),
            "image_size": list(sample.image_size),
            "table_marker_id": int(sample.table_marker_id),
            "table_marker_corners_px": (
                None
                if sample.table_marker_corners_px is None
                else np.asarray(sample.table_marker_corners_px, dtype=np.float64).tolist()
            ),
            "camera_T_table": np.asarray(sample.camera_T_table, dtype=np.float64).tolist(),
            "valid_hands": valid_hands,
            "image_path": image_path.name,
            "sha256": _sha256(image_path),
        }
        _write_json_atomic(self.meta_path, self._meta)

    def _chunk_arrays(self) -> dict[str, np.ndarray]:
        samples = self._samples
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "units_json": np.asarray(
                json.dumps(self.scene.units, sort_keys=True), dtype=np.str_
            ),
            "timestamps": np.asarray([sample.timestamp for sample in samples], dtype=np.float64),
            "camera_T_table": np.stack(
                [np.asarray(sample.camera_T_table, dtype=np.float64) for sample in samples], axis=0
            ),
            "table_valid": np.asarray([sample.table_valid for sample in samples], dtype=np.bool_),
            "image_size": np.asarray([sample.image_size for sample in samples], dtype=np.int32),
            "table_marker_corners_px": np.stack(
                [
                    np.full((4, 2), np.nan, dtype=np.float64)
                    if sample.table_marker_corners_px is None
                    else np.asarray(sample.table_marker_corners_px, dtype=np.float64).reshape(4, 2)
                    for sample in samples
                ],
                axis=0,
            ),
        }
        for hand_id in self.scene.hand_ids:
            hands = [sample.hands[hand_id] for sample in samples]
            prefix = f"hands/{hand_id}/"
            arrays.update(
                {
                    prefix + "pose_timestamps": np.asarray(
                        [hand.pose_timestamp for hand in hands], dtype=np.float64
                    ),
                    prefix + "joint_timestamps": np.asarray(
                        [hand.joint_timestamp for hand in hands], dtype=np.float64
                    ),
                    prefix + "transform_table_hand": np.stack(
                        [np.asarray(hand.transform_table_hand, dtype=np.float64) for hand in hands],
                        axis=0,
                    ),
                    prefix + "joint_angles_raw": np.stack(
                        [np.asarray(hand.joint_angles_raw, dtype=np.float64) for hand in hands],
                        axis=0,
                    ),
                    prefix + "joint_angles_dexalign": np.stack(
                        [np.asarray(hand.joint_angles_dexalign, dtype=np.float64) for hand in hands],
                        axis=0,
                    ),
                    prefix + "joint_angles": np.stack(
                        [np.asarray(hand.joint_angles, dtype=np.float64) for hand in hands], axis=0
                    ),
                    prefix + "pose_valid": np.asarray(
                        [hand.pose_valid for hand in hands], dtype=np.bool_
                    ),
                    prefix + "joints_valid": np.asarray(
                        [hand.joints_valid for hand in hands], dtype=np.bool_
                    ),
                    prefix + "joint_age_sec": np.asarray(
                        [hand.joint_age_sec for hand in hands], dtype=np.float64
                    ),
                    prefix + "marker_ids_json": np.asarray(
                        [json.dumps(list(hand.marker_ids)) for hand in hands], dtype=np.str_
                    ),
                    prefix + "reprojection_error_px": np.asarray(
                        [
                            np.nan
                            if hand.reprojection_error_px is None
                            else hand.reprojection_error_px
                            for hand in hands
                        ],
                        dtype=np.float64,
                    ),
                }
            )
        return arrays

    def _flush_chunk(self) -> None:
        if not self._samples:
            return
        chunk_path = self.chunks_dir / f"{self._chunk_index:06d}.npz"
        arrays = self._chunk_arrays()
        with NamedTemporaryFile(
            prefix=f"{chunk_path.stem}.",
            suffix=".npz",
            dir=self.chunks_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(temporary, **arrays)
            with np.load(temporary, allow_pickle=False) as payload:
                if int(payload["timestamps"].shape[0]) != len(self._samples):
                    raise ValueError(f"Incomplete DexSlide chunk write: {temporary}")
            temporary.replace(chunk_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        count = len(self._samples)
        self._meta["chunks"].append(
            {
                "path": str(chunk_path.relative_to(self.session_dir)),
                "sample_count": count,
                "sha256": _sha256(chunk_path),
                "units": dict(self.scene.units),
            }
        )
        self._meta["sample_count"] = int(self._meta["sample_count"]) + count
        self._samples.clear()
        self._chunk_index += 1
        _write_json_atomic(self.meta_path, self._meta)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._flush_chunk()
        finally:
            self._meta["closed_at_unix"] = time.time()
            _write_json_atomic(self.meta_path, self._meta)
            self._closed = True


class DexSlideDatasetReader:
    """Read and validate a chunked DexSlide recording."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir).expanduser().resolve()
        meta_path = self.session_dir / "session_meta.json"
        with meta_path.open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        if int(self.meta.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported DexSlide dataset schema: {meta_path}")
        self.units = dict(self.meta["units"])
        self.hand_ids = tuple(str(value) for value in self.meta["hand_ids"])

    def iter_chunks(self) -> Iterator[dict[str, np.ndarray]]:
        for chunk in self.meta.get("chunks", []):
            chunk_path = self.session_dir / str(chunk["path"])
            if _sha256(chunk_path) != str(chunk["sha256"]):
                raise ValueError(f"DexSlide chunk hash mismatch: {chunk_path}")
            if dict(chunk.get("units", {})) != self.units:
                raise ValueError(f"DexSlide chunk units mismatch: {chunk_path}")
            with np.load(chunk_path, allow_pickle=False) as payload:
                arrays = {key: np.asarray(payload[key]) for key in payload.files}
            if int(arrays["schema_version"]) != int(self.meta["schema_version"]):
                raise ValueError(f"DexSlide chunk schema mismatch: {chunk_path}")
            if json.loads(str(arrays["units_json"])) != self.units:
                raise ValueError(f"DexSlide chunk embedded units mismatch: {chunk_path}")
            yield arrays

    def iter_samples(self) -> Iterator[DexSlideSceneSample]:
        joint_unit = str(self.units["joint_angles"])
        for arrays in self.iter_chunks():
            count = int(arrays["timestamps"].shape[0])
            for index in range(count):
                hands: dict[str, DexSlideHandSample] = {}
                for hand_id in self.hand_ids:
                    prefix = f"hands/{hand_id}/"
                    reprojection = float(arrays[prefix + "reprojection_error_px"][index])
                    hands[hand_id] = DexSlideHandSample(
                        pose_timestamp=float(arrays[prefix + "pose_timestamps"][index]),
                        joint_timestamp=float(arrays[prefix + "joint_timestamps"][index]),
                        transform_table_hand=np.asarray(
                            arrays[prefix + "transform_table_hand"][index], dtype=np.float64
                        ),
                        joint_angles_raw=np.asarray(
                            arrays[prefix + "joint_angles_raw"][index], dtype=np.float64
                        ),
                        joint_angles_dexalign=np.asarray(
                            arrays[prefix + "joint_angles_dexalign"][index], dtype=np.float64
                        ),
                        joint_angles=np.asarray(
                            arrays[prefix + "joint_angles"][index], dtype=np.float64
                        ),
                        joint_unit=joint_unit,
                        pose_valid=bool(arrays[prefix + "pose_valid"][index]),
                        joints_valid=bool(arrays[prefix + "joints_valid"][index]),
                        joint_age_sec=float(arrays[prefix + "joint_age_sec"][index]),
                        marker_ids=tuple(
                            int(value)
                            for value in json.loads(str(arrays[prefix + "marker_ids_json"][index]))
                        ),
                        reprojection_error_px=(
                            None if np.isnan(reprojection) else reprojection
                        ),
                    )
                corners = np.asarray(arrays["table_marker_corners_px"][index], dtype=np.float64)
                yield DexSlideSceneSample(
                    timestamp=float(arrays["timestamps"][index]),
                    camera_T_table=np.asarray(arrays["camera_T_table"][index], dtype=np.float64),
                    table_valid=bool(arrays["table_valid"][index]),
                    image_size=tuple(int(value) for value in arrays["image_size"][index]),
                    hands=hands,
                    table_marker_id=int(
                        self.meta.get("table_marker_id", 0)
                    ),
                    table_marker_corners_px=None if np.isnan(corners).all() else corners,
                    joint_unit=joint_unit,
                )


__all__ = ["DexSlideDatasetReader", "DexSlideRecorder"]
