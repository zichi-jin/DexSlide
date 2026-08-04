"""Direct ArUco world-pose tracking against a fixed reference marker."""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import yaml

from dexslide.kinematics.transforms import (
    invert_transform,
    rotmat_to_quaternion_xyzw,
    rvec_tvec_to_transform,
)
from dexslide.vision.aruco_pose_tracker import (
    _convert_fisheye_intrinsics_resolution,
    _detect_localize_aruco_tags,
    _parse_aruco_config,
    _parse_capture_source,
    _parse_fisheye_intrinsics,
)


def _normalize_target_marker_ids(
    target_marker_ids: Sequence[int] | None,
    table_marker_id: int,
) -> Optional[list[int]]:
    if target_marker_ids is None:
        return None

    out: list[int] = []
    seen: set[int] = set()
    for raw_marker_id in target_marker_ids:
        marker_id = int(raw_marker_id)
        if marker_id == int(table_marker_id) or marker_id in seen:
            continue
        seen.add(marker_id)
        out.append(marker_id)
    return sorted(out)


_transform_from_rvec_tvec = rvec_tvec_to_transform
_invert_transform = invert_transform


def _relative_transform_from_camera_poses(
    t_camera_reference: np.ndarray,
    t_camera_target: np.ndarray,
) -> np.ndarray:
    return _invert_transform(t_camera_reference) @ t_camera_target


def _pose_dict_from_transform(transform: np.ndarray) -> dict[str, Any]:
    quat = rotmat_to_quaternion_xyzw(np.asarray(transform[:3, :3], dtype=np.float64))
    pos = np.asarray(transform[:3, 3], dtype=np.float64)
    return {
        "position_m": [float(x) for x in pos],
        "quaternion_xyzw": [float(x) for x in quat],
        "matrix": np.asarray(transform, dtype=np.float64).tolist(),
    }


def _detect_relevant_aruco_tags(
    *,
    frame_bgr: np.ndarray,
    intr: dict[str, np.ndarray],
    table_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    table_marker_id: int,
    target_marker_ids: Sequence[int] | None,
    same_cfg: bool,
    refine_subpix: bool,
    motion_tolerant: bool,
    corner_refine_mode: str | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    combined: dict[int, dict[str, np.ndarray]] = {}

    if same_cfg:
        all_tags = _detect_localize_aruco_tags(
            img_bgr=frame_bgr,
            aruco_dict=table_cfg["aruco_dict"],
            marker_size_map=table_cfg["marker_size_map"],
            fisheye_intr_dict=intr,
            refine_subpix=refine_subpix,
            motion_tolerant=motion_tolerant,
            corner_refine_mode=corner_refine_mode,
        )
        for marker_id, tag in all_tags.items():
            if marker_id == table_marker_id:
                combined[int(marker_id)] = tag
                continue
            if target_marker_ids is None or marker_id in target_marker_ids:
                combined[int(marker_id)] = tag
        return combined

    table_tags = _detect_localize_aruco_tags(
        img_bgr=frame_bgr,
        aruco_dict=table_cfg["aruco_dict"],
        marker_size_map=table_cfg["marker_size_map"],
        fisheye_intr_dict=intr,
        refine_subpix=refine_subpix,
        motion_tolerant=motion_tolerant,
        corner_refine_mode=corner_refine_mode,
    )
    if table_marker_id in table_tags:
        combined[int(table_marker_id)] = table_tags[int(table_marker_id)]

    target_tags = _detect_localize_aruco_tags(
        img_bgr=frame_bgr,
        aruco_dict=target_cfg["aruco_dict"],
        marker_size_map=target_cfg["marker_size_map"],
        fisheye_intr_dict=intr,
        refine_subpix=refine_subpix,
        motion_tolerant=motion_tolerant,
        corner_refine_mode=corner_refine_mode,
    )
    for marker_id, tag in target_tags.items():
        if marker_id == table_marker_id:
            continue
        if target_marker_ids is not None and marker_id not in target_marker_ids:
            continue
        combined[int(marker_id)] = tag
    return combined


def _build_direct_aruco_frame_result(
    *,
    frame_idx: int,
    image_size: Sequence[int],
    table_marker_id: int,
    target_marker_ids: Sequence[int] | None,
    tag_dict: dict[int, dict[str, np.ndarray]],
    time_wall: float | None = None,
) -> dict[str, Any]:
    detected_ids = sorted(int(marker_id) for marker_id in tag_dict.keys())

    if target_marker_ids is None:
        current_target_ids = [marker_id for marker_id in detected_ids if marker_id != int(table_marker_id)]
    else:
        current_target_ids = [int(marker_id) for marker_id in target_marker_ids]

    table_tag = tag_dict.get(int(table_marker_id))
    table_detected = table_tag is not None
    table_in_camera = None
    camera_in_table = None
    t_camera_table = None
    if table_detected:
        t_camera_table = _transform_from_rvec_tvec(table_tag["rvec"], table_tag["tvec"])
        table_in_camera = _pose_dict_from_transform(t_camera_table)
        camera_in_table = _pose_dict_from_transform(_invert_transform(t_camera_table))

    targets: dict[str, dict[str, Any]] = {}
    n_world_targets = 0
    for target_marker_id in current_target_ids:
        target_tag = tag_dict.get(int(target_marker_id))
        if target_tag is None:
            targets[str(target_marker_id)] = {
                "id": int(target_marker_id),
                "detected": False,
                "target_in_camera": None,
                "target_in_table": None,
                "corners": None,
                "undistorted_corners": None,
                "marker_size_m": None,
            }
            continue

        t_camera_target = _transform_from_rvec_tvec(target_tag["rvec"], target_tag["tvec"])
        target_in_table = None
        if t_camera_table is not None:
            target_in_table = _pose_dict_from_transform(
                _relative_transform_from_camera_poses(t_camera_table, t_camera_target)
            )
            n_world_targets += 1
        targets[str(target_marker_id)] = {
            "id": int(target_marker_id),
            "detected": True,
            "target_in_camera": _pose_dict_from_transform(t_camera_target),
            "target_in_table": target_in_table,
            "corners": np.asarray(target_tag["corners"], dtype=np.float64).reshape(4, 2).tolist(),
            "undistorted_corners": np.asarray(
                target_tag.get("undistorted_corners", target_tag["corners"]),
                dtype=np.float64,
            ).reshape(4, 2).tolist(),
            "marker_size_m": float(target_tag.get("marker_size_m", 0.0)),
            "pose_candidates": [
                {
                    "rvec": np.asarray(candidate["rvec"], dtype=np.float64).reshape(3).tolist(),
                    "tvec": np.asarray(candidate["tvec"], dtype=np.float64).reshape(3).tolist(),
                    "reprojection_error_px": float(candidate.get("reprojection_error_px", 0.0)),
                }
                for candidate in target_tag.get("pose_candidates", [])
                if isinstance(candidate, dict)
                and "rvec" in candidate
                and "tvec" in candidate
            ],
        }

    return {
        "frame_idx": int(frame_idx),
        "time_wall": float(time.time() if time_wall is None else time_wall),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "table_marker_id": int(table_marker_id),
        "configured_target_marker_ids": (
            None if target_marker_ids is None else [int(x) for x in target_marker_ids]
        ),
        "detected_ids": detected_ids,
        "table_detected": bool(table_detected),
        "table_in_camera": table_in_camera,
        "camera_in_table": camera_in_table,
        "targets": targets,
        "n_world_targets": int(n_world_targets),
    }


class DirectArucoTracker:
    """Track target ArUco markers in the frame of a fixed reference marker."""

    def __init__(
        self,
        source: str | int,
        camera_intrinsics: str | Path,
        table_aruco_yaml: str | Path,
        table_marker_id: int,
        target_marker_ids: Sequence[int] | None = None,
        target_aruco_yaml: str | Path | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        buffer_size: int = 2,
        num_workers: int = 1,
        refine_subpix: bool = True,
        motion_tolerant: bool = True,
        corner_refine_mode: str | None = None,
    ):
        self.source = source
        self.camera_intrinsics = Path(camera_intrinsics)
        self.table_aruco_yaml = Path(table_aruco_yaml)
        self.target_aruco_yaml = (
            Path(target_aruco_yaml) if target_aruco_yaml is not None else self.table_aruco_yaml
        )
        self.table_marker_id = int(table_marker_id)
        self.target_marker_ids = _normalize_target_marker_ids(target_marker_ids, self.table_marker_id)
        if target_marker_ids is not None and not self.target_marker_ids:
            raise ValueError("target_marker_ids must contain at least one id different from table_marker_id")
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = int(buffer_size)
        self.num_workers = max(1, int(num_workers))
        self.refine_subpix = bool(refine_subpix)
        self.motion_tolerant = bool(motion_tolerant)
        self.corner_refine_mode = None if corner_refine_mode is None else str(corner_refine_mode)

        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.latest_result: dict[str, Any] | None = None
        self.latest_error: str | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def snapshot(self) -> dict[str, Any] | None:
        with self.lock:
            return None if self.latest_result is None else copy.deepcopy(self.latest_result)

    def error(self) -> str | None:
        with self.lock:
            return self.latest_error

    def _load_aruco_cfg(self, yaml_path: Path) -> dict[str, Any]:
        with yaml_path.open("r", encoding="utf-8") as handle:
            return _parse_aruco_config(yaml.safe_load(handle))

    def _worker(self) -> None:
        try:
            cv2.setNumThreads(self.num_workers)

            table_cfg = self._load_aruco_cfg(self.table_aruco_yaml)
            target_cfg = table_cfg if self.table_aruco_yaml.resolve() == self.target_aruco_yaml.resolve() else self._load_aruco_cfg(self.target_aruco_yaml)
            with self.camera_intrinsics.open("r", encoding="utf-8") as handle:
                raw_intr = _parse_fisheye_intrinsics(json.load(handle))

            cap_source = _parse_capture_source(self.source)
            if isinstance(cap_source, str) and cap_source.startswith("/dev/"):
                cap = cv2.VideoCapture(cap_source, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(cap_source)

            if self.width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
            if self.height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
            if self.fps is not None:
                cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.buffer_size))

            if not cap.isOpened():
                raise RuntimeError(f"Failed to open capture source: {self.source}")

            intr: dict[str, np.ndarray] | None = None
            intr_resolution: tuple[int, int] | None = None
            frame_idx = 0
            try:
                while self.running:
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        time.sleep(0.01)
                        continue

                    height, width = frame_bgr.shape[:2]
                    resolution = (int(width), int(height))
                    if intr is None or intr_resolution != resolution:
                        intr = _convert_fisheye_intrinsics_resolution(raw_intr, target_resolution=resolution)
                        intr_resolution = resolution

                    assert intr is not None
                    tag_dict = _detect_relevant_aruco_tags(
                        frame_bgr=frame_bgr,
                        intr=intr,
                        table_cfg=table_cfg,
                        target_cfg=target_cfg,
                        table_marker_id=self.table_marker_id,
                        target_marker_ids=self.target_marker_ids,
                        same_cfg=self.table_aruco_yaml.resolve() == self.target_aruco_yaml.resolve(),
                        refine_subpix=self.refine_subpix,
                        motion_tolerant=self.motion_tolerant,
                        corner_refine_mode=self.corner_refine_mode,
                    )
                    frame_result = _build_direct_aruco_frame_result(
                        frame_idx=frame_idx,
                        image_size=[width, height],
                        table_marker_id=self.table_marker_id,
                        target_marker_ids=self.target_marker_ids,
                        tag_dict=tag_dict,
                    )

                    with self.lock:
                        self.latest_result = frame_result
                        self.latest_error = None
                    frame_idx += 1
            finally:
                cap.release()
        except Exception as exc:  # pragma: no cover - runtime path
            with self.lock:
                self.latest_error = f"{type(exc).__name__}: {exc}"
            print(f"❌ DirectArucoTracker stopped with error: {type(exc).__name__}: {exc}")
        finally:
            self.running = False
