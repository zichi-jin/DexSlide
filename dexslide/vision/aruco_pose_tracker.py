"""Realtime ArUco pose tracker with optional per-group position fusion."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _parse_capture_source(source: str | int) -> int | str:
    src = str(source).strip()
    if src.lstrip("-").isdigit():
        return int(src)
    return os.path.expanduser(src)


def _parse_fisheye_intrinsics(json_data: dict) -> dict[str, np.ndarray]:
    if "K" in json_data and "D" in json_data:
        k = np.asarray(json_data["K"], dtype=np.float64).reshape(3, 3)
        d = np.asarray(json_data["D"], dtype=np.float64).reshape(-1, 1)
        if "DIM" in json_data:
            dim = np.asarray(json_data["DIM"], dtype=np.int64).reshape(2)
            w, h = int(dim[0]), int(dim[1])
        else:
            w = int(json_data.get("image_width", 0))
            h = int(json_data.get("image_height", 0))
        return {"DIM": np.array([w, h], dtype=np.int64), "K": k, "D": d}

    if "intrinsics" not in json_data:
        raise ValueError("Invalid intrinsics json: expect OpenCamera style with key 'intrinsics'.")

    intr_data = json_data["intrinsics"]
    h = int(json_data["image_height"])
    w = int(json_data["image_width"])
    f = float(intr_data["focal_length"])
    px = float(intr_data["principal_pt_x"])
    py = float(intr_data["principal_pt_y"])
    kb8 = [
        float(intr_data["radial_distortion_1"]),
        float(intr_data["radial_distortion_2"]),
        float(intr_data["radial_distortion_3"]),
        float(intr_data["radial_distortion_4"]),
    ]
    return {
        "DIM": np.array([w, h], dtype=np.int64),
        "K": np.array([[f, 0.0, px], [0.0, f, py], [0.0, 0.0, 1.0]], dtype=np.float64),
        "D": np.array([kb8], dtype=np.float64).T,
    }


def _convert_fisheye_intrinsics_resolution(
    opencv_intr_dict: dict[str, np.ndarray],
    target_resolution: tuple[int, int],
) -> dict[str, np.ndarray]:
    iw, ih = opencv_intr_dict["DIM"]
    if int(iw) <= 0 or int(ih) <= 0:
        out = copy.deepcopy(opencv_intr_dict)
        out["DIM"] = np.array([target_resolution[0], target_resolution[1]], dtype=np.int64)
        return out

    iK = opencv_intr_dict["K"]
    ifx = iK[0, 0]
    ify = iK[1, 1]
    ipx = iK[0, 2]
    ipy = iK[1, 2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2.0)) / ih * oh + (ow / 2.0)
    opy = ipy / ih * oh
    oK = np.array([[ofx, 0.0, opx], [0.0, ofy, opy], [0.0, 0.0, 1.0]], dtype=np.float64)

    out = copy.deepcopy(opencv_intr_dict)
    out["DIM"] = np.array([ow, oh], dtype=np.int64)
    out["K"] = oK
    return out


def _parse_aruco_config(aruco_config_dict: dict[str, Any]) -> dict[str, Any]:
    aruco_dict_cfg = aruco_config_dict.get("aruco_dict", {})
    predefined = aruco_dict_cfg.get("predefined")
    if predefined is None:
        raise ValueError("aruco_yaml missing 'aruco_dict.predefined'")
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, str(predefined)))

    n_markers = len(aruco_dict.bytesList)
    marker_size_map_raw = aruco_config_dict.get("marker_size_map", {})
    default_size = marker_size_map_raw.get("default", None)

    marker_size_map: dict[int, float] = {}
    for marker_id in range(n_markers):
        size = default_size
        if marker_id in marker_size_map_raw:
            size = marker_size_map_raw[marker_id]
        elif str(marker_id) in marker_size_map_raw:
            size = marker_size_map_raw[str(marker_id)]
        if size is not None:
            marker_size_map[marker_id] = float(size)

    return {"aruco_dict": aruco_dict, "marker_size_map": marker_size_map}


def _detect_localize_aruco_tags(
    img_bgr: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    marker_size_map: dict[int, float],
    fisheye_intr_dict: dict[str, np.ndarray],
    refine_subpix: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    k = fisheye_intr_dict["K"]
    d = fisheye_intr_dict["D"]
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(image=img_bgr, dictionary=aruco_dict, parameters=param)
    if ids is None or len(corners) == 0:
        return {}

    tag_dict: dict[int, dict[str, np.ndarray]] = {}
    for this_id_arr, this_corners in zip(ids, corners):
        this_id = int(this_id_arr[0])
        if this_id not in marker_size_map:
            continue
        marker_size_m = marker_size_map[this_id]

        if d.size == 4:
            undistorted = cv2.fisheye.undistortPoints(this_corners, k, d, P=k)
        else:
            undistorted = cv2.undistortPoints(this_corners, k, d, P=k)

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            undistorted,
            marker_size_m,
            k,
            np.zeros((1, 5), dtype=np.float64),
        )
        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": this_corners.squeeze(),
        }
    return tag_dict


def _offset_along_marker_negative_z(
    rvec: np.ndarray,
    tvec: np.ndarray,
    offset_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rot_mat, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    z_axis_cam = rot_mat[:, 2]
    tvec_offset = tvec - offset_scale * z_axis_cam
    return tvec_offset, z_axis_cam


def _eval_and_fuse_positions(
    marker_entries: list[dict[str, Any]],
    merge_pos_threshold: float,
) -> dict[str, Any]:
    n = len(marker_entries)
    if n == 0:
        return {
            "within_threshold": False,
            "max_pairwise_dist": None,
            "fused_position": None,
            "rep_rvec": None,
            "rep_marker_id": None,
            "mode": "no_marker",
        }

    if n == 1:
        e = marker_entries[0]
        return {
            "within_threshold": True,
            "max_pairwise_dist": 0.0,
            "fused_position": e["tvec_offset"].copy(),
            "rep_rvec": e["rvec"].copy(),
            "rep_marker_id": e["id"],
            "mode": "single",
        }

    positions = np.stack([e["tvec_offset"] for e in marker_entries], axis=0)
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    max_pairwise_dist = float(np.max(dists))
    within_threshold = max_pairwise_dist <= merge_pos_threshold
    if within_threshold:
        rep = marker_entries[0]
        return {
            "within_threshold": True,
            "max_pairwise_dist": max_pairwise_dist,
            "fused_position": np.mean(positions, axis=0),
            "rep_rvec": rep["rvec"].copy(),
            "rep_marker_id": rep["id"],
            "mode": "mean_position_only",
        }
    return {
        "within_threshold": False,
        "max_pairwise_dist": max_pairwise_dist,
        "fused_position": None,
        "rep_rvec": None,
        "rep_marker_id": None,
        "mode": "disagree",
    }


def _to_float_list(v: np.ndarray | list[float]) -> list[float]:
    return [float(x) for x in np.asarray(v).reshape(-1)]


def _fusion_to_output_dict(fusion: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": fusion["mode"],
        "within_threshold": bool(fusion["within_threshold"]),
        "max_pairwise_dist": (
            None if fusion["max_pairwise_dist"] is None else float(fusion["max_pairwise_dist"])
        ),
        "rep_marker_id": (None if fusion["rep_marker_id"] is None else int(fusion["rep_marker_id"])),
        "fused_position": (
            None if fusion["fused_position"] is None else _to_float_list(fusion["fused_position"])
        ),
        "rep_rvec": (None if fusion["rep_rvec"] is None else _to_float_list(fusion["rep_rvec"])),
    }


def _parse_fusion_groups_config(path: str | Path) -> dict[str, list[int]]:
    cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
    if cfg is None:
        return {}

    groups_obj = cfg["groups"] if isinstance(cfg, dict) and ("groups" in cfg) else cfg

    groups: dict[str, list[int]] = {}
    if isinstance(groups_obj, dict):
        raw_items = list(groups_obj.items())
    elif isinstance(groups_obj, list):
        raw_items = []
        for i, item in enumerate(groups_obj):
            if not isinstance(item, dict) or ("name" not in item) or ("ids" not in item):
                raise ValueError(
                    f"Invalid fusion group entry index={i}, expected dict with keys 'name' and 'ids'."
                )
            raw_items.append((item["name"], item["ids"]))
    else:
        raise ValueError(
            "Invalid fusion groups config. Expected dict/list, e.g. {groups: {g1: [10,11], g2: [20,21]}}."
        )

    for raw_name, raw_ids in raw_items:
        group_name = str(raw_name)
        if isinstance(raw_ids, (int, np.integer)):
            ids = [int(raw_ids)]
        elif isinstance(raw_ids, (list, tuple, set)):
            ids = [int(x) for x in raw_ids]
        else:
            raise ValueError(
                f"Invalid ids for group '{group_name}', expected int/list/tuple/set, got {type(raw_ids)}"
            )
        ids = sorted(set(ids))
        if not ids:
            raise ValueError(f"Fusion group '{group_name}' has no marker ids.")
        groups[group_name] = ids
    return groups


class ArucoPoseTracker:
    """Background tracker that produces latest per-group fused marker pose snapshot."""

    def __init__(
        self,
        source: str | int,
        camera_intrinsics: str | Path,
        aruco_yaml: str | Path,
        offset_scale: float = 0.0,
        merge_pos_threshold: float = 0.03,
        fusion_groups_yaml: str | Path | None = None,
        warning_cooldown_sec: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        buffer_size: int = 2,
        num_workers: int = 1,
        refine_subpix: bool = True,
    ):
        self.source = source
        self.camera_intrinsics = Path(camera_intrinsics)
        self.aruco_yaml = Path(aruco_yaml)
        self.offset_scale = float(offset_scale)
        self.merge_pos_threshold = float(merge_pos_threshold)
        self.fusion_groups_yaml = Path(fusion_groups_yaml) if fusion_groups_yaml else None
        self.warning_cooldown_sec = float(warning_cooldown_sec)
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = int(buffer_size)
        self.num_workers = int(num_workers)
        self.refine_subpix = bool(refine_subpix)

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

    def _worker(self) -> None:
        try:
            cv2.setNumThreads(self.num_workers)

            aruco_cfg = _parse_aruco_config(yaml.safe_load(open(self.aruco_yaml, "r", encoding="utf-8")))
            aruco_dict = aruco_cfg["aruco_dict"]
            marker_size_map = aruco_cfg["marker_size_map"]
            raw_intr = _parse_fisheye_intrinsics(
                json.load(open(self.camera_intrinsics, "r", encoding="utf-8"))
            )

            fusion_groups: dict[str, list[int]] | None = None
            if self.fusion_groups_yaml is not None:
                fusion_groups = _parse_fusion_groups_config(self.fusion_groups_yaml)
                if not fusion_groups:
                    raise ValueError(f"No valid groups found in: {self.fusion_groups_yaml}")

            cap_source = _parse_capture_source(self.source)
            if isinstance(cap_source, str) and cap_source.startswith("/dev/video"):
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
                raise RuntimeError(f"Failed to open aruco source: {self.source}")

            intr = None
            frame_idx = 0
            last_warning_time_by_group: dict[str, float] = {}
            try:
                while self.running:
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        time.sleep(0.01)
                        continue

                    h, w = frame_bgr.shape[:2]
                    if intr is None:
                        intr = _convert_fisheye_intrinsics_resolution(raw_intr, target_resolution=(w, h))

                    tag_dict = _detect_localize_aruco_tags(
                        img_bgr=frame_bgr,
                        aruco_dict=aruco_dict,
                        marker_size_map=marker_size_map,
                        fisheye_intr_dict=intr,
                        refine_subpix=self.refine_subpix,
                    )

                    marker_entries = []
                    for marker_id in sorted(tag_dict.keys()):
                        tag = tag_dict[marker_id]
                        rvec = np.asarray(tag["rvec"], dtype=np.float64).reshape(3)
                        tvec = np.asarray(tag["tvec"], dtype=np.float64).reshape(3)
                        tvec_offset, z_axis_cam = _offset_along_marker_negative_z(
                            rvec=rvec, tvec=tvec, offset_scale=self.offset_scale
                        )
                        marker_entries.append(
                            {
                                "id": int(marker_id),
                                "rvec": rvec,
                                "tvec": tvec,
                                "tvec_offset": tvec_offset,
                                "z_axis_cam": z_axis_cam,
                            }
                        )

                    marker_entries_by_id = {m["id"]: m for m in marker_entries}
                    if fusion_groups is None:
                        group_entries_map: dict[str, list[dict[str, Any]]] = {"all": marker_entries}
                        group_config_ids: dict[str, list[int] | None] = {"all": None}
                    else:
                        group_entries_map = {}
                        group_config_ids = {}
                        for group_name, cfg_ids in fusion_groups.items():
                            group_config_ids[group_name] = cfg_ids
                            group_entries_map[group_name] = [
                                marker_entries_by_id[mid] for mid in cfg_ids if mid in marker_entries_by_id
                            ]

                    group_fusions: dict[str, dict[str, Any]] = {}
                    now_mono = time.monotonic()
                    for group_name, entries in group_entries_map.items():
                        fusion = _eval_and_fuse_positions(entries, self.merge_pos_threshold)
                        group_fusions[group_name] = fusion
                        if len(entries) > 1 and (not fusion["within_threshold"]):
                            last_t = last_warning_time_by_group.get(group_name, -1e9)
                            if now_mono - last_t >= self.warning_cooldown_sec:
                                ids = [m["id"] for m in entries]
                                print(
                                    f"⚠️ aruco group[{group_name}] disagree: "
                                    f"ids={ids}, max_pairwise_dist={fusion['max_pairwise_dist']:.4f}m, "
                                    f"threshold={self.merge_pos_threshold:.4f}m"
                                )
                                last_warning_time_by_group[group_name] = now_mono

                    frame_result: dict[str, Any] = {
                        "frame_idx": int(frame_idx),
                        "time_wall": float(time.time()),
                        "n_markers": int(len(marker_entries)),
                        "offset_scale": float(self.offset_scale),
                        "merge_pos_threshold": float(self.merge_pos_threshold),
                        "fusion_groups_enabled": bool(fusion_groups is not None),
                        "markers": [
                            {
                                "id": int(m["id"]),
                                "rvec": _to_float_list(m["rvec"]),
                                "tvec": _to_float_list(m["tvec"]),
                                "tvec_offset": _to_float_list(m["tvec_offset"]),
                            }
                            for m in marker_entries
                        ],
                        "groups": {},
                    }
                    for group_name, fusion in group_fusions.items():
                        detected_ids = [m["id"] for m in group_entries_map[group_name]]
                        frame_result["groups"][group_name] = {
                            "configured_ids": group_config_ids[group_name],
                            "detected_ids": detected_ids,
                            "n_detected": int(len(detected_ids)),
                            "fusion": _fusion_to_output_dict(fusion),
                        }
                    frame_result["fused"] = (
                        frame_result["groups"]["all"]["fusion"] if "all" in frame_result["groups"] else None
                    )

                    with self.lock:
                        self.latest_result = frame_result
                        self.latest_error = None
                    frame_idx += 1
            finally:
                cap.release()
        except Exception as ex:  # pragma: no cover - runtime path
            with self.lock:
                self.latest_error = f"{type(ex).__name__}: {ex}"
            print(f"❌ ArucoPoseTracker stopped with error: {type(ex).__name__}: {ex}")
        finally:
            self.running = False
