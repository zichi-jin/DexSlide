"""Vision backends that turn camera frames into scene pose frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from dexslide.kinematics.glove_pose_filter import (
    GlovePoseFilter,
    GlovePoseFilterConfig,
)
from dexslide.kinematics.transforms import rvec_tvec_to_transform
from dexslide.vision.aruco_pose_tracker import (
    _convert_fisheye_intrinsics_resolution,
    _detect_localize_aruco_tags,
    _parse_aruco_config,
    _parse_fisheye_intrinsics,
    stabilize_marker_pose,
)
from dexslide.vision.camera.stream import CameraStream
from dexslide.vision.direct_aruco_tracker import _build_direct_aruco_frame_result
from dexslide.vision.marker_body_pose_tracker import MarkerBodyPoseTracker
from dexslide.vision.types import VisionHandPose, VisionSceneFrame


def _nan_transform() -> np.ndarray:
    return np.full((4, 4), np.nan, dtype=np.float64)


class OpenCVSceneVision:
    """Estimate table and hand poses from frames supplied by ``CameraStream``."""

    def __init__(
        self,
        *,
        camera: CameraStream,
        intrinsics_file: Path,
        table_aruco_file: Path,
        table_marker_id: int,
        hands: Mapping[str, Any],
        num_workers: int,
        pose_solver: str,
        smoothing_alpha: float,
        pose_filter_enabled: bool,
        outlier_threshold_m: float,
        reprojection_error_threshold_px: float,
        detection_scale: float,
    ) -> None:
        self._camera = camera
        self._intrinsics_file = intrinsics_file
        self._table_aruco_file = table_aruco_file
        self._table_marker_id = int(table_marker_id)
        self._hands = dict(hands)
        self._num_workers = max(1, int(num_workers))
        self._detection_scale = float(detection_scale)
        if not 0.1 <= self._detection_scale <= 1.0:
            raise ValueError("aruco_detection_scale must be between 0.1 and 1.0")
        self._raw_intrinsics: dict[str, np.ndarray] | None = None
        self._intrinsics: dict[str, np.ndarray] | None = None
        self._intrinsics_resolution: tuple[int, int] | None = None
        self._detection_intrinsics: dict[str, np.ndarray] | None = None
        self._detection_resolution: tuple[int, int] | None = None
        self._table_cfg: dict[str, Any] | None = None
        self._table_dictionary_name = ""
        self._detector_groups: dict[str, dict[str, Any]] = {}
        self._pose_filter_enabled = bool(pose_filter_enabled)
        self._pose_trackers = {
            hand_id: MarkerBodyPoseTracker(
                runtime.marker_config,
                pose_solver=pose_solver,
                smoothing_alpha=(smoothing_alpha if self._pose_filter_enabled else 1.0),
                outlier_threshold_m=outlier_threshold_m,
                reprojection_error_threshold_px=reprojection_error_threshold_px,
            )
            for hand_id, runtime in self._hands.items()
        }
        hand_filter_config = GlovePoseFilterConfig(
            position_time_constant_s=0.05,
            rotation_time_constant_s=0.1,
        )
        self._hand_pose_filters = (
            {hand_id: GlovePoseFilter(hand_filter_config) for hand_id in self._hands}
            if self._pose_filter_enabled
            else {}
        )
        self._last_camera_T_table: np.ndarray | None = None
        self._table_pose_filter = (GlovePoseFilter() if self._pose_filter_enabled else None)

    @property
    def intrinsics(self) -> dict[str, np.ndarray] | None:
        if self._intrinsics is None:
            return None
        return {key: np.asarray(value).copy() for key, value in self._intrinsics.items()}

    def start(self) -> None:
        cv2.setNumThreads(self._num_workers)
        with self._intrinsics_file.open("r", encoding="utf-8") as handle:
            self._raw_intrinsics = _parse_fisheye_intrinsics(json.load(handle))
        with self._table_aruco_file.open("r", encoding="utf-8") as handle:
            table_document = yaml.safe_load(handle)
        if not isinstance(table_document, dict):
            raise ValueError(f"Invalid table ArUco config: {self._table_aruco_file}")
        self._table_cfg = _parse_aruco_config(table_document)
        self._table_dictionary_name = str(
            table_document.get("aruco_dict", {}).get("predefined", "")
        )
        self._build_detector_groups()
        self._camera.start()

    def _build_detector_groups(self) -> None:
        assert self._table_cfg is not None
        groups: dict[str, dict[str, Any]] = {
            self._table_dictionary_name: {
                "aruco_dict": self._table_cfg["aruco_dict"],
                "marker_size_map": dict(self._table_cfg["marker_size_map"]),
            }
        }
        for runtime in self._hands.values():
            config = runtime.marker_config
            group = groups.setdefault(
                config.aruco_predefined,
                {
                    "aruco_dict": cv2.aruco.getPredefinedDictionary(
                        getattr(cv2.aruco, config.aruco_predefined)
                    ),
                    "marker_size_map": {},
                },
            )
            for marker_id in config.marker_ids():
                group["marker_size_map"][int(marker_id)] = float(config.aruco_bound_size_m)
        self._detector_groups = groups

    def stop(self) -> None:
        self._camera.stop()
        for tracker in self._pose_trackers.values():
            tracker.reset()
        for pose_filter in self._hand_pose_filters.values():
            pose_filter.reset()
        self._last_camera_T_table = None
        if self._table_pose_filter is not None:
            self._table_pose_filter.reset()

    def read(self) -> VisionSceneFrame:
        if self._raw_intrinsics is None:
            raise RuntimeError("OpenCVSceneVision has not been started")
        timestamp, frame_bgr = self._camera.read()
        height, width = frame_bgr.shape[:2]
        resolution = (int(width), int(height))
        if self._intrinsics is None or self._intrinsics_resolution != resolution:
            self._intrinsics = _convert_fisheye_intrinsics_resolution(
                self._raw_intrinsics,
                target_resolution=resolution,
            )
            self._intrinsics_resolution = resolution
        detection_resolution = (
            max(2, int(round(width * self._detection_scale))),
            max(2, int(round(height * self._detection_scale))),
        )
        scale_x = width / detection_resolution[0]
        scale_y = height / detection_resolution[1]
        if (
            self._detection_intrinsics is None
            or self._detection_resolution != detection_resolution
        ):
            self._detection_intrinsics = _convert_fisheye_intrinsics_resolution(
                self._raw_intrinsics,
                target_resolution=detection_resolution,
            )
            self._detection_resolution = detection_resolution
        detection_frame = frame_bgr
        if detection_resolution != resolution:
            detection_frame = cv2.resize(
                frame_bgr,
                detection_resolution,
                interpolation=cv2.INTER_AREA,
            )

        detected_by_dictionary: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        for dictionary_name, detector_config in self._detector_groups.items():
            detected_by_dictionary[dictionary_name] = _detect_localize_aruco_tags(
                img_bgr=detection_frame,
                aruco_dict=detector_config["aruco_dict"],
                marker_size_map=detector_config["marker_size_map"],
                fisheye_intr_dict=self._detection_intrinsics,
                refine_subpix=False,
                motion_tolerant=False,
            )

        table_tags = detected_by_dictionary.get(self._table_dictionary_name, {})
        table_tag = table_tags.get(self._table_marker_id)
        if table_tag is not None:
            stable_transform = stabilize_marker_pose(
                table_tag,
                previous_transform=(
                    self._last_camera_T_table if self._pose_filter_enabled else None
                ),
                pose_filter=self._table_pose_filter,
                timestamp_s=timestamp,
            )
            if stable_transform is not None:
                self._last_camera_T_table = stable_transform.copy()
        table_valid = table_tag is not None
        camera_t_table = (
            rvec_tvec_to_transform(table_tag["rvec"], table_tag["tvec"])
            if table_tag is not None
            else _nan_transform()
        )
        table_corners = (
            None
            if table_tag is None
            else np.asarray(table_tag["corners"], dtype=np.float64).reshape(4, 2)
            * np.asarray([scale_x, scale_y], dtype=np.float64)
        )

        hand_poses: dict[str, VisionHandPose] = {}
        for hand_id, runtime in self._hands.items():
            hand_tags = detected_by_dictionary.get(runtime.marker_config.aruco_predefined, {})
            tag_dict: dict[int, dict[str, Any]] = {}
            if table_tag is not None:
                tag_dict[self._table_marker_id] = table_tag
            for marker_id in runtime.marker_config.marker_ids():
                if marker_id in hand_tags:
                    tag_dict[int(marker_id)] = hand_tags[int(marker_id)]
            frame_result = _build_direct_aruco_frame_result(
                frame_idx=0,
                image_size=detection_resolution,
                table_marker_id=self._table_marker_id,
                target_marker_ids=runtime.marker_config.marker_ids(),
                tag_dict=tag_dict,
                time_wall=timestamp,
            )
            tracker_result = self._pose_trackers[hand_id].update(
                frame_result=frame_result,
                camera_matrix=np.asarray(self._detection_intrinsics["K"], dtype=np.float64),
            )
            body_pose = (
                tracker_result.smoothed_pose
                if self._pose_filter_enabled
                else tracker_result.raw_pose
            )
            marker_corners_px = {
                int(marker_id): np.asarray(hand_tags[int(marker_id)]["corners"], dtype=np.float64).reshape(4, 2)
                * np.asarray([scale_x, scale_y], dtype=np.float64)
                for marker_id in runtime.marker_config.marker_ids()
                if int(marker_id) in hand_tags
            }
            if not table_valid or body_pose is None:
                pose_filter = self._hand_pose_filters.get(hand_id)
                if pose_filter is not None:
                    pose_filter.reset()
                hand_poses[hand_id] = VisionHandPose(
                    transform_table_hand=_nan_transform(),
                    valid=False,
                    marker_corners_px=marker_corners_px,
                )
                continue
            if self._pose_filter_enabled:
                filtered_body_result = self._hand_pose_filters[hand_id].update(
                    body_pose.transform_table_cube,
                    timestamp,
                )
                filtered_body_transform = filtered_body_result.transform_table_hand
            else:
                filtered_body_transform = np.asarray(
                    body_pose.transform_table_cube,
                    dtype=np.float64,
                ).reshape(4, 4)
            if filtered_body_transform is None:
                hand_poses[hand_id] = VisionHandPose(
                    transform_table_hand=_nan_transform(),
                    valid=False,
                    marker_corners_px=marker_corners_px,
                )
                continue
            transform_table_hand = (
                np.asarray(filtered_body_transform, dtype=np.float64).reshape(4, 4)
                @ runtime.marker_config.body_to_wrist_transform()
            )
            hand_poses[hand_id] = VisionHandPose(
                transform_table_hand=transform_table_hand,
                valid=True,
                transform_table_body=np.asarray(filtered_body_transform, dtype=np.float64).reshape(4, 4).copy(),
                marker_ids=tuple(int(value) for value in body_pose.source_marker_ids),
                marker_corners_px=marker_corners_px,
                reprojection_error_px=float(body_pose.mean_reprojection_error_px),
            )

        return VisionSceneFrame(
            timestamp=timestamp,
            frame_bgr=frame_bgr,
            image_size=resolution,
            camera_T_table=camera_t_table,
            table_valid=table_valid,
            table_marker_corners_px=table_corners,
            hands=hand_poses,
        )


def make_opencv_scene_vision(
    *,
    camera_config: Mapping[str, Any],
    table_aruco_file: Path,
    table_marker_id: int,
    hands: Mapping[str, Any],
    stream_config: Mapping[str, Any],
) -> OpenCVSceneVision:
    """Build the default vision backend from resolved scene configuration."""

    camera = CameraStream(
        source=camera_config["source"],
        width=int(camera_config["width"]),
        height=int(camera_config["height"]),
        fps=float(camera_config["fps"]),
        buffer_size=int(camera_config.get("buffer_size", 2)),
        fourcc=camera_config.get("fourcc"),
    )
    return OpenCVSceneVision(
        camera=camera,
        intrinsics_file=Path(camera_config["intrinsics_file"]),
        table_aruco_file=table_aruco_file,
        table_marker_id=table_marker_id,
        hands=hands,
        num_workers=int(camera_config.get("num_workers", 1)),
        pose_solver=str(stream_config.get("pose_solver", "joint_pnp")),
        smoothing_alpha=float(stream_config.get("pose_smoothing", 0.35)),
        pose_filter_enabled=bool(stream_config.get("pose_filter_enabled", True)),
        outlier_threshold_m=0.001 * float(stream_config.get("pose_outlier_threshold_mm", 20.0)),
        reprojection_error_threshold_px=float(
            stream_config.get("reprojection_error_threshold_px", 5.0)
        ),
        detection_scale=float(stream_config.get("aruco_detection_scale", 0.5)),
    )
