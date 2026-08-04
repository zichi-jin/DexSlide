"""Realtime multi-hand DexSlide scene API.

The scene owns acquisition only. Recording, visualization, and teleoperation are
independent consumers of the immutable samples emitted here.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np

from dexslide.communications import (
    hand_joint_communication,
    resolve_joint_port,
)
from dexslide.paths import DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE
from dexslide.serial_angles import AngleStreamReader, load_calibration, make_joint_order
from dexslide.vision.marker_body_model import HandCubeOverlayConfig
from dexslide.vision.marker_body_model_impl import marker_to_wrist_asset_transforms
from dexslide.vision.camera.config import resolve_camera_config
from dexslide.vision.scene_backend import make_opencv_scene_vision
from dexslide.vision.types import VisionHandPose, VisionSceneFrame


SCHEMA_VERSION = 1
JOINT_UNITS = {"deg", "rad"}
JOINT_MODES = {"raw", "dexalign"}


def _nan_transform() -> np.ndarray:
    return np.full((4, 4), np.nan, dtype=np.float64)

def _as_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _as_serializable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _as_serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _as_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_serializable(item) for item in value]
    return value


def _resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _required_file(payload: Mapping[str, Any], key: str, *, base_dir: Path, context: str) -> Path:
    raw_value = str(payload.get(key, "")).strip()
    if not raw_value:
        raise ValueError(f"Missing {context}.{key}")
    path = _resolve_path(raw_value, base_dir=base_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Configured file does not exist for {context}.{key}: {path}")
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_joint_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_json_object(path)
    scale = np.asarray(payload.get("joint_scale"), dtype=np.float64).reshape(-1)
    bias = np.asarray(payload.get("joint_bias_rad"), dtype=np.float64).reshape(-1)
    if scale.shape != (20,) or bias.shape != (20,):
        raise ValueError(
            f"DexAlign joint calibration must contain 20 scales and 20 joint_bias_rad values: {path}"
        )
    if not np.isfinite(scale).all() or not np.isfinite(bias).all():
        raise ValueError(f"DexAlign joint calibration contains non-finite values: {path}")
    return scale, bias


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class DexSlideHandSample:
    pose_timestamp: float
    joint_timestamp: float
    transform_table_hand: np.ndarray
    joint_angles_raw: np.ndarray
    joint_angles_dexalign: np.ndarray
    joint_angles: np.ndarray
    joint_unit: str
    pose_valid: bool
    joints_valid: bool
    joint_age_sec: float
    marker_ids: tuple[int, ...] = ()
    reprojection_error_px: float | None = None
    transform_table_body: np.ndarray | None = None
    marker_corners_px: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return bool(self.pose_valid and self.joints_valid)

    def to_dict(self) -> dict[str, Any]:
        return _as_serializable(
            {
                "pose_timestamp": self.pose_timestamp,
                "joint_timestamp": self.joint_timestamp,
                "transform_table_hand": self.transform_table_hand,
                "joint_angles_raw": self.joint_angles_raw,
                "joint_angles_dexalign": self.joint_angles_dexalign,
                "joint_angles": self.joint_angles,
                "joint_unit": self.joint_unit,
                "pose_valid": self.pose_valid,
                "joints_valid": self.joints_valid,
                "valid": self.valid,
                "joint_age_sec": self.joint_age_sec,
                "marker_ids": self.marker_ids,
                "transform_table_body": self.transform_table_body,
                "marker_corners_px": self.marker_corners_px,
                "reprojection_error_px": self.reprojection_error_px,
            }
        )


@dataclass(frozen=True)
class DexSlideSceneSample:
    timestamp: float
    camera_T_table: np.ndarray
    table_valid: bool
    image_size: tuple[int, int]
    hands: dict[str, DexSlideHandSample]
    table_marker_id: int
    table_marker_corners_px: np.ndarray | None = None
    schema_version: int = SCHEMA_VERSION
    joint_unit: str = "deg"

    @property
    def units(self) -> dict[str, str]:
        return {
            "joint_angles": self.joint_unit,
            "translation": "m",
            "timestamp": "s",
            "image_coordinates": "px",
            "reprojection_error": "px",
        }

    def to_dict(self) -> dict[str, Any]:
        return _as_serializable(
            {
                "schema_version": self.schema_version,
                "units": self.units,
                "timestamp": self.timestamp,
                "camera_T_table": self.camera_T_table,
                "table_valid": self.table_valid,
                "image_size": self.image_size,
                "table_marker_id": self.table_marker_id,
                "table_marker_corners_px": self.table_marker_corners_px,
                "hands": {hand_id: sample.to_dict() for hand_id, sample in self.hands.items()},
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


class JointReader(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def wait_for_first_sample(self, timeout_sec: float) -> None: ...
    def snapshot_nearest_rad20(self, timestamp: float) -> tuple[np.ndarray, float, str]: ...


class VisionBackend(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> VisionSceneFrame: ...


@dataclass(frozen=True)
class _HandRuntime:
    hand_id: str
    communication_hand: str
    skeleton_file: Path
    glove_calibration_file: Path
    tags_to_marker_file: Path
    marker_to_hand_file: Path
    joint_calibration_file: Path | None
    dexalign_session: Path | None
    max_sample_age_sec: float
    startup_timeout_sec: float
    joint_scale: np.ndarray
    joint_bias_rad: np.ndarray
    marker_config: HandCubeOverlayConfig



class DexSlideScene:
    """Own a shared camera plus one independent joint stream per configured hand."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        config_path: str | Path | None = None,
        joint_unit: str | None = None,
        joint_mode: str | None = None,
        pose_filter_enabled: bool | None = None,
        joint_readers: Mapping[str, JointReader] | None = None,
        vision_backend: VisionBackend | None = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.config_path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else (Path.cwd() / "dexslide_streaming.json").resolve()
        )
        self.base_dir = self.config_path.parent
        if int(self.config.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported DexSlide streaming schema_version: {self.config.get('schema_version')!r}"
            )

        defaults = self.config.get("stream", {})
        if not isinstance(defaults, dict):
            raise ValueError("stream must be an object in DexSlide streaming config")
        self.joint_unit = str(joint_unit or defaults.get("joint_unit", "deg")).lower()
        self.joint_mode = str(joint_mode or defaults.get("joint_mode", "dexalign")).lower()
        configured_pose_filter = defaults.get("pose_filter_enabled", True)
        if not isinstance(configured_pose_filter, bool):
            raise ValueError("stream.pose_filter_enabled must be a boolean")
        if pose_filter_enabled is not None and not isinstance(pose_filter_enabled, bool):
            raise ValueError("pose_filter_enabled override must be a boolean or None")
        self.pose_filter_enabled = (
            configured_pose_filter
            if pose_filter_enabled is None
            else pose_filter_enabled
        )
        if self.joint_unit not in JOINT_UNITS:
            raise ValueError(f"joint_unit must be one of {sorted(JOINT_UNITS)}")
        if self.joint_mode not in JOINT_MODES:
            raise ValueError(f"joint_mode must be one of {sorted(JOINT_MODES)}")

        communications_raw = str(
            self.config.get("communications_file", DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE)
        )
        self.communications_file = _resolve_path(communications_raw, base_dir=self.base_dir)
        if not self.communications_file.is_file():
            raise FileNotFoundError(
                f"DexSlide communications config not found: {self.communications_file}"
            )

        self._joint_order = make_joint_order()
        self.joint_order = tuple(str(item["id"]) for item in self._joint_order)
        self._hands = self._load_hands()
        self._validate_marker_conflicts()
        self._camera_config, self._world_config = self._load_scene_config()
        self._validate_table_marker_conflicts()
        self._update_effective_config()
        self._joint_readers = dict(joint_readers or self._make_joint_readers())
        missing_readers = sorted(set(self._hands) - set(self._joint_readers))
        if missing_readers:
            raise ValueError(f"Missing joint readers for configured hands: {missing_readers}")
        self._vision = vision_backend or self._make_vision_backend()
        self._latest_color_frame: np.ndarray | None = None
        self._started = False

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        joint_unit: str | None = None,
        joint_mode: str | None = None,
        pose_filter_enabled: bool | None = None,
        joint_readers: Mapping[str, JointReader] | None = None,
        vision_backend: VisionBackend | None = None,
    ) -> "DexSlideScene":
        config_path = Path(path).expanduser().resolve()
        config = _load_json_object(config_path)
        return cls(
            config,
            config_path=config_path,
            joint_unit=joint_unit,
            joint_mode=joint_mode,
            pose_filter_enabled=pose_filter_enabled,
            joint_readers=joint_readers,
            vision_backend=vision_backend,
        )

    def _load_hands(self) -> dict[str, _HandRuntime]:
        hands_payload = self.config.get("hands")
        if not isinstance(hands_payload, dict) or not hands_payload:
            raise ValueError("DexSlide streaming config must enable at least one hand")
        runtimes: dict[str, _HandRuntime] = {}
        for raw_hand_id, raw_payload in hands_payload.items():
            hand_id = str(raw_hand_id).strip()
            if not hand_id or not isinstance(raw_payload, dict):
                raise ValueError(f"Invalid hand config for {raw_hand_id!r}")
            context = f"hands.{hand_id}"
            communication_hand = str(raw_payload.get("communication_hand", hand_id)).strip().lower()
            skeleton_file = _required_file(
                raw_payload, "skeleton_file", base_dir=self.base_dir, context=context
            )
            glove_file = _required_file(
                raw_payload, "glove_calibration_file", base_dir=self.base_dir, context=context
            )
            tags_file = _required_file(
                raw_payload, "tags_to_marker_file", base_dir=self.base_dir, context=context
            )
            marker_to_hand_file = _required_file(
                raw_payload, "marker_to_hand_file", base_dir=self.base_dir, context=context
            )

            dexalign_raw = str(raw_payload.get("dexalign_session", "")).strip()
            dexalign_session = (
                None if not dexalign_raw else _resolve_path(dexalign_raw, base_dir=self.base_dir)
            )
            joint_calibration_raw = str(raw_payload.get("joint_calibration_file", "")).strip()
            joint_calibration_file = (
                None
                if not joint_calibration_raw
                else _resolve_path(joint_calibration_raw, base_dir=self.base_dir)
            )
            if joint_calibration_file is not None and not joint_calibration_file.is_file():
                raise FileNotFoundError(
                    f"Configured file does not exist for {context}.joint_calibration_file: "
                    f"{joint_calibration_file}"
                )
            if self.joint_mode == "dexalign" and joint_calibration_file is None:
                raise ValueError(
                    f"{context}.joint_calibration_file is required for joint_mode='dexalign'"
                )
            if dexalign_session is not None:
                if not dexalign_session.is_dir():
                    raise FileNotFoundError(f"DexAlign session does not exist: {dexalign_session}")
                session_assets = [skeleton_file, marker_to_hand_file]
                if joint_calibration_file is not None:
                    session_assets.append(joint_calibration_file)
                outside = [str(path) for path in session_assets if not _path_is_within(path, dexalign_session)]
                if outside:
                    raise ValueError(
                        f"{context} mixes assets outside declared DexAlign session "
                        f"{dexalign_session}: {outside}"
                    )

            if joint_calibration_file is None:
                scale = np.ones(20, dtype=np.float64)
                bias = np.zeros(20, dtype=np.float64)
            else:
                scale, bias = _load_joint_calibration(joint_calibration_file)

            marker_config = HandCubeOverlayConfig.load(tags_file)
            _initial, _result, active_marker_to_hand = marker_to_wrist_asset_transforms(
                marker_to_hand_file
            )
            if active_marker_to_hand is None:
                raise ValueError(
                    f"Marker-to-hand config has neither result nor initial_guess: {marker_to_hand_file}"
                )
            marker_config.set_body_to_wrist_transform(active_marker_to_hand)
            marker_config.hand = hand_id

            communication = hand_joint_communication(
                communication_hand,
                path=self.communications_file,
            )
            max_sample_age_sec = float(
                raw_payload.get("max_sample_age_sec", communication["max_sample_age_sec"])
            )
            startup_timeout_sec = float(
                raw_payload.get("startup_timeout_sec", communication["startup_timeout_sec"])
            )
            if max_sample_age_sec <= 0.0 or startup_timeout_sec <= 0.0:
                raise ValueError(
                    f"{context}.max_sample_age_sec and startup_timeout_sec must be positive"
                )
            runtimes[hand_id] = _HandRuntime(
                hand_id=hand_id,
                communication_hand=communication_hand,
                skeleton_file=skeleton_file,
                glove_calibration_file=glove_file,
                tags_to_marker_file=tags_file,
                marker_to_hand_file=marker_to_hand_file,
                joint_calibration_file=joint_calibration_file,
                dexalign_session=dexalign_session,
                max_sample_age_sec=max_sample_age_sec,
                startup_timeout_sec=startup_timeout_sec,
                joint_scale=scale,
                joint_bias_rad=bias,
                marker_config=marker_config,
            )
        return runtimes

    def _validate_marker_conflicts(self) -> None:
        marker_sets: dict[str, dict[int, str]] = {}
        for hand_id, runtime in self._hands.items():
            dictionary_markers = marker_sets.setdefault(runtime.marker_config.aruco_predefined, {})
            for marker_id in runtime.marker_config.marker_ids():
                other_hand = dictionary_markers.get(marker_id)
                if other_hand is not None:
                    raise ValueError(
                        "Hands using the same ArUco dictionary cannot share marker IDs: "
                        f"dictionary={runtime.marker_config.aruco_predefined}, marker_id={marker_id}, "
                        f"hands={other_hand},{hand_id}"
                    )
                dictionary_markers[marker_id] = hand_id

    def _load_scene_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        camera_payload = self.config.get("camera")
        world_payload = self.config.get("world")
        if not isinstance(camera_payload, dict) or not isinstance(world_payload, dict):
            raise ValueError("DexSlide streaming config requires camera and world objects")
        table_file = _required_file(
            world_payload,
            "table_aruco_file",
            base_dir=self.base_dir,
            context="world",
        )
        camera = resolve_camera_config(
            camera_payload,
            base_dir=self.base_dir,
            communications_file=self.communications_file,
        )
        world = dict(world_payload)
        world["table_aruco_file"] = table_file
        world["table_marker_id"] = int(world_payload.get("table_marker_id", 0))
        return camera, world

    def _validate_table_marker_conflicts(self) -> None:
        table_marker_id = int(self._world_config["table_marker_id"])
        conflicting_hands = [
            hand_id
            for hand_id, runtime in self._hands.items()
            if table_marker_id in runtime.marker_config.marker_ids()
        ]
        if conflicting_hands:
            raise ValueError(
                "The table marker ID cannot also be a hand marker ID in the shared scene: "
                f"table_marker_id={table_marker_id}, hands={conflicting_hands}"
            )

    def _make_joint_readers(self) -> dict[str, JointReader]:
        readers: dict[str, JointReader] = {}
        for hand_id, runtime in self._hands.items():
            communication = hand_joint_communication(
                runtime.communication_hand,
                path=self.communications_file,
            )
            calibration = load_calibration(runtime.glove_calibration_file, self._joint_order)
            readers[hand_id] = AngleStreamReader(
                port=resolve_joint_port(runtime.communication_hand, path=self.communications_file),
                baud=int(communication["baud"]),
                mode=str(communication["mode"]),
                joint_order=self._joint_order,
                calibration=calibration,
                buffer_size=int(self.config.get("joint_buffer_size", 512)),
            )
        return readers

    def _update_effective_config(self) -> None:
        """Persist the fully resolved runtime configuration for provenance snapshots."""

        self.config["communications_file"] = str(self.communications_file)
        stream = dict(self.config.get("stream", {}))
        stream["joint_unit"] = self.joint_unit
        stream["joint_mode"] = self.joint_mode
        stream["pose_filter_enabled"] = self.pose_filter_enabled
        self.config["stream"] = stream
        self.config["camera"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self._camera_config.items()
        }
        self.config["world"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self._world_config.items()
        }
        effective_hands: dict[str, dict[str, Any]] = {}
        for hand_id, runtime in self._hands.items():
            effective_hands[hand_id] = {
                "communication_hand": runtime.communication_hand,
                "skeleton_file": str(runtime.skeleton_file),
                "glove_calibration_file": str(runtime.glove_calibration_file),
                "tags_to_marker_file": str(runtime.tags_to_marker_file),
                "marker_to_hand_file": str(runtime.marker_to_hand_file),
                "joint_calibration_file": (
                    None
                    if runtime.joint_calibration_file is None
                    else str(runtime.joint_calibration_file)
                ),
                "dexalign_session": (
                    None if runtime.dexalign_session is None else str(runtime.dexalign_session)
                ),
                "max_sample_age_sec": runtime.max_sample_age_sec,
                "startup_timeout_sec": runtime.startup_timeout_sec,
            }
        self.config["hands"] = effective_hands

    def _make_vision_backend(self) -> VisionBackend:
        return make_opencv_scene_vision(
            camera_config=self._camera_config,
            table_aruco_file=self._world_config["table_aruco_file"],
            table_marker_id=self._world_config["table_marker_id"],
            hands=self._hands,
            stream_config=self.config.get("stream", {}),
        )

    @property
    def hand_ids(self) -> tuple[str, ...]:
        return tuple(self._hands.keys())

    @property
    def hand_skeleton_files(self) -> dict[str, Path]:
        """Resolved skeleton assets for visualization consumers."""
        return {hand_id: runtime.skeleton_file for hand_id, runtime in self._hands.items()}

    @property
    def hand_handedness(self) -> dict[str, str]:
        """Configured handedness used by each skeleton model."""
        return {
            hand_id: str(getattr(runtime.marker_config, "hand", hand_id))
            for hand_id, runtime in self._hands.items()
        }

    @property
    def table_marker_id(self) -> int:
        return int(self._world_config["table_marker_id"])

    @property
    def units(self) -> dict[str, str]:
        return {
            "joint_angles": self.joint_unit,
            "translation": "m",
            "timestamp": "s",
            "image_coordinates": "px",
            "reprojection_error": "px",
        }

    @property
    def effective_calibration(self) -> dict[str, dict[str, Any]]:
        return {
            hand_id: {
                "effective_T_marker_hand": runtime.marker_config.body_to_wrist_transform().tolist(),
                "effective_joint_scale": runtime.joint_scale.tolist(),
                "effective_joint_bias": (
                    np.rad2deg(runtime.joint_bias_rad).tolist()
                    if self.joint_unit == "deg"
                    else runtime.joint_bias_rad.tolist()
                ),
                "effective_joint_bias_rad": runtime.joint_bias_rad.tolist(),
                "joint_unit": self.joint_unit,
                "joint_order": list(self.joint_order),
                "handedness": hand_id,
            }
            for hand_id, runtime in self._hands.items()
        }

    @property
    def config_files(self) -> dict[str, Path]:
        files = {
            "streaming_config": self.config_path,
            "communications": self.communications_file,
            "camera_intrinsics": Path(self._camera_config["intrinsics_file"]),
            "table_aruco": Path(self._world_config["table_aruco_file"]),
        }
        for hand_id, runtime in self._hands.items():
            files.update(
                {
                    f"{hand_id}.skeleton": runtime.skeleton_file,
                    f"{hand_id}.glove_calibration": runtime.glove_calibration_file,
                    f"{hand_id}.tags_to_marker": runtime.tags_to_marker_file,
                    f"{hand_id}.marker_to_hand": runtime.marker_to_hand_file,
                }
            )
            if runtime.joint_calibration_file is not None:
                files[f"{hand_id}.joint_calibration"] = runtime.joint_calibration_file
        return files

    def start(self) -> "DexSlideScene":
        if self._started:
            return self
        started_readers: list[JointReader] = []
        try:
            for hand_id, reader in self._joint_readers.items():
                reader.start()
                started_readers.append(reader)
                reader.wait_for_first_sample(self._hands[hand_id].startup_timeout_sec)
            self._vision.start()
        except Exception:
            for reader in reversed(started_readers):
                reader.stop()
            raise
        self._started = True
        return self

    def close(self) -> None:
        if not self._started:
            return
        try:
            self._vision.stop()
        finally:
            for reader in self._joint_readers.values():
                reader.stop()
            self._started = False

    def __enter__(self) -> "DexSlideScene":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def latest_color_frame(self) -> np.ndarray | None:
        return None if self._latest_color_frame is None else self._latest_color_frame.copy()

    def latest_intrinsics(self) -> dict[str, np.ndarray] | None:
        intrinsics = getattr(self._vision, "intrinsics", None)
        if intrinsics is None:
            return None
        return intrinsics

    def sample(self) -> DexSlideSceneSample:
        if not self._started:
            self.start()
        vision = self._vision.read()
        self._latest_color_frame = (
            None if vision.frame_bgr is None else np.asarray(vision.frame_bgr).copy()
        )
        hands: dict[str, DexSlideHandSample] = {}
        for hand_id, runtime in self._hands.items():
            raw_rad, joint_timestamp, _raw_line = self._joint_readers[
                hand_id
            ].snapshot_nearest_rad20(vision.timestamp)
            raw_rad = np.asarray(raw_rad, dtype=np.float64).reshape(20)
            dexalign_rad = runtime.joint_scale * raw_rad + runtime.joint_bias_rad
            selected_rad = dexalign_rad if self.joint_mode == "dexalign" else raw_rad
            if self.joint_unit == "deg":
                raw_output = np.rad2deg(raw_rad)
                dexalign_output = np.rad2deg(dexalign_rad)
                selected_output = np.rad2deg(selected_rad)
            else:
                raw_output = raw_rad.copy()
                dexalign_output = dexalign_rad.copy()
                selected_output = selected_rad.copy()
            joint_age_sec = (
                float("inf")
                if joint_timestamp <= 0.0
                else abs(float(vision.timestamp) - float(joint_timestamp))
            )
            joints_valid = bool(
                joint_timestamp > 0.0 and joint_age_sec <= runtime.max_sample_age_sec
            )
            vision_hand = vision.hands.get(hand_id)
            pose_valid = bool(vision.table_valid and vision_hand is not None and vision_hand.valid)
            hands[hand_id] = DexSlideHandSample(
                pose_timestamp=float(vision.timestamp),
                joint_timestamp=float(joint_timestamp),
                transform_table_hand=(
                    _nan_transform()
                    if not pose_valid or vision_hand is None
                    else np.asarray(vision_hand.transform_table_hand, dtype=np.float64).reshape(4, 4).copy()
                ),
                joint_angles_raw=raw_output,
                joint_angles_dexalign=dexalign_output,
                joint_angles=selected_output,
                joint_unit=self.joint_unit,
                pose_valid=pose_valid,
                joints_valid=joints_valid,
                joint_age_sec=joint_age_sec,
                marker_ids=() if vision_hand is None else tuple(vision_hand.marker_ids),
                reprojection_error_px=(
                    None if vision_hand is None else vision_hand.reprojection_error_px
                ),
                transform_table_body=(
                    None
                    if vision_hand is None or vision_hand.transform_table_body is None
                    else np.asarray(vision_hand.transform_table_body, dtype=np.float64).reshape(4, 4).copy()
                ),
                marker_corners_px=(
                    {}
                    if vision_hand is None
                    else {
                        int(marker_id): np.asarray(corners, dtype=np.float64).reshape(4, 2).copy()
                        for marker_id, corners in vision_hand.marker_corners_px.items()
                    }
                ),
            )
        return DexSlideSceneSample(
            timestamp=float(vision.timestamp),
            camera_T_table=np.asarray(vision.camera_T_table, dtype=np.float64).reshape(4, 4).copy(),
            table_valid=bool(vision.table_valid),
            image_size=tuple(int(value) for value in vision.image_size),
            hands=hands,
            table_marker_id=self.table_marker_id,
            table_marker_corners_px=(
                None
                if vision.table_marker_corners_px is None
                else np.asarray(vision.table_marker_corners_px, dtype=np.float64).reshape(4, 2).copy()
            ),
            joint_unit=self.joint_unit,
        )

    def samples(self, rate_hz: float | None = None) -> Iterator[DexSlideSceneSample]:
        period = 0.0 if rate_hz is None or float(rate_hz) <= 0.0 else 1.0 / float(rate_hz)
        next_deadline = time.monotonic()
        while True:
            if period > 0.0:
                delay = next_deadline - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
            yield self.sample()
            if period > 0.0:
                next_deadline = max(next_deadline + period, time.monotonic())


__all__ = [
    "DexSlideHandSample",
    "DexSlideScene",
    "DexSlideSceneSample",
    "VisionHandPose",
    "VisionSceneFrame",
]
