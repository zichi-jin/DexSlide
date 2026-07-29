#!/usr/bin/env python3
"""DexSlide wrist-pose incremental teleop bridge for JAKA compliant servo mode."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROBOT_MANIP_ROOT = Path(__file__).resolve().parents[1]
DEXSLIDE_ROOT = ROBOT_MANIP_ROOT.parent
if str(DEXSLIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEXSLIDE_ROOT))

from dexslide.communications import (
    camera_communication,
    hand_joint_communication,
    resolve_camera_source,
    resolve_joint_port,
    resolve_realsense_serial,
)
from dexslide.camera_profile import load_camera_stream_profile
from dexslide.paths import (
    DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE,
    DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE,
    DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE,
    DEFAULT_LEFT_TAGS_TO_MARKER_FILE,
)
from dexslide.vision.aruco_pose_tracker import (
    _convert_fisheye_intrinsics_resolution,
    _parse_aruco_config,
    _parse_capture_source,
    _parse_fisheye_intrinsics,
)
from dexslide.kinematics.glove_pose_filter import GlovePoseFilter
from dexslide.kinematics.transforms import make_transform, transform_to_rvec_tvec
from dexslide.visualization.aruco_overlay import (
    draw_axes as _draw_axes_overlay,
    draw_hud as _draw_hud,
    draw_marker_outline as _draw_marker_outline,
    marker_id_to_color as _marker_id_to_color,
)
from dexslide.world_pose.direct_aruco_tracker import (
    _build_direct_aruco_frame_result,
    _detect_relevant_aruco_tags,
)
from dexslide.world_pose.hand_cube_overlay import (
    CubePoseEstimate,
    HandCubeOverlayConfig,
    marker_to_wrist_asset_transforms,
)
from dexslide.world_pose.marker_body_pose_tracker import MarkerBodyPoseTracker

JAKA_SDK_ROOT = ROBOT_MANIP_ROOT / "JAKA_control" / "JAKA_dependecies" / "x86_64-linux-gnu"
PAYLOAD_CONFIG_PATH = ROBOT_MANIP_ROOT / "JAKA_control" / "config" / "jaka_s5_orcahand_payload.json"
DEFAULT_MAPPING_FILE = DEXSLIDE_ROOT / "assets" / "teleop_robot_mappings" / "workspace_axis_mapping.json"
DEFAULT_DEXALIGN_SESSION_DIR = (
    DEXSLIDE_ROOT / "assets" / "calibration" / "dexalign" / "test_left_001"
)

DEFAULT_IP = "192.168.99.44"
DEFAULT_SENSOR_BRAND = 10
DEFAULT_FORCE_DAMPING_N = 1.0
DEFAULT_TORQUE_DAMPING_NM = 10.0
DEFAULT_TRANSLATION_REBOUND_FK = 0.5
DEFAULT_ROTATION_REBOUND_FK = 10.0
DEFAULT_SERVO_STEP_NUM = 1
DEFAULT_LOOP_HZ = 20.0
DEFAULT_PRINT_INTERVAL_S = 1.0
DEFAULT_SAFE_START_POSITION_TOL_MM = 5.0
DEFAULT_SAFE_START_ANGLE_TOL_DEG = 6.0
DEFAULT_SAFE_START_SPEED_MM_S = 60.0
DEFAULT_BODY_SMOOTHING = 0.65
DEFAULT_BODY_OUTLIER_THRESHOLD_MM = 20.0
DEFAULT_BODY_REPROJECTION_THRESHOLD_PX = 6.0
DEFAULT_TABLE_MARKER_ID = 0
DEFAULT_SHOW_OVERLAY = False
DEFAULT_OVERLAY_WINDOW_NAME = "DexSlide Teleop AR"
DEFAULT_TABLE_AXIS_LENGTH_M = 0.08
DEFAULT_BODY_AXIS_LENGTH_M = 0.05
DEFAULT_HAND_AXIS_LENGTH_M = 0.04
FT_CTRL_TOOL_FRAME = 0
SDK_POWER_ON_WAIT_S = 1.0
SDK_ENABLE_WAIT_S = 1.0
SDK_SENSOR_BRAND_WAIT_S = 1.0
SDK_COMPLIANCE_SWITCH_WAIT_S = 4.0
ZERO_SENSOR_SETTLE_S = 0.6


@dataclass(frozen=True)
class IdentifiedPayload:
    mass_kg: float
    centroid_mm: tuple[float, float, float]


@dataclass(frozen=True)
class DexAlignSessionPaths:
    session_dir: Path
    marker2hand_file: Path
    skeleton_file: Path
    joint_calibration_file: Path


@dataclass(frozen=True)
class WorkspaceAxisMapping:
    robot_from_table_transform: np.ndarray
    same_pose_fixed_rotation: np.ndarray
    mirror_translation_sign: np.ndarray
    mirror_rotation_sign: np.ndarray
    translation_scale: np.ndarray
    rotation_scale: np.ndarray
    safe_start_pose_mmdeg: tuple[float, float, float, float, float, float]
    translation_deadband_mm: float
    rotation_deadband_deg: float
    max_translation_step_mm: float
    max_rotation_step_deg: float
    frame_jump_reject_mm: float
    frame_jump_reject_deg: float
    anchor_stable_frames: int

    @property
    def robot_from_table_rotation(self) -> np.ndarray:
        return np.asarray(self.robot_from_table_transform[:3, :3], dtype=np.float64)


@dataclass(frozen=True)
class WristPoseSample:
    transform_table_hand: np.ndarray
    frame_idx: int
    time_wall: float
    source_marker_ids: tuple[int, ...]


@dataclass(frozen=True)
class TeleopAnchorState:
    glove_anchor_transform: np.ndarray
    robot_anchor_transform: np.ndarray
    previous_desired_robot_transform: np.ndarray
    anchor_frame_idx: int


def smooth_wrist_pose_sample(
    pose_filter: GlovePoseFilter,
    sample: WristPoseSample,
) -> WristPoseSample:
    filtered = pose_filter.update(sample.transform_table_hand, sample.time_wall)
    if filtered.transform_table_hand is None:
        raise RuntimeError("GlovePoseFilter unexpectedly returned no pose for a fresh observation")
    return WristPoseSample(
        transform_table_hand=np.asarray(filtered.transform_table_hand, dtype=np.float64).reshape(4, 4),
        frame_idx=sample.frame_idx,
        time_wall=sample.time_wall,
        source_marker_ids=sample.source_marker_ids,
    )


@dataclass(frozen=True)
class OrcaHandTeleopPlaceholder:
    glove_joint_port: str
    skeleton_file: Path
    joint_calibration_file: Path

    def describe(self) -> str:
        return (
            f"[orcahand] placeholder enabled: joint_port={self.glove_joint_port} "
            f"skeleton={self.skeleton_file} joint_calibration={self.joint_calibration_file} "
            "当前阶段不发手指指令，只保留后续接入位置"
        )

    def update(self) -> None:
        return None


def load_jkrc() -> object:
    lib_path = JAKA_SDK_ROOT / "libjakaAPI.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"libjakaAPI.so not found: {lib_path}")
    ctypes.CDLL(str(lib_path))
    if str(JAKA_SDK_ROOT) not in sys.path:
        sys.path.insert(0, str(JAKA_SDK_ROOT))
    import jkrc  # type: ignore

    return jkrc


def result_code(result: object) -> int:
    if isinstance(result, tuple):
        return int(result[0])
    if isinstance(result, int):
        return int(result)
    raise RuntimeError(f"未知 SDK 返回值：{result!r}")


def ensure_ok(result: object, action: str) -> None:
    code = result_code(result)
    if code != 0:
        raise RuntimeError(f"{action} 失败，返回码：{code}，原始结果：{result!r}")


def maybe_set_sensor_brand(robot: object, sensor_brand: int) -> None:
    if sensor_brand == DEFAULT_SENSOR_BRAND:
        return
    setter = getattr(robot, "set_torsenosr_brand", None)
    if setter is None:
        return
    ensure_ok(setter(sensor_brand), "set_torsenosr_brand")
    time.sleep(SDK_SENSOR_BRAND_WAIT_S)


def read_tcp_pose(robot: object) -> tuple[float, float, float, float, float, float]:
    for method_name in ("get_actual_tcp_position", "get_tcp_position"):
        method = getattr(robot, method_name, None)
        if method is None:
            continue
        result = method()
        ensure_ok(result, method_name)
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(f"{method_name} 返回值异常：{result!r}")
        return tuple(float(value) for value in result[1])
    raise AttributeError("robot does not provide get_actual_tcp_position/get_tcp_position")


def load_saved_payload_snapshot() -> IdentifiedPayload | None:
    if not PAYLOAD_CONFIG_PATH.exists():
        return None
    try:
        payload_file = json.loads(PAYLOAD_CONFIG_PATH.read_text(encoding="utf-8"))
        payload_raw = payload_file["payload"]
        mass_kg = float(payload_raw["mass_kg"])
        centroid_mm = tuple(float(value) for value in payload_raw["centroid_mm"])
        if payload_file.get("valid") is not True or mass_kg <= 0.0 or len(centroid_mm) != 3:
            return None
        return IdentifiedPayload(mass_kg=mass_kg, centroid_mm=centroid_mm)
    except Exception:
        return None


def write_payload(robot: object, payload: IdentifiedPayload) -> None:
    centroid = list(payload.centroid_mm)
    applied = False
    setter = getattr(robot, "set_torq_sensor_tool_payload", None)
    if setter is not None:
        ensure_ok(setter(payload.mass_kg, centroid), "set_torq_sensor_tool_payload")
        applied = True
    setter = getattr(robot, "set_payload", None)
    if setter is not None:
        ensure_ok(setter(payload.mass_kg, centroid), "set_payload")
        applied = True
    if not applied:
        raise AttributeError("robot does not provide set_torq_sensor_tool_payload/set_payload")


def rotation_y_deg(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def rotation_z_deg(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_dexalign_session_paths(session_dir: str | Path) -> DexAlignSessionPaths:
    root = Path(session_dir).expanduser().resolve()
    marker2hand_file = root / "optimized_marker2hand.json"
    skeleton_file = root / "optimized_skeleton.json"
    joint_calibration_file = root / "optimized_joint_calibration.json"
    missing = [path.name for path in (marker2hand_file, skeleton_file, joint_calibration_file) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"DexAlign session `{root}` 缺少文件：{', '.join(missing)}"
        )
    return DexAlignSessionPaths(
        session_dir=root,
        marker2hand_file=marker2hand_file,
        skeleton_file=skeleton_file,
        joint_calibration_file=joint_calibration_file,
    )


def load_workspace_axis_mapping(path: str | Path) -> WorkspaceAxisMapping:
    mapping_path = Path(path).expanduser().resolve()
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported workspace axis mapping schema_version in {mapping_path}")

    table_to_robot = payload.get("table_to_robot", {})
    rotation_matrix = np.asarray(table_to_robot.get("rotation_matrix"), dtype=np.float64).reshape(3, 3)
    translation_m = np.asarray(table_to_robot.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    robot_from_table = np.eye(4, dtype=np.float64)
    robot_from_table[:3, :3] = rotation_matrix
    robot_from_table[:3, 3] = translation_m

    fixed_rotation_payload = payload.get("glove_to_orca_same_pose_fixed_rotation", {})
    same_pose_fixed_rotation = np.asarray(
        fixed_rotation_payload.get("rotation_matrix"),
        dtype=np.float64,
    ).reshape(3, 3)

    mirror = payload.get("mirror", {})
    translation_sign = np.asarray(mirror.get("translation_sign"), dtype=np.float64).reshape(3)
    rotation_sign = np.asarray(mirror.get("rotation_vector_sign"), dtype=np.float64).reshape(3)
    translation_scale = np.asarray(payload.get("translation_scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
    rotation_scale = np.asarray(payload.get("rotation_scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(3)
    safe_start_pose_mmdeg = tuple(float(x) for x in payload.get("safe_start_pose_mmdeg", []))
    if len(safe_start_pose_mmdeg) != 6:
        raise ValueError(f"`safe_start_pose_mmdeg` must contain 6 values in {mapping_path}")

    return WorkspaceAxisMapping(
        robot_from_table_transform=robot_from_table,
        same_pose_fixed_rotation=same_pose_fixed_rotation,
        mirror_translation_sign=translation_sign,
        mirror_rotation_sign=rotation_sign,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        safe_start_pose_mmdeg=safe_start_pose_mmdeg,  # type: ignore[arg-type]
        translation_deadband_mm=float(payload.get("translation_deadband_mm", 0.8)),
        rotation_deadband_deg=float(payload.get("rotation_deadband_deg", 0.6)),
        max_translation_step_mm=float(payload.get("max_translation_step_mm", 3.0)),
        max_rotation_step_deg=float(payload.get("max_rotation_step_deg", 2.0)),
        frame_jump_reject_mm=float(payload.get("frame_jump_reject_mm", 25.0)),
        frame_jump_reject_deg=float(payload.get("frame_jump_reject_deg", 18.0)),
        anchor_stable_frames=max(1, int(payload.get("anchor_stable_frames", 5))),
    )


def load_hand_pose_config(
    tags_to_marker_file: str | Path,
    marker2hand_file: str | Path,
) -> HandCubeOverlayConfig:
    config = HandCubeOverlayConfig.load(tags_to_marker_file)
    _initial, _result, active = marker_to_wrist_asset_transforms(marker2hand_file)
    if active is None:
        raise ValueError(f"marker2hand 文件缺少可用的 initial_guess/result：{marker2hand_file}")
    config.set_body_to_wrist_transform(active)
    return config


def shortest_angle_delta_rad(current: float, target: float) -> float:
    return float((current - target + math.pi) % (2.0 * math.pi) - math.pi)


def rpy_to_rotation_matrix(rpy_rad: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    # JAKA SDK 将 TCP 姿态字段命名为 Rpy。这里按常见 yaw-pitch-roll 外旋 / XYZ 内旋约定处理：
    # R = Rz(rz) @ Ry(ry) @ Rx(rx)
    rx, ry, rz = [float(value) for value in np.asarray(rpy_rad, dtype=np.float64).reshape(3)]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = float(-rot[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)

    if abs(cy) > 1e-8:
        rx = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        rz = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        rx = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        rz = 0.0
    return (rx, ry, rz)


def pose_mmrad_to_transform(pose: tuple[float, float, float, float, float, float]) -> np.ndarray:
    rotation = rpy_to_rotation_matrix(pose[3:6])
    translation = np.asarray(pose[:3], dtype=np.float64)
    return make_transform(rotation, translation)


def transform_to_pose_mmrad(transform: np.ndarray) -> tuple[float, float, float, float, float, float]:
    mat = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = rotation_matrix_to_rpy(mat[:3, :3])
    translation = tuple(float(value) for value in mat[:3, 3])
    return (
        translation[0],
        translation[1],
        translation[2],
        rotation[0],
        rotation[1],
        rotation[2],
    )


def pose_is_close(
    current_pose: tuple[float, float, float, float, float, float],
    target_pose: tuple[float, float, float, float, float, float],
    *,
    position_tol_mm: float,
    angle_tol_deg: float,
) -> bool:
    angle_tol_rad = math.radians(float(angle_tol_deg))
    position_ok = all(abs(current_pose[idx] - target_pose[idx]) <= position_tol_mm for idx in range(3))
    rotation_ok = all(
        abs(shortest_angle_delta_rad(current_pose[idx], target_pose[idx])) <= angle_tol_rad
        for idx in range(3, 6)
    )
    return position_ok and rotation_ok


def format_pose_mmdeg(pose: tuple[float, float, float, float, float, float]) -> str:
    return (
        f"x={pose[0]:.1f} mm, y={pose[1]:.1f} mm, z={pose[2]:.1f} mm, "
        f"rx={math.degrees(pose[3]):.2f} deg, ry={math.degrees(pose[4]):.2f} deg, rz={math.degrees(pose[5]):.2f} deg"
    )


def make_safe_start_pose_sdk(mmdeg_pose: tuple[float, float, float, float, float, float]) -> tuple[float, float, float, float, float, float]:
    return (
        float(mmdeg_pose[0]),
        float(mmdeg_pose[1]),
        float(mmdeg_pose[2]),
        math.radians(float(mmdeg_pose[3])),
        math.radians(float(mmdeg_pose[4])),
        math.radians(float(mmdeg_pose[5])),
    )


def clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= max(float(max_norm), 1e-12):
        return vec
    return vec * (float(max_norm) / norm)


def compute_transform_delta(
    previous_transform: np.ndarray,
    current_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous = np.asarray(previous_transform, dtype=np.float64).reshape(4, 4)
    current = np.asarray(current_transform, dtype=np.float64).reshape(4, 4)
    delta_translation_m = current[:3, 3] - previous[:3, 3]
    delta_rotation = current[:3, :3] @ previous[:3, :3].T
    delta_rotvec, _ = cv2.Rodrigues(delta_rotation)
    return delta_translation_m, np.asarray(delta_rotvec, dtype=np.float64).reshape(3)


def reflection_matrix_from_mapping(mapping: WorkspaceAxisMapping) -> np.ndarray | None:
    signs = np.asarray(mapping.mirror_translation_sign, dtype=np.float64).reshape(3)
    if not np.allclose(np.abs(signs), np.ones(3), atol=1e-9):
        return None
    reflection = np.diag(signs)
    expected_rotvec_sign = float(np.linalg.det(reflection)) * signs
    if not np.allclose(mapping.mirror_rotation_sign, expected_rotvec_sign, atol=1e-9):
        return None
    return reflection


def map_translation_delta_to_robot_mm(
    delta_translation_table_m: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    delta_table = np.asarray(delta_translation_table_m, dtype=np.float64).reshape(3)
    mapped_table = mapping.mirror_translation_sign * (mapping.translation_scale * delta_table)
    return 1000.0 * (mapping.robot_from_table_rotation @ mapped_table)


def map_rotation_delta_to_robot_rad(
    delta_rotation_table_rad: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    delta_table = np.asarray(delta_rotation_table_rad, dtype=np.float64).reshape(3)
    mapped_table = mapping.mirror_rotation_sign * (mapping.rotation_scale * delta_table)
    return mapping.robot_from_table_rotation @ mapped_table


def map_rotation_delta_matrix_to_robot(
    delta_rotation_table: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    reflection = reflection_matrix_from_mapping(mapping)
    if reflection is not None and np.allclose(mapping.rotation_scale, np.ones(3), atol=1e-9):
        mirrored_in_table = reflection @ np.asarray(delta_rotation_table, dtype=np.float64).reshape(3, 3) @ reflection
        return mapping.robot_from_table_rotation @ mirrored_in_table @ mapping.robot_from_table_rotation.T

    delta_rotvec_table, _ = cv2.Rodrigues(np.asarray(delta_rotation_table, dtype=np.float64).reshape(3, 3))
    mapped_rotvec_robot = map_rotation_delta_to_robot_rad(delta_rotvec_table.reshape(3), mapping)
    mapped_rotation_robot, _ = cv2.Rodrigues(mapped_rotvec_robot.reshape(3, 1))
    return np.asarray(mapped_rotation_robot, dtype=np.float64).reshape(3, 3)


def build_servo_increment(
    delta_translation_table_m: np.ndarray,
    delta_rotation_table_rad: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    delta_translation_robot_mm = map_translation_delta_to_robot_mm(delta_translation_table_m, mapping)
    delta_rotation_robot_rad = map_rotation_delta_to_robot_rad(delta_rotation_table_rad, mapping)

    if float(np.linalg.norm(delta_translation_robot_mm)) < mapping.translation_deadband_mm:
        delta_translation_robot_mm = np.zeros(3, dtype=np.float64)
    else:
        delta_translation_robot_mm = clamp_norm(delta_translation_robot_mm, mapping.max_translation_step_mm)

    if math.degrees(float(np.linalg.norm(delta_rotation_robot_rad))) < mapping.rotation_deadband_deg:
        delta_rotation_robot_rad = np.zeros(3, dtype=np.float64)
    else:
        delta_rotation_robot_rad = clamp_norm(
            delta_rotation_robot_rad,
            math.radians(mapping.max_rotation_step_deg),
        )

    return np.concatenate([delta_translation_robot_mm, delta_rotation_robot_rad])


def build_servo_increment_from_robot_delta(
    delta_translation_robot_mm: np.ndarray,
    delta_rotation_robot_rad: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    translation = np.asarray(delta_translation_robot_mm, dtype=np.float64).reshape(3)
    rotation = np.asarray(delta_rotation_robot_rad, dtype=np.float64).reshape(3)

    if float(np.linalg.norm(translation)) < mapping.translation_deadband_mm:
        translation = np.zeros(3, dtype=np.float64)
    else:
        translation = clamp_norm(translation, mapping.max_translation_step_mm)

    if math.degrees(float(np.linalg.norm(rotation))) < mapping.rotation_deadband_deg:
        rotation = np.zeros(3, dtype=np.float64)
    else:
        rotation = clamp_norm(rotation, math.radians(mapping.max_rotation_step_deg))

    return np.concatenate([translation, rotation])


def build_desired_robot_transform(
    current_glove_transform: np.ndarray,
    anchor_state: TeleopAnchorState,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    glove_anchor = np.asarray(anchor_state.glove_anchor_transform, dtype=np.float64).reshape(4, 4)
    robot_anchor = np.asarray(anchor_state.robot_anchor_transform, dtype=np.float64).reshape(4, 4)
    current_glove = np.asarray(current_glove_transform, dtype=np.float64).reshape(4, 4)

    delta_translation_table_m = current_glove[:3, 3] - glove_anchor[:3, 3]
    delta_translation_robot_mm = map_translation_delta_to_robot_mm(delta_translation_table_m, mapping)
    desired_translation_robot_mm = robot_anchor[:3, 3] + delta_translation_robot_mm

    delta_rotation_table = current_glove[:3, :3] @ glove_anchor[:3, :3].T
    delta_rotation_robot = map_rotation_delta_matrix_to_robot(delta_rotation_table, mapping)
    desired_rotation_robot = delta_rotation_robot @ robot_anchor[:3, :3]
    return make_transform(desired_rotation_robot, desired_translation_robot_mm)


def build_servo_increment_from_desired_transforms(
    previous_desired_robot_transform: np.ndarray,
    current_desired_robot_transform: np.ndarray,
    mapping: WorkspaceAxisMapping,
) -> np.ndarray:
    delta_translation_robot_mm, delta_rotation_robot_rad = compute_transform_delta(
        previous_desired_robot_transform,
        current_desired_robot_transform,
    )
    return build_servo_increment_from_robot_delta(
        delta_translation_robot_mm,
        delta_rotation_robot_rad,
        mapping,
    )


def format_increment(delta: np.ndarray) -> str:
    values = np.asarray(delta, dtype=np.float64).reshape(6)
    return (
        f"dx={values[0]:.2f} mm, dy={values[1]:.2f} mm, dz={values[2]:.2f} mm, "
        f"drx={math.degrees(values[3]):.2f} deg, dry={math.degrees(values[4]):.2f} deg, "
        f"drz={math.degrees(values[5]):.2f} deg"
    )


class GloveWristPoseTracker:
    def __init__(
        self,
        *,
        camera_source: str,
        camera_intrinsics: str | Path,
        table_aruco_yaml: str | Path,
        table_marker_id: int,
        hand_pose_config: HandCubeOverlayConfig,
        width: int,
        height: int,
        fps: int,
        body_pose_solver: str,
        smoothing_alpha: float,
        outlier_threshold_mm: float,
        reprojection_threshold_px: float,
        show_overlay: bool,
        overlay_window_name: str,
        table_axis_length_m: float,
        body_axis_length_m: float,
        hand_axis_length_m: float,
    ) -> None:
        self.camera_source = str(camera_source)
        self.table_marker_id = int(table_marker_id)
        self.hand_pose_config = hand_pose_config
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.show_overlay = bool(show_overlay)
        self.overlay_window_name = str(overlay_window_name)
        self.table_axis_length_m = float(table_axis_length_m)
        self.body_axis_length_m = float(body_axis_length_m)
        self.hand_axis_length_m = float(hand_axis_length_m)

        with Path(table_aruco_yaml).expanduser().resolve().open("r", encoding="utf-8") as handle:
            self.table_cfg = _parse_aruco_config(yaml.safe_load(handle))
        self.target_cfg = hand_pose_config.build_target_aruco_config()
        with Path(camera_intrinsics).expanduser().resolve().open("r", encoding="utf-8") as handle:
            self.raw_intrinsics = _parse_fisheye_intrinsics(json.load(handle))

        self.pose_tracker = MarkerBodyPoseTracker(
            hand_pose_config,
            pose_solver=str(body_pose_solver),
            smoothing_alpha=float(smoothing_alpha),
            outlier_threshold_m=0.001 * float(outlier_threshold_mm),
            reprojection_error_threshold_px=float(reprojection_threshold_px),
            enable_diagnostics=False,
        )
        self.capture = self._open_capture(self.camera_source)
        self.intrinsics_scaled: dict[str, np.ndarray] | None = None
        self.intrinsics_resolution: tuple[int, int] | None = None
        self.frame_idx = 0
        self.last_status = "starting"
        self.close_requested = False
        self.last_frame_bgr: np.ndarray | None = None
        self.last_frame_result: dict[str, object] | None = None
        self.last_tag_dict: dict[int, dict[str, np.ndarray]] = {}
        self.last_cube_pose: CubePoseEstimate | None = None
        self.last_transform_table_hand: np.ndarray | None = None

        if self.show_overlay:
            cv2.namedWindow(self.overlay_window_name, cv2.WINDOW_NORMAL)

    def _open_capture(self, source: str) -> cv2.VideoCapture:
        parsed_source = _parse_capture_source(source)
        if isinstance(parsed_source, str) and parsed_source.startswith("/dev/"):
            capture = cv2.VideoCapture(parsed_source, cv2.CAP_V4L2)
        else:
            capture = cv2.VideoCapture(parsed_source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, float(self.fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开相机源：{source}")
        return capture

    def close(self) -> None:
        self.capture.release()
        if self.show_overlay:
            try:
                cv2.destroyWindow(self.overlay_window_name)
            except Exception:
                pass

    def poll(self) -> WristPoseSample | None:
        ok, frame_bgr = self.capture.read()
        if not ok or frame_bgr is None:
            self.last_status = "capture_empty"
            return None

        self.last_frame_bgr = frame_bgr.copy()

        height, width = frame_bgr.shape[:2]
        resolution = (int(width), int(height))
        if self.intrinsics_scaled is None or self.intrinsics_resolution != resolution:
            self.intrinsics_scaled = _convert_fisheye_intrinsics_resolution(
                self.raw_intrinsics,
                target_resolution=resolution,
            )
            self.intrinsics_resolution = resolution

        assert self.intrinsics_scaled is not None
        tag_dict = _detect_relevant_aruco_tags(
            frame_bgr=frame_bgr,
            intr=self.intrinsics_scaled,
            table_cfg=self.table_cfg,
            target_cfg=self.target_cfg,
            table_marker_id=self.table_marker_id,
            target_marker_ids=self.hand_pose_config.marker_ids(),
            same_cfg=False,
            refine_subpix=True,
            motion_tolerant=True,
            corner_refine_mode=None,
        )
        frame_result = _build_direct_aruco_frame_result(
            frame_idx=self.frame_idx,
            image_size=[width, height],
            table_marker_id=self.table_marker_id,
            target_marker_ids=self.hand_pose_config.marker_ids(),
            tag_dict=tag_dict,
        )
        self.frame_idx += 1
        self.last_frame_result = frame_result
        self.last_tag_dict = tag_dict

        tracker_result = self.pose_tracker.update(
            frame_result=frame_result,
            camera_matrix=self.intrinsics_scaled["K"],
        )
        pose_estimate = tracker_result.smoothed_pose
        self.last_cube_pose = pose_estimate
        if pose_estimate is None:
            self.last_transform_table_hand = None
            if not bool(frame_result.get("table_detected", False)):
                self.last_status = "table_lost"
            else:
                self.last_status = "marker_body_lost"
            return None

        transform_table_marker = np.asarray(pose_estimate.transform_table_cube, dtype=np.float64).reshape(4, 4)
        transform_table_hand = transform_table_marker @ self.hand_pose_config.body_to_wrist_transform()
        self.last_transform_table_hand = transform_table_hand.copy()
        self.last_status = f"ok markers={list(pose_estimate.source_marker_ids)}"
        return WristPoseSample(
            transform_table_hand=transform_table_hand,
            frame_idx=int(frame_result["frame_idx"]),
            time_wall=float(frame_result["time_wall"]),
            source_marker_ids=tuple(int(x) for x in pose_estimate.source_marker_ids),
        )

    def update_overlay(self, hud_lines: list[str]) -> None:
        if not self.show_overlay or self.last_frame_bgr is None or self.intrinsics_scaled is None:
            return

        vis = self.last_frame_bgr.copy()
        frame_result = self.last_frame_result or {}

        for marker_id, tag in sorted(self.last_tag_dict.items(), key=lambda item: int(item[0])):
            color = (0, 255, 255) if int(marker_id) == int(self.table_marker_id) else _marker_id_to_color(int(marker_id))
            label = f"T{marker_id}" if int(marker_id) == int(self.table_marker_id) else f"M{marker_id}"
            corners = np.asarray(tag.get("corners"), dtype=np.float64).reshape(4, 2)
            _draw_marker_outline(vis, corners, color, label)

        transform_camera_table: np.ndarray | None = None
        table_in_camera = frame_result.get("table_in_camera")
        if isinstance(table_in_camera, dict) and table_in_camera.get("matrix") is not None:
            transform_camera_table = np.asarray(table_in_camera["matrix"], dtype=np.float64).reshape(4, 4)
            rvec_table, tvec_table = transform_to_rvec_tvec(transform_camera_table)
            _draw_axes_overlay(
                vis,
                self.intrinsics_scaled,
                rvec_table,
                tvec_table,
                axis_length=self.table_axis_length_m,
                label="T",
                label_color=(0, 255, 255),
            )

        if self.last_cube_pose is not None and transform_camera_table is not None:
            transform_camera_body = transform_camera_table @ np.asarray(
                self.last_cube_pose.transform_table_cube,
                dtype=np.float64,
            ).reshape(4, 4)
            rvec_body, tvec_body = transform_to_rvec_tvec(transform_camera_body)
            _draw_axes_overlay(
                vis,
                self.intrinsics_scaled,
                rvec_body,
                tvec_body,
                axis_length=self.body_axis_length_m,
                label="B",
                label_color=(80, 255, 80),
            )

            transform_camera_hand = transform_camera_body @ self.hand_pose_config.body_to_wrist_transform()
            rvec_hand, tvec_hand = transform_to_rvec_tvec(transform_camera_hand)
            _draw_axes_overlay(
                vis,
                self.intrinsics_scaled,
                rvec_hand,
                tvec_hand,
                axis_length=self.hand_axis_length_m,
                label="H",
                label_color=(255, 255, 255),
            )

        _draw_hud(vis, hud_lines)
        cv2.imshow(self.overlay_window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self.close_requested = True


class JakaIncrementalTeleopRobot:
    def __init__(self, args: argparse.Namespace, mapping: WorkspaceAxisMapping) -> None:
        self.args = args
        self.mapping = mapping
        self.robot: object | None = None
        self.servo_enabled = False
        self.compliance_enabled = False
        self.move_mode_abs = 0
        self.move_mode_incr = 1
        self.last_tcp_pose: tuple[float, float, float, float, float, float] | None = None

    def startup(self) -> tuple[float, float, float, float, float, float]:
        jkrc = load_jkrc()
        self.move_mode_abs = int(getattr(getattr(jkrc, "MoveMode", object()), "ABS", 0))
        self.move_mode_incr = int(getattr(getattr(jkrc, "MoveMode", object()), "INCR", 1))
        self.robot = jkrc.RC(str(self.args.ip).strip())
        robot = self.robot

        login = getattr(robot, "login", None)
        if login is not None:
            ensure_ok(login(), "login")
        else:
            ensure_ok(robot.log_in(), "log_in")

        ensure_ok(robot.power_on(), "power_on")
        time.sleep(SDK_POWER_ON_WAIT_S)
        ensure_ok(robot.enable_robot(), "enable_robot")
        time.sleep(SDK_ENABLE_WAIT_S)

        self._ensure_safe_start_pose()
        maybe_set_sensor_brand(robot, self.args.sensor_brand)
        ensure_ok(robot.set_torque_sensor_mode(1), "set_torque_sensor_mode")

        if not self.args.no_saved_payload:
            payload = load_saved_payload_snapshot()
            if payload is not None:
                write_payload(robot, payload)

        zero_fn = getattr(robot, "zero_end_sensor", None)
        if zero_fn is None:
            raise AttributeError("robot does not provide zero_end_sensor")
        ensure_ok(zero_fn(), "zero_end_sensor")
        time.sleep(ZERO_SENSOR_SETTLE_S)

        set_frame = getattr(robot, "set_ft_ctrl_frame", None)
        if set_frame is not None:
            ensure_ok(set_frame(FT_CTRL_TOOL_FRAME), "set_ft_ctrl_frame(tool)")

        for axis_index in range(6):
            is_translation_axis = axis_index < 3
            enable_axis = is_translation_axis or self.args.enable_rotation_compliance
            damping = self.args.force_damping_n if is_translation_axis else self.args.torque_damping_nm
            rebound_fk = (
                self.args.translation_rebound_fk if is_translation_axis else self.args.rotation_rebound_fk
            )
            if not enable_axis:
                rebound_fk = 0.0
            ensure_ok(
                robot.set_admit_ctrl_config(axis_index, 1 if enable_axis else 0, damping, 0.0, 0, rebound_fk),
                f"set_admit_ctrl_config(axis={axis_index})",
            )

        ensure_ok(robot.servo_move_enable(True), "servo_move_enable(on)")
        self.servo_enabled = True
        ensure_ok(robot.set_compliant_type(1, 0), "set_compliant_type(init)")
        time.sleep(SDK_COMPLIANCE_SWITCH_WAIT_S)
        ensure_ok(robot.set_compliant_type(0, 1), "set_compliant_type(constant_force)")
        self.compliance_enabled = True
        self.last_tcp_pose = read_tcp_pose(robot)
        return self.last_tcp_pose

    def _ensure_safe_start_pose(self) -> None:
        assert self.robot is not None
        safe_pose = make_safe_start_pose_sdk(self.mapping.safe_start_pose_mmdeg)
        current_pose = read_tcp_pose(self.robot)
        if pose_is_close(
            current_pose,
            safe_pose,
            position_tol_mm=self.args.safe_start_position_tol_mm,
            angle_tol_deg=self.args.safe_start_angle_tol_deg,
        ):
            print(f"[jaka] safe start already satisfied: {format_pose_mmdeg(current_pose)}")
            return

        print("[jaka] current TCP is not in the validated droop pose, moving to safe start...")
        print(f"[jaka] current: {format_pose_mmdeg(current_pose)}")
        print(f"[jaka] target : {format_pose_mmdeg(safe_pose)}")
        ensure_ok(
            self.robot.linear_move(safe_pose, self.move_mode_abs, True, float(self.args.safe_start_speed_mm_s)),
            "linear_move(safe_start)",
        )
        confirmed_pose = read_tcp_pose(self.robot)
        if not pose_is_close(
            confirmed_pose,
            safe_pose,
            position_tol_mm=self.args.safe_start_position_tol_mm,
            angle_tol_deg=self.args.safe_start_angle_tol_deg,
        ):
            raise RuntimeError(
                "机械臂未能可靠到达安全下垂初始姿态，已拒绝继续开启柔顺。"
                f" current={format_pose_mmdeg(confirmed_pose)} target={format_pose_mmdeg(safe_pose)}"
            )
        print(f"[jaka] safe start confirmed: {format_pose_mmdeg(confirmed_pose)}")
        self.last_tcp_pose = confirmed_pose

    def send_increment(self, delta: np.ndarray) -> None:
        assert self.robot is not None
        ensure_ok(
            self.robot.servo_p(tuple(float(value) for value in np.asarray(delta, dtype=np.float64).reshape(6)), self.move_mode_incr, self.args.servo_step_num),
            "servo_p",
        )
        if self.last_tcp_pose is not None:
            last_transform = pose_mmrad_to_transform(self.last_tcp_pose)
            delta_transform = make_transform(
                cv2.Rodrigues(np.asarray(delta[3:], dtype=np.float64).reshape(3, 1))[0],
                np.asarray(delta[:3], dtype=np.float64).reshape(3),
            )
            next_transform = delta_transform @ last_transform
            self.last_tcp_pose = transform_to_pose_mmrad(next_transform)

    def read_current_tcp_transform(self) -> np.ndarray:
        assert self.robot is not None
        self.last_tcp_pose = read_tcp_pose(self.robot)
        return pose_mmrad_to_transform(self.last_tcp_pose)

    def shutdown(self) -> None:
        if self.robot is None:
            return
        try:
            if self.compliance_enabled:
                try:
                    ensure_ok(self.robot.set_compliant_type(0, 0), "set_compliant_type(reset)")
                    disable_force_control = getattr(self.robot, "disable_force_control", None)
                    if disable_force_control is not None:
                        ensure_ok(disable_force_control(), "disable_force_control")
                except Exception:
                    pass
            if self.servo_enabled:
                try:
                    ensure_ok(self.robot.servo_move_enable(False), "servo_move_enable(off)")
                except Exception:
                    pass
            if self.args.power_off_on_exit:
                try:
                    disable_robot = getattr(self.robot, "disable_robot", None)
                    if disable_robot is not None:
                        ensure_ok(disable_robot(), "disable_robot")
                    power_off = getattr(self.robot, "power_off", None)
                    if power_off is not None:
                        ensure_ok(power_off(), "power_off")
                except Exception:
                    pass
        finally:
            try:
                logout = getattr(self.robot, "logout", None)
                if logout is not None:
                    ensure_ok(logout(), "logout")
                else:
                    ensure_ok(self.robot.log_out(), "log_out")
            except Exception:
                pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DexSlide wrist pose incremental teleop bridge for JAKA compliant servo mode."
    )
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--sensor-brand", type=int, default=DEFAULT_SENSOR_BRAND)
    parser.add_argument("--force-damping-n", type=float, default=DEFAULT_FORCE_DAMPING_N)
    parser.add_argument("--torque-damping-nm", type=float, default=DEFAULT_TORQUE_DAMPING_NM)
    parser.add_argument("--translation-rebound-fk", type=float, default=DEFAULT_TRANSLATION_REBOUND_FK)
    parser.add_argument("--rotation-rebound-fk", type=float, default=DEFAULT_ROTATION_REBOUND_FK)
    parser.add_argument("--enable-rotation-compliance", action="store_true")
    parser.add_argument("--no-saved-payload", action="store_true")
    parser.add_argument("--servo-step-num", type=int, default=DEFAULT_SERVO_STEP_NUM)
    parser.add_argument("--loop-hz", type=float, default=DEFAULT_LOOP_HZ)
    parser.add_argument("--print-interval-s", type=float, default=DEFAULT_PRINT_INTERVAL_S)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--power-off-on-exit", action="store_true")
    parser.add_argument("--show-overlay", action="store_true", default=DEFAULT_SHOW_OVERLAY)
    parser.add_argument("--overlay-window-name", default=DEFAULT_OVERLAY_WINDOW_NAME)
    parser.add_argument("--table-axis-length-m", type=float, default=DEFAULT_TABLE_AXIS_LENGTH_M)
    parser.add_argument("--body-axis-length-m", type=float, default=DEFAULT_BODY_AXIS_LENGTH_M)
    parser.add_argument("--hand-axis-length-m", type=float, default=DEFAULT_HAND_AXIS_LENGTH_M)

    parser.add_argument("--communications-file", default=str(DEFAULT_DEXSLIDE_COMMUNICATIONS_FILE))
    parser.add_argument("--camera-name", default="primary")
    parser.add_argument("--camera-intrinsics", default=str(DEFAULT_DIRECT_ARUCO_CAMERA_INTRINSICS_FILE))
    parser.add_argument("--table-aruco-yaml", default=str(DEFAULT_DIRECT_ARUCO_TABLE_CONFIG_FILE))
    parser.add_argument("--table-marker-id", type=int, default=DEFAULT_TABLE_MARKER_ID)
    parser.add_argument("--tags-to-marker", default=str(DEFAULT_LEFT_TAGS_TO_MARKER_FILE))
    parser.add_argument("--dexalign-session", default=str(DEFAULT_DEXALIGN_SESSION_DIR))
    parser.add_argument("--mapping-file", default=str(DEFAULT_MAPPING_FILE))
    parser.add_argument("--body-pose-solver", default="joint_pnp", choices=["joint_pnp", "marker_average"])
    parser.add_argument("--body-smoothing", type=float, default=DEFAULT_BODY_SMOOTHING)
    parser.add_argument("--body-outlier-threshold-mm", type=float, default=DEFAULT_BODY_OUTLIER_THRESHOLD_MM)
    parser.add_argument("--body-reprojection-threshold-px", type=float, default=DEFAULT_BODY_REPROJECTION_THRESHOLD_PX)

    parser.add_argument("--safe-start-position-tol-mm", type=float, default=DEFAULT_SAFE_START_POSITION_TOL_MM)
    parser.add_argument("--safe-start-angle-tol-deg", type=float, default=DEFAULT_SAFE_START_ANGLE_TOL_DEG)
    parser.add_argument("--safe-start-speed-mm-s", type=float, default=DEFAULT_SAFE_START_SPEED_MM_S)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if (
        args.force_damping_n < 0.0
        or args.torque_damping_nm < 0.0
        or args.translation_rebound_fk < 0.0
        or args.rotation_rebound_fk < 0.0
    ):
        raise SystemExit("导纳参数不能小于 0")
    if args.servo_step_num < 1:
        raise SystemExit("--servo-step-num 必须大于等于 1")
    if args.loop_hz <= 0.0:
        raise SystemExit("--loop-hz 必须大于 0")
    if args.print_interval_s <= 0.0:
        raise SystemExit("--print-interval-s 必须大于 0")
    if args.safe_start_position_tol_mm <= 0.0 or args.safe_start_angle_tol_deg <= 0.0:
        raise SystemExit("safe start 容差必须大于 0")
    if args.safe_start_speed_mm_s <= 0.0:
        raise SystemExit("--safe-start-speed-mm-s 必须大于 0")
    if not 0.0 <= args.body_smoothing <= 1.0:
        raise SystemExit("--body-smoothing 必须在 [0, 1] 范围内")
    if args.body_outlier_threshold_mm <= 0.0 or args.body_reprojection_threshold_px <= 0.0:
        raise SystemExit("marker body 阈值必须大于 0")
    if args.table_axis_length_m <= 0.0 or args.body_axis_length_m <= 0.0 or args.hand_axis_length_m <= 0.0:
        raise SystemExit("overlay 轴长度必须大于 0")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    mapping = load_workspace_axis_mapping(args.mapping_file)
    dexalign_paths = load_dexalign_session_paths(args.dexalign_session)
    hand_pose_config = load_hand_pose_config(args.tags_to_marker, dexalign_paths.marker2hand_file)

    camera_cfg = camera_communication(args.camera_name, path=args.communications_file)
    camera_stream = load_camera_stream_profile()
    hand_joint_cfg = hand_joint_communication("left", path=args.communications_file)
    camera_source = resolve_camera_source(args.camera_name, path=args.communications_file)
    camera_serial = resolve_realsense_serial(args.camera_name, path=args.communications_file)
    reserved_joint_port = resolve_joint_port("left", path=args.communications_file)
    orcahand_placeholder = OrcaHandTeleopPlaceholder(
        glove_joint_port=reserved_joint_port,
        skeleton_file=dexalign_paths.skeleton_file,
        joint_calibration_file=dexalign_paths.joint_calibration_file,
    )

    print(f"[dexalign] session={dexalign_paths.session_dir}")
    print(f"[dexalign] marker2hand={dexalign_paths.marker2hand_file}")
    print(f"[dexalign] skeleton={dexalign_paths.skeleton_file} (reserved for later hand joint teleop)")
    print(
        f"[dexalign] joint_calibration={dexalign_paths.joint_calibration_file} "
        "(reserved for later OrcaHand finger mapping)"
    )
    print(f"[camera] backend={camera_cfg['backend']} serial={camera_serial} source={camera_source}")
    print(
        f"[camera] width={camera_stream.width} height={camera_stream.height} fps={camera_stream.fps} "
        f"table_marker_id={args.table_marker_id}"
    )
    print(
        f"[glove-joints] reserved_port={reserved_joint_port} baud={hand_joint_cfg['baud']} "
        "当前阶段不消费关节流，只预留后续接口"
    )
    print(orcahand_placeholder.describe())
    print(
        "[mapping] same-pose fixed rotation stored but not re-applied to incremental motion, "
        f"safe_start_pose={mapping.safe_start_pose_mmdeg}"
    )

    pose_stream = GloveWristPoseTracker(
        camera_source=camera_source,
        camera_intrinsics=args.camera_intrinsics,
        table_aruco_yaml=args.table_aruco_yaml,
        table_marker_id=args.table_marker_id,
        hand_pose_config=hand_pose_config,
        width=camera_stream.width,
        height=camera_stream.height,
        fps=int(camera_stream.fps),
        body_pose_solver=args.body_pose_solver,
        smoothing_alpha=args.body_smoothing,
        outlier_threshold_mm=args.body_outlier_threshold_mm,
        reprojection_threshold_px=args.body_reprojection_threshold_px,
        show_overlay=args.show_overlay,
        overlay_window_name=args.overlay_window_name,
        table_axis_length_m=args.table_axis_length_m,
        body_axis_length_m=args.body_axis_length_m,
        hand_axis_length_m=args.hand_axis_length_m,
    )

    robot: JakaIncrementalTeleopRobot | None = None
    if not args.dry_run:
        robot = JakaIncrementalTeleopRobot(args, mapping)
        startup_pose = robot.startup()
        print(f"[jaka] compliant teleop ready: {format_pose_mmdeg(startup_pose)}")
    else:
        print("[jaka] dry-run mode: no robot connection, only printing mapped increments")

    glove_pose_filter = GlovePoseFilter()
    loop_period_s = 1.0 / float(args.loop_hz)
    latest_filtered_sample: WristPoseSample | None = None
    anchor_candidate_sample: WristPoseSample | None = None
    anchor_state: TeleopAnchorState | None = None
    valid_streak = 0
    teleop_armed = False
    tracking_lost_reported = False
    last_status_time = 0.0

    try:
        while True:
            loop_start = time.monotonic()
            sample = pose_stream.poll()

            if sample is None:
                if anchor_state is None:
                    anchor_candidate_sample = None
                    valid_streak = 0
                elif not tracking_lost_reported:
                    print(f"[teleop] tracking lost -> pause but keep anchor ({pose_stream.last_status})")
                    tracking_lost_reported = True
            else:
                tracking_lost_reported = False
                filtered_sample = smooth_wrist_pose_sample(glove_pose_filter, sample)
                latest_filtered_sample = filtered_sample

                if not teleop_armed:
                    if anchor_candidate_sample is None:
                        anchor_candidate_sample = filtered_sample
                        valid_streak = 1
                        print(
                            f"[teleop] valid wrist pose recovered, waiting {mapping.anchor_stable_frames} stable frames..."
                        )
                    else:
                        delta_translation_table_m, delta_rotation_table_rad = compute_transform_delta(
                            anchor_candidate_sample.transform_table_hand,
                            filtered_sample.transform_table_hand,
                        )
                        raw_translation_robot_mm = map_translation_delta_to_robot_mm(delta_translation_table_m, mapping)
                        raw_rotation_robot_rad = map_rotation_delta_to_robot_rad(delta_rotation_table_rad, mapping)
                        raw_translation_norm_mm = float(np.linalg.norm(raw_translation_robot_mm))
                        raw_rotation_norm_deg = math.degrees(float(np.linalg.norm(raw_rotation_robot_rad)))

                        if (
                            raw_translation_norm_mm > mapping.frame_jump_reject_mm
                            or raw_rotation_norm_deg > mapping.frame_jump_reject_deg
                        ):
                            print(
                                "[teleop] pose jump observed while arming -> restart stability counter "
                                f"(translation={raw_translation_norm_mm:.2f} mm, rotation={raw_rotation_norm_deg:.2f} deg)"
                            )
                            valid_streak = 1
                        else:
                            valid_streak += 1
                        anchor_candidate_sample = filtered_sample

                    if valid_streak >= mapping.anchor_stable_frames:
                        if robot is not None:
                            robot_anchor_transform = robot.read_current_tcp_transform()
                        else:
                            robot_anchor_transform = np.eye(4, dtype=np.float64)
                        assert anchor_candidate_sample is not None
                        anchor_state = TeleopAnchorState(
                            glove_anchor_transform=np.asarray(
                                anchor_candidate_sample.transform_table_hand,
                                dtype=np.float64,
                            ).reshape(4, 4).copy(),
                            robot_anchor_transform=robot_anchor_transform.copy(),
                            previous_desired_robot_transform=robot_anchor_transform.copy(),
                            anchor_frame_idx=anchor_candidate_sample.frame_idx,
                        )
                        teleop_armed = True
                        print(
                            "[teleop] anchor acquired, incremental teleop armed "
                            f"(glove_frame={anchor_candidate_sample.frame_idx})"
                        )
                else:
                    assert anchor_state is not None
                    desired_robot_transform = build_desired_robot_transform(
                        filtered_sample.transform_table_hand,
                        anchor_state,
                        mapping,
                    )
                    increment = build_servo_increment_from_desired_transforms(
                        anchor_state.previous_desired_robot_transform,
                        desired_robot_transform,
                        mapping,
                    )
                    if float(np.linalg.norm(increment)) > 0.0:
                        if robot is not None:
                            robot.send_increment(increment)
                        else:
                            print(f"[dry-run] {format_increment(increment)}")
                    anchor_state = TeleopAnchorState(
                        glove_anchor_transform=anchor_state.glove_anchor_transform,
                        robot_anchor_transform=anchor_state.robot_anchor_transform,
                        previous_desired_robot_transform=desired_robot_transform,
                        anchor_frame_idx=anchor_state.anchor_frame_idx,
                    )
                    orcahand_placeholder.update()

            now = time.monotonic()
            if now - last_status_time >= float(args.print_interval_s):
                status_text = "armed" if teleop_armed else "waiting"
                extra = pose_stream.last_status
                if latest_filtered_sample is not None:
                    extra += (
                        f" frame={latest_filtered_sample.frame_idx} "
                        f"markers={list(latest_filtered_sample.source_marker_ids)}"
                    )
                if anchor_state is not None:
                    extra += f" anchor_frame={anchor_state.anchor_frame_idx}"
                print(f"[teleop] status={status_text} tracker={extra}")
                last_status_time = now

            pose_stream.update_overlay(
                [
                    f"teleop={'armed' if teleop_armed else 'waiting'}",
                    f"tracker={pose_stream.last_status}",
                    (
                        f"anchor_frame={anchor_state.anchor_frame_idx}"
                        if anchor_state is not None
                        else "anchor_frame=-"
                    ),
                    "keys=q/esc quit",
                ]
            )
            if pose_stream.close_requested:
                print("[teleop] overlay requested exit")
                break

            sleep_s = loop_period_s - (time.monotonic() - loop_start)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[teleop] interrupted by user")
    finally:
        pose_stream.close()
        if robot is not None:
            robot.shutdown()


if __name__ == "__main__":
    main()
