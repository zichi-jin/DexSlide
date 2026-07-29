"""Helpers for estimating a hand-mounted marker body pose and overlaying a glove skeleton."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _normalize_quaternion_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quaternion_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quaternion_xyzw(rot: np.ndarray) -> np.ndarray:
    mat = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (mat[2, 1] - mat[1, 2]) / s
        qy = (mat[0, 2] - mat[2, 0]) / s
        qz = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        qw = (mat[2, 1] - mat[1, 2]) / s
        qx = 0.25 * s
        qy = (mat[0, 1] + mat[1, 0]) / s
        qz = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        qw = (mat[0, 2] - mat[2, 0]) / s
        qx = (mat[0, 1] + mat[1, 0]) / s
        qy = 0.25 * s
        qz = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        qw = (mat[1, 0] - mat[0, 1]) / s
        qx = (mat[0, 2] + mat[2, 0]) / s
        qy = (mat[1, 2] + mat[2, 1]) / s
        qz = 0.25 * s
    return _normalize_quaternion_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))


def make_transform(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(trans, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rot = np.asarray(transform[:3, :3], dtype=np.float64)
    trans = np.asarray(transform[:3, 3], dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ trans)
    return inv


def transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    rot = np.asarray(transform[:3, :3], dtype=np.float64)
    trans = np.asarray(transform[:3, 3], dtype=np.float64)
    return (pts @ rot.T) + trans[None, :]


def transform_to_rvec_tvec(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot = np.asarray(transform[:3, :3], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rot)
    return rvec.reshape(3), np.asarray(transform[:3, 3], dtype=np.float64).reshape(3)


def _load_structured_document(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Invalid hand overlay asset `{cfg_path}`: expected a mapping object.")
    return document


def _asset_prefix_from_path(path: str | Path) -> str | None:
    stem = Path(path).expanduser().resolve().stem
    for suffix in ("_tags2marker", "_marker2wrist", "_marker2wrist_dataset"):
        if stem.endswith(suffix):
            prefix = stem[: -len(suffix)].strip()
            return prefix or None
    if stem.startswith("hand_marker_body_"):
        prefix = stem[len("hand_marker_body_") :].strip()
        return prefix or None
    return None


def resolve_hand_overlay_asset_paths(
    path: str | Path,
    *,
    hand: str | None = None,
) -> dict[str, Path]:
    asset_path = Path(path).expanduser().resolve()
    prefix = _asset_prefix_from_path(asset_path)
    hand_name = str(hand).strip().lower() if hand is not None else ""
    if not prefix:
        prefix = hand_name or "left"
    base_dir = asset_path.parent
    tags_to_marker = base_dir / f"{prefix}_tags2marker.json"
    marker_to_wrist = base_dir / f"{prefix}_marker2wrist.json"
    marker_to_wrist_dataset = base_dir / f"{prefix}_marker2wrist_dataset.json"
    return {
        "tags_to_marker": tags_to_marker,
        "marker_to_wrist": marker_to_wrist,
        "marker_to_wrist_dataset": marker_to_wrist_dataset,
    }


def _resolve_relative_asset_path(
    base_path: str | Path,
    raw_value: str | Path | None,
    fallback_path: Path,
) -> Path:
    if raw_value is None or not str(raw_value).strip():
        return fallback_path
    candidate = Path(str(raw_value)).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base_path).expanduser().resolve().parent / candidate
    return candidate.resolve()


def _transform_from_marker_to_wrist_entry(
    entry: dict[str, Any],
    *,
    entry_name: str,
    asset_path: str | Path,
) -> np.ndarray:
    if not isinstance(entry, dict):
        raise ValueError(
            f"Invalid marker->wrist asset `{Path(asset_path).resolve()}`: `{entry_name}` must be a mapping."
        )
    trans = entry.get("trans")
    rot = entry.get("rot")
    if trans is None or rot is None:
        raise ValueError(
            f"Invalid marker->wrist asset `{Path(asset_path).resolve()}`: `{entry_name}` must contain `trans` and `rot`."
        )
    return make_transform(
        np.asarray(rot, dtype=np.float64).reshape(3, 3),
        np.asarray(trans, dtype=np.float64).reshape(3),
    )


def load_marker_to_wrist_asset(path: str | Path) -> dict[str, Any]:
    return _load_structured_document(path)


def marker_to_wrist_asset_transforms(
    path_or_document: str | Path | dict[str, Any],
    *,
    asset_path: str | Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if isinstance(path_or_document, dict):
        document = path_or_document
        resolved_asset_path = Path("." if asset_path is None else asset_path).expanduser().resolve()
    else:
        resolved_asset_path = Path(path_or_document).expanduser().resolve()
        document = load_marker_to_wrist_asset(resolved_asset_path)

    initial_guess = None
    if isinstance(document.get("initial_guess"), dict):
        initial_guess = _transform_from_marker_to_wrist_entry(
            document["initial_guess"],
            entry_name="initial_guess",
            asset_path=resolved_asset_path,
        )
    result = None
    if isinstance(document.get("result"), dict):
        result = _transform_from_marker_to_wrist_entry(
            document["result"],
            entry_name="result",
            asset_path=resolved_asset_path,
        )
    active = result if result is not None else initial_guess
    return initial_guess, result, active


def marker_to_wrist_entry_from_transform(
    transform: np.ndarray,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "trans": [
            float(x) for x in np.asarray(transform[:3, 3], dtype=np.float64).reshape(3)
        ],
        "rot": np.asarray(transform[:3, :3], dtype=np.float64).reshape(3, 3).tolist(),
        "trans_unit": "m",
    }
    if extra_fields:
        payload.update(extra_fields)
    return payload


def _parse_markers_from_document(
    document: dict[str, Any],
    *,
    cfg_path: str | Path,
) -> tuple[dict[int, "MarkerMount"], float]:
    marker_faces_raw = document.get("marker_face_id", document.get("marker_faces"))
    if not isinstance(marker_faces_raw, dict) or not marker_faces_raw:
        raise ValueError(f"Invalid marker body config `{Path(cfg_path).resolve()}`: missing marker_face_id.")

    marker_square_size_m = 0.001 * float(
        document.get("marker_square_size_mm", document.get("marker_square_size", 40.0))
    )
    marker_center_radius_m = 0.5 * marker_square_size_m * (1.0 + math.sqrt(2.0))

    markers: dict[int, MarkerMount] = {}
    for raw_marker_id, raw_payload in marker_faces_raw.items():
        marker_id = int(raw_marker_id)
        if isinstance(raw_payload, dict):
            rows = raw_payload.get("rot", raw_payload.get("axes_rows_body"))
            if rows is None:
                raise ValueError(
                    f"Invalid marker_face_id[{marker_id}] in `{Path(cfg_path).resolve()}`: missing rot matrix."
                )
            raw_translation = raw_payload.get("p_mm", raw_payload.get("p"))
            translation_m = None
            if raw_translation is not None:
                translation_m = 0.001 * np.asarray(raw_translation, dtype=np.float64).reshape(3)
        else:
            rows = raw_payload
            translation_m = None
        try:
            marker = MarkerMount(
                marker_id=marker_id,
                axes_rows_body=np.asarray(rows, dtype=np.float64).reshape(3, 3),
                translation_body_marker_m=translation_m,
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid marker_face_id[{marker_id}] in `{Path(cfg_path).resolve()}`: {exc}"
            ) from exc
        if marker.translation_body_marker_m is None:
            try:
                marker = MarkerMount(
                    marker_id=marker_id,
                    axes_rows_body=marker.axes_rows_body,
                    translation_body_marker_m=marker.normal_body * marker_center_radius_m,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid marker_face_id[{marker_id}] in `{Path(cfg_path).resolve()}`: {exc}"
                ) from exc
        markers[marker_id] = marker
    return markers, marker_square_size_m


def compose_overlay_joint_angles(
    raw_joint_angles: np.ndarray,
    joint_zero_rad: np.ndarray,
    joint_base_render_rad: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(raw_joint_angles, dtype=np.float64).reshape(-1)
    zero = np.asarray(joint_zero_rad, dtype=np.float64).reshape(-1)
    base = np.asarray(joint_base_render_rad, dtype=np.float64).reshape(-1)
    if raw.shape[0] != 20 or zero.shape[0] != 20 or base.shape[0] != 20:
        raise ValueError(
            f"Expected raw/zero/base joint vectors to all have length 20, got "
            f"{raw.shape[0]}, {zero.shape[0]}, {base.shape[0]}"
        )
    return (raw - zero) + base


def average_rotation_matrices(
    rotations: list[np.ndarray],
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    if not rotations:
        return np.eye(3, dtype=np.float64)
    if weights is None:
        weights_arr = np.ones(len(rotations), dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights_arr.shape[0] != len(rotations):
            raise ValueError(
                f"Expected {len(rotations)} rotation weights, got {weights_arr.shape[0]}"
            )
        weights_arr = np.maximum(weights_arr, 0.0)
        if float(np.sum(weights_arr)) <= 1e-12:
            weights_arr = np.ones(len(rotations), dtype=np.float64)
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for rot, weight in zip(rotations, weights_arr):
        accumulator += float(weight) * np.asarray(rot, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(accumulator)
    mean_rot = u @ vt
    if np.linalg.det(mean_rot) < 0.0:
        u[:, -1] *= -1.0
        mean_rot = u @ vt
    return mean_rot


def _sanitize_weights(values: list[float] | np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if weights.size == 0:
        return weights
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float64)
    return weights


def _normalize_marker_axes_rows(rows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float64).reshape(3, 3)
    out = matrix.copy()
    for idx in range(3):
        norm = float(np.linalg.norm(out[idx]))
        if norm < 1e-9:
            raise ValueError("Marker axes matrix contains a near-zero row.")
        out[idx] /= norm
    if abs(float(np.dot(out[0], out[1]))) > 5e-3:
        raise ValueError("Marker x/y axes are not orthogonal.")
    if abs(float(np.dot(out[0], out[2]))) > 5e-3:
        raise ValueError("Marker x/z axes are not orthogonal.")
    if abs(float(np.dot(out[1], out[2]))) > 5e-3:
        raise ValueError("Marker y/z axes are not orthogonal.")
    handed = float(np.dot(np.cross(out[0], out[1]), out[2]))
    if handed < 0.995:
        raise ValueError("Marker axes must be right-handed and satisfy x × y = z.")
    return out


def _marker_pose_weight(target_data: dict[str, Any]) -> float:
    pose_camera = target_data.get("target_in_camera")
    if pose_camera is None:
        return 1.0
    transform_camera_marker = np.asarray(pose_camera["matrix"], dtype=np.float64).reshape(4, 4)
    rot = transform_camera_marker[:3, :3]
    trans = transform_camera_marker[:3, 3]
    depth_m = max(abs(float(trans[2])), 1e-6)
    frontal = max(abs(float(rot[2, 2])), 1e-3)
    return float(frontal / (depth_m * depth_m))


def _marker_square_object_points(marker_size_m: float) -> np.ndarray:
    half_size = 0.5 * float(marker_size_m)
    return np.array(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MarkerMount:
    marker_id: int
    axes_rows_body: np.ndarray
    translation_body_marker_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        try:
            normalized_axes = _normalize_marker_axes_rows(self.axes_rows_body)
        except ValueError as exc:
            raise ValueError(
                f"Invalid axes_rows_body for marker_id={int(self.marker_id)}: {exc}"
            ) from exc
        object.__setattr__(self, "axes_rows_body", normalized_axes)
        if self.translation_body_marker_m is not None:
            translation = np.asarray(self.translation_body_marker_m, dtype=np.float64).reshape(3)
            object.__setattr__(self, "translation_body_marker_m", translation)

    @property
    def rotation_body_marker(self) -> np.ndarray:
        return np.asarray(self.axes_rows_body, dtype=np.float64).reshape(3, 3).T

    @property
    def normal_body(self) -> np.ndarray:
        return np.asarray(self.axes_rows_body, dtype=np.float64).reshape(3, 3)[2]

    def body_to_marker_transform(self, center_radius_m: float) -> np.ndarray:
        translation = self.translation_body_marker_m
        if translation is None:
            translation = self.normal_body * float(center_radius_m)
        return make_transform(self.rotation_body_marker, translation)

    def to_yaml_payload(self, *, translation_mm: np.ndarray | None = None) -> list[list[float]] | dict[str, Any]:
        axes_rows = np.asarray(self.axes_rows_body, dtype=np.float64).tolist()
        if translation_mm is None:
            return axes_rows
        return {
            "rot": axes_rows,
            "p_mm": [float(x) for x in np.asarray(translation_mm, dtype=np.float64).reshape(3)],
        }


@dataclass
class CubePoseEstimate:
    transform_table_cube: np.ndarray
    source_marker_ids: list[int]
    max_position_deviation_m: float
    solver_mode: str = "joint_pnp"
    mean_reprojection_error_px: float = 0.0
    max_reprojection_error_px: float = 0.0


@dataclass(frozen=True)
class MarkerBodyConsistencyItem:
    marker_id: int
    peer_position_error_m: float
    peer_rotation_error_deg: float
    fused_position_error_m: float | None = None
    fused_rotation_error_deg: float | None = None
    reprojection_mean_error_px: float | None = None
    reprojection_max_error_px: float | None = None


@dataclass(frozen=True)
class MarkerBodyConsistencyReport:
    marker_ids: list[int]
    items: list[MarkerBodyConsistencyItem]


@dataclass(frozen=True)
class _MarkerObservation:
    marker_id: int
    image_points_px: np.ndarray
    object_points_body_m: np.ndarray
    transform_camera_body_single: np.ndarray
    weight: float


@dataclass(frozen=True)
class _MarkerPoseBranchCandidate:
    marker_id: int
    candidate_index: int
    transform_camera_marker: np.ndarray
    transform_camera_body: np.ndarray
    reprojection_error_px: float


@dataclass
class HandCubeOverlayConfig:
    hand: str = "left"
    aruco_predefined: str = "DICT_4X4_50"
    aruco_bound_size_m: float = 0.03
    marker_square_size_m: float = 0.04
    markers: dict[int, MarkerMount] = field(default_factory=dict)
    marker_skeleton_translation_m: list[float] = field(
        default_factory=lambda: [-0.00265, 0.09, -0.0701]
    )
    marker_skeleton_quaternion_xyzw: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0]
    )
    joint_zero_rad: list[float] = field(default_factory=lambda: [0.0] * 20)
    joint_base_render_rad: list[float] = field(default_factory=lambda: [0.0] * 20)

    @property
    def marker_center_radius_m(self) -> float:
        return 0.5 * float(self.marker_square_size_m) * (1.0 + math.sqrt(2.0))

    def body_to_wrist_transform(self) -> np.ndarray:
        return make_transform(
            quaternion_xyzw_to_rotmat(np.asarray(self.marker_skeleton_quaternion_xyzw, dtype=np.float64)),
            np.asarray(self.marker_skeleton_translation_m, dtype=np.float64),
        )

    def set_body_to_wrist_transform(self, transform: np.ndarray) -> None:
        self.marker_skeleton_translation_m = [
            float(x) for x in np.asarray(transform[:3, 3], dtype=np.float64).reshape(3)
        ]
        self.marker_skeleton_quaternion_xyzw = [
            float(x) for x in rotmat_to_quaternion_xyzw(transform[:3, :3])
        ]

    # Backward-compatible aliases for the existing overlay/calibration flow.
    def cube_to_wrist_transform(self) -> np.ndarray:
        return self.body_to_wrist_transform()

    def set_cube_to_wrist_transform(self, transform: np.ndarray) -> None:
        self.set_body_to_wrist_transform(transform)

    @property
    def cube_to_wrist_translation_m(self) -> list[float]:
        return self.marker_skeleton_translation_m

    @property
    def cube_to_wrist_quaternion_xyzw(self) -> list[float]:
        return self.marker_skeleton_quaternion_xyzw

    def set_joint_zero(self, joint_zero_rad: np.ndarray) -> None:
        values = np.asarray(joint_zero_rad, dtype=np.float64).reshape(-1)
        if values.shape[0] != 20:
            raise ValueError(f"Expected 20 glove joints for joint_zero_rad, got {values.shape[0]}")
        self.joint_zero_rad = [float(x) for x in values]

    def set_joint_base_render(self, joint_base_render_rad: np.ndarray) -> None:
        values = np.asarray(joint_base_render_rad, dtype=np.float64).reshape(-1)
        if values.shape[0] != 20:
            raise ValueError(
                f"Expected 20 glove joints for joint_base_render_rad, got {values.shape[0]}"
            )
        self.joint_base_render_rad = [float(x) for x in values]

    def marker_ids(self) -> list[int]:
        return sorted(int(marker_id) for marker_id in self.markers.keys())

    def build_target_aruco_config(self) -> dict[str, Any]:
        return {
            "aruco_dict": cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.aruco_predefined)),
            "marker_size_map": {
                int(marker_id): float(self.aruco_bound_size_m) for marker_id in self.marker_ids()
            },
        }

    def to_yaml_dict(self) -> dict[str, Any]:
        skeleton_rot = quaternion_xyzw_to_rotmat(
            np.asarray(self.marker_skeleton_quaternion_xyzw, dtype=np.float64)
        )
        marker_face_id: dict[int, Any] = {}
        for marker_id, marker in sorted(self.markers.items(), key=lambda item: int(item[0])):
            translation_mm = None
            if marker.translation_body_marker_m is not None:
                translation_mm = 1000.0 * np.asarray(marker.translation_body_marker_m, dtype=np.float64)
            marker_face_id[int(marker_id)] = marker.to_yaml_payload(translation_mm=translation_mm)
        return {
            "hand": str(self.hand),
            "aruco_dict": {"predefined": str(self.aruco_predefined)},
            "aruco_bound_size_mm": float(self.aruco_bound_size_m) * 1000.0,
            "marker_square_size_mm": float(self.marker_square_size_m) * 1000.0,
            "marker_face_id": marker_face_id,
            "marker_skeleton_translate_mm": [
                float(x) * 1000.0 for x in self.marker_skeleton_translation_m
            ],
            "marker_skeleton_rotation": np.asarray(skeleton_rot, dtype=np.float64).tolist(),
            "joint_zero_rad": [float(x) for x in self.joint_zero_rad],
            "joint_base_render_rad": [float(x) for x in self.joint_base_render_rad],
        }

    def save(self, path: str | Path) -> None:
        out_path = Path(path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.to_yaml_dict(),
                handle,
                allow_unicode=True,
                sort_keys=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> HandCubeOverlayConfig:
        cfg_path = Path(path).expanduser().resolve()
        document = _load_structured_document(cfg_path)

        marker_to_wrist_document: dict[str, Any] | None = None
        marker_to_wrist_path: Path | None = None
        if "marker_face_id" not in document and "marker_faces" not in document:
            if "initial_guess" not in document and "result" not in document:
                raise ValueError(
                    f"Invalid hand overlay asset `{cfg_path}`: expected marker_face_id or marker->wrist entries."
                )
            asset_paths = resolve_hand_overlay_asset_paths(
                cfg_path,
                hand=str(document.get("hand", "")).strip().lower() or None,
            )
            tags_to_marker_path = _resolve_relative_asset_path(
                cfg_path,
                document.get("tags2marker_path"),
                asset_paths["tags_to_marker"],
            )
            document = _load_structured_document(tags_to_marker_path)
            cfg_path = tags_to_marker_path
            marker_to_wrist_document = _load_structured_document(path)
            marker_to_wrist_path = Path(path).expanduser().resolve()

        markers, marker_square_size_m = _parse_markers_from_document(document, cfg_path=cfg_path)

        if marker_to_wrist_document is None:
            if "marker_skeleton_translate_mm" in document or "marker_skeleton_rotation" in document:
                skeleton_rot = np.asarray(
                    document.get(
                        "marker_skeleton_rotation",
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ),
                    dtype=np.float64,
                ).reshape(3, 3)
                body_to_wrist_transform = make_transform(
                    skeleton_rot,
                    np.array(
                        [
                            0.001 * float(x)
                            for x in document.get("marker_skeleton_translate_mm", [-2.65, 90.0, -70.1])
                        ],
                        dtype=np.float64,
                    ).reshape(3),
                )
            else:
                asset_paths = resolve_hand_overlay_asset_paths(
                    cfg_path,
                    hand=str(document.get("hand", "")).strip().lower() or None,
                )
                marker_to_wrist_path = _resolve_relative_asset_path(
                    cfg_path,
                    document.get("marker2wrist_path"),
                    asset_paths["marker_to_wrist"],
                )
                marker_to_wrist_document = _load_structured_document(marker_to_wrist_path)

        if marker_to_wrist_document is not None:
            active_entry_name = "result" if isinstance(marker_to_wrist_document.get("result"), dict) else "initial_guess"
            active_entry = marker_to_wrist_document.get(active_entry_name)
            if active_entry is None:
                raise ValueError(
                    f"Invalid marker->wrist asset `{marker_to_wrist_path}`: missing `initial_guess` and `result`."
                )
            body_to_wrist_transform = _transform_from_marker_to_wrist_entry(
                active_entry,
                entry_name=active_entry_name,
                asset_path=marker_to_wrist_path if marker_to_wrist_path is not None else cfg_path,
            )

        cfg = cls(
            hand=str(document.get("hand", "left")),
            aruco_predefined=str(document.get("aruco_dict", {}).get("predefined", "DICT_4X4_50")),
            aruco_bound_size_m=0.001
            * float(document.get("aruco_bound_size_mm", document.get("aruco_bound_size", 30.0))),
            marker_square_size_m=marker_square_size_m,
            markers=markers,
            marker_skeleton_translation_m=[
                float(x) for x in np.asarray(body_to_wrist_transform[:3, 3], dtype=np.float64).reshape(3)
            ],
            marker_skeleton_quaternion_xyzw=[
                float(x) for x in rotmat_to_quaternion_xyzw(body_to_wrist_transform[:3, :3])
            ],
            joint_zero_rad=[float(x) for x in document.get("joint_zero_rad", [0.0] * 20)],
            joint_base_render_rad=[
                float(x) for x in document.get("joint_base_render_rad", [0.0] * 20)
            ],
        )
        if len(cfg.joint_zero_rad) != 20:
            raise ValueError(
                f"Invalid marker body config `{cfg_path}`: joint_zero_rad must have length 20."
            )
        if len(cfg.joint_base_render_rad) != 20:
            raise ValueError(
                f"Invalid marker body config `{cfg_path}`: joint_base_render_rad must have length 20."
            )
        return cfg


def try_load_hand_cube_overlay_config(path: str | Path) -> HandCubeOverlayConfig | None:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        return None
    try:
        return HandCubeOverlayConfig.load(cfg_path)
    except Exception as exc:
        print(
            f"⚠️ hand overlay config invalid, cannot load: {cfg_path} "
            f"({type(exc).__name__}: {exc})"
        )
        return None


def _pose_candidate_dicts_from_tag_data(target_data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = target_data.get("pose_candidates")
    if isinstance(raw_candidates, list) and raw_candidates:
        return raw_candidates
    return [
        {
            "rvec": np.asarray(target_data["rvec"], dtype=np.float64).reshape(3),
            "tvec": np.asarray(target_data["tvec"], dtype=np.float64).reshape(3),
            "reprojection_error_px": float(target_data.get("reprojection_error_px", 0.0)),
        }
    ]


def _marker_body_pose_distance_score(
    transform_a: np.ndarray,
    transform_b: np.ndarray,
    *,
    position_scale_mm: float = 6.0,
    rotation_scale_deg: float = 7.5,
) -> float:
    pos_mm = float(
        np.linalg.norm(
            np.asarray(transform_a[:3, 3], dtype=np.float64).reshape(3)
            - np.asarray(transform_b[:3, 3], dtype=np.float64).reshape(3)
        )
    ) * 1000.0
    rot_deg = _rotation_angle_deg(transform_a[:3, :3], transform_b[:3, :3])
    return pos_mm / max(float(position_scale_mm), 1e-6) + rot_deg / max(float(rotation_scale_deg), 1e-6)


def _average_marker_body_pose_candidates(
    candidates: list[_MarkerPoseBranchCandidate],
) -> np.ndarray | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return np.asarray(candidates[0].transform_camera_body, dtype=np.float64).reshape(4, 4).copy()
    weights = _sanitize_weights(
        [1.0 / max(1.0 + float(candidate.reprojection_error_px), 1e-6) for candidate in candidates]
    )
    positions = np.stack(
        [
            np.asarray(candidate.transform_camera_body[:3, 3], dtype=np.float64).reshape(3)
            for candidate in candidates
        ],
        axis=0,
    )
    rotations = [
        np.asarray(candidate.transform_camera_body[:3, :3], dtype=np.float64).reshape(3, 3)
        for candidate in candidates
    ]
    mean_position = np.average(positions, axis=0, weights=weights)
    mean_rotation = average_rotation_matrices(rotations, weights)
    return make_transform(mean_rotation, mean_position)


def _build_marker_pose_branch_candidates(
    tag_dict: dict[int, dict[str, Any]],
    config: HandCubeOverlayConfig,
) -> dict[int, list[_MarkerPoseBranchCandidate]]:
    candidates_by_marker: dict[int, list[_MarkerPoseBranchCandidate]] = {}
    for marker_id, marker_mount in config.markers.items():
        target_data = tag_dict.get(int(marker_id))
        if not target_data:
            continue
        transform_body_marker = marker_mount.body_to_marker_transform(config.marker_center_radius_m)
        transform_marker_body = invert_transform(transform_body_marker)
        marker_candidates: list[_MarkerPoseBranchCandidate] = []
        for candidate_index, candidate in enumerate(_pose_candidate_dicts_from_tag_data(target_data)):
            rvec = np.asarray(candidate["rvec"], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(candidate["tvec"], dtype=np.float64).reshape(3)
            rot, _ = cv2.Rodrigues(rvec)
            transform_camera_marker = make_transform(rot, tvec)
            marker_candidates.append(
                _MarkerPoseBranchCandidate(
                    marker_id=int(marker_id),
                    candidate_index=int(candidate_index),
                    transform_camera_marker=transform_camera_marker,
                    transform_camera_body=transform_camera_marker @ transform_marker_body,
                    reprojection_error_px=float(candidate.get("reprojection_error_px", 0.0)),
                )
            )
        if marker_candidates:
            candidates_by_marker[int(marker_id)] = marker_candidates
    return candidates_by_marker


def _select_marker_body_branch_anchor(
    candidates_by_marker: dict[int, list[_MarkerPoseBranchCandidate]],
    *,
    reference_camera_body: np.ndarray | None,
) -> _MarkerPoseBranchCandidate | None:
    best_candidate = None
    best_score = float("inf")
    for marker_id, marker_candidates in candidates_by_marker.items():
        for candidate in marker_candidates:
            score = 0.15 * float(candidate.reprojection_error_px)
            if reference_camera_body is not None:
                score += 2.0 * _marker_body_pose_distance_score(
                    candidate.transform_camera_body,
                    reference_camera_body,
                )
            for other_marker_id, other_candidates in candidates_by_marker.items():
                if other_marker_id == marker_id:
                    continue
                best_other = min(
                    (
                        _marker_body_pose_distance_score(
                            candidate.transform_camera_body,
                            other_candidate.transform_camera_body,
                        )
                        + 0.15 * float(other_candidate.reprojection_error_px)
                    )
                    for other_candidate in other_candidates
                )
                score += float(best_other)
            if score < best_score:
                best_score = float(score)
                best_candidate = candidate
    return best_candidate


def _select_marker_body_branch_candidates(
    candidates_by_marker: dict[int, list[_MarkerPoseBranchCandidate]],
    *,
    reference_camera_body: np.ndarray | None,
) -> dict[int, _MarkerPoseBranchCandidate]:
    selected: dict[int, _MarkerPoseBranchCandidate] = {}
    for marker_id, marker_candidates in candidates_by_marker.items():
        if reference_camera_body is None:
            best = min(marker_candidates, key=lambda candidate: float(candidate.reprojection_error_px))
        else:
            best = min(
                marker_candidates,
                key=lambda candidate: (
                    _marker_body_pose_distance_score(
                        candidate.transform_camera_body,
                        reference_camera_body,
                    )
                    + 0.15 * float(candidate.reprojection_error_px)
                ),
            )
        selected[int(marker_id)] = best
    return selected


def resolve_marker_body_tag_pose_branches(
    tag_dict: dict[int, dict[str, Any]],
    config: HandCubeOverlayConfig,
    *,
    reference_camera_body: np.ndarray | None = None,
) -> dict[str, Any]:
    candidates_by_marker = _build_marker_pose_branch_candidates(tag_dict, config)
    if not candidates_by_marker:
        return {
            "resolved_marker_ids": [],
            "anchor_marker_id": None,
            "anchor_candidate_index": None,
            "reference_camera_body": None,
        }

    anchor = _select_marker_body_branch_anchor(
        candidates_by_marker,
        reference_camera_body=reference_camera_body,
    )
    selection_reference = reference_camera_body
    if anchor is not None:
        selection_reference = np.asarray(anchor.transform_camera_body, dtype=np.float64).reshape(4, 4)

    selected = _select_marker_body_branch_candidates(
        candidates_by_marker,
        reference_camera_body=selection_reference,
    )
    consensus_camera_body = _average_marker_body_pose_candidates(list(selected.values()))
    if consensus_camera_body is not None:
        selected = _select_marker_body_branch_candidates(
            candidates_by_marker,
            reference_camera_body=consensus_camera_body,
        )
        consensus_camera_body = _average_marker_body_pose_candidates(list(selected.values()))

    for marker_id, candidate in selected.items():
        rvec, tvec = transform_to_rvec_tvec(candidate.transform_camera_marker)
        tag_dict[int(marker_id)]["rvec"] = np.asarray(rvec, dtype=np.float64).reshape(3)
        tag_dict[int(marker_id)]["tvec"] = np.asarray(tvec, dtype=np.float64).reshape(3)
        tag_dict[int(marker_id)]["selected_pose_candidate_index"] = int(candidate.candidate_index)
        tag_dict[int(marker_id)]["selected_pose_reprojection_error_px"] = float(
            candidate.reprojection_error_px
        )

    return {
        "resolved_marker_ids": sorted(int(marker_id) for marker_id in selected.keys()),
        "anchor_marker_id": (None if anchor is None else int(anchor.marker_id)),
        "anchor_candidate_index": (None if anchor is None else int(anchor.candidate_index)),
        "reference_camera_body": (
            None
            if consensus_camera_body is None
            else np.asarray(consensus_camera_body, dtype=np.float64).reshape(4, 4)
        ),
    }


def _build_marker_observations(
    frame_result: dict[str, Any],
    config: HandCubeOverlayConfig,
) -> list[_MarkerObservation]:
    targets = frame_result.get("targets", {})
    observations: list[_MarkerObservation] = []
    for marker_id, marker_mount in config.markers.items():
        target_data = targets.get(str(marker_id))
        if not target_data or not bool(target_data.get("detected", False)):
            continue

        pose_camera = target_data.get("target_in_camera")
        undistorted_corners = target_data.get("undistorted_corners")
        if pose_camera is None or undistorted_corners is None:
            continue

        transform_camera_marker = np.asarray(pose_camera["matrix"], dtype=np.float64).reshape(4, 4)
        transform_body_marker = marker_mount.body_to_marker_transform(config.marker_center_radius_m)
        object_points_body_m = transform_points(
            transform_body_marker,
            _marker_square_object_points(config.aruco_bound_size_m),
        )
        observations.append(
            _MarkerObservation(
                marker_id=int(marker_id),
                image_points_px=np.asarray(undistorted_corners, dtype=np.float64).reshape(4, 2),
                object_points_body_m=object_points_body_m,
                transform_camera_body_single=(
                    transform_camera_marker @ invert_transform(transform_body_marker)
                ),
                weight=_marker_pose_weight(target_data),
            )
        )
    return observations


def _seed_camera_body_pose(
    observations: list[_MarkerObservation],
    *,
    outlier_threshold_m: float,
) -> tuple[np.ndarray | None, list[_MarkerObservation], float]:
    if not observations:
        return None, [], 0.0

    positions = np.stack([obs.transform_camera_body_single[:3, 3] for obs in observations], axis=0)
    weights = _sanitize_weights([obs.weight for obs in observations])
    keep_mask = np.ones(len(observations), dtype=bool)
    if len(observations) >= 3 and float(outlier_threshold_m) > 0.0:
        median_position = np.median(positions, axis=0)
        distances_to_median = np.linalg.norm(positions - median_position[None, :], axis=1)
        keep_mask = distances_to_median <= float(outlier_threshold_m)
        if not np.any(keep_mask):
            keep_mask[int(np.argmin(distances_to_median))] = True

    filtered = [obs for obs, keep in zip(observations, keep_mask) if keep]
    filtered_positions = positions[keep_mask]
    filtered_weights = _sanitize_weights(weights[keep_mask])
    filtered_rotations = [
        obs.transform_camera_body_single[:3, :3]
        for obs, keep in zip(observations, keep_mask)
        if keep
    ]
    mean_position = np.average(filtered_positions, axis=0, weights=filtered_weights)
    mean_rotation = average_rotation_matrices(filtered_rotations, filtered_weights)
    diffs = filtered_positions - mean_position[None, :]
    max_position_deviation_m = float(np.max(np.linalg.norm(diffs, axis=-1)))
    return make_transform(mean_rotation, mean_position), filtered, max_position_deviation_m


def _solve_body_pose_camera_from_observations(
    observations: list[_MarkerObservation],
    *,
    camera_matrix: np.ndarray,
    initial_transform: np.ndarray | None,
    reprojection_error_threshold_px: float,
) -> np.ndarray | None:
    if not observations:
        return None

    object_points = np.ascontiguousarray(
        np.concatenate([obs.object_points_body_m for obs in observations], axis=0).astype(np.float64)
    )
    image_points = np.ascontiguousarray(
        np.concatenate([obs.image_points_px for obs in observations], axis=0).astype(np.float64)
    )
    camera = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3))
    dist = np.zeros((1, 5), dtype=np.float64)

    rvec_guess = None
    tvec_guess = None
    if initial_transform is not None:
        rvec_guess, tvec_guess = transform_to_rvec_tvec(initial_transform)
        rvec_guess = np.ascontiguousarray(rvec_guess.reshape(3, 1), dtype=np.float64)
        tvec_guess = np.ascontiguousarray(tvec_guess.reshape(3, 1), dtype=np.float64)

    if len(observations) >= 2 and object_points.shape[0] >= 8:
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera,
                dist,
                rvec=rvec_guess,
                tvec=tvec_guess,
                useExtrinsicGuess=initial_transform is not None,
                iterationsCount=120,
                reprojectionError=max(1.5, float(reprojection_error_threshold_px)),
                confidence=0.995,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            success = False
            rvec = None
            tvec = None
            inliers = None
        if success and rvec is not None and tvec is not None:
            if inliers is not None and len(inliers) >= 4:
                inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
                refine_success, refined_rvec, refined_tvec = cv2.solvePnP(
                    object_points[inlier_idx],
                    image_points[inlier_idx],
                    camera,
                    dist,
                    rvec=rvec,
                    tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if refine_success:
                    rvec = refined_rvec
                    tvec = refined_tvec
            rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
            return make_transform(rot, np.asarray(tvec, dtype=np.float64).reshape(3))

    solve_flags = cv2.SOLVEPNP_ITERATIVE if initial_transform is not None else getattr(
        cv2,
        "SOLVEPNP_SQPNP",
        cv2.SOLVEPNP_ITERATIVE,
    )
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera,
        dist,
        rvec=rvec_guess,
        tvec=tvec_guess,
        useExtrinsicGuess=initial_transform is not None,
        flags=solve_flags,
    )
    if not success:
        return None
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return make_transform(rot, np.asarray(tvec, dtype=np.float64).reshape(3))


def _compute_marker_reprojection_errors(
    observations: list[_MarkerObservation],
    *,
    transform_camera_body: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[dict[str, float | int]]:
    if not observations:
        return []

    rvec, tvec = transform_to_rvec_tvec(transform_camera_body)
    camera = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3))
    dist = np.zeros((1, 5), dtype=np.float64)
    marker_errors: list[dict[str, float | int]] = []
    for obs in observations:
        projected, _ = cv2.projectPoints(
            np.ascontiguousarray(obs.object_points_body_m.reshape(-1, 1, 3), dtype=np.float64),
            np.ascontiguousarray(rvec.reshape(3, 1), dtype=np.float64),
            np.ascontiguousarray(tvec.reshape(3, 1), dtype=np.float64),
            camera,
            dist,
        )
        projected_2d = projected.reshape(-1, 2)
        residuals = np.linalg.norm(projected_2d - obs.image_points_px, axis=1)
        marker_errors.append(
            {
                "marker_id": int(obs.marker_id),
                "mean_error_px": float(np.mean(residuals)),
                "max_error_px": float(np.max(residuals)),
            }
        )
    return marker_errors


def _rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rel))
    cos_theta = float(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _collect_per_marker_table_body_transforms(
    frame_result: dict[str, Any],
    config: HandCubeOverlayConfig,
) -> list[tuple[int, np.ndarray, float]]:
    per_marker_body_transforms: list[tuple[int, np.ndarray, float]] = []
    for marker_id, marker_mount in config.markers.items():
        target_data = frame_result.get("targets", {}).get(str(marker_id))
        if not target_data:
            continue
        pose = target_data.get("target_in_table")
        if pose is None:
            continue
        transform_table_marker = np.asarray(pose["matrix"], dtype=np.float64).reshape(4, 4)
        transform_body_marker = marker_mount.body_to_marker_transform(config.marker_center_radius_m)
        transform_table_body = transform_table_marker @ invert_transform(transform_body_marker)
        per_marker_body_transforms.append(
            (int(marker_id), transform_table_body, _marker_pose_weight(target_data))
        )
    return per_marker_body_transforms


def _estimate_body_pose_by_table_average(
    frame_result: dict[str, Any],
    config: HandCubeOverlayConfig,
    *,
    outlier_threshold_m: float,
) -> CubePoseEstimate | None:
    per_marker_body_transforms = _collect_per_marker_table_body_transforms(frame_result, config)
    if not per_marker_body_transforms:
        return None

    marker_ids = [marker_id for marker_id, _transform, _weight in per_marker_body_transforms]
    transforms = [transform for _marker_id, transform, _weight in per_marker_body_transforms]
    weights = _sanitize_weights(
        [weight for _marker_id, _transform, weight in per_marker_body_transforms]
    )
    positions = np.stack([transform[:3, 3] for transform in transforms], axis=0)

    keep_mask = np.ones(len(per_marker_body_transforms), dtype=bool)
    if len(per_marker_body_transforms) >= 3 and float(outlier_threshold_m) > 0.0:
        median_position = np.median(positions, axis=0)
        distances_to_median = np.linalg.norm(positions - median_position[None, :], axis=1)
        keep_mask = distances_to_median <= float(outlier_threshold_m)
        if not np.any(keep_mask):
            keep_mask[int(np.argmin(distances_to_median))] = True

    filtered_positions = positions[keep_mask]
    filtered_weights = _sanitize_weights(weights[keep_mask])
    filtered_marker_ids = [marker_id for marker_id, keep in zip(marker_ids, keep_mask) if keep]
    filtered_rotations = [transform[:3, :3] for transform, keep in zip(transforms, keep_mask) if keep]

    mean_position = np.average(filtered_positions, axis=0, weights=filtered_weights)
    diffs = filtered_positions - mean_position[None, :]
    max_position_deviation_m = float(np.max(np.linalg.norm(diffs, axis=-1)))
    mean_rotation = average_rotation_matrices(filtered_rotations, filtered_weights)
    return CubePoseEstimate(
        transform_table_cube=make_transform(mean_rotation, mean_position),
        source_marker_ids=filtered_marker_ids,
        max_position_deviation_m=max_position_deviation_m,
        solver_mode="marker_average",
    )


def diagnose_marker_body_consistency(
    frame_result: dict[str, Any],
    config: HandCubeOverlayConfig,
    *,
    fused_pose: CubePoseEstimate | None = None,
    camera_matrix: np.ndarray | None = None,
) -> MarkerBodyConsistencyReport | None:
    per_marker_body_transforms = _collect_per_marker_table_body_transforms(frame_result, config)
    if not per_marker_body_transforms:
        return None

    marker_ids = [marker_id for marker_id, _transform, _weight in per_marker_body_transforms]
    transforms = [transform for _marker_id, transform, _weight in per_marker_body_transforms]
    weights = _sanitize_weights([weight for _marker_id, _transform, weight in per_marker_body_transforms])

    reproj_by_marker: dict[int, tuple[float, float]] = {}
    if (
        fused_pose is not None
        and camera_matrix is not None
        and frame_result.get("table_in_camera") is not None
    ):
        transform_camera_table = np.asarray(
            frame_result["table_in_camera"]["matrix"],
            dtype=np.float64,
        ).reshape(4, 4)
        transform_camera_body = transform_camera_table @ fused_pose.transform_table_cube
        observations = _build_marker_observations(frame_result, config)
        reproj_errors = _compute_marker_reprojection_errors(
            observations,
            transform_camera_body=transform_camera_body,
            camera_matrix=camera_matrix,
        )
        reproj_by_marker = {
            int(item["marker_id"]): (
                float(item["mean_error_px"]),
                float(item["max_error_px"]),
            )
            for item in reproj_errors
        }

    items: list[MarkerBodyConsistencyItem] = []
    for idx, (marker_id, transform_table_body, _weight) in enumerate(per_marker_body_transforms):
        if len(transforms) <= 1:
            peer_position_error_m = 0.0
            peer_rotation_error_deg = 0.0
        else:
            other_indices = [j for j in range(len(transforms)) if j != idx]
            other_weights = _sanitize_weights([weights[j] for j in other_indices])
            pair_position_errors = np.asarray(
                [
                    np.linalg.norm(
                        np.asarray(transform_table_body[:3, 3], dtype=np.float64)
                        - np.asarray(transforms[j][:3, 3], dtype=np.float64)
                    )
                    for j in other_indices
                ],
                dtype=np.float64,
            )
            pair_rotation_errors = np.asarray(
                [
                    _rotation_angle_deg(
                        transform_table_body[:3, :3],
                        transforms[j][:3, :3],
                    )
                    for j in other_indices
                ],
                dtype=np.float64,
            )
            peer_position_error_m = float(np.average(pair_position_errors, weights=other_weights))
            peer_rotation_error_deg = float(np.average(pair_rotation_errors, weights=other_weights))

        fused_position_error_m = None
        fused_rotation_error_deg = None
        if fused_pose is not None:
            fused_position_error_m = float(
                np.linalg.norm(
                    np.asarray(transform_table_body[:3, 3], dtype=np.float64)
                    - np.asarray(fused_pose.transform_table_cube[:3, 3], dtype=np.float64)
                )
            )
            fused_rotation_error_deg = _rotation_angle_deg(
                fused_pose.transform_table_cube[:3, :3],
                transform_table_body[:3, :3],
            )

        reproj_mean_error_px = None
        reproj_max_error_px = None
        if marker_id in reproj_by_marker:
            reproj_mean_error_px, reproj_max_error_px = reproj_by_marker[marker_id]

        items.append(
            MarkerBodyConsistencyItem(
                marker_id=int(marker_id),
                peer_position_error_m=peer_position_error_m,
                peer_rotation_error_deg=peer_rotation_error_deg,
                fused_position_error_m=fused_position_error_m,
                fused_rotation_error_deg=fused_rotation_error_deg,
                reprojection_mean_error_px=reproj_mean_error_px,
                reprojection_max_error_px=reproj_max_error_px,
            )
        )

    return MarkerBodyConsistencyReport(
        marker_ids=marker_ids,
        items=sorted(
            items,
            key=lambda item: (
                float(item.peer_rotation_error_deg),
                float(item.peer_position_error_m),
                float(-item.marker_id),
            ),
            reverse=True,
        ),
    )


def estimate_cube_pose_in_table(
    frame_result: dict[str, Any],
    config: HandCubeOverlayConfig,
    *,
    outlier_threshold_m: float = 0.02,
    camera_matrix: np.ndarray | None = None,
    reprojection_error_threshold_px: float = 5.0,
    pose_solver: str = "joint_pnp",
) -> CubePoseEstimate | None:
    solver = str(pose_solver).strip().lower()
    if solver not in {"joint_pnp", "marker_average"}:
        raise ValueError(
            f"Unsupported pose_solver `{pose_solver}`. Expected one of: joint_pnp, marker_average."
        )

    average_estimate = _estimate_body_pose_by_table_average(
        frame_result,
        config,
        outlier_threshold_m=outlier_threshold_m,
    )
    if solver == "marker_average":
        return average_estimate

    if camera_matrix is None:
        return average_estimate

    table_pose = frame_result.get("table_in_camera")
    if table_pose is None:
        return average_estimate

    observations = _build_marker_observations(frame_result, config)
    if not observations:
        return average_estimate

    seed_camera_body, active_observations, seed_spread_m = _seed_camera_body_pose(
        observations,
        outlier_threshold_m=outlier_threshold_m,
    )
    if seed_camera_body is None or not active_observations:
        return average_estimate

    final_transform_camera_body = seed_camera_body
    final_observations = list(active_observations)
    final_errors = _compute_marker_reprojection_errors(
        final_observations,
        transform_camera_body=final_transform_camera_body,
        camera_matrix=camera_matrix,
    )

    while active_observations:
        solved_transform = _solve_body_pose_camera_from_observations(
            active_observations,
            camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
            initial_transform=seed_camera_body,
            reprojection_error_threshold_px=reprojection_error_threshold_px,
        )
        if solved_transform is None:
            break

        current_errors = _compute_marker_reprojection_errors(
            active_observations,
            transform_camera_body=solved_transform,
            camera_matrix=camera_matrix,
        )
        final_transform_camera_body = solved_transform
        final_observations = list(active_observations)
        final_errors = current_errors

        if len(active_observations) <= 1:
            break

        worst_error = max(
            current_errors,
            key=lambda item: (float(item["mean_error_px"]), float(item["max_error_px"])),
        )
        if float(worst_error["mean_error_px"]) <= float(reprojection_error_threshold_px):
            break

        active_observations = [
            obs for obs in active_observations if obs.marker_id != int(worst_error["marker_id"])
        ]
        seed_camera_body, active_observations, seed_spread_m = _seed_camera_body_pose(
            active_observations,
            outlier_threshold_m=outlier_threshold_m,
        )
        if seed_camera_body is None or not active_observations:
            break

    transform_camera_table = np.asarray(table_pose["matrix"], dtype=np.float64).reshape(4, 4)
    transform_table_body = invert_transform(transform_camera_table) @ final_transform_camera_body
    max_position_deviation_m = seed_spread_m
    if final_observations:
        final_positions = np.stack(
            [obs.transform_camera_body_single[:3, 3] for obs in final_observations],
            axis=0,
        )
        diffs = final_positions - final_transform_camera_body[:3, 3][None, :]
        max_position_deviation_m = float(np.max(np.linalg.norm(diffs, axis=-1)))

    return CubePoseEstimate(
        transform_table_cube=transform_table_body,
        source_marker_ids=[obs.marker_id for obs in final_observations],
        max_position_deviation_m=max_position_deviation_m,
        solver_mode="joint_pnp",
        mean_reprojection_error_px=(
            0.0 if not final_errors else float(np.mean([err["mean_error_px"] for err in final_errors]))
        ),
        max_reprojection_error_px=(
            0.0 if not final_errors else float(np.max([err["max_error_px"] for err in final_errors]))
        ),
    )
