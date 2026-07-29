"""Marker-body model, configuration, and asset serialization."""


from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from dexslide.kinematics.transforms import (
    average_rotation_matrices,
    invert_transform,
    make_transform,
    quaternion_xyzw_to_rotmat,
    rotmat_to_quaternion_xyzw,
    transform_points,
    transform_to_rvec_tvec,
)
from dexslide.vision.marker_geometry import (
    marker_square_object_points,
    normalize_marker_axes_rows,
)


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


def _sanitize_weights(values: list[float] | np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if weights.size == 0:
        return weights
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float64)
    return weights


def _normalize_marker_axes_rows(rows: np.ndarray) -> np.ndarray:
    return normalize_marker_axes_rows(rows)


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
    return marker_square_object_points(marker_size_m)


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



