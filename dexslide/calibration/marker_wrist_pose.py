from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexslide.kinematics.transforms import rvec_tvec_to_transform
from dexslide.vision.marker_body_solver import (
    _build_marker_observations, _compute_marker_reprojection_errors,
    _seed_camera_body_pose, _solve_body_pose_camera_from_observations,
    resolve_marker_body_tag_pose_branches,
)
from dexslide.vision.marker_body_model import HandCubeOverlayConfig


def _make_camera_frame_result_from_tag_dict(tag_dict: dict[int, dict[str, Any]]) -> dict[str, Any]:
    targets: dict[str, dict[str, Any]] = {}
    for marker_id, tag in tag_dict.items():
        targets[str(int(marker_id))] = {
            "detected": True,
            "target_in_camera": {"matrix": rvec_tvec_to_transform(tag["rvec"], tag["tvec"]).tolist()},
            "undistorted_corners": np.asarray(tag.get("undistorted_corners", tag.get("corners")), dtype=np.float64).reshape(4, 2).tolist(),
        }
    return {"targets": targets}

@dataclass(frozen=True)
class BodyPoseEstimate:
    transform_camera_body: np.ndarray
    source_marker_ids: tuple[int, ...]
    max_position_deviation_m: float
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    resolved_tag_dict: dict[int, dict[str, Any]]


def _default_hand_overlay_config_path(hand: str) -> Path:
    requested = str(hand).strip().lower()
    if requested in {"left", "right"}:
        return DIRECT_ARUCO_CALIBRATION_DIR / f"{requested}_tags2marker.json"
    return DEFAULT_HAND_MARKER_BODY_LEFT_CONFIG_FILE


def _default_output_config_path(config_path: Path) -> Path:
    asset_paths = resolve_hand_overlay_asset_paths(config_path)
    return asset_paths["marker_to_wrist"]


def _default_output_report_path(config_path: Path) -> Path:
    asset_paths = resolve_hand_overlay_asset_paths(config_path)
    return asset_paths["marker_to_wrist_dataset"]


def _optional_path_arg(raw: str | None) -> Path | None:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _rs_intrinsics_to_opencv_dict(intr: Any) -> dict[str, np.ndarray]:
    return {
        "DIM": np.array([int(intr.width), int(intr.height)], dtype=np.int64),
        "K": np.array(
            [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "D": np.asarray(list(intr.coeffs), dtype=np.float64).reshape(-1, 1),
    }


def _make_camera_frame_result_from_tag_dict(tag_dict: dict[int, dict[str, Any]]) -> dict[str, Any]:
    targets: dict[str, dict[str, Any]] = {}
    for marker_id, tag in tag_dict.items():
        targets[str(int(marker_id))] = {
            "detected": True,
            "target_in_camera": {
                "matrix": rvec_tvec_to_transform(tag["rvec"], tag["tvec"]).tolist(),
            },
            "undistorted_corners": np.asarray(
                tag.get("undistorted_corners", tag.get("corners")),
                dtype=np.float64,
            ).reshape(4, 2).tolist(),
        }
    return {"targets": targets}


def _estimate_body_pose_in_camera_from_tag_dict(
    tag_dict: dict[int, dict[str, Any]],
    config: HandCubeOverlayConfig,
    camera_matrix: np.ndarray,
    *,
    reference_camera_body: np.ndarray | None = None,
    outlier_threshold_m: float,
    reprojection_error_threshold_px: float,
) -> BodyPoseEstimate | None:
    resolved_tag_dict: dict[int, dict[str, Any]] = copy.deepcopy(tag_dict)
    resolve_marker_body_tag_pose_branches(
        resolved_tag_dict,
        config,
        reference_camera_body=reference_camera_body,
    )

    frame_result = _make_camera_frame_result_from_tag_dict(resolved_tag_dict)
    observations = _build_marker_observations(frame_result, config)
    if not observations:
        return None

    seed_camera_body, active_observations, seed_spread_m = _seed_camera_body_pose(
        observations,
        outlier_threshold_m=outlier_threshold_m,
    )
    if seed_camera_body is None or not active_observations:
        return None

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

    max_position_deviation_m = seed_spread_m
    if final_observations:
        final_positions = np.stack(
            [obs.transform_camera_body_single[:3, 3] for obs in final_observations],
            axis=0,
        )
        diffs = final_positions - final_transform_camera_body[:3, 3][None, :]
        max_position_deviation_m = float(np.max(np.linalg.norm(diffs, axis=-1)))

    return BodyPoseEstimate(
        transform_camera_body=np.asarray(final_transform_camera_body, dtype=np.float64).reshape(4, 4).copy(),
        source_marker_ids=tuple(int(obs.marker_id) for obs in final_observations),
        max_position_deviation_m=float(max_position_deviation_m),
        mean_reprojection_error_px=(
            0.0 if not final_errors else float(np.mean([err["mean_error_px"] for err in final_errors]))
        ),
        max_reprojection_error_px=(
            0.0 if not final_errors else float(np.max([err["max_error_px"] for err in final_errors]))
        ),
        resolved_tag_dict=resolved_tag_dict,
    )
