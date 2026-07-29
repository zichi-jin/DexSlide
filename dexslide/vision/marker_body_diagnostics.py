"""Marker-body consistency diagnostics and table-pose estimation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dexslide.kinematics.transforms import average_rotation_matrices, invert_transform, make_transform, transform_points
from dexslide.vision.marker_body_model_impl import (
    CubePoseEstimate, HandCubeOverlayConfig, MarkerBodyConsistencyItem, MarkerBodyConsistencyReport,
    _marker_pose_weight,
    _sanitize_weights,
)
from dexslide.vision.marker_body_solver import (
    _build_marker_observations,
    _compute_marker_reprojection_errors,
    _rotation_angle_deg,
    _seed_camera_body_pose,
    _solve_body_pose_camera_from_observations,
)

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
