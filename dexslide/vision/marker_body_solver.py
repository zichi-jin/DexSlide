"""Marker-body pose branch resolution and PnP solver."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dexslide.kinematics.transforms import average_rotation_matrices, invert_transform, make_transform, transform_points, transform_to_rvec_tvec
from dexslide.vision.marker_body_model_impl import (
    HandCubeOverlayConfig, MarkerMount, _MarkerObservation, _MarkerPoseBranchCandidate,
    _marker_pose_weight, _marker_square_object_points, _sanitize_weights,
)
from dexslide.vision.pnp import front_facing_pose_candidates


def _rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    relative = np.asarray(rot_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))

def _pose_candidate_dicts_from_tag_data(target_data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = target_data.get("pose_candidates")
    if isinstance(raw_candidates, list) and raw_candidates:
        candidates = raw_candidates
    else:
        candidates = [
            {
                "rvec": np.asarray(target_data["rvec"], dtype=np.float64).reshape(3),
                "tvec": np.asarray(target_data["tvec"], dtype=np.float64).reshape(3),
                "reprojection_error_px": float(target_data.get("reprojection_error_px", 0.0)),
            }
        ]
    selected = front_facing_pose_candidates(candidates)
    indexed: list[dict[str, Any]] = []
    for candidate in selected:
        source_index = next(
            index for index, original in enumerate(candidates) if original is candidate
        )
        indexed.append({**candidate, "_source_candidate_index": int(source_index)})
    return indexed


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
                    candidate_index=int(
                        candidate.get("_source_candidate_index", candidate_index)
                    ),
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
    selected_camera_marker: dict[int, np.ndarray] = {}
    candidate_tags: dict[int, dict[str, Any]] = {}
    for marker_id, target_data in targets.items():
        if not isinstance(target_data, dict) or not bool(target_data.get("detected", False)):
            continue
        pose_camera = target_data.get("target_in_camera")
        if not isinstance(pose_camera, dict) or pose_camera.get("matrix") is None:
            continue
        transform_camera_marker = np.asarray(
            pose_camera["matrix"], dtype=np.float64
        ).reshape(4, 4)
        rvec, _ = cv2.Rodrigues(transform_camera_marker[:3, :3])
        entry: dict[str, Any] = {
            "rvec": rvec.reshape(3),
            "tvec": transform_camera_marker[:3, 3].reshape(3),
        }
        raw_candidates = target_data.get("pose_candidates")
        if isinstance(raw_candidates, list) and raw_candidates:
            entry["pose_candidates"] = raw_candidates
        candidate_tags[int(marker_id)] = entry
        selected_camera_marker[int(marker_id)] = transform_camera_marker

    if any(isinstance(item.get("pose_candidates"), list) for item in candidate_tags.values()):
        resolve_marker_body_tag_pose_branches(candidate_tags, config)
        for marker_id, tag_data in candidate_tags.items():
            rvec = np.asarray(tag_data["rvec"], dtype=np.float64).reshape(3, 1)
            rotation, _ = cv2.Rodrigues(rvec)
            selected_camera_marker[int(marker_id)] = make_transform(
                rotation,
                np.asarray(tag_data["tvec"], dtype=np.float64).reshape(3),
            )

    observations: list[_MarkerObservation] = []
    for marker_id, marker_mount in config.markers.items():
        target_data = targets.get(str(marker_id))
        if not target_data or not bool(target_data.get("detected", False)):
            continue

        undistorted_corners = target_data.get("undistorted_corners")
        pose_camera = selected_camera_marker.get(int(marker_id))
        if pose_camera is None or undistorted_corners is None:
            continue

        transform_camera_marker = np.asarray(pose_camera, dtype=np.float64).reshape(4, 4)
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
