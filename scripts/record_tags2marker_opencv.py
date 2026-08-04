#!/usr/bin/env python3
"""Record raw OpenCV ArUco IPPE poses and tags2marker centroid transforms.

This is intentionally independent from the DexSlide runtime API.  It opens the
configured camera with ``cv2.VideoCapture``, detects only marker IDs declared in
``tags2marker.json``, records every IPPE pose candidate, and converts every
candidate from ``camera_T_tag`` to ``camera_T_body`` using the configured
``body_T_tag`` transform.

No pose filtering, branch rejection, temporal selection, or multi-marker fusion
is applied.  The resulting JSONL is meant for diagnosing the raw planar-PnP
ambiguity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREAMING_CONFIG = REPO_ROOT / "assets" / "dexslide_streaming.json"
DEFAULT_TAGS2MARKER = (
    REPO_ROOT / "assets" / "calibration" / "direct_aruco" / "left_tags2marker.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "build" / "tags2marker_opencv_trace.jsonl"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_relative(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _parse_source(value: str | int) -> str | int:
    text = str(value).strip()
    return int(text) if text.lstrip("-").isdigit() else str(Path(text).expanduser())


def _configured_camera_source(
    streaming_config: dict[str, Any],
    *,
    streaming_path: Path,
) -> str | int:
    camera = streaming_config.get("camera", {})
    if isinstance(camera, dict) and str(camera.get("source", "")).strip():
        return _parse_source(camera["source"])

    communications_value = streaming_config.get(
        "communications_file",
        "dexslide_communications.json",
    )
    communications_path = _resolve_relative(
        str(communications_value),
        base_dir=streaming_path.parent,
    )
    communications = _load_json(communications_path)
    primary = communications.get("camera", {}).get("primary", {})
    if not isinstance(primary, dict):
        raise ValueError(f"Missing camera.primary in {communications_path}")
    source = str(primary.get("stable_opencv_source", "")).strip()
    if not source:
        source = str(primary.get("opencv_source", "")).strip()
    if not source:
        raise ValueError(f"No OpenCV camera source configured in {communications_path}")
    return _parse_source(source)


def _parse_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int], str]:
    payload = _load_json(path)
    if "K" in payload and "D" in payload:
        camera_matrix = np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(payload["D"], dtype=np.float64).reshape(-1, 1)
        if "DIM" in payload:
            width, height = (int(value) for value in payload["DIM"])
        else:
            width = int(payload.get("image_width", 0))
            height = int(payload.get("image_height", 0))
        model = "fisheye" if distortion.size == 4 else "opencv"
        return camera_matrix, distortion, (width, height), model

    intrinsics = payload.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError(f"Unsupported camera intrinsics format: {path}")
    width = int(payload["image_width"])
    height = int(payload["image_height"])
    focal = float(intrinsics["focal_length"])
    camera_matrix = np.array(
        [
            [focal, 0.0, float(intrinsics["principal_pt_x"])],
            [0.0, focal, float(intrinsics["principal_pt_y"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(
        [
            float(intrinsics.get("radial_distortion_1", 0.0)),
            float(intrinsics.get("radial_distortion_2", 0.0)),
            float(intrinsics.get("radial_distortion_3", 0.0)),
            float(intrinsics.get("radial_distortion_4", 0.0)),
        ],
        dtype=np.float64,
    ).reshape(4, 1)
    return camera_matrix, distortion, (width, height), "fisheye"


def _scale_camera_matrix(
    camera_matrix: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        return np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    scaled = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _marker_object_points(marker_size_m: float) -> np.ndarray:
    half = 0.5 * float(marker_size_m)
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    return _make_transform(rotation.T, -(rotation.T @ translation))


def _body_to_tag_transform(marker_payload: dict[str, Any]) -> np.ndarray:
    axes_rows_body = np.asarray(marker_payload["rot"], dtype=np.float64).reshape(3, 3)
    rotation_body_tag = axes_rows_body.T
    translation_body_tag_m = 0.001 * np.asarray(
        marker_payload["p_mm"],
        dtype=np.float64,
    ).reshape(3)
    determinant = float(np.linalg.det(rotation_body_tag))
    if not math.isclose(determinant, 1.0, abs_tol=1e-5):
        raise ValueError(f"tags2marker rotation is not right-handed: det={determinant}")
    return _make_transform(rotation_body_tag, translation_body_tag_m)


def _undistort_corners(
    corners_px: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    distortion_model: str,
) -> np.ndarray:
    corners = np.asarray(corners_px, dtype=np.float64).reshape(1, 4, 2)
    if distortion_model == "fisheye":
        return cv2.fisheye.undistortPoints(
            corners,
            camera_matrix,
            distortion,
            P=camera_matrix,
        ).reshape(4, 2)
    return cv2.undistortPoints(
        corners,
        camera_matrix,
        distortion,
        P=camera_matrix,
    ).reshape(4, 2)


def _solve_ippe_candidates(
    corners_px: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
) -> list[dict[str, Any]]:
    result = cv2.solvePnPGeneric(
        _marker_object_points(marker_size_m),
        np.asarray(corners_px, dtype=np.float64).reshape(4, 2),
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.zeros((1, 5), dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    success = bool(result[0]) if result else False
    if not success:
        return []
    rvecs = result[1]
    tvecs = result[2]
    reprojection_errors = result[3] if len(result) >= 4 else None
    candidates: list[dict[str, Any]] = []
    for opencv_index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        error_px = 0.0
        if reprojection_errors is not None and len(reprojection_errors) > opencv_index:
            error_px = float(np.asarray(reprojection_errors[opencv_index]).reshape(-1)[0])
        candidates.append(
            {
                "opencv_index": int(opencv_index),
                "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
                "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
                "reprojection_error_px": error_px,
            }
        )
    return candidates


def _candidate_record(
    candidate: dict[str, Any],
    *,
    body_T_tag: np.ndarray,
) -> dict[str, Any]:
    rvec = np.asarray(candidate["rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(candidate["tvec"], dtype=np.float64).reshape(3)
    rotation_camera_tag, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    camera_T_tag = _make_transform(rotation_camera_tag, tvec)
    camera_T_body = camera_T_tag @ _invert_transform(body_T_tag)
    z_axis_camera = np.asarray(rotation_camera_tag[:, 2], dtype=np.float64).reshape(3)
    position_norm = float(np.linalg.norm(tvec))
    z_dot = float(np.dot(z_axis_camera, tvec))
    z_cosine = z_dot / position_norm if position_norm > 1e-12 else None
    return {
        "opencv_index": int(candidate["opencv_index"]),
        "reprojection_error_px": float(candidate["reprojection_error_px"]),
        "rvec": rvec.tolist(),
        "tvec_m": tvec.tolist(),
        "tvec_z_m": float(tvec[2]),
        "rotation_matrix_camera_tag": np.asarray(rotation_camera_tag).tolist(),
        "rotation_det_camera_tag": float(np.linalg.det(rotation_camera_tag)),
        "z_axis_camera": z_axis_camera.tolist(),
        "z_dot_camera_to_tag": z_dot,
        "z_cosine_camera_to_tag": z_cosine,
        "camera_T_tag": camera_T_tag.tolist(),
        "centroid": {
            "camera_T_body": camera_T_body.tolist(),
            "rotation_matrix_camera_body": camera_T_body[:3, :3].tolist(),
            "rotation_det_camera_body": float(np.linalg.det(camera_T_body[:3, :3])),
            "position_camera_body_m": camera_T_body[:3, 3].tolist(),
        },
    }


def _detector_parameters(corner_refine: str) -> cv2.aruco.DetectorParameters:
    parameters = cv2.aruco.DetectorParameters()
    if corner_refine == "none":
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    elif corner_refine == "subpix":
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    elif corner_refine == "apriltag":
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    else:
        raise ValueError(f"Unsupported corner refinement mode: {corner_refine}")
    return parameters


def _fourcc_text(value: float) -> str:
    encoded = int(value)
    return "".join(chr((encoded >> (8 * index)) & 0xFF) for index in range(4))


def _project_points(
    points: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    distortion_model: str,
) -> np.ndarray:
    rotation = np.ascontiguousarray(
        np.asarray(transform[:3, :3], dtype=np.float64).reshape(3, 3)
    )
    translation = np.ascontiguousarray(
        np.asarray(transform[:3, 3], dtype=np.float64).reshape(3, 1)
    )
    rvec, _ = cv2.Rodrigues(rotation)
    rvec = np.ascontiguousarray(rvec)
    object_points = np.ascontiguousarray(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    )
    if distortion_model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            translation,
            camera_matrix,
            distortion,
        )
    else:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            translation,
            camera_matrix,
            distortion,
        )
    return projected.reshape(-1, 2)


def _draw_axes(
    image: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    distortion_model: str,
    *,
    length_m: float,
    thickness: int,
) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [length_m, 0.0, 0.0],
            [0.0, length_m, 0.0],
            [0.0, 0.0, length_m],
        ],
        dtype=np.float64,
    )
    projected = np.rint(
        _project_points(
            points,
            transform,
            camera_matrix,
            distortion,
            distortion_model,
        )
    ).astype(int)
    origin = tuple(projected[0])
    for endpoint, color in zip(
        projected[1:],
        ((0, 0, 255), (0, 255, 0), (255, 0, 0)),
    ):
        cv2.line(image, origin, tuple(endpoint), color, thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record raw OpenCV ArUco IPPE candidates and their tags2marker "
            "centroid transforms without using the DexSlide runtime API."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_STREAMING_CONFIG))
    parser.add_argument("--tags2marker", default=str(DEFAULT_TAGS2MARKER))
    parser.add_argument("--intrinsics", default="", help="Override camera intrinsics JSON")
    parser.add_argument("--source", default="", help="Override OpenCV camera source")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--fourcc", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=30.0,
        help="Recording duration. Use 0 to run until q/Esc.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--corner-refine",
        choices=["none", "subpix", "apriltag"],
        default="subpix",
    )
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_sec < 0.0:
        raise SystemExit("--duration-sec must be >= 0")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    streaming_path = Path(args.config).expanduser().resolve()
    tags_path = Path(args.tags2marker).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    streaming = _load_json(streaming_path)
    tags_document = _load_json(tags_path)
    camera_config = streaming.get("camera", {})
    if not isinstance(camera_config, dict):
        raise ValueError("streaming config camera must be an object")

    source = (
        _parse_source(args.source)
        if str(args.source).strip()
        else _configured_camera_source(streaming, streaming_path=streaming_path)
    )
    width = int(args.width or camera_config.get("width", 0))
    height = int(args.height or camera_config.get("height", 0))
    fps = float(args.fps or camera_config.get("fps", 0.0))
    fourcc = str(args.fourcc or camera_config.get("fourcc", "")).strip()
    buffer_size = int(camera_config.get("buffer_size", 2))

    intrinsics_value = str(args.intrinsics).strip() or str(camera_config["intrinsics_file"])
    intrinsics_path = _resolve_relative(
        intrinsics_value,
        base_dir=streaming_path.parent,
    )
    raw_camera_matrix, distortion, calibration_size, distortion_model = _parse_intrinsics(
        intrinsics_path
    )

    dictionary_name = str(tags_document.get("aruco_dict", {}).get("predefined", "")).strip()
    if not dictionary_name or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Invalid ArUco dictionary in {tags_path}: {dictionary_name!r}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        _detector_parameters(args.corner_refine),
    )

    marker_payloads = {
        int(marker_id): payload
        for marker_id, payload in tags_document.get("marker_face_id", {}).items()
    }
    if not marker_payloads:
        raise ValueError(f"No marker_face_id entries in {tags_path}")
    marker_size_m = 0.001 * float(tags_document["aruco_bound_size_mm"])
    body_T_tags = {
        marker_id: _body_to_tag_transform(payload)
        for marker_id, payload in marker_payloads.items()
    }

    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera source: {source!r}")
    if fourcc:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
    if width > 0:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0.0:
        camera.set(cv2.CAP_PROP_FPS, fps)
    if buffer_size > 0:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

    actual_mode = {
        "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(camera.get(cv2.CAP_PROP_FPS)),
        "fourcc": _fourcc_text(camera.get(cv2.CAP_PROP_FOURCC)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    frames_with_markers = 0
    candidate_count = 0
    marker_counts: Counter[int] = Counter()
    started_wall = time.time()
    started_monotonic = time.monotonic()
    effective_camera_matrix: np.ndarray | None = None
    effective_image_size: tuple[int, int] | None = None

    print(
        "[opencv-tags2marker] "
        f"source={source!r} requested={width}x{height}@{fps:g} fourcc={fourcc or 'default'}"
    )
    print(f"[opencv-tags2marker] tags2marker={tags_path}")
    print(
        "[opencv-tags2marker] "
        f"intrinsics={intrinsics_path} "
        f"calibration={calibration_size[0]}x{calibration_size[1]} "
        f"model={distortion_model}"
    )
    print(f"[opencv-tags2marker] output={output_path}")
    print(
        "[opencv-tags2marker] records all IPPE candidates; "
        "no filtering, branch selection, or fusion"
    )

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            session_record = {
                "record_type": "session",
                "schema_version": 1,
                "created_at": datetime.now().astimezone().isoformat(),
                "opencv_version": cv2.__version__,
                "streaming_config": str(streaming_path),
                "tags2marker_file": str(tags_path),
                "intrinsics_file": str(intrinsics_path),
                "camera": {
                    "source": source,
                    "requested_width": width,
                    "requested_height": height,
                    "requested_fps": fps,
                    "requested_fourcc": fourcc or None,
                    "calibration_image_size": list(calibration_size),
                    "distortion_model": distortion_model,
                },
                "aruco": {
                    "dictionary": dictionary_name,
                    "marker_size_m": marker_size_m,
                    "configured_marker_ids": sorted(marker_payloads),
                    "corner_refine": args.corner_refine,
                    "pose_solver": "SOLVEPNP_IPPE_SQUARE",
                },
                "transform_semantics": {
                    "configured": "body_T_tag",
                    "derived": "camera_T_body = camera_T_tag @ inverse(body_T_tag)",
                },
            }
            output_file.write(json.dumps(session_record, ensure_ascii=False) + "\n")

            while True:
                ok, frame = camera.read()
                timestamp_wall = time.time()
                timestamp_monotonic = time.monotonic()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame read failed")
                frame_height, frame_width = frame.shape[:2]
                image_size = (int(frame_width), int(frame_height))
                if effective_camera_matrix is None or effective_image_size != image_size:
                    effective_camera_matrix = _scale_camera_matrix(
                        raw_camera_matrix,
                        calibration_size,
                        image_size,
                    )
                    effective_image_size = image_size

                corners_list, ids, _rejected = detector.detectMarkers(frame)
                observations: list[dict[str, Any]] = []
                detected_ids: list[int] = []
                display = frame.copy() if not args.no_display else None

                if ids is not None:
                    for marker_id_array, corners_raw in zip(ids, corners_list):
                        marker_id = int(marker_id_array[0])
                        if marker_id not in body_T_tags:
                            continue
                        corners_px = np.asarray(corners_raw, dtype=np.float64).reshape(4, 2)
                        undistorted_corners = _undistort_corners(
                            corners_px,
                            effective_camera_matrix,
                            distortion,
                            distortion_model,
                        )
                        raw_candidates = _solve_ippe_candidates(
                            undistorted_corners,
                            marker_size_m,
                            effective_camera_matrix,
                        )
                        candidate_records = [
                            _candidate_record(candidate, body_T_tag=body_T_tags[marker_id])
                            for candidate in raw_candidates
                        ]
                        selected_candidate_index = None
                        if candidate_records:
                            selected_candidate_index = int(
                                min(
                                    range(len(candidate_records)),
                                    key=lambda index: candidate_records[index][
                                        "reprojection_error_px"
                                    ],
                                )
                            )
                        observations.append(
                            {
                                "marker_id": marker_id,
                                "corners_px": corners_px.tolist(),
                                "undistorted_corners_px": undistorted_corners.tolist(),
                                "body_T_tag": body_T_tags[marker_id].tolist(),
                                "selected_candidate_list_index": selected_candidate_index,
                                "candidates": candidate_records,
                            }
                        )
                        detected_ids.append(marker_id)
                        marker_counts[marker_id] += 1
                        candidate_count += len(candidate_records)

                        if display is not None:
                            cv2.polylines(
                                display,
                                [np.rint(corners_px).astype(np.int32)],
                                True,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            if selected_candidate_index is not None:
                                selected = candidate_records[selected_candidate_index]
                                _draw_axes(
                                    display,
                                    np.asarray(selected["camera_T_tag"], dtype=np.float64),
                                    effective_camera_matrix,
                                    distortion,
                                    distortion_model,
                                    length_m=0.018,
                                    thickness=1,
                                )
                                _draw_axes(
                                    display,
                                    np.asarray(
                                        selected["centroid"]["camera_T_body"],
                                        dtype=np.float64,
                                    ),
                                    effective_camera_matrix,
                                    distortion,
                                    distortion_model,
                                    length_m=0.028,
                                    thickness=3,
                                )

                detected_ids.sort()
                if detected_ids:
                    frames_with_markers += 1
                frame_record = {
                    "record_type": "frame",
                    "frame_index": frame_count,
                    "timestamp_wall_s": timestamp_wall,
                    "timestamp_monotonic_s": timestamp_monotonic,
                    "elapsed_s": timestamp_monotonic - started_monotonic,
                    "image_size": [frame_width, frame_height],
                    "camera_matrix": effective_camera_matrix.tolist(),
                    "detected_ids": detected_ids,
                    "observations": observations,
                }
                output_file.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
                frame_count += 1
                if frame_count % 30 == 0:
                    output_file.flush()
                    elapsed = max(timestamp_monotonic - started_monotonic, 1e-9)
                    print(
                        "[opencv-tags2marker] "
                        f"frames={frame_count} measured_fps={frame_count / elapsed:.1f} "
                        f"detected={dict(sorted(marker_counts.items()))}"
                    )

                if display is not None:
                    cv2.imshow("OpenCV tags2marker recorder", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                if args.max_frames is not None and frame_count >= args.max_frames:
                    break
                if args.duration_sec > 0.0 and (
                    timestamp_monotonic - started_monotonic >= args.duration_sec
                ):
                    break

            output_file.flush()
    finally:
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = max(time.monotonic() - started_monotonic, 1e-9)
    summary = {
        "schema_version": 1,
        "trace": str(output_path),
        "frames": frame_count,
        "frames_with_markers": frames_with_markers,
        "candidate_records": candidate_count,
        "duration_sec": elapsed,
        "measured_fps": frame_count / elapsed,
        "marker_detection_counts": {
            str(marker_id): count for marker_id, count in sorted(marker_counts.items())
        },
        "actual_camera_mode": actual_mode,
        "started_wall_s": started_wall,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[opencv-tags2marker] complete "
        f"frames={frame_count} measured_fps={frame_count / elapsed:.1f}"
    )
    print(f"[opencv-tags2marker] trace={output_path}")
    print(f"[opencv-tags2marker] summary={summary_path}")


if __name__ == "__main__":
    main()
