"""OpenCV rendering helpers for the direct-ArUco camera overlay."""

from __future__ import annotations

import cv2
import numpy as np


HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
]
PALM_BONES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 5), (5, 9), (9, 13), (13, 17)]
HAND_BONE_COLORS = {
    "thumb": (80, 120, 255), "index": (255, 120, 80), "middle": (80, 255, 120),
    "ring": (40, 220, 220), "pinky": (220, 80, 220),
}


def project_points(object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, intr: dict[str, np.ndarray]) -> np.ndarray:
    """Project 3D points with pinhole or fisheye OpenCV intrinsics."""
    obj = np.ascontiguousarray(np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3))
    rvec = np.ascontiguousarray(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    tvec = np.ascontiguousarray(np.asarray(tvec, dtype=np.float64).reshape(3, 1))
    k = np.ascontiguousarray(np.asarray(intr["K"], dtype=np.float64).reshape(3, 3))
    d = np.ascontiguousarray(np.asarray(intr["D"], dtype=np.float64))
    if d.size == 4:
        image_points, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, k, d)
    else:
        image_points, _ = cv2.projectPoints(obj, rvec, tvec, k, d)
    return image_points.reshape(-1, 2)


def draw_axes(image: np.ndarray, intr: dict[str, np.ndarray], rvec: np.ndarray, tvec: np.ndarray, axis_length: float, label: str | None, label_color: tuple[int, int, int]) -> None:
    axis_points = np.array([[0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], [0.0, axis_length, 0.0], [0.0, 0.0, axis_length]], dtype=np.float64)
    projected = project_points(axis_points, rvec, tvec, intr)
    origin = tuple(np.round(projected[0]).astype(int))
    for idx, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), start=1):
        endpoint = tuple(np.round(projected[idx]).astype(int))
        cv2.line(image, origin, endpoint, color, 2, lineType=cv2.LINE_AA)
        cv2.circle(image, endpoint, 3, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, origin, 4, label_color, -1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(image, label, (origin[0] + 6, origin[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2, lineType=cv2.LINE_AA)


def draw_marker_outline(image: np.ndarray, corners: np.ndarray, color: tuple[int, int, int], text: str | None) -> None:
    points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    polygon = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
    if text:
        anchor = tuple(np.round(points[0]).astype(int))
        cv2.putText(image, text, (anchor[0] + 4, anchor[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, lineType=cv2.LINE_AA)


def draw_hud(image: np.ndarray, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        y = 26 + index * 24
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (24, 24, 24), 1, lineType=cv2.LINE_AA)


def marker_id_to_color(marker_id: int) -> tuple[int, int, int]:
    palette = [(80, 255, 80), (80, 180, 255), (180, 120, 255), (255, 180, 80), (255, 120, 180)]
    return palette[marker_id % len(palette)]


def _depth_scaled_int(depth_m: float, *, near_depth_m: float, far_depth_m: float, near_value: int, far_value: int) -> int:
    alpha = float(np.clip((depth_m - near_depth_m) / (far_depth_m - near_depth_m), 0.0, 1.0))
    return max(1, int(round(near_value + alpha * (far_value - near_value))))


def _hand_bone_color(start_idx: int) -> tuple[int, int, int]:
    if start_idx <= 3:
        return HAND_BONE_COLORS["thumb"]
    if start_idx <= 7:
        return HAND_BONE_COLORS["index"]
    if start_idx <= 11:
        return HAND_BONE_COLORS["middle"]
    if start_idx <= 15:
        return HAND_BONE_COLORS["ring"]
    return HAND_BONE_COLORS["pinky"]


def draw_projected_hand(image: np.ndarray, intr: dict[str, np.ndarray], camera_points: np.ndarray, *, draw_axes_enabled: bool, axis_rvec: np.ndarray | None, axis_tvec: np.ndarray | None, axis_length_m: float) -> None:
    """Draw a 21-landmark hand whose points are already in camera coordinates."""
    camera_points = np.asarray(camera_points, dtype=np.float64)
    projected = project_points(camera_points, np.zeros(3), np.zeros(3), intr)
    visible = camera_points[:, 2] > 1e-4
    for index_a, index_b in PALM_BONES:
        if visible[index_a] and visible[index_b]:
            depth_m = 0.5 * float(camera_points[index_a, 2] + camera_points[index_b, 2])
            cv2.line(image, tuple(np.round(projected[index_a]).astype(int)), tuple(np.round(projected[index_b]).astype(int)), (96, 96, 96), _depth_scaled_int(depth_m, near_depth_m=0.18, far_depth_m=1.20, near_value=4, far_value=1), lineType=cv2.LINE_AA)
    for index_a, index_b in HAND_BONES:
        if visible[index_a] and visible[index_b]:
            depth_m = 0.5 * float(camera_points[index_a, 2] + camera_points[index_b, 2])
            cv2.line(image, tuple(np.round(projected[index_a]).astype(int)), tuple(np.round(projected[index_b]).astype(int)), _hand_bone_color(max(index_a, index_b)), _depth_scaled_int(depth_m, near_depth_m=0.18, far_depth_m=1.20, near_value=5, far_value=2), lineType=cv2.LINE_AA)
    for index, point in enumerate(projected):
        if not visible[index]:
            continue
        radius = _depth_scaled_int(float(camera_points[index, 2]), near_depth_m=0.18, far_depth_m=1.20, near_value=6 if index == 0 else 5, far_value=3 if index == 0 else 2)
        cv2.circle(image, tuple(np.round(point).astype(int)), radius, (32, 32, 32) if index == 0 else (245, 245, 245), -1, lineType=cv2.LINE_AA)
    if draw_axes_enabled and axis_rvec is not None and axis_tvec is not None:
        draw_axes(image, intr, axis_rvec, axis_tvec, axis_length_m, None, (32, 32, 32))
