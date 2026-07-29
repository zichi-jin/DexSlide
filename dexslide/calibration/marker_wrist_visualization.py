from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dexslide.kinematics.transforms import transform_to_rvec_tvec
from dexslide.visualization.aruco_overlay import draw_axes, draw_marker_outline

def _estimate_bbox_xyxy(
    keypoints_2d: np.ndarray,
    image_shape: tuple[int, ...],
    padding_px: int = 24,
) -> tuple[int, int, int, int]:
    height, width = int(image_shape[0]), int(image_shape[1])
    pts = np.asarray(keypoints_2d, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return (0, 0, max(0, width - 1), max(0, height - 1))
    x0 = int(np.floor(float(np.min(pts[:, 0])))) - int(padding_px)
    y0 = int(np.floor(float(np.min(pts[:, 1])))) - int(padding_px)
    x1 = int(np.ceil(float(np.max(pts[:, 0])))) + int(padding_px)
    y1 = int(np.ceil(float(np.max(pts[:, 1])))) + int(padding_px)
    x0 = int(np.clip(x0, 0, max(0, width - 1)))
    y0 = int(np.clip(y0, 0, max(0, height - 1)))
    x1 = int(np.clip(x1, 0, max(0, width - 1)))
    y1 = int(np.clip(y1, 0, max(0, height - 1)))
    return (x0, y0, x1, y1)


def _draw_projected_axes(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    transform_camera_body: np.ndarray,
    *,
    axis_length_m: float,
    label: str,
) -> None:
    rvec, tvec = transform_to_rvec_tvec(transform_camera_body)
    draw_axes(
        image_bgr,
        {
            "K": np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
            "D": np.zeros((1, 5), dtype=np.float64),
        },
        rvec,
        tvec,
        axis_length_m,
        label,
        (255, 255, 255),
    )


def _draw_marker_outlines(image_bgr: np.ndarray, tag_dict: dict[int, dict[str, Any]]) -> None:
    for marker_id, tag in sorted(tag_dict.items(), key=lambda item: int(item[0])):
        corners = np.asarray(tag.get("corners"), dtype=np.float64).reshape(4, 2)
        draw_marker_outline(image_bgr, corners, (0, 215, 255), str(int(marker_id)))


def _draw_status_lines(image_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image_bgr.copy()
    line_height = 24
    box_height = max(36, 12 + line_height * len(lines))
    cv2.rectangle(out, (8, 8), (900, box_height), (15, 15, 15), -1)
    cv2.rectangle(out, (8, 8), (900, box_height), (90, 90, 90), 1)
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (18, 32 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return out


def _compose_display_frame(color_bgr: np.ndarray, depth_bgr: np.ndarray, *, view: str) -> np.ndarray:
    if view == "color":
        return color_bgr
    if view == "depth":
        return depth_bgr
    if view != "split":
        raise ValueError(f"Unsupported view: {view}")
    return np.hstack([color_bgr, depth_bgr])


