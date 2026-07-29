"""RealSense depth sampling and camera/body point conversion."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexslide.kinematics.transforms import invert_transform

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


def sample_depth_frame_m(depth_frame: Any, u: int, v: int, *, window_radius: int) -> float | None:
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    samples: list[float] = []
    for dv in range(-int(window_radius), int(window_radius) + 1):
        for du in range(-int(window_radius), int(window_radius) + 1):
            uu = int(np.clip(u + du, 0, max(0, width - 1)))
            vv = int(np.clip(v + dv, 0, max(0, height - 1)))
            z_m = float(depth_frame.get_distance(uu, vv))
            if z_m > 0.0:
                samples.append(z_m)
    return None if not samples else float(np.median(np.asarray(samples, dtype=np.float64)))


def deproject_keypoint_points(depth_frame: Any, keypoints_2d: np.ndarray, *, landmark_indices: tuple[int, ...], window_radius: int) -> np.ndarray | None:
    if rs is None:
        raise RuntimeError("pyrealsense2 is required for depth deprojection")
    keypoints = np.asarray(keypoints_2d, dtype=np.float64).reshape(-1, 2)
    if keypoints.shape[0] <= max(int(idx) for idx in landmark_indices):
        return None
    intr = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    points_camera_xyz: list[np.ndarray] = []
    for landmark_idx in landmark_indices:
        uv = keypoints[int(landmark_idx)]
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        z_m = sample_depth_frame_m(depth_frame, u, v, window_radius=window_radius)
        if z_m is None or z_m <= 0.0:
            return None
        point_xyz = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))
        points_camera_xyz.append(np.asarray(point_xyz, dtype=np.float64).reshape(3))
    return np.asarray(points_camera_xyz, dtype=np.float64).reshape(len(landmark_indices), 3)


def camera_points_to_body_points(transform_camera_body: np.ndarray, camera_points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(camera_points_xyz, dtype=np.float64).reshape(-1, 3)
    points_h = np.ones((points.shape[0], 4), dtype=np.float64)
    points_h[:, :3] = points
    return ((invert_transform(transform_camera_body) @ points_h.T).T[:, :3]).reshape(points.shape[0], 3)


def wrist_body_point(transform_camera_body: np.ndarray, wrist_camera_xyz: np.ndarray) -> np.ndarray:
    return np.asarray(camera_points_to_body_points(transform_camera_body, wrist_camera_xyz)[0], dtype=np.float64).reshape(3)

