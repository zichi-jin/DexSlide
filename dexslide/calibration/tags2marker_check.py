"""实时检查 tags2marker.json 的 ArUco 几何配置。

运行示例：
python -m dexslide.calibration.tags2marker_check --source 0
按 Esc 或 q 退出。彩色坐标轴表示每个 tag 按配置变换后得到的 marker frame。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from dexslide.kinematics.transforms import (
    invert_transform,
    rvec_tvec_to_transform,
    transform_to_rvec_tvec,
)
from dexslide.vision.aruco_pose_tracker import (
    _convert_fisheye_intrinsics_resolution,
    _detect_localize_aruco_tags,
    _parse_capture_source,
    _parse_fisheye_intrinsics,
    _parse_aruco_config,
)
from dexslide.visualization.aruco_overlay import project_points
from dexslide.world_pose.hand_cube_overlay import (
    resolve_marker_body_tag_pose_branches,
    try_load_hand_cube_overlay_config,
)


def _load_json_with_comments(path: Path) -> dict:
    text = re.sub(r"(?m)^\s*//.*$", "", path.read_text(encoding="utf-8"))
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def _transform_body_to_marker(rows: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Build T_body_marker from the config's axes_rows_body and p_mm.

    The JSON stores marker axes expressed as rows in the body frame.  Therefore
    the rotation matrix used by homogeneous transforms is ``rows.T``.  ``p_mm``
    is the marker-origin position expressed in the body frame.
    """
    out = np.eye(4)
    out[:3, :3] = np.asarray(rows, dtype=float).reshape(3, 3).T
    out[:3, 3] = np.asarray(p, dtype=float) / 1000.0
    return out


def _draw_axis(image, pose, k, d, length=0.025, colors=None, thickness=2):
    pts = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]], dtype=float)
    rvec, tvec = transform_to_rvec_tvec(pose)
    p = project_points(pts, rvec, tvec, {"K": k, "D": d}).astype(int)
    o = tuple(p[0])
    if colors is None:
        colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
    for q, color in zip(p[1:], colors):
        cv2.line(image, o, tuple(q), color, thickness)
    return o


class _Pose3DPlot:
    """实时显示每个 tag 反算出的 camera_T_body。"""

    def __init__(self, axis_length_mm: float = 20.0):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.axis_length_mm = axis_length_mm
        plt.ion()
        self.fig = plt.figure("tags2marker 3D")
        self.ax = self.fig.add_subplot(111, projection="3d")

    def update(self, poses: dict[int, np.ndarray]) -> None:
        ax = self.ax
        ax.clear()
        points = []
        for marker_id, pose in sorted(poses.items()):
            origin = pose[:3, 3] * 1000.0
            points.append(origin)
            ax.scatter(*origin, color="k", s=18)
            ax.text(*origin, str(marker_id), fontsize=8)
            for axis, color in enumerate(("r", "g", "b")):
                direction = pose[:3, axis] * self.axis_length_mm
                points.append(origin + direction)
                ax.quiver(*origin, *direction, color=color, linewidth=1.5, arrow_length_ratio=0.12)

        if points:
            points_array = np.asarray(points)
            center = np.mean(points_array, axis=0)
            half_range = max(30.0, float(np.max(np.ptp(points_array, axis=0))) / 2.0 + 10.0)
            ax.set_xlim(center[0] - half_range, center[0] + half_range)
            ax.set_ylim(center[1] - half_range, center[1] + half_range)
            ax.set_zlim(center[2] - half_range, center[2] + half_range)
            # ax.set_xlim(50, 200)
            # ax.set_ylim(-30, 50)
            # ax.set_zlim(250, 470)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("camera X (mm)")
        ax.set_ylabel("camera Y (mm)")
        ax.set_zlabel("camera Z (mm)")
        ax.set_title("Recovered body frames: tag ID at each origin")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        self.plt.close(self.fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("assets/calibration/direct_aruco/left_tags2marker.json"))
    parser.add_argument("--camera-intrinsics", type=Path, default=Path("assets/calibration/direct_aruco/d435i_intrinsic.json"))
    parser.add_argument("--source", default="/dev/video4", help="相机编号或视频路径")
    parser.add_argument("--position-threshold-mm", type=float, default=5.0)
    parser.add_argument("--rotation-threshold-deg", type=float, default=8.0)
    parser.add_argument("--3d-axis-length-mm", type=float, default=20.0)
    args = parser.parse_args()

    cfg = _load_json_with_comments(args.config)
    intr_raw = _load_json_with_comments(args.camera_intrinsics)
    intr = _parse_fisheye_intrinsics(intr_raw)
    aruco = _parse_aruco_config(cfg)
    marker_ids = {int(k): v for k, v in cfg["marker_face_id"].items()}
    # ArUco 位姿解算使用黑色 ArUco 边界尺寸；40 mm 外部面尺寸只用于 p_mm 几何。
    marker_size = float(cfg.get("aruco_bound_size_mm", cfg.get("aruco_bound_size", 30.0))) / 1000.0
    aruco["marker_size_map"] = {mid: marker_size for mid in marker_ids}
    overlay_cfg = try_load_hand_cube_overlay_config(args.config)

    source = _parse_capture_source(args.source)
    # Linux 摄像头优先使用 V4L2，避免 OpenCV 在默认 backend 间切换时触发 Qt 线程警告。
    if (isinstance(source, int) or str(source).startswith("/dev/video")) and hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: {args.source}")
    cv2.namedWindow("tags2marker check", cv2.WINDOW_NORMAL)
    pose_plot = _Pose3DPlot(getattr(args, "3d_axis_length_mm"))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        cam_intr = _convert_fisheye_intrinsics_resolution(intr, (w, h))
        tags = _detect_localize_aruco_tags(
            img_bgr=frame, aruco_dict=aruco["aruco_dict"], marker_size_map=aruco["marker_size_map"],
            fisheye_intr_dict=cam_intr, refine_subpix=True, motion_tolerant=True,
        )
        # 平面 ArUco 的 IPPE 会给出两个姿态分支。主 overlay 会根据所有
        # marker 的几何一致性选择分支；这里复用同一逻辑，否则原始坐标轴
        # 可能与主 overlay 看起来相反。
        if tags and overlay_cfg is not None:
            resolve_marker_body_tag_pose_branches(tags, overlay_cfg)
        poses = {}
        tag_poses = {}
        for mid, entry in tags.items():
            camera_tag = rvec_tvec_to_transform(entry["rvec"], entry["tvec"])
            tag_poses[mid] = camera_tag
            # tags2marker 配置实际保存 T_body_marker；要从相机下的 tag
            # 反推出 body，必须乘以其逆矩阵，而不是正向矩阵。
            body_to_marker = _transform_body_to_marker(
                np.asarray(marker_ids[mid]["rot"]), np.asarray(marker_ids[mid]["p_mm"])
            )
            poses[mid] = camera_tag @ invert_transform(body_to_marker)
            cv2.polylines(frame, [entry["corners"].astype(int)], True, (255, 255, 0), 2)
        if poses:
            pose_plot.update(poses)
            origin = np.mean([p[:3, 3] for p in poses.values()], axis=0)
            ref = next(iter(poses.values()))[:3, :3]
            for mid, pose in poses.items():
                pos_err = np.linalg.norm(pose[:3, 3] - origin) * 1000.0
                rot_err = np.degrees(np.arccos(np.clip((np.trace(ref.T @ pose[:3, :3]) - 1) / 2, -1, 1)))
                bad = pos_err > args.position_threshold_mm or rot_err > args.rotation_threshold_deg
                color = (0, 0, 255) if bad else (0, 220, 0)
                # 细线显示 OpenCV 原始 tag frame；粗线显示按配置还原的 body frame。
                _draw_axis(
                    frame,
                    tag_poses[mid],
                    cam_intr["K"],
                    cam_intr["D"],
                    length=0.018,
                    colors=((255, 255, 0), (255, 0, 255), (0, 255, 255)),
                    thickness=1,
                )
                pixel = _draw_axis(frame, pose, cam_intr["K"], cam_intr["D"], thickness=3)
                text = f"id {mid}: {'BAD' if bad else 'OK'} {pos_err:.1f}mm {rot_err:.1f}deg"
                cv2.putText(frame, text, (pixel[0] + 5, pixel[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(
                frame,
                "No configured ArUco marker detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        cv2.imshow("tags2marker check", frame)
        if not pose_plot.is_open():
            break
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()
    pose_plot.close()


if __name__ == "__main__":
    main()
