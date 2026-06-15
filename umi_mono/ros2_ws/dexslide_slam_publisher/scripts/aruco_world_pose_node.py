#!/usr/bin/env python3
import json
import os
import sys
import time
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster


def _resolve_repo_root() -> str:
    candidates = []

    env_root = os.environ.get("DEXSLIDE_UMI_MONO_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            Path("/home/jzq/MyJob/DexSlide/umi_mono"),
            Path("/data/codes/DexSlide/umi_mono"),
        ]
    )

    here = Path(__file__).resolve()
    try:
        candidates.append(here.parents[5])
    except IndexError:
        pass

    for candidate in candidates:
        if candidate.joinpath("umi/common/cv_util.py").is_file():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
            return candidate_str

    fallback = str(Path("/home/jzq/MyJob/DexSlide/umi_mono"))
    if fallback not in sys.path:
        sys.path.append(fallback)
    return fallback


DEFAULT_REPO_ROOT = _resolve_repo_root()

from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


def stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pose_msg_to_mat(msg: PoseStamped) -> np.ndarray:
    q = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    n = np.linalg.norm(q)
    if n <= 1e-12:
        raise ValueError("PoseStamped contains a zero quaternion")
    q /= n
    x, y, z, w = q
    rot = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = [
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
    ]
    return mat


def tvec_rvec_to_mat(tvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return mat


def quat_xyzw_from_mat(rot: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return tuple(float(x) for x in q)


class PoseBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.times: List[float] = []
        self.poses: List[np.ndarray] = []

    def append(self, t: float, mat: np.ndarray) -> None:
        self.times.append(t)
        self.poses.append(mat)
        if len(self.times) > self.max_size:
            extra = len(self.times) - self.max_size
            del self.times[:extra]
            del self.poses[:extra]

    def nearest(self, t: float, max_dt: float) -> Optional[Tuple[float, np.ndarray]]:
        if not self.times:
            return None
        idx = bisect_left(self.times, t)
        candidates = []
        if idx < len(self.times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best_idx = min(candidates, key=lambda i: abs(self.times[i] - t))
        dt = abs(self.times[best_idx] - t)
        if dt > max_dt + 1e-6:
            return None
        return dt, self.poses[best_idx]

    def nearest_dt(self, t: float) -> Optional[float]:
        if not self.times:
            return None
        idx = bisect_left(self.times, t)
        candidates = []
        if idx < len(self.times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best_idx = min(candidates, key=lambda i: abs(self.times[i] - t))
        return abs(self.times[best_idx] - t)


class ArucoWorldPoseNode(Node):
    def __init__(self):
        super().__init__("aruco_world_pose_node")
        repo_root = self.declare_parameter("repo_root", DEFAULT_REPO_ROOT).value
        if repo_root not in sys.path:
            sys.path.append(repo_root)

        self.image_topic = self.declare_parameter(
            "image_topic", "/camera/camera/color/image_raw"
        ).value
        self.slam_pose_topic = self.declare_parameter(
            "slam_pose_topic", "/dexslide/slam/pose"
        ).value
        self.output_pose_topic = self.declare_parameter(
            "output_pose_topic", "/dexslide/aruco/world_pose"
        ).value
        self.world_frame = self.declare_parameter("world_frame", "world").value
        self.marker_frame = self.declare_parameter("marker_frame", "aruco_marker").value
        self.target_marker_id = int(
            self.declare_parameter("target_marker_id", 10).value
        )
        self.max_pose_dt = float(self.declare_parameter("max_pose_dt", 0.2).value)
        self.input_pose_is_twc = bool(
            self.declare_parameter("input_pose_is_twc", True).value
        )
        self.broadcast_tf = bool(self.declare_parameter("broadcast_tf", True).value)
        pose_buffer_size = int(self.declare_parameter("pose_buffer_size", 256).value)

        camera_intrinsics = self.declare_parameter(
            "camera_intrinsics",
            os.path.join(repo_root, "example/calibration/d435i_960_540.json"),
        ).value
        aruco_yaml = self.declare_parameter(
            "aruco_yaml",
            os.path.join(repo_root, "example/calibration/aruco_config_wrist.yaml"),
        ).value
        tx_slam_tag = self.declare_parameter("tx_slam_tag", "").value
        if not tx_slam_tag:
            raise ValueError(
                "Parameter tx_slam_tag is required for world-frame ArUco output"
            )

        self.t_world_slam = self.load_world_slam(tx_slam_tag)
        with open(camera_intrinsics, "r") as f:
            self.raw_intr = parse_fisheye_intrinsics(json.load(f))
        with open(aruco_yaml, "r") as f:
            aruco_config = parse_aruco_config(yaml.safe_load(f))
        self.aruco_dict = aruco_config["aruco_dict"]
        self.marker_size_map: Dict[int, float] = aruco_config["marker_size_map"]
        if self.target_marker_id not in self.marker_size_map:
            raise ValueError(
                f"target_marker_id={self.target_marker_id} is not in marker_size_map"
            )

        self.bridge = CvBridge()
        self.pose_buffer = PoseBuffer(max_size=pose_buffer_size)
        self.intr_by_resolution: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self.pose_rx_count = 0
        self.image_rx_count = 0
        self.publish_count = 0
        self.detect_count = 0
        self.target_detect_count = 0
        self.last_pose_stamp: Optional[float] = None
        self.last_image_stamp: Optional[float] = None
        self.last_publish_stamp: Optional[float] = None
        self.last_pose_match_dt: Optional[float] = None
        self.last_detected_ids: List[int] = []
        self.last_no_publish_reason = "startup"
        self.last_status_log_time = 0.0
        self.status_log_period_s = 2.0

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pose_pub = self.create_publisher(PoseStamped, self.output_pose_topic, qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            PoseStamped, self.slam_pose_topic, self.on_slam_pose, qos
        )
        self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.create_timer(self.status_log_period_s, self.log_status)

        self.get_logger().info(
            f"Tracking ArUco-{self.target_marker_id} on {self.image_topic}; "
            f"publishing {self.output_pose_topic} in frame '{self.world_frame}'"
        )

    def load_world_slam(self, path_str: str) -> np.ndarray:
        path = Path(path_str).expanduser().absolute()
        if not path.is_file():
            raise FileNotFoundError(f"tx_slam_tag file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        if "tx_slam_tag" not in data:
            raise KeyError(f"'tx_slam_tag' missing in {path}")
        t_slam_tag = np.asarray(data["tx_slam_tag"], dtype=np.float64)
        if t_slam_tag.shape != (4, 4):
            raise ValueError(
                f"Expected 4x4 tx_slam_tag in {path}, got {t_slam_tag.shape}"
            )
        return np.linalg.inv(t_slam_tag)

    def on_slam_pose(self, msg: PoseStamped) -> None:
        try:
            mat = pose_msg_to_mat(msg)
        except ValueError as e:
            self.get_logger().warn(str(e))
            return
        if not self.input_pose_is_twc:
            mat = np.linalg.inv(mat)
        pose_t = stamp_to_float(msg.header.stamp)
        self.pose_buffer.append(pose_t, mat)
        self.pose_rx_count += 1
        self.last_pose_stamp = pose_t

    def intrinsics_for(self, width: int, height: int) -> Dict[str, np.ndarray]:
        key = (width, height)
        if key not in self.intr_by_resolution:
            self.intr_by_resolution[key] = convert_fisheye_intrinsics_resolution(
                opencv_intr_dict=self.raw_intr,
                target_resolution=key,
            )
        return self.intr_by_resolution[key]

    def on_image(self, msg: Image) -> None:
        image_t = stamp_to_float(msg.header.stamp)
        self.image_rx_count += 1
        self.last_image_stamp = image_t
        pose_match = self.pose_buffer.nearest(image_t, self.max_pose_dt)
        if pose_match is None:
            nearest_dt = self.pose_buffer.nearest_dt(image_t)
            self.last_pose_match_dt = nearest_dt
            if nearest_dt is None:
                self.last_no_publish_reason = "no_slam_pose_received"
            else:
                self.last_no_publish_reason = (
                    f"slam_pose_stale(dt={nearest_dt:.3f}s > {self.max_pose_dt:.3f}s)"
                )
            return
        pose_dt, t_slam_camera = pose_match
        self.last_pose_match_dt = pose_dt

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.last_no_publish_reason = f"image_convert_failed({e})"
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        height, width = frame_bgr.shape[:2]
        intr = self.intrinsics_for(width, height)
        try:
            tag_dict = detect_localize_aruco_tags(
                img=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                aruco_dict=self.aruco_dict,
                marker_size_map=self.marker_size_map,
                fisheye_intr_dict=intr,
                refine_subpix=True,
            )
        except Exception as e:
            self.last_no_publish_reason = f"aruco_detect_failed({e})"
            self.get_logger().error(f"Aruco detection failed: {e}")
            return
        self.last_detected_ids = sorted(int(x) for x in tag_dict.keys())
        if tag_dict:
            self.detect_count += 1
        if self.target_marker_id not in tag_dict:
            if self.last_detected_ids:
                self.last_no_publish_reason = (
                    "target_marker_missing(detected_ids="
                    f"{self.last_detected_ids}, target={self.target_marker_id})"
                )
            else:
                self.last_no_publish_reason = "no_aruco_detected"
            return

        self.target_detect_count += 1
        tag = tag_dict[self.target_marker_id]
        t_camera_marker = tvec_rvec_to_mat(tag["tvec"], tag["rvec"])
        t_world_marker = self.t_world_slam @ t_slam_camera @ t_camera_marker
        self.publish_world_marker(msg.header.stamp, t_world_marker)

    def publish_world_marker(self, stamp, mat: np.ndarray) -> None:
        qx, qy, qz, qw = quat_xyzw_from_mat(mat[:3, :3])
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.pose.position.x = float(mat[0, 3])
        msg.pose.position.y = float(mat[1, 3])
        msg.pose.position.z = float(mat[2, 3])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pose_pub.publish(msg)
        self.publish_count += 1
        self.last_publish_stamp = stamp_to_float(stamp)
        self.last_no_publish_reason = (
            f"published(target={self.target_marker_id}, pose_dt={self.last_pose_match_dt:.3f}s)"
        )

        if self.broadcast_tf:
            tf_msg = TransformStamped()
            tf_msg.header = msg.header
            tf_msg.child_frame_id = (
                self.marker_frame
                if self.marker_frame
                else f"aruco_{self.target_marker_id}"
            )
            tf_msg.transform.translation.x = msg.pose.position.x
            tf_msg.transform.translation.y = msg.pose.position.y
            tf_msg.transform.translation.z = msg.pose.position.z
            tf_msg.transform.rotation = msg.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

    def log_status(self) -> None:
        now = time.monotonic()
        if now - self.last_status_log_time < self.status_log_period_s:
            return
        self.last_status_log_time = now

        if self.last_pose_stamp is None:
            pose_age_str = "none"
        elif self.last_image_stamp is None:
            pose_age_str = "waiting_image"
        else:
            pose_age_str = f"{abs(self.last_image_stamp - self.last_pose_stamp):.3f}s"

        if self.last_pose_match_dt is None:
            pose_match_dt_str = "none"
        else:
            pose_match_dt_str = f"{self.last_pose_match_dt:.3f}s"

        self.get_logger().info(
            "Aruco status: "
            f"poses={self.pose_rx_count}, "
            f"images={self.image_rx_count}, "
            f"aruco_frames={self.detect_count}, "
            f"target_hits={self.target_detect_count}, "
            f"publishes={self.publish_count}, "
            f"buffer={len(self.pose_buffer.times)}, "
            f"last_pose_gap={pose_age_str}, "
            f"last_match_dt={pose_match_dt_str}, "
            f"last_ids={self.last_detected_ids}, "
            f"last_result={self.last_no_publish_reason}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArucoWorldPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
