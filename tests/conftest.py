import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ROS_LOG_DIR", "/tmp/roslog")
Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def rclpy_initialized():
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def fake_pose_msg():
    from geometry_msgs.msg import PoseStamped

    def _make(
        t: float,
        tx: float = 0.0,
        ty: float = 0.0,
        tz: float = 0.0,
        qx: float = 0.0,
        qy: float = 0.0,
        qz: float = 0.0,
        qw: float = 1.0,
    ) -> PoseStamped:
        msg = PoseStamped()
        sec = int(t)
        nanosec = int(round((t - sec) * 1e9))
        if nanosec >= 1000000000:
            sec += 1
            nanosec -= 1000000000
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.pose.position.x = tx
        msg.pose.position.y = ty
        msg.pose.position.z = tz
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    return _make


@pytest.fixture
def subscriber_factory(rclpy_initialized):
    from dexslide.world_pose import SlamPoseSubscriber

    created = []

    def _make(**kwargs):
        node_name = kwargs.pop("node_name", f"dexslide_test_{len(created)}")
        sub = SlamPoseSubscriber(node_name=node_name, **kwargs)
        created.append(sub)
        return sub

    yield _make

    for sub in created:
        try:
            if getattr(sub, "_executor", None) is not None:
                sub._executor.shutdown()
        except Exception:
            pass
        try:
            if getattr(sub, "_node", None) is not None:
                sub._node.destroy_node()
        except Exception:
            pass
