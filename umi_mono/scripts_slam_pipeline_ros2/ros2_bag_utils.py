import pathlib
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


ROSBAGS_INSTALL_HINT = "Missing dependency 'rosbags'. Install with: pip install rosbags"


def _load_any_reader():
    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except ImportError as exc:
        raise RuntimeError(ROSBAGS_INSTALL_HINT) from exc
    return AnyReader


def _load_default_typestore():
    try:
        from rosbags.typesys import Stores, get_typestore  # type: ignore
    except ImportError as exc:
        raise RuntimeError(ROSBAGS_INSTALL_HINT) from exc
    return get_typestore(Stores.ROS2_HUMBLE)


def _open_any_reader(paths: Sequence[pathlib.Path]):
    AnyReader = _load_any_reader()
    return AnyReader(paths, default_typestore=_load_default_typestore())


def ensure_rosbags_available() -> None:
    _load_any_reader()


def is_ros2_bag_dir(path: pathlib.Path) -> bool:
    """A ROS2 bag directory contains metadata.yaml."""
    try:
        return path.is_dir() and path.joinpath("metadata.yaml").is_file()
    except OSError:
        return False


def discover_ros2_bag_dirs(session_dir: pathlib.Path) -> List[pathlib.Path]:
    """List ROS2 bag directories directly under session dir."""
    if not session_dir.is_dir():
        return []
    return sorted(
        [p for p in session_dir.iterdir() if is_ros2_bag_dir(p)], key=lambda p: p.name
    )


def get_available_topics(bag_dir: pathlib.Path) -> List[str]:
    with _open_any_reader([bag_dir]) as reader:
        return sorted({conn.topic for conn in reader.connections})


def iter_deserialized_messages(
    bag_dir: pathlib.Path,
    topics: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, float, object]]:
    """Yield (topic, timestamp_sec, msg) from a ROS2 bag directory."""
    with _open_any_reader([bag_dir]) as reader:
        topic_set = set(topics) if topics else None
        if topic_set is None:
            connections = list(reader.connections)
        else:
            connections = [c for c in reader.connections if c.topic in topic_set]
        for connection, timestamp_ns, rawdata in reader.messages(
            connections=connections
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)
            yield connection.topic, float(timestamp_ns) * 1e-9, msg


def _flatten_msg_data(data: object) -> np.ndarray:
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, memoryview):
        arr = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, (bytes, bytearray)):
        arr = np.frombuffer(data, dtype=np.uint8)
    else:
        arr = np.asarray(data)
    return arr.reshape(-1)


def image_msg_to_bgr(msg: object) -> np.ndarray:
    """Convert sensor_msgs/Image-like message to BGR uint8 image."""
    height = int(getattr(msg, "height"))
    width = int(getattr(msg, "width"))
    encoding = str(getattr(msg, "encoding", "")).lower()
    step = int(getattr(msg, "step"))

    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image shape: height={height}, width={width}")

    flat = _flatten_msg_data(getattr(msg, "data"))

    if encoding in {"bgr8", "rgb8", "bgra8", "rgba8"}:
        channels = 4 if encoding in {"bgra8", "rgba8"} else 3
        expected = height * step
        if flat.size < expected:
            raise ValueError(
                f"Image buffer too small for {encoding}: {flat.size} < {expected}"
            )
        row = flat[:expected].reshape(height, step)
        img = row[:, : width * channels].reshape(height, width, channels)
        if encoding == "bgr8":
            return img.copy()
        if encoding == "rgb8":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    if encoding == "mono8":
        expected = height * step
        if flat.size < expected:
            raise ValueError(
                f"Image buffer too small for {encoding}: {flat.size} < {expected}"
            )
        row = flat[:expected].reshape(height, step)
        gray = row[:, :width]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported image encoding: {encoding}")


def image_msg_to_array(msg: object) -> np.ndarray:
    """Return image as ndarray. Currently normalized to BGR for compatibility."""
    return image_msg_to_bgr(msg)


def extract_scalar_from_msg(msg: object) -> float:
    """Extract numeric scalar from common ROS std_msgs-style messages."""
    if hasattr(msg, "data"):
        data = getattr(msg, "data")
        if isinstance(data, (float, int, np.floating, np.integer)):
            return float(data)
        data_np = np.asarray(data)
        if data_np.size == 1:
            return float(data_np.reshape(-1)[0])
    raise ValueError(f"Cannot extract scalar from message type {type(msg)}")
