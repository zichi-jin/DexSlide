import concurrent.futures
import json
import os
import pathlib
import shutil
import sys
from datetime import datetime

import click
import cv2

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

from scripts_slam_pipeline_ros2.ros2_bag_utils import (
    discover_ros2_bag_dirs,
    ensure_rosbags_available,
    get_available_topics,
    image_msg_to_bgr,
    iter_deserialized_messages,
)


def _format_imu_sample(ts: float, value, first_ts: float):
    timestamp_str = (
        datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )
    return {
        "date": timestamp_str,
        "value": value,
        "cts": 1000.0 * (ts - first_ts),
    }


def process_ros2_bag(
    bag_dir: pathlib.Path,
    output_dir: pathlib.Path,
    image_topic: str,
    accel_topic: str,
    gyro_topic: str,
    imu_topic: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    imu_data = {
        "1": {
            "device name": "Intel RealSense D435i",
            "streams": {
                "ACCL": {"name": "Accelerometer", "units": "m/s2", "samples": []},
                "GYRO": {"name": "Gyroscope", "units": "rad/s", "samples": []},
                "GORI": {"name": "Orientation", "units": "quaternion", "samples": []},
            },
        }
    }

    images = []
    image_timestamps = []
    accel_samples = []
    gyro_samples = []

    expected_topics = {image_topic, accel_topic, gyro_topic, imu_topic}
    available_topics = get_available_topics(bag_dir)
    print(f"[{bag_dir.name}] topics: {available_topics}")

    matched_topics = [t for t in expected_topics if t in available_topics]
    if not matched_topics:
        raise RuntimeError(
            f"None of expected topics found in {bag_dir}. expected={sorted(expected_topics)}"
        )

    for topic, ts, msg in iter_deserialized_messages(bag_dir, topics=matched_topics):
        if topic == image_topic:
            cv_img = image_msg_to_bgr(msg)
            images.append(cv_img)
            image_timestamps.append(ts)
        elif topic == accel_topic:
            accel_samples.append(
                (
                    ts,
                    [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                    ],
                )
            )
        elif topic == gyro_topic:
            gyro_samples.append(
                (
                    ts,
                    [
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ],
                )
            )
        elif topic == imu_topic:
            accel_samples.append(
                (
                    ts,
                    [
                        msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z,
                    ],
                )
            )
            gyro_samples.append(
                (
                    ts,
                    [
                        msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z,
                    ],
                )
            )

    if not images:
        raise RuntimeError(f"No image frames found on topic {image_topic} in {bag_dir}")

    accel_samples.sort(key=lambda x: x[0])
    gyro_samples.sort(key=lambda x: x[0])

    if gyro_samples:
        first_ts = gyro_samples[0][0]
    elif accel_samples:
        first_ts = accel_samples[0][0]
    else:
        first_ts = image_timestamps[0]

    for ts, value in accel_samples:
        if ts < first_ts:
            continue
        imu_data["1"]["streams"]["ACCL"]["samples"].append(
            _format_imu_sample(ts, value, first_ts)
        )

    for ts, value in gyro_samples:
        imu_data["1"]["streams"]["GYRO"]["samples"].append(
            _format_imu_sample(ts, value, first_ts)
        )
        imu_data["1"]["streams"]["GORI"]["samples"].append(
            _format_imu_sample(ts, [1, 0, 0, 0], first_ts)
        )

    imu_json_path = output_dir.joinpath("imu_data.json")
    with imu_json_path.open("w") as f:
        json.dump(imu_data, f, indent=2)

    actual_fps = 30.0
    if len(image_timestamps) > 1:
        duration = image_timestamps[-1] - image_timestamps[0]
        if duration > 0:
            actual_fps = (len(image_timestamps) - 1) / duration
        print(
            f"Actual video FPS: {actual_fps:.2f} ({len(image_timestamps)} frames, duration {duration:.2f}s)"
        )
    else:
        print("Not enough frames to calculate FPS, fallback to 30 FPS.")

    h, w, _ = images[0].shape
    video_path = output_dir.joinpath("raw_video.mp4")
    print(f"Saving mp4 video to {video_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, actual_fps, (w, h))
    for img in images:
        out.write(img)
    out.release()


def move_largest_video_to_mapping(demos_dir: pathlib.Path, session_dir: pathlib.Path):
    mapping_dir = demos_dir / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    mapping_video = mapping_dir / "raw_video.mp4"
    if mapping_video.exists():
        print(f"{mapping_video} already exists, skip moving.")
        return

    demo_dirs = [x for x in demos_dir.iterdir() if x.is_dir() and x.name.startswith("data")]
    video_info = []

    for demo_dir in demo_dirs:
        video_path = demo_dir / "raw_video.mp4"
        imu_path = demo_dir / "imu_data.json"
        if video_path.exists():
            video_info.append(
                {
                    "video": video_path,
                    "imu": imu_path,
                    "size": video_path.stat().st_size,
                    "folder": demo_dir,
                }
            )

    if not video_info:
        print("No data* videos found for mapping promotion.")
        return

    largest = max(video_info, key=lambda x: x["size"])
    largest_name = largest["folder"].name
    print(f"Promoting {largest_name} to mapping (largest data* video by size).")

    src_bag_dir = session_dir / largest_name
    dst_bag_dir = session_dir / "mapping"
    if src_bag_dir.is_dir() and not dst_bag_dir.exists():
        shutil.move(str(src_bag_dir), str(dst_bag_dir))

    shutil.move(str(largest["video"]), str(mapping_video))

    if largest["imu"].exists():
        mapping_imu = mapping_dir / "imu_data.json"
        shutil.move(str(largest["imu"]), str(mapping_imu))

    try:
        shutil.rmtree(largest["folder"])
    except Exception as e:
        print(f"Failed to delete folder {largest['folder']}: {e}")


def process_single_bag(args):
    bag_dir, demo_dir, image_topic, accel_topic, gyro_topic, imu_topic = args
    bag_name = bag_dir.name
    output_dir = demo_dir.joinpath(bag_name)
    video_path = output_dir.joinpath("raw_video.mp4")
    imu_path = output_dir.joinpath("imu_data.json")

    if output_dir.exists() and video_path.is_file() and imu_path.is_file():
        print(f"{output_dir} already exists, skipping {bag_name}.")
        return None
    if output_dir.exists():
        print(f"{output_dir} exists but is incomplete, removing stale outputs for {bag_name}.")
        shutil.rmtree(output_dir, ignore_errors=True)

    try:
        process_ros2_bag(
            bag_dir=bag_dir,
            output_dir=output_dir,
            image_topic=image_topic,
            accel_topic=accel_topic,
            gyro_topic=gyro_topic,
            imu_topic=imu_topic,
        )
        print(f"Processed {bag_name} to {output_dir}")
        return str(output_dir)
    except Exception as e:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Error processing {bag_name}: {e}")
        return None


def build_process_args(
    bag_dirs,
    demo_dir: pathlib.Path,
    image_topic: str,
    accel_topic: str,
    gyro_topic: str,
    imu_topic: str,
):
    return [
        (bag_dir, demo_dir, image_topic, accel_topic, gyro_topic, imu_topic)
        for bag_dir in bag_dirs
    ]


@click.command(help="Process ROS2 bag directories under session path.")
@click.option("--input_path", required=True)
@click.option("--demo_dir", required=True)
@click.option(
    "--image_topic", default="/camera/camera/color/image_raw", show_default=True
)
@click.option("--accel_topic", default="/camera/camera/accel/sample", show_default=True)
@click.option("--gyro_topic", default="/camera/camera/gyro/sample", show_default=True)
@click.option("--imu_topic", default="/camera/camera/imu", show_default=True)
@click.option(
    "--max_workers",
    default=4,
    help="Maximum number of worker threads for parallel processing",
)
def main(
    input_path, demo_dir, image_topic, accel_topic, gyro_topic, imu_topic, max_workers
):
    session_path = pathlib.Path(os.path.expanduser(input_path)).absolute()
    demo_dir = pathlib.Path(os.path.expanduser(demo_dir)).absolute()
    demo_dir.mkdir(parents=True, exist_ok=True)

    bag_dirs = discover_ros2_bag_dirs(session_path)
    if not bag_dirs:
        print("No ROS2 bag directories (metadata.yaml) found in the input path.")
        return

    try:
        ensure_rosbags_available()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return

    print(
        f"Found {len(bag_dirs)} ROS2 bag directories. Processing with {max_workers} workers..."
    )

    process_args = build_process_args(
        bag_dirs=bag_dirs,
        demo_dir=demo_dir,
        image_topic=image_topic,
        accel_topic=accel_topic,
        gyro_topic=gyro_topic,
        imu_topic=imu_topic,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_bag, process_args))

    successful_results = [r for r in results if r is not None]
    print(
        f"Successfully processed {len(successful_results)} out of {len(process_args)} bag directories."
    )

    move_largest_video_to_mapping(demos_dir=demo_dir, session_dir=session_path)


if __name__ == "__main__":
    main()
