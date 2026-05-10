import sys
import os
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import json
import shutil
import pathlib
import click
from datetime import datetime
import concurrent.futures
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def process_rosbag(bag_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    imu_data = {
    "1": {
        "device name": "Intel RealSense D435i",
        "streams": {
            "ACCL": {
                "name": "Accelerometer",
                "units": "m/s2",
                "samples": []
            },
            "GYRO": {
                "name": "Gyroscope",
                "units": "rad/s",
                "samples": []
            },
            "GORI": {
                "name": "Orientation",
                "units": "quaternion",
                "samples": []
            }
            }
        }
    }
    images = []
    images2 = []
    image_timestamps = []
    image_timestamps2 = []
    accel_dict = {}
    gyro_dict = {}
    print(bag.get_type_and_topic_info()[1].keys())

    for topic, msg, t in bag.read_messages():
        if topic == '/camera/accel/sample':
            accel_dict[t.to_sec()] = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        elif topic == '/camera/gyro/sample':
            gyro_dict[t.to_sec()] = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        elif topic == '/camera/color/image_raw':
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            images.append(cv_img)
            image_timestamps.append(t.to_sec())
        elif topic == '/camera2/color/image_raw':
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            images2.append(cv_img)
            image_timestamps2.append(t.to_sec())
        else:
            continue
    # for topic, msg, t in bag.read_messages(topics=["/camera1/color/image_raw"]):
    #     cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     images.append(cv_img)
    #     image_timestamps.append(t.to_sec())
    # output_dir_path = pathlib.Path(output_dir)
    # mapping_mask_path = output_dir_path.parent.parent / 'sideview_mask.png'
    # if not mapping_mask_path.exists():
    #     first_frame = images2[0]
    #     cv2.imwrite(mapping_mask_path, first_frame)

    first_ts = next(iter(gyro_dict))

    for ts, value in accel_dict.items():
        if (ts - first_ts < 0):
            continue
        timestamp_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        imu_data["1"]["streams"]["ACCL"]["samples"].append({
            "date": timestamp_str,
            "value": value,
            "cts": 1000 * (ts - first_ts)
        })

    # 直接保存所有陀螺仪数据
    for ts, value in gyro_dict.items():
        timestamp_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        imu_data["1"]["streams"]["GYRO"]["samples"].append({
            "date": timestamp_str,
            "value": value,
            "cts": 1000 * (ts - first_ts)
        })

    for ts in gyro_dict.keys():
        timestamp_str = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        imu_data["1"]["streams"]["GORI"]["samples"].append({
            "date": timestamp_str,
            "value": [1, 0, 0, 0],
            "cts": 1000 * (ts - first_ts)
        })

    # 保存IMU数据
    imu_json_path = os.path.join(output_dir, 'imu_data.json')
    with open(imu_json_path, 'w') as f:
        json.dump(imu_data, f, indent=2)

    # 计算实际FPS
    actual_fps = None
    if len(image_timestamps) > 1:
        duration = image_timestamps[-1] - image_timestamps[0]
        actual_fps = (len(image_timestamps) - 1) / duration if duration > 0 else None
        print(f"Actual video FPS: {actual_fps:.2f} ({len(image_timestamps)} frames, duration {duration:.2f}s)")
    else:
        print("Not enough frames to calculate FPS.")

    # 保存视频
    if len(images) > 0:
        print("start saving")
        h, w, _ = images[0].shape
        video_path = os.path.join(output_dir, 'raw_video.mp4')
        print(f"mp4 video save to {video_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, actual_fps, (w, h))
        for img in images:
            out.write(img)
        out.release()
    else:
        print("video is none")
    
    actual_fps2 = None
    if len(image_timestamps2) > 1:
        duration2 = image_timestamps2[-1] - image_timestamps2[0]
        actual_fps2 = (len(image_timestamps2) - 1) / duration2 if duration2 > 0 else None
        print(f"Actual video2 FPS: {actual_fps2:.2f} ({len(image_timestamps2)} frames, duration {duration2:.2f}s)")
    else:
        print("Not enough frames to calculate FPS for video2.")

    if len(images2) > 0:
        print("start saving video2")
        h2, w2, _ = images2[0].shape
        video_path2 = os.path.join(output_dir, 'side_video.mp4')
        print(f"mp4 video2 save to {video_path2}")
        fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
        out2 = cv2.VideoWriter(video_path2, fourcc2, actual_fps2, (w2, h2))
        for img2 in images2:
            out2.write(img2)
        out2.release()
    else:
        print("video2 is none")

    # return actual_fps

def move_largest_video_to_mapping(demos_dir):
    # 创建 mapping 目录
    mapping_dir = pathlib.Path(demos_dir) / 'mapping'
    mapping_dir.mkdir(parents=True, exist_ok=True)

    # print(f"Will create mapping_dir: {mapping_dir}")
    # print(f"mapping_dir exists: {mapping_dir.exists()}")

    # 如果mapping目录下已有raw_video.mp4，直接跳过
    mapping_video = mapping_dir / 'raw_video.mp4'
    if mapping_video.exists():
        print(f"{mapping_video} already exists, skip moving.")
        return

    # 获取所有 demo 子目录
    demo_dirs = [x for x in pathlib.Path(demos_dir).iterdir() if x.is_dir()]
    video_info = []

    # 收集所有 raw_video.mp4 的路径和大小
    for demo_dir in demo_dirs:
        video_path = demo_dir / 'raw_video.mp4'
        imu_path = demo_dir / 'imu_data.json'
        if video_path.exists():
            size = video_path.stat().st_size
            video_info.append({'video': video_path, 'imu': imu_path, 'size': size, 'folder': demo_dir})

    if not video_info:
        print("No videos found.")
        return

    # 找到最大的视频
    largest = max(video_info, key=lambda x: x['size'])

    ori_bag = demos_dir.parent / (largest['video'].parent.stem + '.bag')
    map_bag = demos_dir.parent / 'mapping.bag'
    # os.symlink(str(ori_bag), str(map_bag), target_is_directory=False)
    shutil.move(ori_bag, map_bag)

    # 移动最大视频并重命名
    shutil.move(str(largest['video']), str(mapping_video))

    # 移动对应的 imu 文件
    if largest['imu'].exists():
        mapping_imu = mapping_dir / 'imu_data.json'
        shutil.move(str(largest['imu']), str(mapping_imu))

    # 删除原始 demo 文件夹（如果为空则删除，否则只删除视频和imu）
    try:
        shutil.rmtree(largest['folder'])
    except Exception as e:
        print(f"Failed to delete folder {largest['folder']}: {e}")



def process_single_bag(args):
    """Helper function to process a single bag file (for multiprocessing)"""
    bag_file, demo_dir = args
    bag_name = bag_file.stem
    output_dir = demo_dir.joinpath(bag_name)
    
    if output_dir.exists():
        print(f"{output_dir} already exists, skipping {bag_file.name}.")
        return None
    
    try:
        process_rosbag(str(bag_file), str(output_dir))
        print(f"Processed rosbag {bag_file.name} to {output_dir}")
        return str(output_dir)
    except Exception as e:
        print(f"Error processing {bag_file.name}: {e}")
        return None

@click.command(help='Session directories or rosbag files...')
@click.option('--input_path', required=True)
@click.option('--demo_dir', required=True)
@click.option('--max_workers', default=4, help='Maximum number of worker threads for parallel processing')
def main(input_path, demo_dir, max_workers):
    path = pathlib.Path(os.path.expanduser(input_path)).absolute()
    demo_dir = pathlib.Path(os.path.expanduser(demo_dir)).absolute()
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all bag files
    bag_files = list(path.glob('*.bag'))
    if not bag_files:
        print("No .bag files found in the input path.")
        return
    
    print(f"Found {len(bag_files)} bag files. Processing with {max_workers} workers...")
    
    # Prepare arguments for parallel processing
    process_args = [(bag_file, demo_dir) for bag_file in bag_files]
    
    # Process bag files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_bag, process_args))
    
    # # Filter out None results (skipped or failed processing)
    successful_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(successful_results)} out of {len(bag_files)} bag files.")
        #executor.map(process_single_bag, process_args)
    move_largest_video_to_mapping(demo_dir)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
    # session_dir = "/home/bohanfeng/Desktop/liboyan/Imitation_learning/test_video3/"
    # main(session_dir)