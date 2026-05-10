import os
import pathlib
import json
import pickle
import click
import rosbag
import numpy as np
import h5py
import pandas as pd
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R, Slerp
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import threading

# def quat2mat(row_data):
#     """
#     Converts 6 with position and quaternion to a 4x4 transformation matrix.
#     """
#     tvec = row_data[:3]  # x, y, z
#     quat = row_data[3:]  # q_x, q_y, q_z, q_w
#     rot = R.from_quat(quat)
#     mat = np.eye(4)
#     mat[:3, :3] = rot.as_matrix()
#     mat[:3, 3] = tvec
#     return mat

# def mat2quat(mat):
#     """
#     Converts a 4x4 transformation matrix to a quaternion.
#     """
#     rot = R.from_matrix(mat[:3, :3])
#     quat = rot.as_quat()
#     res = np.zeros(7)
#     res[:3] = mat[:3, 3]  # x, y,
#     res[3:] = quat  # q_x, q_y, q_z, q_w
#     return res

def find_first_long_valid_interval(seq, n=5, m=60, ratio=0.15, L=180):
    """
    找到第一个满足条件的长区间，其中：
        - True  = lost (缺失)
        - False = valid (有效)

    条件：
        1. 连续 True (lost) 长度 <= n
        2. 总 True (lost) 数量 <= m
        3. lost_ratio = (#True) / length <= ratio
        4. 区间长度 >= L

    返回第一个满足条件的 (start, end) 区间（左闭右开）
    若无，返回 None
    """
    seq = np.array(seq, dtype=bool)
    N = len(seq)

    # Step 1: Find the first valid start (first False)
    start = 0
    while start < N - L and seq[start]:
        start += 1
    if start >= N:
        return None

    best_start, best_end = None, None
    total_lost = 0
    current_gap = 0
    max_consecutive_lost = 0
    last_valid_in_window = start  # Tracks the last non-lost (False) index

    for end in range(start, N):
        if seq[end]:  # Lost frame (True)
            total_lost += 1
            current_gap += 1
            max_consecutive_lost = max(max_consecutive_lost, current_gap)
        else:  # Valid frame (False)
            current_gap = 0
            last_valid_in_window = end  # Update last valid index

        window_length = end - start + 1

        # Check if current window violates constraints
        while (total_lost > m or 
               max_consecutive_lost > n or 
               (window_length > 0 and total_lost / window_length > ratio)):
            # Move start forward (sliding window adjustment)
            if seq[start]:
                total_lost -= 1
                # Update max_consecutive_lost (may need full rescan, but rare in practice)
                if current_gap > 0:
                    current_gap -= 1
            start += 1
            window_length = end - start + 1
            if start > end:
                break

        # Reset tracking if start overtakes end (invalid window)
        # if start > end:
        #     total_lost = 0
        #     current_gap = 0
        #     max_consecutive_lost = 0
        #     continue

        # If window is large enough, check trimmed length
        if window_length >= L:
            clean_end = last_valid_in_window + 1
            trimmed_length = clean_end - start
            if trimmed_length > L:
                if best_start is None or trimmed_length > (best_end - best_start):
                    best_start, best_end = start, clean_end

    return (best_start, best_end) if best_start is not None else (None, None)


def euler2mat(euler):
    """
    Convert x y z roll, pitch, yaw to a 4x4 transformation matrix.
    """
    mat = np.eye(4)
    t = euler[:3]  # x, y, z
    r = euler[3:]  # roll, pitch, yaw
    rot = R.from_euler('zyx', r, degrees=False)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = t
    return mat

def mat2euler(mat):
    """
    Converts a 4x4 transformation matrix to Euler angles (roll, pitch, yaw).
    """
    rot = R.from_matrix(mat[:3, :3])
    euler = rot.as_euler('zyx', degrees=False)
    t = mat[:3, 3]  # x, y, z
    euler = np.concatenate([t, euler])  # x, y, z, roll, pitch, yaw
    return euler

def tvec_rvec_to_mat(tvec, rvec):
    tx = np.eye(4, dtype=np.float64)
    tx[:3, :3] = R.from_rotvec(rvec).as_matrix()
    tx[:3, 3] = tvec
    return tx


def max_consecutive_missing(valid_mask):
    max_gap = 0
    curr_gap = 0
    for is_valid in valid_mask:
        if is_valid:
            curr_gap = 0
        else:
            curr_gap += 1
            max_gap = max(max_gap, curr_gap)
    return max_gap


def interpolate_fused_world_trajectory(tx_world_raw, valid_mask):
    n_steps = len(valid_mask)
    valid_idx = np.nonzero(valid_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("No valid fused ArUco-10 poses to interpolate.")

    all_idx = np.arange(n_steps, dtype=np.float64)
    valid_idx_f = valid_idx.astype(np.float64)

    pos_valid = tx_world_raw[valid_idx, :3, 3]
    pos_interp = np.zeros((n_steps, 3), dtype=np.float64)
    for i in range(3):
        pos_interp[:, i] = np.interp(all_idx, valid_idx_f, pos_valid[:, i])

    if len(valid_idx) == 1:
        rot_interp = np.tile(tx_world_raw[valid_idx[0], :3, :3], (n_steps, 1, 1))
    else:
        rot_valid = R.from_matrix(tx_world_raw[valid_idx, :3, :3])
        slerp = Slerp(valid_idx_f, rot_valid)
        rot_interp = slerp(all_idx).as_matrix()

    tx_interp = np.tile(np.eye(4, dtype=np.float64), (n_steps, 1, 1))
    tx_interp[:, :3, :3] = rot_interp
    tx_interp[:, :3, 3] = pos_interp
    return tx_interp


def process_bag_file(
        bag_file,
        path,
        total_lost,
        tx_cam_tcp,
        tx_aruco13_slamworld,
        aruco_tag_id,
        tag_detection_name,
        aruco_min_valid_frames,
        aruco_max_missing_ratio,
        aruco_max_consecutive_missing,
        lost_threshold,
        index_counter,
        index_lock):
    """Process a single bag file and save as HDF5"""
    bridge = CvBridge()
    bag_name = bag_file.stem
    demo_path = path / 'demos' / bag_name
    trajectory_path = demo_path / 'camera_trajectory.csv'
    tag_detection_path = demo_path / tag_detection_name
    print(f"Processing {trajectory_path}...")

    # 判断 camera_trajectory.csv 是否存在
    if not trajectory_path.exists():
        print(f"❌ {trajectory_path} does not exist. Skipping {bag_file.name}.")
        return
    df = pd.read_csv(trajectory_path)
    # 1. 读取 camera_trajectory.csv
    if df['is_lost'].all():
        print(f"❌ all frames are lost in {trajectory_path}, skipping {bag_file.name}.")
        return
    first_valid_idx = df.index[df['is_lost'] == False][0]
    end_valid_idx = df.index[df['is_lost'] == False][-1]
    df = df.iloc[first_valid_idx:end_valid_idx+1].reset_index(drop=True)
    if 'frame_idx' not in df.columns:
        df['frame_idx'] = np.arange(first_valid_idx, end_valid_idx + 1)

    lost_count = df['is_lost'].sum()
    if lost_count > total_lost:
        print(f"❌ Too many lost frames ({lost_count}). Skipping {bag_file.name}.")
        return

    max_consecutive_lost = (df['is_lost'] != df['is_lost'].shift()).cumsum()
    lost_streaks = df.groupby(max_consecutive_lost)['is_lost'].sum()
    max_lost_streak = lost_streaks.max() if not lost_streaks.empty else 0
    if max_lost_streak > lost_threshold:
        print(f"❌ Too many consecutive lost frames ({max_lost_streak}). Skipping {bag_file.name}.")
        return

    # 线性插值位置
    for col in ['x', 'y', 'z']:
        valid = ~df['is_lost']
        df[col] = np.interp(np.arange(len(df)), np.where(valid)[0], df.loc[valid, col])

    # SLERP插值四元数
    valid_idx = np.where(~df['is_lost'])[0]
    key_rots = R.from_quat(df.iloc[valid_idx][['q_x', 'q_y', 'q_z', 'q_w']].values)
    slerp = Slerp(valid_idx, key_rots)
    interp_rots = slerp(np.arange(len(df)))
    # quats = interp_rots.as_quat()
    # df[['q_x', 'q_y', 'q_z', 'q_w']] = quats

    # tx_slam_traj: camera trajectory in SLAM frame (cam -> slam)
    tx_slam_traj = np.tile(np.eye(4, dtype=np.float64), (len(df), 1, 1))
    tx_slam_traj[:, :3, :3] = interp_rots.as_matrix()
    tx_slam_traj[:, :3, 3] = df[['x', 'y', 'z']].to_numpy(dtype=np.float64)

    euler = interp_rots.as_euler('zyx', degrees=False)
    selected_cols = ['frame_idx', 'x', 'y', 'z']
    qpos = np.concatenate([df[selected_cols].values, euler], axis=1)
    for i in range(qpos.shape[0]):
        qpos_mat = euler2mat(qpos[i][1:])
        qpos_mat = qpos_mat @ tx_cam_tcp
        qpos[i][1:] = mat2euler(qpos_mat)
    action = qpos.copy()
    # 2. 读取 bag 文件中的图像
    wrist_imgs = []
    wrist_times = []
    sensor = []
    sensor_times = []
    with rosbag.Bag(str(bag_file), 'r') as bag:
        # only read wrist camera and sensor_data
        for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/sensor_data']):
            if topic == '/camera/color/image_raw':
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                wrist_imgs.append(cv_img)
                wrist_times.append(t.to_sec())
            elif topic == '/sensor_data':
                sensor.append(msg.data)
                sensor_times.append(t.to_sec())
    wrist_imgs = np.stack(wrist_imgs) if wrist_imgs else np.empty((0,))
    sensor = np.array(sensor) if sensor else np.empty((0,))  # data in millimeters of gripper width

    # interpolate sensor data (gripper width) to match wrist image timestamps
    # here we already assume wrist images are aligned !!!
    sensor_times = np.array(sensor_times) if sensor_times else np.empty((0,))
    wrist_times = np.array(wrist_times) if wrist_times else np.empty((0,))
    print(f"Found {len(wrist_imgs)} wrist images and {len(sensor)} sensor data points in {bag_file.name}.")

    if sensor_times.size == 0 or sensor.size == 0:
        # No sensor data recorded; fill with zeros (safe default) and warn.
        print(f"⚠️ No /sensor_data in {bag_file.name}; filling sensor values with zeros.")
        interp_sensor = np.zeros_like(wrist_times, dtype=float)
    else:
        interp_sensor = np.interp(wrist_times, sensor_times, sensor)

    cropped_wrist_imgs = wrist_imgs[first_valid_idx:end_valid_idx+1]
    if interp_sensor.size > 0:
        cropped_sensor = interp_sensor[first_valid_idx:end_valid_idx+1]
    else:
        # keep previous behavior if there are truly no wrist timestamps
        raise ValueError("No sensor data found in the bag file.")

    qpos = np.concatenate([qpos, cropped_sensor[:, None]], axis=1)
    action = np.concatenate([action, cropped_sensor[:, None]], axis=1)

    # 3. load per-frame ArUco-10 detection and compose world trajectory
    tx_aruco10_cam = np.tile(np.eye(4, dtype=np.float64), (len(df), 1, 1))
    aruco10_valid = np.zeros(len(df), dtype=np.uint8)
    frame_indices = df['frame_idx'].to_numpy(dtype=np.int64)

    if tag_detection_path.is_file():
        tag_detection_results = pickle.load(open(tag_detection_path, 'rb'))
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx < 0 or frame_idx >= len(tag_detection_results):
                continue
            tag_dict = tag_detection_results[frame_idx].get('tag_dict', {})
            if aruco_tag_id not in tag_dict:
                continue
            tag = tag_dict[aruco_tag_id]
            tvec = np.asarray(tag['tvec'], dtype=np.float64).reshape(3)
            rvec = np.asarray(tag['rvec'], dtype=np.float64).reshape(3)
            tx_aruco10_cam[i] = tvec_rvec_to_mat(tvec=tvec, rvec=rvec)
            aruco10_valid[i] = 1
    else:
        print(f"⚠️ {tag_detection_path} not found. ArUco-{aruco_tag_id} trajectory will be invalid.")

    tx_aruco10_world_raw = np.matmul(
        np.matmul(tx_aruco13_slamworld[None, :, :], tx_slam_traj),
        tx_aruco10_cam
    )

    # quality checks after fusion, before interpolation
    valid_mask = (aruco10_valid == 1)
    n_steps = len(valid_mask)
    n_valid = int(valid_mask.sum())
    missing_ratio = 1.0 - (float(n_valid) / float(max(1, n_steps)))
    max_missing = max_consecutive_missing(valid_mask)
    if n_valid < aruco_min_valid_frames:
        print(
            f"❌ Too few valid ArUco-{aruco_tag_id} frames ({n_valid}/{n_steps}). "
            f"Need >= {aruco_min_valid_frames}. Skipping {bag_file.name}."
        )
        return
    if missing_ratio > aruco_max_missing_ratio:
        print(
            f"❌ Missing ratio too high for ArUco-{aruco_tag_id} "
            f"({missing_ratio:.3f} > {aruco_max_missing_ratio:.3f}). "
            f"Skipping {bag_file.name}."
        )
        return
    if max_missing > aruco_max_consecutive_missing:
        print(
            f"❌ Max consecutive missing too long for ArUco-{aruco_tag_id} "
            f"({max_missing} > {aruco_max_consecutive_missing}). "
            f"Skipping {bag_file.name}."
        )
        return

    # interpolate fused world trajectory
    tx_aruco10_world = interpolate_fused_world_trajectory(
        tx_world_raw=tx_aruco10_world_raw, valid_mask=valid_mask)

    # keep raw measurements with invalid frames zeroed for debugging / analysis
    tx_aruco10_world_raw[~valid_mask] = 0.0
    tx_aruco10_cam[~valid_mask] = 0.0

    aruco10_pose_world = np.zeros((len(df), 7), dtype=np.float32)
    aruco10_pose_world[:, :3] = tx_aruco10_world[:, :3, 3].astype(np.float32)
    aruco10_pose_world[:, 3:] = R.from_matrix(tx_aruco10_world[:, :3, :3]).as_quat().astype(np.float32)

    # Get thread-safe index
    with index_lock:
        current_index = index_counter[0]
        index_counter[0] += 1

    # 4. 保存为 HDF5
    out_path = path / f"episode_{current_index}.hdf5"
    with h5py.File(out_path, 'w') as f:
        obs = f.create_group('observation')
        obs.create_dataset('action', data=action)
        obs.create_dataset('qpos', data=qpos)
        grp = obs.create_group('images')
        grp.create_dataset('wrist', data=cropped_wrist_imgs)
        # side dataset removed
        aruco = obs.create_group('aruco')
        aruco.create_dataset('tx_aruco13_slamworld', data=tx_aruco13_slamworld.astype(np.float32))
        aruco.create_dataset('tx_slam_traj', data=tx_slam_traj.astype(np.float32))
        aruco.create_dataset('tx_aruco10_cam', data=tx_aruco10_cam.astype(np.float32))
        aruco.create_dataset('tx_aruco10_world_raw', data=tx_aruco10_world_raw.astype(np.float32))
        aruco.create_dataset('tx_aruco10_world', data=tx_aruco10_world.astype(np.float32))
        aruco.create_dataset('aruco10_valid', data=aruco10_valid)
        aruco.create_dataset('aruco10_pose_world', data=aruco10_pose_world)
	    
    print(f"Saved {out_path}")

@click.command()
@click.option('--input', required=True)
@click.option('--tx_eye_hand', required=False, default=None, help='Path to 4x4 tx_eye_hand matrix (optional). If not provided, identity is used.')
@click.option('--tx_slam_tag', required=False, default=None, help='Path to tx_slam_tag.json for ArUco-13 world calibration.')
@click.option('--aruco_tag_id', default=10, type=int, help='ArUco marker id to track in wrist camera detections.')
@click.option('--tag_detection_name', default='tag_detection_wrist.pkl', help='Detection filename under each demos/<bag_name> folder.')
@click.option('--aruco_min_valid_frames', default=30, type=int, help='Minimum valid fused ArUco frames required.')
@click.option('--aruco_max_missing_ratio', default=0.75, type=float, help='Maximum allowed missing ratio in fused ArUco trajectory.')
@click.option('--aruco_max_consecutive_missing', default=10, type=int, help='Maximum allowed consecutive missing fused ArUco frames.')
@click.option('--total_lost', default=200, type=int, help='Total number of lost frames allowed in camera_trajectory.csv')
@click.option('--lost_threshold', default=50, type=int, help='Threshold for constant lost frames in camera_trajectory.csv')
@click.option('--max_workers', default=8, type=int, help='Maximum number of worker threads')
def main(input, total_lost, tx_eye_hand, tx_slam_tag, aruco_tag_id, tag_detection_name,
         aruco_min_valid_frames, aruco_max_missing_ratio, aruco_max_consecutive_missing,
         lost_threshold, max_workers):
    path = pathlib.Path(os.path.expanduser(input)).absolute()
    bag_files = [
        f for f in path.glob('*.bag')
        if f.name not in ('aurco.bag', 'mapping.bag')
    ]
    if not bag_files:
        print("No bag files found in the specified directory.")
        return
    
    if tx_eye_hand is None:
        tx_cam_tcp = np.eye(4)
    else:
        eye_hand_path = pathlib.Path(os.path.expanduser(tx_eye_hand)).absolute()
        if not eye_hand_path.is_file():
            raise FileNotFoundError(f"tx_eye_hand file not found: {eye_hand_path}")
        tx_cam_tcp = np.loadtxt(eye_hand_path)
        # Ensure it's a 4x4 matrix
        if tx_cam_tcp.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix in {eye_hand_path}, got shape {tx_cam_tcp.shape}")
        tx_cam_tcp = np.linalg.inv(tx_cam_tcp)

    if tx_slam_tag is None:
        print("⚠️ tx_slam_tag is not provided. Using identity for tx_aruco13_slamworld.")
        tx_aruco13_slamworld = np.eye(4, dtype=np.float64)
    else:
        slam_tag_path = pathlib.Path(os.path.expanduser(tx_slam_tag)).absolute()
        if not slam_tag_path.is_file():
            raise FileNotFoundError(f"tx_slam_tag file not found: {slam_tag_path}")
        slam_tag_data = json.load(open(slam_tag_path, 'r'))
        if 'tx_slam_tag' not in slam_tag_data:
            raise KeyError(f"'tx_slam_tag' key missing in {slam_tag_path}")
        tx_slam_aruco13 = np.asarray(slam_tag_data['tx_slam_tag'], dtype=np.float64)
        if tx_slam_aruco13.shape != (4, 4):
            raise ValueError(f"Expected 4x4 tx_slam_tag in {slam_tag_path}, got shape {tx_slam_aruco13.shape}")
        # world is defined by ArUco-13 frame
        tx_aruco13_slamworld = np.linalg.inv(tx_slam_aruco13)
    
    print(f"Found {len(bag_files)} bag files. Processing with {max_workers} threads...")
    
    # Thread-safe counter for index
    index_counter = [0]
    index_lock = threading.Lock()
    
    if max_workers == 1:
        for bag_file in bag_files:
            process_bag_file(
                bag_file, path, total_lost, tx_cam_tcp, tx_aruco13_slamworld,
                aruco_tag_id, tag_detection_name, aruco_min_valid_frames,
                aruco_max_missing_ratio, aruco_max_consecutive_missing,
                lost_threshold, index_counter, index_lock)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            for bag_file in bag_files:
                while len(futures) >= max_workers:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                futures.add(executor.submit(
                    process_bag_file, bag_file, path, total_lost, tx_cam_tcp, tx_aruco13_slamworld,
                    aruco_tag_id, tag_detection_name, aruco_min_valid_frames,
                    aruco_max_missing_ratio, aruco_max_consecutive_missing,
                    lost_threshold, index_counter, index_lock))
            for future in as_completed(futures):
                future.result()
    
    print(f"Processing complete. Generated {index_counter[0]} episodes.")
        
if __name__ == "__main__":
    main()
