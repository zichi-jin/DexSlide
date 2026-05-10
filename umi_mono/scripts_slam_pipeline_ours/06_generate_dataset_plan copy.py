"""
This script processes single-camera robot demonstrations.
It iterates through each 'demo_*' folder, uses the SLAM trajectory data
to find continuous, valid segments, and saves a SEPARATE 'dataset_plan.pkl'
file inside EACH processed demo folder.
"""

# %%
import sys
import os

# Set up the root directory for module imports
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import pickle
import numpy as np
import json
import math
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import av
from umi.common.pose_util_ours import pose_to_mat, mat_to_quatpose

# %%
def get_bool_segments(bool_seq):
    """
    Finds continuous segments of True or False in a boolean sequence.
    Returns a list of slice objects and their corresponding boolean type.
    """
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = list()
    segment_type = list()
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i+1]
        this_type = bool_seq[start]
        segments.append(slice(start, end))
        segment_type.append(this_type)
    segment_type = np.array(segment_type, dtype=bool)
    return segments, segment_type

# %%
@click.command()
@click.option('-i', '--demo_dir', required=True, help='Project directory containing the "demos" folder.')
@click.option('-to', '--tx_eye_hand', required=True)
@click.option('-ts', '--tx_slam_tag', required=True)
@click.option('-ml', '--min_episode_length', type=int, default=1, help="Minimum number of frames for a valid episode.")
def main(demo_dir, tx_eye_hand, tx_slam_tag, min_episode_length):
    # ======================== STAGE 0: INITIAL SETUP ========================
    print("Starting dataset generation...")
    demos_dir = pathlib.Path(os.path.expanduser(demo_dir)).absolute()
    tx_slam_tag = pathlib.Path(os.path.expanduser(tx_slam_tag)).absolute()
    tx_eye_hand = pathlib.Path(os.path.expanduser(tx_eye_hand)).absolute()
    # Load the transformation from the SLAM map to the reference tag
    slam_tag_path = tx_slam_tag.joinpath('tx_slam_tag.json')
    eye_hand_path = tx_eye_hand.joinpath('handoneye_cali.txt')
    if not eye_hand_path.is_file():
        raise FileNotFoundError(f"tx_slam_tag.json not found in {slam_tag_path.parent}")
    
    tx_slam_tag_data = json.load(open(os.path.expanduser(slam_tag_path), 'r'))
    tx_slam_tag_mat = np.array(tx_slam_tag_data['tx_slam_tag'])
    tx_tag_slam = np.linalg.inv(tx_slam_tag_mat)

    tx_cam_tcp = np.loadtxt(eye_hand_path)
    # Ensure it's a 4x4 matrix
    if tx_cam_tcp.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {eye_hand_path}, got shape {tx_cam_tcp.shape}")
    tx_cam_tcp = np.linalg.inv(tx_cam_tcp)

    # ================== STAGE 1: PROCESS EACH DEMO INDIVIDUALLY =================
    total_episodes_generated = 0
    # Find all demonstration videos
    video_dirs = sorted([x.parent for x in demos_dir.glob('data_*/raw_video.mp4')])
    
    if not video_dirs:
        print("❌ No 'demo_*/raw_video.mp4' files found. Exiting.")
        return

    print(f"Found {len(video_dirs)} demo videos to process.")

    for video_dir in tqdm(video_dirs, desc="Processing Demos"):
        # MODIFICATION: Create a fresh list for each demo folder
        episodes_for_this_demo = []

        # ----- Get metadata for this video -----
        mp4_path = video_dir.joinpath('raw_video.mp4')
        csv_path = video_dir.joinpath('camera_trajectory.csv')

        if not csv_path.is_file():
            # tqdm.write is used to print without breaking the progress bar
            tqdm.write(f"⚠️ Skipping {video_dir.name}, no camera_trajectory.csv found.")
            continue
        
        # check_path = video_dir.joinpath('check_result.txt')
        # if check_path.is_file() and not check_path.open('r').read().startswith('true'):
        #     tqdm.write(f"⚠️ Skipping {video_dir.name}, manually filtered by 'check_result.txt'.")
        #     continue

        with av.open(str(mp4_path), 'r') as container:
            stream = container.streams.video[0]
            fps = stream.average_rate
            dt = 1 / float(fps)
        
        # ----- Load and process trajectory data -----
        csv_df = pd.read_csv(csv_path)
        is_tracked = (~csv_df['is_lost']).to_numpy()
        
        if is_tracked.sum() < min_episode_length:
            tqdm.write(f"⚠️ Skipping {video_dir.name}, not enough valid tracked frames.")
            continue
            
        csv_df.loc[csv_df['is_lost'], ['q_x', 'q_y', 'q_z']] = 0.0
        csv_df.loc[csv_df['is_lost'], 'q_w'] = 1.0
        
        cam_pos = csv_df[['x', 'y', 'z']].to_numpy()
        cam_rot_quat = csv_df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
        cam_rot = Rotation.from_quat(cam_rot_quat)
        
        tx_slam_cam = np.zeros((len(csv_df), 4, 4), dtype=np.float32)
        tx_slam_cam[:, 3, 3] = 1
        tx_slam_cam[:, :3, 3] = cam_pos
        tx_slam_cam[:, :3, :3] = cam_rot.as_matrix()
        tx_tag_cam = tx_tag_slam @ tx_slam_cam
        tx_tag_tcp = tx_tag_cam @ tx_cam_tcp
        pose_tag_tcp = mat_to_quatpose(tx_tag_tcp)

        # ----- Segment trajectory into valid episodes -----
        segment_slices, segment_type = get_bool_segments(is_tracked)
        for s, is_valid in zip(segment_slices, segment_type):
            if not is_valid or (s.stop - s.start) < min_episode_length:
                continue

            start_frame, end_frame = s.start, s.stop
            cameras = [{"video_path": str(mp4_path.relative_to(demos_dir)), "video_start_end": (start_frame, end_frame)}]
            grippers = [{"tcp_pose": pose_tag_tcp[start_frame:end_frame]}]
            episode_timestamps = np.arange(start_frame, end_frame) * dt

            # MODIFICATION: Append to the local list for this demo
            episodes_for_this_demo.append({
                "episode_timestamps": episode_timestamps,
                "grippers": grippers,
                "cameras": cameras
            })
        
        # MODIFICATION: Save the pkl file inside the current demo folder
        if episodes_for_this_demo:
            output_path = video_dir.joinpath('dataset_plan.pkl')
            with output_path.open('wb') as f:
                pickle.dump(episodes_for_this_demo, f)
            total_episodes_generated += len(episodes_for_this_demo)

    # ======================== FINAL SUMMARY ========================
    print(f"\n✅ All processing complete. Generated a total of {total_episodes_generated} episodes across all demos.")

# %%
if __name__ == "__main__":
    main()