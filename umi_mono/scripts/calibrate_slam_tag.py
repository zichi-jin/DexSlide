# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import numpy as np
import pickle
import json
import csv
from collections import Counter
from scipy.spatial.transform import Rotation
from umi.common.pose_util import pose_to_mat


def _parse_bool(value):
    return str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y'}


def _load_trajectory_rows(csv_trajectory):
    with open(csv_trajectory, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty trajectory CSV: {csv_trajectory}')
    return rows


def _select_valid_rows(rows, keyframe_only=False):
    valid_rows = []
    for row in rows:
        if _parse_bool(row['is_lost']):
            continue
        if keyframe_only and (not _parse_bool(row['is_keyframe'])):
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise ValueError('No valid trajectory rows remain after filtering.')
    return valid_rows


def _extract_cam_pose(valid_rows):
    cam_pose_timestamps = np.array(
        [float(row['timestamp']) for row in valid_rows],
        dtype=np.float64
    )
    cam_pos = np.array(
        [[float(row[key]) for key in ('x', 'y', 'z')] for row in valid_rows],
        dtype=np.float32
    )
    cam_rot_quat_xyzw = np.array(
        [[float(row[key]) for key in ('q_x', 'q_y', 'q_z', 'q_w')] for row in valid_rows],
        dtype=np.float32
    )
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:, 3, 3] = 1
    cam_pose[:, :3, 3] = cam_pos
    cam_pose[:, :3, :3] = cam_rot.as_matrix()
    return cam_pose_timestamps, cam_pose


def _geometric_median(points, eps=1e-5, max_iter=256):
    if len(points) == 1:
        return points[0]
    guess = np.mean(points, axis=0)
    for _ in range(max_iter):
        dists = np.linalg.norm(points - guess, axis=1)
        nonzero = dists > eps
        if not np.any(nonzero):
            return guess
        inv_dists = 1.0 / dists[nonzero]
        next_guess = np.sum(points[nonzero] * inv_dists[:, None], axis=0) / np.sum(inv_dists)
        if np.linalg.norm(next_guess - guess) < eps:
            return next_guess
        guess = next_guess
    return guess


def _resolve_tag_id(tag_detection_results, requested_tag_id):
    counts = Counter()
    for result in tag_detection_results:
        counts.update(result.get('tag_dict', {}).keys())
    if not counts:
        raise ValueError('No ArUco detections found in tag_detection.pkl.')

    if requested_tag_id is not None and counts.get(requested_tag_id, 0) > 0:
        return requested_tag_id

    most_common_tag_id, num_hits = counts.most_common(1)[0]
    if requested_tag_id is not None:
        print(
            f"Requested tag_id={requested_tag_id} not found. "
            f"Falling back to most common detected tag_id={most_common_tag_id} "
            f"({num_hits} frames)."
        )
    else:
        print(f"Auto-selected tag_id={most_common_tag_id} from {num_hits} detections.")
    return most_common_tag_id

# %%
@click.command()
@click.option('-d', '--tag_detection', required=True, help='Tag detection pkl path')
@click.option('-c', '--csv_trajectory', default=None, help='CSV trajectory from SLAM (not mapping)')
@click.option('-o', '--output', required=True, help='output json')
@click.option('-tid', '--tag_id', type=int, default=None)
@click.option('-k', '--keyframe_only', is_flag=True, default=False)
def main(tag_detection, csv_trajectory, output, tag_id, keyframe_only):
    """
    Please use camera_trajectory.csv produced by re-localizing (initializing)
    the mapping video with the map_atlas.osa produced by mapping run.
    This is much more accurate than the mapping_camera_trajectory.csv produced by
    mapping run itself.
    """

    # load
    tag_detection_results = pickle.load(open(tag_detection, 'rb'))
    tag_id = _resolve_tag_id(tag_detection_results, tag_id)
    rows = _load_trajectory_rows(csv_trajectory)

    # filter pose
    valid_rows = _select_valid_rows(rows, keyframe_only=keyframe_only)
    cam_pose_timestamps, cam_pose = _extract_cam_pose(valid_rows)
    # match tum data to video idx
    video_timestamps = np.array([x['time'] for x in tag_detection_results])
    tum_video_idxs = list()
    for t in cam_pose_timestamps:
        if np.min(np.abs(video_timestamps - t)) > 0.03:
            print(f"Warning: timestamp {t} not found in video timestamps, skipping.")
            tum_video_idxs.append(None)
            continue
        tum_video_idxs.append(np.argmin(np.abs(video_timestamps - t)))

    # find corresponding tag detection
    all_tx_slam_tag = list()
    all_idxs = list()
    count = 0
    for tum_idx, video_idx in enumerate(tum_video_idxs):
        # print(count)
        if video_idx is None:
            print(f"Skipping tum_idx {tum_idx} with None video_idx")
            continue
        count += 1
        td = tag_detection_results[video_idx]
        tag_dict = td['tag_dict']
        if tag_id not in tag_dict:
            continue
        
        tag = tag_dict[tag_id]
        pose = np.concatenate([tag['tvec'], tag['rvec']])
        tx_cam_tag = pose_to_mat(pose)
        tx_slam_cam = cam_pose[tum_idx]
        # filter cam pose
        dist_to_cam = np.linalg.norm(tx_cam_tag[:3,3])
        if (dist_to_cam < 0.3) or  (dist_to_cam > 2):
            continue
        
        # filter tag location in image
        corners = tag['corners']
        tag_center_pix = corners.mean(axis=0)
        img_center = np.array([960, 540], dtype=np.float32) / 2
        # img_center = np.array([2704, 2028], dtype=np.float32) / 2
        dist_to_center = np.linalg.norm(tag_center_pix - img_center) / img_center[0]
        if dist_to_center > 0.6:  # 0.6
            continue

        tx_slam_tag = tx_slam_cam @ tx_cam_tag
        # print(tx_slam_cam, tx_slam_cam @ tx_cam_tag)
        all_tx_slam_tag.append(tx_slam_tag)
        all_idxs.append(tum_idx)
    all_tx_slam_tag = np.array(all_tx_slam_tag)
    if len(all_tx_slam_tag) == 0:
        raise ValueError(f'No usable detections found for tag_id={tag_id}.')

    # find transform closest to the mean
    all_slam_tag_pos = all_tx_slam_tag[:,:3,3]
    median = _geometric_median(all_slam_tag_pos)
    dists = np.linalg.norm((all_tx_slam_tag[:,:3,3] - median), axis=-1)
    threshold = np.quantile(dists, 0.7)
    is_valid = dists < threshold
    std = all_slam_tag_pos[is_valid].std(axis=0)
    mean = all_slam_tag_pos[is_valid].mean(axis=0)
    dists = np.linalg.norm((all_tx_slam_tag[is_valid][:,:3,3] - mean), axis=-1)
    nn_idx = np.argmin(dists)
    tx_slam_tag = all_tx_slam_tag[is_valid][nn_idx]
    print("Tag detection standard deviation (cm) < 0.9 quantile")
    print(std * 100)

    # save
    result = {
        'tx_slam_tag': tx_slam_tag.tolist(),
        'tag_id': int(tag_id),
    }
    json.dump(result, open(output, 'w'), indent=2)
    print(f"Saved result to {output}")


# %%
if __name__ == "__main__":
    main()
