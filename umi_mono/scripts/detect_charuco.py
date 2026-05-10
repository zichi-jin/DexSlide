# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
from tqdm import tqdm
import yaml
import json
import av
import numpy as np
import cv2
import pickle

from umi.common.cv_util import (
    parse_charuco_config,
    detect_localize_charuco_tags,
)

# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
def main(input, output, intrinsics_json, aruco_yaml, num_workers):
    cv2.setNumThreads(num_workers)
    raise NotImplementedError("This script is not corrected yet.")
    # load aruco config
    board, dictionary = parse_charuco_config(yaml.safe_load(open(aruco_yaml, 'r')))

    # load intrinsics
    # raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))
    
    camera_setting = json.load(open(intrinsics_json, 'r'))
    intrinsics_dict = camera_setting['intrinsics']
    f = intrinsics_dict['focal_length']
    cx = intrinsics_dict['principal_pt_x']
    cy = intrinsics_dict['principal_pt_y']
    intr = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    results = list()
    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        # in_res = np.array([in_stream.height, in_stream.width])[::-1]
        # fisheye_intr = convert_fisheye_intrinsics_resolution(
        #     opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)

        for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
            img = frame.to_ndarray(format='rgb24')
            frame_cts_sec = frame.pts * in_stream.time_base
            tag_dict = detect_localize_charuco_tags(
                img, intr, board, dictionary
            )
            result = {
                'frame_idx': i,
                'time': float(frame_cts_sec),
                'tag_dict': tag_dict
            }
            results.append(result)
    
    # dump
    pickle.dump(results, open(os.path.expanduser(output), 'wb'))

# %%
if __name__ == "__main__":
    main()
