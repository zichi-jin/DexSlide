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
    parse_aruco_config, 
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask
)

# %%
@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-ij', '--intrinsics_json', required=True)
@click.option('-ay', '--aruco_yaml', required=True)
@click.option('-n', '--num_workers', type=int, default=4)
@click.option('-m', '--mask_path', type=str, default=None, help='Path to mask image to avoid detecting tags in the mirror')
def main(input, output, intrinsics_json, aruco_yaml, num_workers, mask_path):
    cv2.setNumThreads(num_workers)

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    if mask_path is not None:
        assert os.path.isfile(mask_path), f"Mask path {mask_path} does not exist."
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert mask.shape[:2] == img.shape[:2], "Mask shape does not match image shape."
        assert np.sum(mask > 0.5) < np.prod(mask.shape) - np.sum(mask > 0.5), "Mask should be mostly zeros."

    # load intrinsics
    intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))

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
            # avoid detecting tags in the mirror
            if mask_path is not None:
                assert mask.shape[:2] == img.shape[:2], "Mask shape does not match image shape."
                img[mask > 0.5] = 0  # invalid parts in value 1/255 in mask
                
            # img = draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=False, finger=False)
            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=intr,  # here we do not use fisheye but custom intrinsics
                refine_subpix=True
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
