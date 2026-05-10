"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2

#from umi.common.cv_util import draw_predefined_mask

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()
    map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    if map_path.exists():
        print(f"Map file {map_path} already exists, skipping.")
        return

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    video_path = mount_target.joinpath('raw_video.mp4')
    json_path = mount_target.joinpath('imu_data.json')
    mask_path = mount_target.joinpath('slam_mask.png')

    # if not no_mask:
    #     mask_write_path = video_dir.joinpath('slam_mask.png')
        # slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
        # slam_mask = draw_predefined_mask(
        #     slam_mask, color=255, mirror=True, gripper=False, finger=True)
        # cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = mount_target.joinpath(map_mount_source.name)

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    setting_local_path = repo_root.joinpath('config', 'RealSense_D435i.yaml').absolute()
    if not setting_local_path.is_file():
        raise FileNotFoundError(f"SLAM setting file not found: {setting_local_path}")
    setting_in_container = '/ORB_SLAM3/Examples/Monocular-Inertial/RealSense_D435i.yaml'

    # run SLAM
    cmd = [
        'docker',
        'run',
        '--rm', # delete after finish
        '--volume', str(video_dir) + ':' + '/data',
        '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
        '--volume', str(setting_local_path) + ':' + str(setting_in_container),
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/RealSense_D435i.yaml',
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ]
    mask_local_path = video_dir.joinpath('slam_mask.png')
    should_use_mask = (not no_mask) and mask_local_path.is_file()
    if (not no_mask) and (not should_use_mask):
        print(f"⚠️ slam_mask.png not found in {video_dir}; running SLAM without mask.")

    if should_use_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])
        
    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(result)


# %%
if __name__ == "__main__":
    main()
