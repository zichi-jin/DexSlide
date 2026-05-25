"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
def main(session_dir, calibration_dir):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline_ours')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        demo_dir = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')                 
        aurco_dir = demo_dir.joinpath('aurco')
        map_path = mapping_dir.joinpath('map_atlas.osa')
        
        print("############## 00_process_videos #############")
        # script_path = script_dir.joinpath("00_process_videos.py")
        # assert script_path.is_file()
        # cmd = [
        #     'python3', str(script_path),
        #     '--input_path', str(session),
        #     '--demo_dir', str(demo_dir),
        # ]
        # print(cmd)
        # result = subprocess.run(cmd)
   
        # assert result.returncode == 0

        print("############# 02_create_map ###########")
        script_path = script_dir.joinpath("02_create_map.py")
        assert script_path.is_file()
        assert mapping_dir.is_dir()
        # print(f"Map path: {map_path}")
        if not map_path.is_file():
            cmd = [
                'python3', str(script_path),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path),
                # '--no_mask',  # Add this option if you want to skip masking
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0
            assert map_path.is_file()

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path),
            '--max_lost_frames', '600',
            # '--num_workers', '2'
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04a_detect_aruco_of_base ###########")
        # detect world marker (ArUco-13) in aurco demo for SLAM-to-world calibration
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        camera_intrinsics = calibration_dir.joinpath('d435i_960_540.json')
        aruco_config = calibration_dir.joinpath('aruco_config.yaml')
        assert camera_intrinsics.is_file()
        assert aruco_config.is_file()
        assert aurco_dir.is_dir()
        cmd = [
            'python', str(script_path),
            str(aurco_dir),
            '--camera_intrinsics', str(camera_intrinsics),
            '--aruco_yaml', str(aruco_config),
            '--output', 'tag_detection.pkl'
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04b_detect_aruco_ongripper ###########")
        # detect wrist-visible marker (ArUco-10) in each demo wrist video
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        camera_intrinsics = calibration_dir.joinpath('d435i_960_540.json')
        aruco_config = calibration_dir.joinpath('aruco_config_wrist.yaml')
        assert camera_intrinsics.is_file()
        assert aruco_config.is_file()
        input_dirs = [x for x in demo_dir.iterdir() if x.is_dir() and x.name not in ('aurco', 'mapping')]
        if len(input_dirs) > 0:
            cmd = [
                'python', str(script_path),
                *[str(x) for x in input_dirs],
                '--camera_intrinsics', str(camera_intrinsics),
                '--aruco_yaml', str(aruco_config),
                '--video', 'raw_video.mp4',
                '--output', 'tag_detection_wrist.pkl'
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0
        else:
            print("No demo folders for 04b wrist marker detection.")

        print("############# 05_run_calibrations ###########")
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--aurco_dir', str(aurco_dir)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--demo_dir', str(demo_dir),
            '--tx_slam_tag', str(aurco_dir.joinpath('tx_slam_tag.json'))
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 07_run_dataset_plan ###########")
        script_path = script_dir.joinpath("07_storage_hdf5.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session),
            '--tx_slam_tag', str(aurco_dir.joinpath('tx_slam_tag.json')),
            '--aruco_tag_id', '10',
            '--tag_detection_name', 'tag_detection_wrist.pkl',
            '--max_workers', '1',
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        # print("############# 08_generate_replay_buffer_from_rosbag ###########")
        # script_path = script_dir.joinpath("08_generate_replay_buffer_from_rosbag.py")
        # print(*[str(x) for x in list(session.rglob("*.hdf5"))])
        # assert script_path.is_file()
        # cmd = [
        #     'python', str(script_path),
        #     *[str(x) for x in list(session.rglob("*.hdf5"))],
        #     '--out_res', '224,224',
        #     '--compression_level', '99',
        #     '--in_res', '960, 540',
        #     '--output', str(demo_dir.joinpath('replay_buffer.zarr.zip'))
        # ]
        # result = subprocess.run(cmd)
        # assert result.returncode == 0
## 
if __name__ == "__main__":
    # session_dir = "/home/bohanfeng/Desktop/liboyan/Imitation_learning/test_video3/"
    # main(session_dir=session_dir, calibration_dir=None)
    main()
