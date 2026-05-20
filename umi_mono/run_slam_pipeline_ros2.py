"""
Main script for UMI SLAM pipeline (ROS2 bag variant).
python run_slam_pipeline_ros2.py <session_dir>
"""

import os
import pathlib
import subprocess
import sys

import click


ROOT_DIR = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


@click.command()
@click.argument("session_dir", nargs=-1)
@click.option("-c", "--calibration_dir", type=str, default=None)
@click.option(
    "--image_topic", default="/camera/camera/color/image_raw", show_default=True
)
@click.option("--accel_topic", default="/camera/camera/accel/sample", show_default=True)
@click.option("--gyro_topic", default="/camera/camera/gyro/sample", show_default=True)
@click.option("--imu_topic", default="/camera/camera/imu", show_default=True)
@click.option("--sensor_topic", default="/sensor_data", show_default=True)
@click.option("--stage00_max_workers", default=4, show_default=True)
@click.option("--stage07_max_workers", default=1, show_default=True)
def main(
    session_dir,
    calibration_dir,
    image_topic,
    accel_topic,
    gyro_topic,
    imu_topic,
    sensor_topic,
    stage00_max_workers,
    stage07_max_workers,
):
    script_dir_ros1 = ROOT_DIR.joinpath("scripts_slam_pipeline_ours")
    script_dir_ros2 = ROOT_DIR.joinpath("scripts_slam_pipeline_ros2")

    if calibration_dir is None:
        calibration_dir_path = ROOT_DIR.joinpath("example", "calibration")
    else:
        calibration_dir_path = pathlib.Path(calibration_dir)

    if not calibration_dir_path.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calibration_dir_path}")

    for session in session_dir:
        session_path = pathlib.Path(os.path.expanduser(session)).absolute()
        demo_dir = session_path.joinpath("demos")
        mapping_dir = demo_dir.joinpath("mapping")
        aurco_dir = demo_dir.joinpath("aurco")
        map_path = mapping_dir.joinpath("map_atlas.osa")

        print("############## 00_process_videos_ros2 #############")
        script_path = script_dir_ros2.joinpath("00_process_videos_ros2.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        cmd = [
            sys.executable,
            str(script_path),
            "--input_path",
            str(session_path),
            "--demo_dir",
            str(demo_dir),
            "--image_topic",
            image_topic,
            "--accel_topic",
            accel_topic,
            "--gyro_topic",
            gyro_topic,
            "--imu_topic",
            imu_topic,
            "--max_workers",
            str(stage00_max_workers),
        ]
        print(cmd)
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_create_map ###########")
        script_path = script_dir_ros1.joinpath("02_create_map.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        if not mapping_dir.is_dir():
            raise FileNotFoundError(mapping_dir)

        if not map_path.is_file():
            cmd = [
                sys.executable,
                str(script_path),
                "--input_dir",
                str(mapping_dir),
                "--map_path",
                str(map_path),
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0
            assert map_path.is_file()

        print("############# 03_batch_slam ###########")
        script_path = script_dir_ros1.joinpath("03_batch_slam.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        cmd = [
            sys.executable,
            str(script_path),
            "--input_dir",
            str(demo_dir),
            "--map_path",
            str(map_path),
            "--max_lost_frames",
            "600",
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04a_detect_aruco_of_base ###########")
        script_path = script_dir_ros1.joinpath("04_detect_aruco.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        camera_intrinsics = calibration_dir_path.joinpath("d435i_960_540.json")
        aruco_config = calibration_dir_path.joinpath("aruco_config.yaml")
        if not camera_intrinsics.is_file():
            raise FileNotFoundError(camera_intrinsics)
        if not aruco_config.is_file():
            raise FileNotFoundError(aruco_config)
        if not aurco_dir.is_dir():
            raise FileNotFoundError(aurco_dir)
        cmd = [
            sys.executable,
            str(script_path),
            str(aurco_dir),
            "--camera_intrinsics",
            str(camera_intrinsics),
            "--aruco_yaml",
            str(aruco_config),
            "--output",
            "tag_detection.pkl",
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04b_detect_aruco_ongripper ###########")
        script_path = script_dir_ros1.joinpath("04_detect_aruco.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        camera_intrinsics = calibration_dir_path.joinpath("d435i_960_540.json")
        aruco_config = calibration_dir_path.joinpath("aruco_config_wrist.yaml")
        if not camera_intrinsics.is_file():
            raise FileNotFoundError(camera_intrinsics)
        if not aruco_config.is_file():
            raise FileNotFoundError(aruco_config)
        input_dirs = [
            x
            for x in demo_dir.iterdir()
            if x.is_dir() and x.name not in ("aurco", "mapping")
        ]
        if len(input_dirs) > 0:
            cmd = [
                sys.executable,
                str(script_path),
                *[str(x) for x in input_dirs],
                "--camera_intrinsics",
                str(camera_intrinsics),
                "--aruco_yaml",
                str(aruco_config),
                "--video",
                "raw_video.mp4",
                "--output",
                "tag_detection_wrist.pkl",
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0
        else:
            print("No demo folders for 04b wrist marker detection.")

        print("############# 05_run_calibrations ###########")
        script_path = script_dir_ros1.joinpath("05_run_calibrations.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        cmd = [
            sys.executable,
            str(script_path),
            "--aurco_dir",
            str(aurco_dir),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir_ros1.joinpath("06_generate_dataset_plan.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        cmd = [
            sys.executable,
            str(script_path),
            "--demo_dir",
            str(demo_dir),
            "--tx_slam_tag",
            str(aurco_dir.joinpath("tx_slam_tag.json")),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 07_run_dataset_plan_ros2 ###########")
        script_path = script_dir_ros2.joinpath("07_storage_hdf5_ros2.py")
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(session_path),
            "--tx_slam_tag",
            str(aurco_dir.joinpath("tx_slam_tag.json")),
            "--aruco_tag_id",
            "10",
            "--tag_detection_name",
            "tag_detection_wrist.pkl",
            "--image_topic",
            image_topic,
            "--sensor_topic",
            sensor_topic,
            "--max_workers",
            str(stage07_max_workers),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0


if __name__ == "__main__":
    main()
