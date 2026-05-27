"""
Partial ROS2 pipeline for:
1. ROS2 bag file processing into demos/
2. mapping map_atlas.osa creation
3. aurco re-localization against the mapping atlas
4. desk ArUco detection in aurco
5. tx_slam_tag.json generation

This intentionally does NOT run:
- batch SLAM for data_* demos
- wrist ArUco detection on demos
- dataset plan generation
- hdf5 export / replay buffer generation
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import click


ROOT_DIR = pathlib.Path(__file__).resolve().parent
os.chdir(ROOT_DIR)


def _python_cmd(script_path: pathlib.Path, *args: str) -> list[str]:
    return [sys.executable, str(script_path), *args]


def _run_or_fail(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    print(cmd)
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _relocalize_single_demo(
    demo_dir: pathlib.Path,
    map_path: pathlib.Path,
    docker_image: str,
    no_docker_pull: bool,
    no_mask: bool,
) -> None:
    csv_path = demo_dir / "camera_trajectory.csv"
    if csv_path.is_file():
        print(f"{csv_path} already exists, skip aurco relocalization.")
        return

    raw_video = demo_dir / "raw_video.mp4"
    imu_json = demo_dir / "imu_data.json"
    if not raw_video.is_file():
        raise FileNotFoundError(raw_video)
    if not imu_json.is_file():
        raise FileNotFoundError(imu_json)
    if not map_path.is_file():
        raise FileNotFoundError(map_path)

    if not no_docker_pull:
        _run_or_fail(["docker", "pull", docker_image])

    repo_root = ROOT_DIR
    setting_local_path = repo_root / "config" / "RealSense_D435i.yaml"
    if not setting_local_path.is_file():
        raise FileNotFoundError(setting_local_path)
    setting_in_container = "/ORB_SLAM3/Examples/Monocular-Inertial/RealSense_D435i.yaml"

    mount_target = pathlib.Path("/data")
    map_mount_target = pathlib.Path("/map") / map_path.name
    cmd = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{demo_dir}:/data",
        "--volume",
        f"{map_path.parent}:{map_mount_target.parent}",
        "--volume",
        f"{setting_local_path}:{setting_in_container}",
        docker_image,
        "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
        "--vocabulary",
        "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "--setting",
        setting_in_container,
        "--input_video",
        str(mount_target / "raw_video.mp4"),
        "--input_imu_json",
        str(mount_target / "imu_data.json"),
        "--output_trajectory_csv",
        str(mount_target / "camera_trajectory.csv"),
        "--load_map",
        str(map_mount_target),
        "--max_lost_frames",
        "600",
    ]
    mask_local_path = map_path.parent / "slam_mask.png"
    if (not no_mask) and mask_local_path.is_file():
        cmd.extend(["--mask_img", str(pathlib.Path("/map") / "slam_mask.png")])
    elif not no_mask:
        print(f"⚠️ slam_mask.png not found in {map_path.parent}; running aurco relocalization without mask.")

    stdout_path = demo_dir / "slam_stdout.txt"
    stderr_path = demo_dir / "slam_stderr.txt"
    result = subprocess.run(
        cmd,
        cwd=str(demo_dir),
        stdout=stdout_path.open("w"),
        stderr=stderr_path.open("w"),
    )
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected aurco camera trajectory not found: {csv_path}")


@click.command()
@click.argument("session_dir", nargs=-1)
@click.option("-c", "--calibration_dir", type=str, default=None)
@click.option(
    "--image_topic", default="/camera/camera/color/image_raw", show_default=True
)
@click.option("--accel_topic", default="/camera/camera/accel/sample", show_default=True)
@click.option("--gyro_topic", default="/camera/camera/gyro/sample", show_default=True)
@click.option("--imu_topic", default="/camera/camera/imu", show_default=True)
@click.option("--stage00_max_workers", default=4, show_default=True)
@click.option("--docker_image", default="chicheng/orb_slam3:latest", show_default=True)
@click.option("--no_docker_pull", is_flag=True, default=False, help="Skip docker pull for SLAM stages.")
@click.option("--no_mask", is_flag=True, default=False, help="Run mapping/relocalization without slam mask.")
def main(
    session_dir,
    calibration_dir,
    image_topic,
    accel_topic,
    gyro_topic,
    imu_topic,
    stage00_max_workers,
    docker_image,
    no_docker_pull,
    no_mask,
):
    if not session_dir:
        raise SystemExit("No session_dir provided.")

    script_dir_ros1 = ROOT_DIR / "scripts_slam_pipeline_ours"
    script_dir_ros2 = ROOT_DIR / "scripts_slam_pipeline_ros2"

    if calibration_dir is None:
        calibration_dir_path = ROOT_DIR / "example" / "calibration"
    else:
        calibration_dir_path = pathlib.Path(calibration_dir)
    if not calibration_dir_path.is_dir():
        raise FileNotFoundError(f"Calibration dir not found: {calibration_dir_path}")

    for session in session_dir:
        session_path = pathlib.Path(os.path.expanduser(session)).absolute()
        demo_dir = session_path / "demos"
        mapping_dir = demo_dir / "mapping"
        aurco_dir = demo_dir / "aurco"
        map_path = mapping_dir / "map_atlas.osa"

        print("############## 00_process_videos_ros2 #############")
        script_path = script_dir_ros2 / "00_process_videos_ros2.py"
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        _run_or_fail(
            _python_cmd(
                script_path,
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
            )
        )

        print("############# 02_create_map ###########")
        script_path = script_dir_ros1 / "02_create_map.py"
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        if not mapping_dir.is_dir():
            raise FileNotFoundError(mapping_dir)
        if not map_path.is_file():
            cmd = _python_cmd(
                script_path,
                "--input_dir",
                str(mapping_dir),
                "--map_path",
                str(map_path),
                "--docker_image",
                docker_image,
            )
            if no_docker_pull:
                cmd.append("--no_docker_pull")
            if no_mask:
                cmd.append("--no_mask")
            _run_or_fail(cmd)

        print("############# 03_relocalize_aurco_only ###########")
        if not aurco_dir.is_dir():
            raise FileNotFoundError(aurco_dir)
        _relocalize_single_demo(
            demo_dir=aurco_dir,
            map_path=map_path,
            docker_image=docker_image,
            no_docker_pull=no_docker_pull,
            no_mask=no_mask,
        )

        print("############# 04_detect_aruco_of_base ###########")
        script_path = script_dir_ros1 / "04_detect_aruco.py"
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        camera_intrinsics = calibration_dir_path / "d435i_960_540.json"
        aruco_config = calibration_dir_path / "aruco_config.yaml"
        if not camera_intrinsics.is_file():
            raise FileNotFoundError(camera_intrinsics)
        if not aruco_config.is_file():
            raise FileNotFoundError(aruco_config)
        _run_or_fail(
            _python_cmd(
                script_path,
                str(aurco_dir),
                "--camera_intrinsics",
                str(camera_intrinsics),
                "--aruco_yaml",
                str(aruco_config),
                "--output",
                "tag_detection.pkl",
            )
        )

        print("############# 05_run_calibrations ###########")
        script_path = script_dir_ros1 / "05_run_calibrations.py"
        if not script_path.is_file():
            raise FileNotFoundError(script_path)
        _run_or_fail(
            _python_cmd(
                script_path,
                "--aurco_dir",
                str(aurco_dir),
            )
        )

        tx_slam_tag = aurco_dir / "tx_slam_tag.json"
        if not tx_slam_tag.is_file():
            raise FileNotFoundError(tx_slam_tag)
        print(f"Desk ArUco global pose saved to {tx_slam_tag}")


if __name__ == "__main__":
    main()
