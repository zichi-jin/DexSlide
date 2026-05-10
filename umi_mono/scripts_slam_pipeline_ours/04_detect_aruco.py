import sys
import os
import pathlib
import subprocess
import click

@click.command()
@click.argument('input_dirs', nargs=-1)
@click.option('-ci', '--camera_intrinsics', required=True, help='Camera intrinsics json file')
@click.option('-ac', '--aruco_yaml', required=True, help='Aruco config yaml file')
@click.option('--video', default='raw_video.mp4')
@click.option('--output', default='tag_detection.pkl', help='Output pkl file for tag detection results')
def main(input_dirs, camera_intrinsics, aruco_yaml, video, output):
    for input_dir in input_dirs: 
        input_dir = pathlib.Path(input_dir)
        video_path = input_dir.joinpath(video)
        pkl_path = input_dir.joinpath(output)

        if not video_path.is_file():
            print(f"❌ {video_path} not found!")
            continue
        if pkl_path.is_file():
            print(f"tag_detection.pkl already exists, skipping.")
            continue

        script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco.py')

        cmd = [
            'python', str(script_path),
            '--input', str(video_path),
            '--output', str(pkl_path),
            '--intrinsics_json', camera_intrinsics,
            '--aruco_yaml', aruco_yaml,
            '--num_workers', '8'
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("Done!")
        else:
            print("Failed!")

if __name__ == "__main__":
    main()