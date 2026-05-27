"""
python scripts_slam_pipeline/05_run_calibrations.py data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
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

# %%
@click.command()
@click.option('--aurco_dir', required=True)
def main(aurco_dir):
    script_dir = pathlib.Path(__file__).parent.parent.joinpath('scripts')
    aurco_dir = pathlib.Path(aurco_dir).absolute()
    slam_tag_path = aurco_dir.joinpath('tx_slam_tag.json')
    if slam_tag_path.is_file():
        print(f"tx_slam_tag.json already exists in {aurco_dir}, skipping calibration.")
        return
    # run slam tag calibration
    script_path = script_dir.joinpath('calibrate_slam_tag.py')
    assert script_path.is_file()
    tag_path = aurco_dir.joinpath('tag_detection.pkl')
    assert tag_path.is_file()
    csv_path = aurco_dir.joinpath('camera_trajectory.csv')
    # if not csv_path.is_file():
    #     csv_path = aurco_dir.joinpath('mapping_camera_trajectory.csv')
    #     print("camera_trajectory.csv not found! using mapping_camera_trajectory.csv")
    assert csv_path.is_file()
    
    cmd = [
        sys.executable, str(script_path),
        '--tag_detection', str(tag_path),
        '--csv_trajectory', str(csv_path),
        '--output', str(slam_tag_path),
        # '--keyframe_only'
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    
    # No need gripper range calibration 
    # Since for robotiq the range is fixed ... 
    # run gripper range calibration
    # script_path = script_dir.joinpath('calibrate_gripper_range.py')
    # assert script_path.is_file()
    
    # for gripper_dir in demos_dir.glob("gripper_calibration*"):
    #     gripper_range_path = gripper_dir.joinpath('gripper_range.json')
    #     tag_path = gripper_dir.joinpath('tag_detection.pkl')
    #     assert tag_path.is_file()
    #     cmd = [
    #         'python', str(script_path),
    #         '--input', str(tag_path),
    #         '--output', str(gripper_range_path)
    #     ]
    #     subprocess.run(cmd)

            
# %%
if __name__ == "__main__":
    main()
