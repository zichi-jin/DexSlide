from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "vocab",
                default_value="/home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt",
            ),
            DeclareLaunchArgument(
                "settings",
                default_value="/home/jzq/MyJob/DexSlide/umi_mono/config/RealSense_D435i_online.yaml",
            ),
            DeclareLaunchArgument("map_atlas", default_value=""),
            DeclareLaunchArgument(
                "image_topic", default_value="/camera/camera/color/image_raw"
            ),
            DeclareLaunchArgument(
                "accel_topic", default_value="/camera/camera/accel/sample"
            ),
            DeclareLaunchArgument(
                "gyro_topic", default_value="/camera/camera/gyro/sample"
            ),
            DeclareLaunchArgument("pose_topic", default_value="/dexslide/slam/pose"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument(
                "camera_frame", default_value="camera_color_optical_frame"
            ),
            DeclareLaunchArgument("enable_aruco_world", default_value="false"),
            DeclareLaunchArgument("tx_slam_tag", default_value=""),
            DeclareLaunchArgument(
                "camera_intrinsics",
                default_value="/home/jzq/MyJob/DexSlide/umi_mono/example/calibration/d435i_960_540.json",
            ),
            DeclareLaunchArgument(
                "aruco_yaml",
                default_value="/home/jzq/MyJob/DexSlide/umi_mono/example/calibration/aruco_config_wrist.yaml",
            ),
            DeclareLaunchArgument("target_marker_id", default_value="10"),
            DeclareLaunchArgument("world_frame", default_value="world"),
            DeclareLaunchArgument("marker_frame", default_value="aruco_marker"),
            DeclareLaunchArgument(
                "aruco_world_pose_topic", default_value="/dexslide/aruco/world_pose"
            ),
            DeclareLaunchArgument("aruco_max_pose_dt", default_value="0.2"),
            DeclareLaunchArgument("max_lost_frames", default_value="900"),
            DeclareLaunchArgument("accel_gyro_pair_window_s", default_value="0.020"),
            DeclareLaunchArgument("activate_localization_mode", default_value="false"),
            Node(
                package="dexslide_slam_publisher",
                executable="realsense_topic_slam_node",
                parameters=[
                    {
                        "vocab": LaunchConfiguration("vocab"),
                        "settings": LaunchConfiguration("settings"),
                        "map_atlas": LaunchConfiguration("map_atlas"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "accel_topic": LaunchConfiguration("accel_topic"),
                        "gyro_topic": LaunchConfiguration("gyro_topic"),
                        "pose_topic": LaunchConfiguration("pose_topic"),
                        "map_frame": LaunchConfiguration("map_frame"),
                        "camera_frame": LaunchConfiguration("camera_frame"),
                        "max_lost_frames": ParameterValue(
                            LaunchConfiguration("max_lost_frames"), value_type=int
                        ),
                        "accel_gyro_pair_window_s": ParameterValue(
                            LaunchConfiguration("accel_gyro_pair_window_s"),
                            value_type=float,
                        ),
                        "activate_localization_mode": ParameterValue(
                            LaunchConfiguration("activate_localization_mode"),
                            value_type=bool,
                        ),
                    }
                ],
                output="screen",
            ),
            Node(
                package="dexslide_slam_publisher",
                executable="aruco_world_pose_node.py",
                condition=IfCondition(LaunchConfiguration("enable_aruco_world")),
                parameters=[
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "slam_pose_topic": LaunchConfiguration("pose_topic"),
                        "output_pose_topic": LaunchConfiguration(
                            "aruco_world_pose_topic"
                        ),
                        "world_frame": LaunchConfiguration("world_frame"),
                        "marker_frame": LaunchConfiguration("marker_frame"),
                        "tx_slam_tag": LaunchConfiguration("tx_slam_tag"),
                        "camera_intrinsics": LaunchConfiguration("camera_intrinsics"),
                        "aruco_yaml": LaunchConfiguration("aruco_yaml"),
                        "target_marker_id": ParameterValue(
                            LaunchConfiguration("target_marker_id"), value_type=int
                        ),
                        "max_pose_dt": ParameterValue(
                            LaunchConfiguration("aruco_max_pose_dt"), value_type=float
                        ),
                        "input_pose_is_twc": True,
                    }
                ],
                output="screen",
            ),
        ]
    )
