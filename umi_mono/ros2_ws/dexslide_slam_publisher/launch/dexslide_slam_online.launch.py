from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _build_actions(context):
    vocab = LaunchConfiguration("vocab").perform(context)
    settings = LaunchConfiguration("settings").perform(context)
    map_atlas = LaunchConfiguration("map_atlas").perform(context)
    exposure_us = LaunchConfiguration("exposure_us").perform(context)
    pose_topic = LaunchConfiguration("pose_topic")
    zmq_endpoint = LaunchConfiguration("zmq_endpoint")
    log_to_file = LaunchConfiguration("log_to_file").perform(context).lower()

    output_mode = "log" if log_to_file == "true" else "screen"

    cmd = [
        "/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Examples/Monocular-Inertial/realsense_online",
        "-v",
        vocab,
        "-s",
        settings,
        "--publisher",
        "zmq",
        "--exposure_us",
        exposure_us,
    ]
    if map_atlas:
        cmd.extend(["-l", map_atlas])

    slam_proc = ExecuteProcess(
        cmd=cmd,
        output=output_mode,
    )
    bridge_node = Node(
        package="dexslide_slam_publisher",
        executable="pose_publisher_node",
        parameters=[
            {
                "pose_topic": pose_topic,
                "zmq_endpoint": zmq_endpoint,
            }
        ],
        output=output_mode,
    )
    shutdown_reason = "dexslide_slam_online component exited"
    return [
        slam_proc,
        bridge_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=slam_proc,
                on_exit=[EmitEvent(event=Shutdown(reason=shutdown_reason))],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=bridge_node,
                on_exit=[EmitEvent(event=Shutdown(reason=shutdown_reason))],
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "vocab",
                default_value="/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt",
            ),
            DeclareLaunchArgument(
                "settings",
                default_value="/data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml",
            ),
            DeclareLaunchArgument("map_atlas", default_value=""),
            DeclareLaunchArgument("exposure_us", default_value="0"),
            DeclareLaunchArgument("pose_topic", default_value="/dexslide/slam/pose"),
            DeclareLaunchArgument("zmq_endpoint", default_value="tcp://127.0.0.1:5555"),
            DeclareLaunchArgument("log_to_file", default_value="false"),
            OpaqueFunction(function=_build_actions),
        ]
    )
