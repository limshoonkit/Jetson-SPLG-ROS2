import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    oak_component = ComposableNode(
        package='oakd_ros2_minimal',
        namespace='',
        plugin='uosm::depthai::OakCamera',
        name='oak_node',
        parameters=[{
                'monoResolution': '400p',
                'lrFPS': 30,
                'syncThreshold': 10,
                'convInterleaved': False,
                'convGetBaseDeviceTimestamp': False,
                'convUpdateROSBaseTimeOnToRosMsg': True,
                'convReverseSocketOrder': True,
            }],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    oak_container = ComposableNodeContainer(
        name='oak_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        arguments=['--ros-args', '--log-level', 'info'],
        output='screen',
        composable_node_descriptions=[oak_component]
    )

    # Create and return launch description
    return LaunchDescription([oak_container])
