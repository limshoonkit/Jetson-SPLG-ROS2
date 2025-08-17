import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    pkg_dir = get_package_share_directory('splg')
    weights_dir = os.path.join(pkg_dir, 'weights', 'fp16', '400_640')
    engine_path = os.path.join(weights_dir, 'superpoint_lightglue_b2_h400_w640_kp256.engine')

    splg_component = ComposableNode(
        package='splg',
        namespace='',
        plugin='uosm::SuperPointLightGlueTrt',
        name='splg_node',
        parameters=[
            {
                'engine_path': engine_path,
                'input_height': 400,
                'input_width': 640,
                'max_keypoints': 256,
                'profile_inference': True,
                'use_gpu_preprocessing': True,
                'frame_skip_mode': 'every_nth',  # Options: 'every_nth', 'rate_limit', 'none'
                'frame_skip_n': 3,  # For every_nth mode
                'max_process_rate_hz': 20.0,  # For rate_limit mode
                'use_unified_memory': True,
            }
        ],
        remappings=[
            ('/image1', '/left/image_rect'),
            ('/image2', '/right/image_rect'),
        ],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

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

    splg_trt_container = ComposableNodeContainer(
        name='splg_trt_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        arguments=['--ros-args', '--log-level', 'info'],
        output='screen',
        composable_node_descriptions=[
            splg_component, 
            oak_component
        ]
    )

    # Create and return launch description
    return LaunchDescription([splg_trt_container])