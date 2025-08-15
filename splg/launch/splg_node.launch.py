from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('splg')
    weights_dir = os.path.join(pkg_dir, 'weights')
    engine_path = os.path.join(weights_dir, 'superpoint_lightglue_pipeline_b2_h400_w640_kp512.engine')
    
    # Create node
    splg_node = Node(
        package='splg',
        executable='splg_node',
        name='splg_node',
        parameters=[
            {
                'engine_path': engine_path,
                'input_height': 400,
                'input_width': 640,
                'max_keypoints': 512,
                'profile_inference': False,
                'use_gpu_preprocessing': True,
                'frame_skip_mode': 'every_nth',  # Options: 'every_nth', 'rate_limit', 'none'
                'frame_skip_n': 2,  # For every_nth mode
                'max_process_rate_hz': 20.0,  # For rate_limit mode
            }
        ],
        remappings=[
            ('/image1', '/left/image_rect'),
            ('/image2', '/right/image_rect'),
        ],
        output='screen',
        emulate_tty=True,
    )
    
    return LaunchDescription([
        splg_node,
    ])