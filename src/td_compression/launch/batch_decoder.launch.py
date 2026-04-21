import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_compression = get_package_share_directory('td_compression')

    rviz_config_file = os.path.join(pkg_compression, 'config', 'decode_octomap.rviz')

    batch_decoder= Node(
        package='td_compression',
        executable='batch_decoder',
        name='batch_decoder',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'config_path': os.path.join(pkg_compression, 'config', 'SR_128.yaml'),
            'model_path': os.path.join(pkg_compression, 'weights', 'SR_128.pt'),
            'input_dir': '/home/hampek/uni/x7014e/data/keyframes/sr_B_route2/SR_128'
        }]
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{
            'use_sim_time': False 
        }],
        additional_env={'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/liboctomap.so'}
    )


    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    return LaunchDescription([
        batch_decoder,
        rviz2_node,
        static_tf_node
    ])