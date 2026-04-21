import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, RegisterEventHandler, EmitEvent, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('td_compression')
    rviz_config_file = os.path.join(pkg_share, 'config', 'octo_rviz_exp.rviz')

    input_bag_arg = DeclareLaunchArgument(
        'input_bag',
        default_value='/home/hampek/uni/x7014e/data/bags/sr_B_route1/sr_B_route1.mcap'
    )
    
    output_bag_arg = DeclareLaunchArgument(
        'output_bag',
        default_value='/home/hampek/uni/x7014e/data/bags/sr_B_route1_MAP'
    )

    input_bag = LaunchConfiguration('input_bag')
    output_bag = LaunchConfiguration('output_bag')

    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', input_bag, '--clock'],
        output='screen'
    )

    record_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', 
            '-o', output_bag, 
            '/tf', '/tf_static', '/octomap_binary', '/clock'
        ],
        output='screen'
    )

    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        parameters=[{
            'use_sim_time': True,
            'resolution': 0.25,
            'frame_id': 'chinook/odom',
            'base_frame_id': 'chinook/base',
            'sensor_model.max_range': 10.0,
            'sensor_model.hit': 0.8,
            'sensor_model.miss': 0.49,
            'sensor_model.max' : 0.99,
            'sensor_model.min' : 0.12,
        }],
        remappings=[
            ('cloud_in', '/chinook/ouster/points')
        ],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        parameters=[{'use_sim_time': True}],
        output='screen',
        arguments=['-d', rviz_config_file],
        additional_env={'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/liboctomap.so'}
    )

    shutdown_on_bag_finish = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=play_bag,
            on_exit=[
                TimerAction(
                    period=5.0,
                    actions=[
                        EmitEvent(event=Shutdown(reason='Bag playback finished.'))
                    ]
                )
            ]
        )
    )

    return LaunchDescription([
        input_bag_arg,
        output_bag_arg,
        octomap_node,
        rviz_node,
        record_bag,
        play_bag,
        shutdown_on_bag_finish
    ])