import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, EmitEvent, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_share = get_package_share_directory('td_compression')
    bag_path = '/home/hampek/uni/x7014e/data/bags/sr_B_route2_MAP/sr_B_route2_MAP_0.mcap'

    base_weight_dir = os.path.join(pkg_share, 'weights')
    base_config_dir = os.path.join(pkg_share, 'config')

    base_out_dir = '/home/hampek/uni/x7014e/data/keyframes/sr_B_route2'

    rviz_config_file = os.path.join(pkg_share, 'config', 'encode_octomap.rviz')

    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--rate', '3.0', '--clock'],
        output='screen'
    )


    model_a_node = Node(
        package='td_compression',
        executable='online_encoder',
        name='SR_128',
        parameters=[{
            'use_sim_time': True,
            'config_path': os.path.join(base_config_dir, 'SR_128.yaml'),
            'model_path': os.path.join(base_weight_dir, 'SR_128.ckpt'),
            'output_dir': os.path.join(base_out_dir, 'SR_128')
        }],
        output='screen'
    )

    # model_b_node = Node(
    #     package='td_compression',
    #     executable='online_encoder',
    #     name='encoder_model_b',
    #     parameters=[{
    #         'use_sim_time': True,
    #         'config_path': os.path.join(base_config_dir, 'r025_256_512x4.yaml'),
    #         'model_path': os.path.join(base_weight_dir, 'r025_256_512x4.ckpt'),
    #         'output_dir': os.path.join(base_out_dir, 'results_model_b')
    #     }],
    #     output='screen'
    # )

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


    shutdown_on_bag_finish = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=play_bag,
            on_exit=[
                TimerAction(
                    period=5.0,
                    actions=[EmitEvent(event=Shutdown(reason='Benchmark complete. Shutting down.'))]
                )
            ]
        )
    )

    return LaunchDescription([
        play_bag,
        model_a_node,
        rviz2_node
        # model_b_node,
        # shutdown_on_bag_finish
    ])