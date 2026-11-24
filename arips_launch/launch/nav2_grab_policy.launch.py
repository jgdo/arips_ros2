"""Navigation (nav2.launch.py) plus the arips_actions servers with the learned GrabDoorHandle.

GrabDoorHandle is served by grab_door_handle_policy_server: the doorhandle model reaches the
pre-grasp pose through arips_lerobot's command_rate_limiter, then MoveIt moves forward and
turns the gripper. Use this instead of nav2.launch.py plus a separately started
arips_actions.launch.py (e.g. the component_manager components full_navigation and
arips_actions); the two must not run at the same time.

Arguments: those of nav2.launch.py (params_file, use_sim_time, autostart, use_respawn,
use_composition) plus
  policy_path: pretrained_model folder of the doorhandle checkpoint (default: the
      doorhandle_pull_right_30_all_new_aug training run in this workspace; it reaches the
      handle in Gazebo but about 15 mm too high, so the grasp does not catch the lever yet).
  limiter_max_speed: command_rate_limiter max_speed [m/s, rad/s].
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_share = get_package_share_directory('arips_launch')
    actions_share = get_package_share_directory('arips_actions')
    # <ws>/install/arips_actions -> <ws>
    workspace = os.path.dirname(os.path.dirname(get_package_prefix('arips_actions')))
    default_policy = os.path.join(
        workspace, 'src', 'arips_lerobot_data', 'train', 'doorhandle_pull_right_30_all_new_aug',
        'checkpoints', 'last', 'pretrained_model')

    nav2_arguments = {
        'params_file': os.path.join(launch_share, 'params', 'nav2.yaml'),
        'use_sim_time': 'false',
        'autostart': 'true',
        'use_respawn': 'false',
        'use_composition': 'false',
    }
    declarations = [DeclareLaunchArgument(name, default_value=default)
                    for name, default in nav2_arguments.items()]
    declarations += [
        DeclareLaunchArgument('policy_path', default_value=default_policy,
                              description='pretrained_model folder of the doorhandle checkpoint'),
        DeclareLaunchArgument('limiter_max_speed', default_value='0.3',
                              description='command_rate_limiter max_speed [m/s, rad/s]'),
    ]

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(launch_share, 'launch', 'nav2.launch.py')),
        launch_arguments={name: LaunchConfiguration(name) for name in nav2_arguments}.items(),
    )
    actions = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(actions_share, 'launch', 'arips_actions.launch.py')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'sim': 'true',
            'grab_policy': 'true',
            'policy_path': LaunchConfiguration('policy_path'),
            'limiter_max_speed': LaunchConfiguration('limiter_max_speed'),
        }.items(),
    )
    return LaunchDescription(declarations + [actions, nav2])
