from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_filepath = PathJoinSubstitution([
        FindPackageShare('arips_launch'),
        'params',
        'teleop_xbox.yaml',
    ])

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('teleop_twist_joy'),
                    'launch',
                    'teleop-launch.py',
                ])
            ),
            launch_arguments={
                'joy_config': 'xbox',
                'config_filepath': config_filepath,
            }.items(),
        ),
    ])
