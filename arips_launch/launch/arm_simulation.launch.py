import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    real_launch_file = os.path.join(
        get_package_share_directory('arips_description'), 'launch', 'real.launch.py'
    )
    servo_control_launch_file = os.path.join(
        get_package_share_directory('arips_launch'), 'launch', 'servo_control.launch.py'
    )
    move_group_launch_file = os.path.join(
        get_package_share_directory('arips_moveit_config'), 'launch', 'move_group.launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(real_launch_file),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(servo_control_launch_file),
            launch_arguments={'enable_mock_mode': 'true'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(move_group_launch_file),
        ),
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
        ),
    ])
