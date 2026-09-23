from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['sudo', 'shutdown', 'now'],
            output='screen'
        )
    ])
