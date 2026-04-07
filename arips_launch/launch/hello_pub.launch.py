from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=['ros2', 'topic', 'pub', '/hello', 'std_msgs/msg/String', 'data: hello'])
    ])