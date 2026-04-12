from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(cmd=['ros2', 'topic', 'pub', '/base_battery_enable_for_sec', 'std_msgs/msg/UInt32', 'data: 2'])
    ])