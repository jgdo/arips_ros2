import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('arips_leader_arm')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', 'leader_description.launch.py'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', 'leader_hw.launch.py'))),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_leader_base_plate',
            arguments=[
                '--x', '-0.4', '--y', '0', '--z', '0.8',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'leader_base_plate',
            ],
        ),
    ])
