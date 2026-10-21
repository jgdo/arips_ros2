import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('arips_leader_arm')

    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_config', default_value=os.path.join(share, 'rviz', 'leader_arm.rviz'),
            description='RViz configuration file'),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='Start joint_state_publisher_gui to move the joints'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(share, 'launch', 'leader_description.launch.py'))),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(LaunchConfiguration('gui')),
            remappings=[('robot_description', '/leader_robot_description')],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            output='screen',
        ),
    ])
