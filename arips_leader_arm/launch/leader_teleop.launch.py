import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('arips_leader_arm'),
        'config', 'leader_teleop.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Parameter file for the leader teleop node'),
        Node(
            package='arips_leader_arm',
            executable='leader_teleop',
            name='leader_teleop',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
