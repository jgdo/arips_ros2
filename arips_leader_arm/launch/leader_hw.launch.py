import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('arips_leader_arm'),
        'config', 'adc_joint_state.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Parameter file for the ADC joint state node'),
        Node(
            package='arips_leader_arm',
            executable='adc_joint_state_node',
            name='adc_joint_state',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
        ),
    ])
