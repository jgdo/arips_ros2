import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory('arips_description'), 'urdf', 'arips_onshape.urdf.xacro')
    params_file = os.path.join(
        get_package_share_directory('arips_teleop'), 'config', 'teleop.yaml')

    return LaunchDescription([
        Node(
            package='arips_teleop',
            executable='arm_teleop_node',
            output='screen',
            parameters=[
                {'robot_description': ParameterValue(
                    Command(['xacro ', urdf_path]), value_type=str)},
                params_file,
            ],
        ),
    ])
