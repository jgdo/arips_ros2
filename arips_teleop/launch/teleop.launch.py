import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory('arips_description'), 'urdf', 'arips_onshape.urdf.xacro')
    params_file = os.path.join(
        get_package_share_directory('arips_teleop'), 'config', 'teleop.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'kinect_up_button', default_value='button11',
            description="Joy input raising the kinect: 'buttonN', 'axisN' or '-axisN'"),
        DeclareLaunchArgument(
            'kinect_down_button', default_value='button12',
            description="Joy input lowering the kinect: 'buttonN', 'axisN' or '-axisN'"),
        Node(
            package='arips_teleop',
            executable='arm_teleop_node',
            output='screen',
            parameters=[
                {'robot_description': ParameterValue(
                    Command(['xacro ', urdf_path]), value_type=str)},
                params_file,
                {
                    'kinect_up_button': ParameterValue(
                        LaunchConfiguration('kinect_up_button'), value_type=str),
                    'kinect_down_button': ParameterValue(
                        LaunchConfiguration('kinect_down_button'), value_type=str),
                },
            ],
        ),
    ])
