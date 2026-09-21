import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    usb_port = LaunchConfiguration('usb_port')
    baudrate = LaunchConfiguration('baudrate')
    joint_config_file = LaunchConfiguration('joint_config_file')
    arm_enabled = LaunchConfiguration('arm_enabled')
    kinect_enabled = LaunchConfiguration('kinect_enabled')

    xacro_file = os.path.join(
        get_package_share_directory('arips_description'), 'urdf', 'arips_onshape.urdf.xacro'
    )
    controllers_file = os.path.join(
        get_package_share_directory('arips_launch'), 'config', 'ros2_control_controllers.yaml'
    )
    default_joint_config_file = os.path.join(
        get_package_share_directory('arips_launch'), 'config', 'servos.yaml'
    )

    robot_description = ParameterValue(
        Command([
            'xacro ', xacro_file,
            ' use_ros2_control:=true',
            ' usb_port:=', usb_port,
            ' baudrate:=', baudrate,
            ' joint_config_file:=', joint_config_file,
            ' arm_enabled:=', arm_enabled,
            ' kinect_enabled:=', kinect_enabled,
        ]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'usb_port',
            default_value='/dev/feetech_servo',
            description='Serial device for the feetech servo bus',
        ),
        DeclareLaunchArgument(
            'baudrate',
            default_value='1000000',
            description='Feetech bus baud rate (currently ignored by feetech_ros2_driver v0.2.2, which hardcodes 1,000,000 baud)',
        ),
        DeclareLaunchArgument(
            'joint_config_file',
            default_value=default_joint_config_file,
            description='Path to the feetech per-joint calibration YAML',
        ),
        DeclareLaunchArgument(
            'arm_enabled',
            default_value='true',
            description='Spawn the arm_forward_position_controller',
        ),
        DeclareLaunchArgument(
            'kinect_enabled',
            default_value='true',
            description='Spawn the kinect_forward_position_controller',
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[{'robot_description': robot_description}, controllers_file],
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['feetech_joint_state_broadcaster'],
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['arm_forward_position_controller'],
            condition=IfCondition(arm_enabled),
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['kinect_forward_position_controller'],
            condition=IfCondition(kinect_enabled),
            output='screen',
        ),
    ])
