import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    serial_port = LaunchConfiguration('serial_port')
    baud_rate = LaunchConfiguration('baud_rate')
    arm_enabled = LaunchConfiguration('arm_enabled')
    kinect_enabled = LaunchConfiguration('kinect_enabled')

    xacro_file = os.path.join(
        get_package_share_directory('arips_description'), 'urdf', 'arips_onshape.urdf.xacro'
    )
    controllers_file = os.path.join(
        get_package_share_directory('arips_launch'), 'config', 'ros2_control_controllers.yaml'
    )

    robot_description = Command([
        'xacro ', xacro_file,
        ' use_ros2_control:=true',
        ' serial_port:=', serial_port,
        ' baud_rate:=', baud_rate,
        ' arm_enabled:=', arm_enabled,
        ' kinect_enabled:=', kinect_enabled,
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_port',
            default_value='/dev/feetech_servo',
            description='Serial device for the feetech servo bus',
        ),
        DeclareLaunchArgument(
            'baud_rate',
            default_value='1000000',
            description='Feetech bus baud rate (currently ignored by feetech_ros2_driver v0.2.2, which hardcodes 1,000,000 baud)',
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

        # publish robot_description as a latched string topic, without the
        # tf/joint_states side effects of robot_state_publisher
        ExecuteProcess(
            cmd=['python3', '-c', '''
import sys
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

rclpy.init()
node = Node("feetech_robot_description_publisher")
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
publisher = node.create_publisher(String, "/feetech_robot_description", qos)
publisher.publish(String(data=sys.argv[1]))
try:
    rclpy.spin(node)
except (KeyboardInterrupt, ExternalShutdownException):
    pass
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
''', robot_description],
            output='screen',
        ),

        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[controllers_file],
            remappings=[
                    ("/robot_description", "/feetech_robot_description"),
                ],
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
