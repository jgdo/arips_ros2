from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():    
    component_manager_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arips_launch'),
                'launch',
                'component_manager.launch.py',
            ])
        )
    )

    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arips_description'),
                'launch',
                'real.launch.py',
            ])
        ),
    )

    dashboard_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arips_web_dashboard'),
                'launch',
                'dashboard.launch.py',
            ])
        )
    )

    return LaunchDescription([
        component_manager_launch,
        description_launch,
        dashboard_launch,

        Node(package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_wheel_center_to_footprint',
            arguments=['--x', '-0.1', '--y', '0', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'arips_wheel_center', '--child-frame-id', 'base_footprint'],
        ),

        Node(package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_footprint_to_base_link',
            arguments=['--x', '-0.1', '--y', '0', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'base_footprint', '--child-frame-id', 'base_link'],
        ),

        Node(package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_kinect_mount_to_kinect_link',
            arguments=['--x', '0', '--y', '0', '--z', '0.03',
                        '--roll', '0', '--pitch', '0', '--yaw', '-1.57079632679',
                        '--frame-id', 'kinect_mount', '--child-frame-id', 'kinect_link'],
        ),

        Node(package='arips_serial_bridge',
             executable='serial_node',
             name='serial_node',
        )
    ])
