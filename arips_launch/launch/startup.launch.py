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
                'display.launch.py',
            ])
        ),
        launch_arguments={'use_gui': 'false'}.items(),
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
            name='static_tf_pub_laser',
            arguments=['--x', '0.1', '--y', '0', '--z', '0',
                        '--roll', '0', '--pitch', '0', '--yaw', '0',
                        '--frame-id', 'base_footprint', '--child-frame-id', 'arips_wheel_center'],
        )
    ])
