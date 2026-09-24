import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    package_share = get_package_share_directory('arips_launch')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_respawn = LaunchConfiguration('use_respawn')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(package_share, 'params', 'nav2_planning_test.yaml'),
        description='Full path to the Nav2 parameters file.')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time.')
    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically activate the Nav2 lifecycle nodes.')
    declare_use_respawn = DeclareLaunchArgument(
        'use_respawn',
        default_value='false',
        description='Respawn Nav2 nodes if they exit.')

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py',
            ])
        ),
        launch_arguments={
            'params_file': params_file,
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'use_respawn': use_respawn,
        }.items(),
    )

    loopback_simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_loopback_sim'),
                'loopback_simulation.launch.py',
            ])
        ),
        launch_arguments={
            'params_file': params_file,
            'scan_frame_id': 'arips_wheel_center',
        }.items(),
    )

    static_publisher_cmd = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=[
            '--x', '0.0', '--y', '0.0', '--z', '0.0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'base_footprint', '--child-frame-id', 'arips_wheel_center']
    )

    static_map_to_odom_publisher_cmd = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            output='screen',
            arguments=[
                '--x', '0.0', '--y', '0.0', '--z', '0.0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'map', '--child-frame-id', 'odom']
        )

    return LaunchDescription([
        declare_params_file,
        declare_use_sim_time,
        declare_autostart,
        declare_use_respawn,
        loopback_simulation_launch,
        navigation_launch,
        static_publisher_cmd,
        static_map_to_odom_publisher_cmd,
    ])