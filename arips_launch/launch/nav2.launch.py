import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = get_package_share_directory('arips_launch')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_respawn = LaunchConfiguration('use_respawn')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(package_share, 'params', 'nav2.yaml'),
        description='Full path to the Nav2 parameters file.')
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
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

    # nav2_bringup's navigation_launch.py no longer starts a lifecycle manager (only
    # bringup_launch.py does), so start our own. node_names/bond_timeout come from params_file.
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            ParameterFile(params_file, allow_substs=True),
            {
                'autostart': ParameterValue(autostart, value_type=bool),
                'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
            },
        ],
    )

    cross_door_step_server = Node(
        package='arips_actions',
        executable='cross_door_step_server',
        name='cross_door_step_server',
        output='screen',
    )

    return LaunchDescription([
        declare_params_file,
        declare_use_sim_time,
        declare_autostart,
        declare_use_respawn,
        navigation_launch,
        lifecycle_manager,
        cross_door_step_server,
    ])
