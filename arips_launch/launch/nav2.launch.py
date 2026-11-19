import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode, ParameterFile, ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = get_package_share_directory('arips_launch')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_respawn = LaunchConfiguration('use_respawn')
    use_composition = LaunchConfiguration('use_composition')

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
    declare_use_composition = DeclareLaunchArgument(
        'use_composition',
        default_value='false',
        description='Run all Nav2 nodes in one component container (one process). With '
                    'separate processes, Fast DDS on this host sometimes never matches some '
                    'endpoints of nodes that start at the same time, which aborts bringup.')

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
            'use_composition': use_composition,
            'container_name': 'nav2_container',
        }.items(),
    )

    # navigation_launch.py loads its nodes into this container when use_composition is
    # true, but does not start it (only bringup_launch.py does).
    nav2_container = Node(
        condition=IfCondition(use_composition),
        package='rclcpp_components',
        executable='component_container',
        name='nav2_container',
        arguments=['--isolated', '--executor-type', 'single-threaded'],
        # The params file must be given to the container too: nodes created inside the
        # components (e.g. the local/global costmaps) only get the container's parameters.
        parameters=[
            ParameterFile(params_file, allow_substs=True),
            {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)},
        ],
        output='screen',
    )

    # nav2_bringup's navigation_launch.py no longer starts a lifecycle manager (only
    # bringup_launch.py does), so start our own. node_names/bond_timeout come from params_file.
    lifecycle_manager_parameters = [
        ParameterFile(params_file, allow_substs=True),
        {
            'autostart': ParameterValue(autostart, value_type=bool),
            'use_sim_time': ParameterValue(use_sim_time, value_type=bool),
        },
    ]
    lifecycle_manager = Node(
        condition=UnlessCondition(use_composition),
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=lifecycle_manager_parameters,
    )
    lifecycle_manager_composed = LoadComposableNodes(
        condition=IfCondition(use_composition),
        target_container='/nav2_container',
        composable_node_descriptions=[
            ComposableNode(
                package='nav2_lifecycle_manager',
                plugin='nav2_lifecycle_manager::LifecycleManager',
                name='lifecycle_manager_navigation',
                parameters=lifecycle_manager_parameters,
            ),
        ],
    )

    return LaunchDescription([
        declare_params_file,
        declare_use_sim_time,
        declare_autostart,
        declare_use_respawn,
        declare_use_composition,
        nav2_container,
        navigation_launch,
        lifecycle_manager,
        lifecycle_manager_composed,
    ])
