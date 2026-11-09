import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.utilities import perform_substitutions
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare

# Static 2D map (in arips_maps/maps) and semantic map (in arips_semantic_map/maps) per world.
# Their frames match the Gazebo world frame.
WORLD_MAPS = {
    'lu': ('lu_5_okt_2026_lyrical.yaml', 'map_lu.yaml'),
}


def start_maps(context):
    world = perform_substitutions(context, [LaunchConfiguration('world')])
    world = os.path.splitext(os.path.basename(world))[0]
    if world not in WORLD_MAPS:
        return [LogInfo(msg=f"No static map for world '{world}', not starting map server and semantic map")]
    static_map, semantic_map = WORLD_MAPS[world]

    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[{
            'yaml_filename': os.path.join(get_package_share_directory('arips_maps'), 'maps', static_map),
            'use_sim_time': True,
        }],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['map_server'],
            'use_sim_time': True,
        }],
    )

    semantic_map_server = Node(
        package='arips_semantic_map',
        executable='semantic_map_server',
        name='semantic_map_server',
        output='screen',
        parameters=[{
            'map_file': os.path.join(
                get_package_share_directory('arips_semantic_map'), 'maps', semantic_map),
            'use_sim_time': True,
        }],
    )

    route_graph_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arips_route_graph'),
                'launch',
                'route_graph.launch.py',
            ])
        )
    )

    return [map_server, lifecycle_manager, semantic_map_server, route_graph_launch]


# Simulation counterpart of arips_launch/startup.launch.py. The robot description,
# the static TFs and the hardware drivers are replaced by sim.launch.py.
# Localization is replaced by ground truth: odom -> arips_wheel_center is the robot's
# world pose from Gazebo and map -> odom is the identity.
def generate_launch_description():
    components_config = os.path.join(
        get_package_share_directory('arips_gazebo'), 'config', 'arips_components_sim.yaml')

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arips_gazebo'),
                'launch',
                'sim.launch.py',
            ])
        ),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'gui': LaunchConfiguration('gui'),
        }.items(),
    )

    dashboard_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arips_web_dashboard'),
                'launch',
                'dashboard.launch.py',
            ])
        ),
        launch_arguments={'port': '8282'}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='rect',
            description='Name of a world in arips_gazebo/worlds (rect, lu, ...) or path to an SDF file',
        ),
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),

        sim_launch,
        dashboard_launch,
        OpaqueFunction(function=start_maps),

        Node(package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_map_to_odom',
            arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

        Node(
            package='component_manager',
            executable='component_manager',
            name='component_manager',
            parameters=[{'config_file': components_config}],
            output='screen',
        ),
    ])
