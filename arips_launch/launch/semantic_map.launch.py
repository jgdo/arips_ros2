from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    semantic_map_file = join(
        get_package_share_directory('arips_semantic_map'),
        'maps',
        'map_lu.yaml',
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

    return LaunchDescription([
        Node(
            package='arips_semantic_map',
            executable='semantic_map_server',
            name='semantic_map_server',
            output='screen',
            parameters=[{'map_file': semantic_map_file}],
        ),
        route_graph_launch,
    ])