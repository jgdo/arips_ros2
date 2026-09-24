from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node


def generate_launch_description():
    semantic_map_file = join(
        get_package_share_directory('arips_semantic_map'),
        'maps',
        'map_lu.yaml',
    )
    static_map_file = join(
        get_package_share_directory('arips_maps'),
        'maps',
        'map_lu_9.9.2026.yaml',
    )

    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[{'yaml_filename': static_map_file}],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'autostart': True},
            {'node_names': ['map_server']},
        ],
    )

    return LaunchDescription([
        Node(
            package='arips_semantic_map',
            executable='semantic_map_server',
            name='semantic_map_server',
            output='screen',
            parameters=[{'map_file': semantic_map_file}],
        ),
        map_server,
        lifecycle_manager,
    ])