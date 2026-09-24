from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('arips_route_graph')
    parameter_file = join(package_share, 'config', 'route_graph.yaml')

    return LaunchDescription([
        Node(
            package='arips_route_graph',
            executable='route_graph_generator',
            name='route_graph_generator',
            parameters=[parameter_file],
            output='screen',
        ),
    ])
