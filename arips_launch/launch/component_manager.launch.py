from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config_file = get_package_share_directory('arips_launch') + '/config/arips_components.yaml'

    return LaunchDescription([
        Node(
            package='component_manager',
            executable='component_manager',
            name='component_manager',
            parameters=[{'config_file': config_file}],
            output='screen',
        ),
    ])
