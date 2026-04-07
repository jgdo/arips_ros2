import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('arips_web_dashboard')
    web_dir = os.path.join(pkg_share, 'web')

    return LaunchDescription([
        # rosbridge websocket server on port 9090
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket.py',
            name='rosbridge_websocket',
            output='screen',
        ),
        # Simple HTTP server serving the dashboard on port 8080
        ExecuteProcess(
            cmd=['python3', '-m', 'http.server', '8080',
                 '--bind', '0.0.0.0',
                 '--directory', web_dir],
            name='web_server',
            output='screen',
        ),
    ])
