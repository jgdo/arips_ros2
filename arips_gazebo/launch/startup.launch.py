import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Simulation counterpart of arips_launch/startup.launch.py. The robot description,
# the static TFs and the hardware drivers are replaced by sim.launch.py.
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
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(
                get_package_share_directory('arips_gazebo'), 'worlds', 'rectangular_room.sdf'),
            description='Gazebo world file',
        ),
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),

        sim_launch,
        dashboard_launch,

        Node(
            package='component_manager',
            executable='component_manager',
            name='component_manager',
            parameters=[{'config_file': components_config}],
            output='screen',
        ),

        Node(package='arips_voice_interface',
             executable='arips_say',
             name='arips_say',
             output='screen'
        ),
    ])
