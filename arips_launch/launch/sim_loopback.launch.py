from os.path import join
from json import dumps

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    semantic_map_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('arips_launch'),
                'launch',
                'semantic_map_with_static_map.launch.py',
            ])
        )
    )

    params_file = LaunchConfiguration('params_file')
    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=join(
            get_package_share_directory('arips_launch'),
            'params',
            'loopback_sim_params.yaml',
        ),
        description='Full path to the loopback simulator parameters file.',
    )

    loopback_simulator = LifecycleNode(
        package='nav2_loopback_sim',
        executable='loopback_simulator',
        name='loopback_simulator',
        namespace='',
        output='screen',
        autostart=True,
        parameters=[
            params_file,
            {'use_sim_time': True},
        ],
    )

    initial_pose = dumps({
        'header': {
            'stamp': {'sec': 1790786323, 'nanosec': 99460408},
            'frame_id': 'map',
        },
        'pose': {
            'pose': {
                'position': {
                    'x': -1.148110032081604,
                    'y': 0.037738800048828125,
                    'z': 0.0,
                },
                'orientation': {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.7062583001994037,
                    'w': 0.7079542452725662,
                },
            },
            'covariance': [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891909122467,
            ],
        },
    })

    publish_initial_pose = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once',
            '/initialpose',
            'geometry_msgs/msg/PoseWithCovarianceStamped',
            initial_pose,
        ],
        output='screen',
    )

    delayed_initial_pose = TimerAction(
        period=4.0,
        actions=[publish_initial_pose],
    )

    # map_to_odom = Node(
    #     package='tf2_ros',
    #     executable='static_transform_publisher',
    #     output='screen',
    #     arguments=[
    #         '--x', '0.0', '--y', '0.0', '--z', '0.0',
    #         '--roll', '0', '--pitch', '0', '--yaw', '0',
    #         '--frame-id', 'map',
    #         '--child-frame-id', 'odom',
    #     ],
    # )

    return LaunchDescription([
        declare_params_file,
        semantic_map_launch,
        loopback_simulator,
        # map_to_odom,
        delayed_initial_pose,
    ])