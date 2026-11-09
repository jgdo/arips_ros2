import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit, OnProcessIO
from launch.substitutions import Command, LaunchConfiguration
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# diff_drive_controller 6.9 repeats this every second while no cmd_vel arrives, which is
# the normal idle state. Its throttle is hard-coded and gz_ros2_control cannot pass a log
# level to the controllers, so the line is dropped from Gazebo's console output instead.
SUPPRESSED_GAZEBO_OUTPUT = (
    'Velocity command timed out. Braking.',
)

# Spawn pose (x, y, yaw) per world, used for the x/y/yaw arguments that are not given
DEFAULT_SPAWN_POSES = {
    'lu': (-6.0, 2.8, 0.0),
}


def world_name(context):
    world = perform_substitutions(context, [LaunchConfiguration('world')])
    return os.path.splitext(os.path.basename(world))[0]


def world_file(context):
    """world is either the name of a file in arips_gazebo/worlds or a path to an SDF file."""
    world = perform_substitutions(context, [LaunchConfiguration('world')])
    if os.sep in world or world.endswith('.sdf'):
        return world
    path = os.path.join(get_package_share_directory('arips_gazebo'), 'worlds', world + '.sdf')
    if not os.path.isfile(path):
        worlds = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.dirname(path)) if f.endswith('.sdf'))
        raise RuntimeError(f"Unknown world '{world}', available: {', '.join(worlds)}")
    return path


def start_gazebo(context):
    gz_args = ['-r', '-v', '3']
    if perform_substitutions(context, [LaunchConfiguration('gui')]).lower() not in ('true', '1'):
        gz_args.append('-s')
    world = world_file(context)

    # Replaces ros_gz_sim/gz_sim.launch.py, whose process output cannot be filtered.
    # Its console output goes to the launch log only and is echoed by the handler below.
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', *gz_args, world],
        name='gazebo',
        output={'both': 'log'},
        on_exit=Shutdown(),
    )

    def echo_filtered(stream):
        # Output arrives in chunks that are not aligned to lines
        partial_line = ''

        def handler(event):
            nonlocal partial_line
            *lines, partial_line = (partial_line + event.text.decode(errors='replace')).split('\n')
            for line in lines:
                if not any(pattern in line for pattern in SUPPRESSED_GAZEBO_OUTPUT):
                    stream.write(f'[{event.process_name}] {line}\n')
            stream.flush()

        return handler

    return [
        gazebo,
        RegisterEventHandler(OnProcessIO(
            target_action=gazebo,
            on_stdout=echo_filtered(sys.stdout),
            on_stderr=echo_filtered(sys.stderr),
        )),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('arips_gazebo')
    urdf_file = os.path.join(pkg_share, 'urdf', 'arips_sim.urdf.xacro')
    bridge_config = os.path.join(pkg_share, 'config', 'bridge.yaml')
    controllers_config = os.path.join(pkg_share, 'config', 'controllers.yaml')

    robot_description = ParameterValue(Command(['xacro ', urdf_file]), value_type=str)

    # Lets Gazebo resolve package://arips_description/meshes/... and package://arips_maps/assets/...
    resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.pathsep.join([
            os.path.dirname(get_package_share_directory('arips_description')),
            os.path.dirname(get_package_share_directory('arips_maps')),
            os.path.dirname(pkg_share),
        ]),
    )

    # Lets Gazebo find the gz_ros2_control system plugin
    plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH', os.environ.get('LD_LIBRARY_PATH', ''))

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': True,
        }],
        output='screen',
    )

    rgb_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='kinect_rgb_image_bridge',
        arguments=['kinect/image'],
        remappings=[('kinect/image', 'kinect/rgb/image_raw')],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    arm_controllers_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            'feetech_joint_state_broadcaster',
            'arm_forward_position_controller',
            'kinect_forward_position_controller',
            '--param-file', controllers_config,
        ],
        output='screen',
    )

    # TwistStamped on /cmd_vel as on the real robot. Its wheel odometry stays on
    # /diff_drive_controller/odom, /csm_odom is Gazebo's ground truth (see bridge.yaml).
    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'diff_drive_controller',
            '--param-file', controllers_config,
            '--controller-ros-args',
            '-r /diff_drive_controller/cmd_vel:=/cmd_vel',
        ],
        output='screen',
    )

    def spawn_robot(context):
        pose = dict(zip(('x', 'y', 'yaw'), DEFAULT_SPAWN_POSES.get(world_name(context), (0.0, 0.0, 0.0))))
        for name in pose:
            value = perform_substitutions(context, [LaunchConfiguration(name)])
            if value:
                pose[name] = value

        spawn = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-topic', 'robot_description',
                '-name', 'arips',
                '-x', str(pose['x']),
                '-y', str(pose['y']),
                '-z', '0.01',
                '-Y', str(pose['yaw']),
            ],
            output='screen',
        )

        # The controller manager only exists once the robot (and its gz_ros2_control plugin) is spawned
        start_controllers = RegisterEventHandler(
            OnProcessExit(
                target_action=spawn,
                on_exit=[arm_controllers_spawner, diff_drive_spawner],
            )
        )
        return [spawn, start_controllers]

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='rect',
            description='Name of a world in arips_gazebo/worlds (rect, lu, ...) or path to an SDF file',
        ),
        DeclareLaunchArgument('gui', default_value='true', description='Start the Gazebo GUI'),
        DeclareLaunchArgument(
            'x', default_value='', description='Spawn x position [m], default depends on the world'),
        DeclareLaunchArgument(
            'y', default_value='', description='Spawn y position [m], default depends on the world'),
        DeclareLaunchArgument(
            'yaw', default_value='', description='Spawn yaw [rad], default depends on the world'),
        resource_path,
        plugin_path,
        OpaqueFunction(function=start_gazebo),
        robot_state_publisher,
        OpaqueFunction(function=spawn_robot),
        bridge,
        rgb_image_bridge,
    ])
