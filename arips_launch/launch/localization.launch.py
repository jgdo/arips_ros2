import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, LogInfo,
                            RegisterEventHandler, ExecuteProcess)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import (AndSubstitution, LaunchConfiguration,
                                  NotSubstitution, PythonExpression)
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch_ros.descriptions import ParameterFile


def generate_launch_description():
    autostart = LaunchConfiguration('autostart')
    use_lifecycle_manager = LaunchConfiguration("use_lifecycle_manager")
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params_file = LaunchConfiguration('slam_params_file')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the slamtoolbox. '
                    'Ignored when use_lifecycle_manager is true.')
    declare_use_lifecycle_manager = DeclareLaunchArgument(
        'use_lifecycle_manager', default_value='false',
        description='Enable bond connection during node activation')
    declare_use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation/Gazebo clock')
    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(get_package_share_directory("arips_launch"),
                                    'params', 'slam_localization.yaml'),
        description='Full path to the ROS2 parameters file to use for the slam_toolbox node')
    declare_bag_file_cmd = DeclareLaunchArgument(
        'bag_file',
        default_value='',
        description='Path to a rosbag to play. Leave empty to disable bag playback.')
    declare_bag_rate_cmd = DeclareLaunchArgument(
        'bag_rate',
        default_value='1',
        description='Playback rate for the rosbag.')

    # Perform substitution `$find-pkg-share`
    slam_params_file_w_subst = ParameterFile(
        slam_params_file,
        allow_substs=True,
    )

    start_localization_slam_toolbox_node = LifecycleNode(
        parameters=[
            slam_params_file_w_subst,
            {
            'use_lifecycle_manager': use_lifecycle_manager,
            'use_sim_time': use_sim_time,
            }
        ],
        package='slam_toolbox',
        executable='localization_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(start_localization_slam_toolbox_node),
            transition_id=Transition.TRANSITION_CONFIGURE
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager)))
    )

    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=start_localization_slam_toolbox_node,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[LifecycleLaunch] Slamtoolbox node is activating."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(start_localization_slam_toolbox_node),
                    transition_id=Transition.TRANSITION_ACTIVATE
                ))
            ]
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager)))
    )

    return LaunchDescription([
        declare_autostart_cmd,
        declare_use_lifecycle_manager,
        declare_use_sim_time_argument,
        declare_slam_params_file_cmd,
        declare_bag_file_cmd,
        declare_bag_rate_cmd,
        start_localization_slam_toolbox_node,
        configure_event,
        activate_event,
    ])
