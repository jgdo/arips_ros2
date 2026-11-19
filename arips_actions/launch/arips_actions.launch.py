"""Starts all arips_actions servers used by the navigation behavior trees.

Arguments:
  use_sim_time: use the simulation clock (Gazebo).
  sim: start the direct (scripted MoveIt) GrabDoorHandle server for the simulation instead
       of the learned-policy one of the real robot.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim = LaunchConfiguration('sim')
    parameters = [{'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}]

    def server(executable, condition=None):
        return Node(
            package='arips_actions',
            executable=executable,
            name=executable,
            output='screen',
            parameters=parameters,
            condition=condition,
        )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use the simulation clock'),
        DeclareLaunchArgument('sim', default_value='false',
                              description='Use the direct GrabDoorHandle server for Gazebo'),
        server('cross_door_step_server'),
        # Laser based door open/closed detection, used by GetDoorState.
        server('door_state_detector'),
        server('get_door_grasping_approach_pose'),
        server('grab_door_handle_server', UnlessCondition(sim)),
        server('grab_door_handle_direct_sim_server', IfCondition(sim)),
        server('drive_around_pivot_server'),
        server('retreat_from_pivot_server'),
        server('drive_forward_server'),
        server('move_arm_server'),
    ])
