"""Starts all arips_actions servers used by the navigation behavior trees.

Arguments:
  use_sim_time: use the simulation clock (Gazebo).
  sim: start the direct (scripted MoveIt) GrabDoorHandle server for the simulation instead
       of the learned-policy one of the real robot.
  grab_policy: start grab_door_handle_policy_server (doorhandle model, no rosetta) together
       with arips_lerobot's command_rate_limiter as GrabDoorHandle server instead of either of
       the above.
  grab_keypoints: start grab_door_handle_keypoint_server (pre-grab pose from the door handle
       keypoints of arips_keypoint_detection's keypoint_detector_node, which must run
       separately) as GrabDoorHandle server instead of the direct sim / real one. Ignored
       with grab_policy.
  policy_path: pretrained_model folder of the doorhandle checkpoint (grab_policy only).
  limiter_max_speed: command_rate_limiter max_speed (grab_policy only).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    sim = LaunchConfiguration('sim')
    grab_policy = LaunchConfiguration('grab_policy')
    grab_keypoints = LaunchConfiguration('grab_keypoints')
    sim_time = {'use_sim_time': ParameterValue(use_sim_time, value_type=bool)}
    parameters = [sim_time]

    def server(executable, condition=None, extra_parameters=()):
        return Node(
            package='arips_actions',
            executable=executable,
            name=executable,
            output='screen',
            parameters=parameters + list(extra_parameters),
            condition=condition,
        )

    def when(expression):
        return IfCondition(PythonExpression(expression))

    policy = ["'", grab_policy, "'.lower() == 'true'"]
    no_policy = ["'", grab_policy, "'.lower() != 'true' and '", grab_keypoints,
                 "'.lower() != 'true'"]
    keypoints = ["'", grab_policy, "'.lower() != 'true' and '", grab_keypoints,
                 "'.lower() == 'true'"]
    no_policy_sim = no_policy + [" and '", sim, "'.lower() == 'true'"]
    no_policy_real = no_policy + [" and '", sim, "'.lower() != 'true'"]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use the simulation clock'),
        DeclareLaunchArgument('sim', default_value='false',
                              description='Use the direct GrabDoorHandle server for Gazebo'),
        DeclareLaunchArgument('grab_policy', default_value='false',
                              description='Use the doorhandle model GrabDoorHandle server'),
        DeclareLaunchArgument('grab_keypoints', default_value='false',
                              description='Use the door handle keypoint GrabDoorHandle server'),
        DeclareLaunchArgument('policy_path', default_value='',
                              description='pretrained_model folder of the doorhandle checkpoint'),
        DeclareLaunchArgument('limiter_max_speed', default_value='0.3',
                              description='command_rate_limiter max_speed [m/s, rad/s]'),
        server('cross_door_step_server'),
        # Laser based door open/closed detection, used by GetDoorState.
        server('door_state_detector'),
        server('get_door_grasping_approach_pose'),
        server('grab_door_handle_server', when(no_policy_real)),
        server('grab_door_handle_direct_sim_server', when(no_policy_sim)),
        server('grab_door_handle_keypoint_server', when(keypoints)),
        server('grab_door_handle_policy_server', when(policy), [
            {'policy_path': LaunchConfiguration('policy_path')}]),
        # Servo for the policy server: rate limits its targets onto the arm command topic and
        # publishes target_reached.
        Node(
            package='arips_lerobot',
            executable='command_rate_limiter',
            name='command_rate_limiter',
            output='screen',
            parameters=[sim_time, {
                'max_speed': ParameterValue(LaunchConfiguration('limiter_max_speed'),
                                            value_type=float)}],
            condition=when(policy),
        ),
        server('drive_around_pivot_server'),
        server('retreat_from_pivot_server'),
        server('drive_forward_server'),
        server('move_arm_server'),
    ])
