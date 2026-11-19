"""MoveArm action server: simple MoveIt arm moves for behavior trees.

Goal modes (see MoveArm.action):
- named_target: move to an SRDF group state. The states (group and joint values) are read
  from /robot_description_semantic, published by move_group.
- linear_distance: move the door gripper (IK link link_3) in a straight Cartesian line along
  its pointing direction (the -y axis of link_3), orientation unchanged.
"""

import math
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
from arips_action_msgs.action import MoveArm
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient, ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def parse_group_states(srdf_xml):
    """Return {state name: (group, {joint: value})} of all group states in an SRDF."""
    states = {}
    for state in ET.fromstring(srdf_xml).iter('group_state'):
        joints = {j.get('name'): float(j.get('value')) for j in state.iter('joint')}
        states[state.get('name')] = (state.get('group'), joints)
    return states


class _Failure(Exception):
    pass


class MoveArmServer(Node):
    def __init__(self):
        super().__init__('move_arm_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('planning_frame', 'base_link'),
                ('linear_group', 'arm'),
                ('ik_link', 'link_3'),
                ('move_group_action', '/move_action'),
                ('execute_trajectory_action', '/execute_trajectory'),
                ('compute_cartesian_path_service', '/compute_cartesian_path'),
                ('srdf_topic', '/robot_description_semantic'),
                ('planning_time_s', 5.0),
                ('planning_attempts', 5),
                ('max_velocity_scaling', 0.5),
                ('max_acceleration_scaling', 0.5),
                ('joint_goal_tolerance', 0.005),
                ('linear_velocity_scaling', 0.1),
                ('linear_acceleration_scaling', 0.1),
                ('linear_max_step', 0.002),
                ('linear_revolute_jump_threshold', 0.3),
                ('linear_prismatic_jump_threshold', 0.05),
                ('avoid_collisions', True),
                ('execution_timeout_s', 60.0),
                ('server_wait_timeout_s', 5.0),
                ('tf_timeout_s', 1.0),
            ],
        )
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._p = p

        self._states = {}
        self.create_subscription(
            String, p('srdf_topic'), self._on_srdf,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._lock = threading.Lock()
        cb = ReentrantCallbackGroup()
        self._move_client = ActionClient(self, MoveGroup, p('move_group_action'), callback_group=cb)
        self._execute_client = ActionClient(
            self, ExecuteTrajectory, p('execute_trajectory_action'), callback_group=cb)
        self._cartesian_client = self.create_client(
            GetCartesianPath, p('compute_cartesian_path_service'), callback_group=cb)
        self._action_server = ActionServer(
            self, MoveArm, 'move_arm',
            execute_callback=self.execute_callback,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=cb)
        self.get_logger().info('MoveArm server ready')

    def _on_srdf(self, msg):
        try:
            self._states = parse_group_states(msg.data)
            self.get_logger().info(f'SRDF group states: {", ".join(sorted(self._states))}')
        except ET.ParseError as error:
            self.get_logger().error(f'Cannot parse SRDF: {error}')

    def execute_callback(self, goal_handle):
        if not self._lock.acquire(blocking=False):
            return self._finish(goal_handle, 'Another MoveArm goal is running')
        try:
            request = goal_handle.request
            if request.named_target:
                self._feedback(goal_handle, f'moving to {request.named_target}')
                self._move_named(goal_handle, request.named_target)
            else:
                self._feedback(goal_handle, f'moving {request.linear_distance:.3f} m linear')
                self._move_linear(goal_handle, request.linear_distance)
        except _Failure as failure:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = MoveArm.Result()
                result.error_code = MoveArm.Result.FAILURE
                result.error_str = 'MoveArm canceled'
                return result
            return self._finish(goal_handle, str(failure))
        finally:
            self._lock.release()
        return self._finish(goal_handle, None)

    # ---- moves --------------------------------------------------------------------------

    def _move_named(self, goal_handle, name):
        if name not in self._states:
            raise _Failure(f'Unknown SRDF group state "{name}" '
                           f'(known: {", ".join(sorted(self._states)) or "none yet"})')
        group, joints = self._states[name]
        constraints = Constraints()
        for joint_name, position in joints.items():
            joint = JointConstraint()
            joint.joint_name = joint_name
            joint.position = position
            joint.tolerance_above = self._p('joint_goal_tolerance')
            joint.tolerance_below = self._p('joint_goal_tolerance')
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        goal = MoveGroup.Goal()
        goal.request.group_name = group
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(constraints)
        goal.request.num_planning_attempts = self._p('planning_attempts')
        goal.request.allowed_planning_time = self._p('planning_time_s')
        goal.request.max_velocity_scaling_factor = self._p('max_velocity_scaling')
        goal.request.max_acceleration_scaling_factor = self._p('max_acceleration_scaling')
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self._run(goal_handle, self._move_client, goal, f'state {name}')

    def _move_linear(self, goal_handle, distance):
        frame = self._p('planning_frame')
        link = self._p('ik_link')
        try:
            tf = self._tf_buffer.lookup_transform(
                frame, link, Time(), timeout=Duration(seconds=self._p('tf_timeout_s')))
        except TransformException as error:
            raise _Failure(f'No transform {frame} -> {link}: {error}')
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = yaw_of(q)
        # The gripper points along the -y axis of link_3 (roll and pitch are zero).
        dx, dy = math.sin(yaw), -math.cos(yaw)
        waypoint = Pose()
        waypoint.position.x = t.x + distance * dx
        waypoint.position.y = t.y + distance * dy
        waypoint.position.z = t.z
        waypoint.orientation.z = math.sin(yaw / 2.0)
        waypoint.orientation.w = math.cos(yaw / 2.0)

        if not self._cartesian_client.wait_for_service(
                timeout_sec=self._p('server_wait_timeout_s')):
            raise _Failure('compute_cartesian_path service not available')
        request = GetCartesianPath.Request()
        request.header.frame_id = frame
        request.start_state.is_diff = True
        request.group_name = self._p('linear_group')
        request.link_name = link
        request.waypoints = [waypoint]
        request.max_step = self._p('linear_max_step')
        request.jump_threshold = 0.0
        request.revolute_jump_threshold = self._p('linear_revolute_jump_threshold')
        request.prismatic_jump_threshold = self._p('linear_prismatic_jump_threshold')
        request.avoid_collisions = self._p('avoid_collisions')
        request.max_velocity_scaling_factor = self._p('linear_velocity_scaling')
        request.max_acceleration_scaling_factor = self._p('linear_acceleration_scaling')
        future = self._cartesian_client.call_async(request)
        self._wait(goal_handle, future,
                   self._p('planning_time_s') + self._p('server_wait_timeout_s'),
                   'planning the linear move')
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS or response.fraction < 0.999:
            raise _Failure(
                f'Linear move of {distance:.3f} m not feasible: only '
                f'{100.0 * response.fraction:.0f}% planned '
                f'(MoveIt error code {response.error_code.val})')
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        self._run(goal_handle, self._execute_client, goal, f'{distance:.3f} m linear')

    # ---- helpers ------------------------------------------------------------------------

    def _run(self, goal_handle, client, goal, what):
        if not client.wait_for_server(timeout_sec=self._p('server_wait_timeout_s')):
            raise _Failure(f'MoveIt action server for {what} not available')
        send_future = client.send_goal_async(goal)
        self._wait(goal_handle, send_future, self._p('server_wait_timeout_s'),
                   f'sending the goal for {what}')
        handle = send_future.result()
        if not handle.accepted:
            raise _Failure(f'MoveIt rejected the goal for {what}')
        result_future = handle.get_result_async()
        try:
            self._wait(goal_handle, result_future, self._p('execution_timeout_s'), what)
        except _Failure:
            handle.cancel_goal_async()
            raise
        code = result_future.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            raise _Failure(f'Moving to {what} failed (MoveIt error code {code})')
        self.get_logger().info(f'Reached {what}')

    def _wait(self, goal_handle, future, timeout, what):
        deadline = time.monotonic() + timeout
        while not future.done():
            if goal_handle.is_cancel_requested:
                raise _Failure('canceled')
            if time.monotonic() > deadline:
                raise _Failure(f'Timed out {what}')
            time.sleep(0.01)

    def _feedback(self, goal_handle, state):
        feedback = MoveArm.Feedback()
        feedback.state = state
        goal_handle.publish_feedback(feedback)
        self.get_logger().info(state)

    def _finish(self, goal_handle, error_str):
        result = MoveArm.Result()
        if error_str is None:
            result.error_code = MoveArm.Result.SUCCESS
            goal_handle.succeed()
        else:
            result.error_code = MoveArm.Result.FAILURE
            result.error_str = error_str
            self.get_logger().error(error_str)
            goal_handle.abort()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MoveArmServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
