"""Action server that moves the arm into a safe pose and retreats from a pivot.

Sequence:

1. Move the MoveIt group ``door_gripper_group`` to the named state
   ``door_gripper_neutral``.
2. Move the MoveIt group ``arm`` to the named state ``door_grasp_folded``.
3. Slowly turn in place until the robot's x axis points away from the pivot.
   The robot turns away from the pivot: pivot on the left -> clockwise,
   pivot on the right -> counter-clockwise.
4. Slowly drive straight forward (default 0.5 m), holding the heading.

Named states are resolved from the SRDF published by move_group on
/robot_description_semantic and sent as joint goals to the MoveGroup action.
"""

import math
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
from arips_action_msgs.action import RetreatFromPivot
from geometry_msgs.msg import Twist, TwistStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _transform_point(transform, x, y, z):
    """Apply a geometry_msgs/Transform to a point."""
    q = transform.rotation
    t = transform.translation
    # v' = v + 2 * q_vec x (q_vec x v + w * v)
    cx = q.y * z - q.z * y + q.w * x
    cy = q.z * x - q.x * z + q.w * y
    cz = q.x * y - q.y * x + q.w * z
    rx = x + 2.0 * (q.y * cz - q.z * cy)
    ry = y + 2.0 * (q.z * cx - q.x * cz)
    rz = z + 2.0 * (q.x * cy - q.y * cx)
    return rx + t.x, ry + t.y, rz + t.z


class _Failure(Exception):
    """Raised inside the execute callback to abort the goal."""


class _Canceled(Exception):
    """Raised inside the execute callback when the goal was canceled."""


class RetreatFromPivotServer(Node):
    def __init__(self):
        super().__init__('retreat_from_pivot_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('base_frame', 'arips_wheel_center'),
                ('gripper_group', 'door_gripper_group'),
                ('gripper_named_state', 'door_gripper_neutral'),
                ('arm_group', 'arm'),
                ('arm_named_state', 'door_grasp_folded'),
                ('move_group_action', '/move_action'),
                ('planning_time_s', 5.0),
                ('planning_attempts', 3),
                ('max_velocity_scaling', 0.5),
                ('max_acceleration_scaling', 0.5),
                ('joint_goal_tolerance', 0.01),
                ('move_timeout_s', 60.0),
                ('server_wait_timeout_s', 10.0),
                ('default_forward_distance', 0.5),
                ('linear_speed', 0.1),
                ('max_angular_speed', 0.2),
                ('min_angular_speed', 0.03),
                ('angular_p_factor', 1.0),
                ('heading_p_factor', 2.0),
                ('rotate_threshold_deg', 1.5),
                ('min_pivot_distance', 0.05),
                ('timeout_factor', 3.0),
                ('control_rate_hz', 20.0),
                ('enable_stamped_cmd_vel', True),
            ],
        )
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._map_frame = p('map_frame')
        self._base_frame = p('base_frame')
        self._gripper_group = p('gripper_group')
        self._gripper_state = p('gripper_named_state')
        self._arm_group = p('arm_group')
        self._arm_state = p('arm_named_state')
        self._planning_time = p('planning_time_s')
        self._planning_attempts = p('planning_attempts')
        self._velocity_scaling = p('max_velocity_scaling')
        self._acceleration_scaling = p('max_acceleration_scaling')
        self._joint_tolerance = p('joint_goal_tolerance')
        self._move_timeout = p('move_timeout_s')
        self._server_wait_timeout = p('server_wait_timeout_s')
        self._default_forward_distance = p('default_forward_distance')
        self._linear_speed = abs(p('linear_speed'))
        self._max_angular_speed = abs(p('max_angular_speed'))
        self._min_angular_speed = abs(p('min_angular_speed'))
        self._angular_p = p('angular_p_factor')
        self._heading_p = p('heading_p_factor')
        self._rotate_threshold = math.radians(p('rotate_threshold_deg'))
        self._min_pivot_distance = p('min_pivot_distance')
        self._timeout_factor = p('timeout_factor')
        control_rate_hz = p('control_rate_hz')
        self._use_twist_stamped = p('enable_stamped_cmd_vel')

        if control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be greater than zero')
        if self._linear_speed <= 0.0 or self._max_angular_speed <= 0.0:
            raise ValueError('linear_speed and max_angular_speed must be non-zero')
        self._control_period = 1.0 / control_rate_hz

        # Only one goal at a time.
        self._busy_lock = threading.Lock()
        self._srdf_lock = threading.Lock()
        self._named_states = None  # (group, state) -> {joint: value}

        callback_group = ReentrantCallbackGroup()
        command_type = TwistStamped if self._use_twist_stamped else Twist
        self._cmd_vel_publisher = self.create_publisher(command_type, '/cmd_vel', 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(
            String, '/robot_description_semantic', self._srdf_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=callback_group)
        self._move_client = ActionClient(
            self, MoveGroup, p('move_group_action'), callback_group=callback_group)
        self._action_server = ActionServer(
            self,
            RetreatFromPivot,
            'retreat_from_pivot',
            execute_callback=self.execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=callback_group,
        )
        self.get_logger().info('retreat_from_pivot action server ready')

    # ------------------------------------------------------------- callbacks

    def _srdf_callback(self, msg):
        states = {}
        try:
            root = ET.fromstring(msg.data)
        except ET.ParseError as error:
            self.get_logger().error(f'Could not parse SRDF: {error}')
            return
        for state in root.iter('group_state'):
            joints = {j.get('name'): float(j.get('value'))
                      for j in state.iter('joint')}
            states[(state.get('group'), state.get('name'))] = joints
        with self._srdf_lock:
            self._named_states = states
        self.get_logger().info(
            f'Loaded {len(states)} named states from SRDF', once=True)

    def _goal_callback(self, _request):
        if self._busy_lock.locked():
            self.get_logger().warn('Rejecting goal: another goal is running')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    # ------------------------------------------------------------------ main

    def execute_callback(self, goal_handle):
        result = RetreatFromPivot.Result()
        with self._busy_lock:
            try:
                self._execute(goal_handle, result)
            except _Canceled:
                self._publish_stop()
                result.error_code = RetreatFromPivot.Result.FAILURE
                result.error_str = 'RetreatFromPivot canceled'
                goal_handle.canceled()
                self.get_logger().info(result.error_str)
                return result
            except _Failure as error:
                self._publish_stop()
                result.error_code = RetreatFromPivot.Result.FAILURE
                result.error_str = str(error)
                goal_handle.abort()
                self.get_logger().error(result.error_str)
                return result
            except Exception:
                self._publish_stop()
                raise
        result.error_code = RetreatFromPivot.Result.SUCCESS
        goal_handle.succeed()
        return result

    def _execute(self, goal_handle, result):
        request = goal_handle.request
        forward_distance = (abs(request.forward_distance)
                            if request.forward_distance != 0.0
                            else self._default_forward_distance)
        self._publish_stop()

        # Pivot into the (fixed) map frame once, so it stays put while we move.
        pivot_x, pivot_y = self._tf(lambda: self._pivot_in_map(request.pivot))
        self.get_logger().info(
            f'RetreatFromPivot: pivot ({pivot_x:.3f}, {pivot_y:.3f}) in '
            f'{self._map_frame}, forward distance {forward_distance:.2f} m')

        # 1 + 2: arm into a safe pose.
        self._move_to_named_state(
            goal_handle, 'gripper_neutral', self._gripper_group, self._gripper_state)
        self._move_to_named_state(
            goal_handle, 'arm_folded', self._arm_group, self._arm_state)

        # 3: turn in place, away from the pivot.
        result.rotated_angle_deg = math.degrees(
            self._rotate_away(goal_handle, pivot_x, pivot_y))

        # 4: drive straight forward.
        result.driven_distance = self._drive_forward(goal_handle, forward_distance)
        _, _, distance, _ = self._tf(lambda: self._pivot_in_base(pivot_x, pivot_y))
        result.final_pivot_distance = distance
        self.get_logger().info(
            f'RetreatFromPivot done: rotated {result.rotated_angle_deg:.1f} deg, '
            f'drove {result.driven_distance:.3f} m, pivot now {distance:.3f} m away')

    # ---------------------------------------------------------------- MoveIt

    def _move_to_named_state(self, goal_handle, phase, group, state):
        self._feedback(goal_handle, phase, 0.0)
        joints = self._resolve_named_state(goal_handle, group, state)
        self.get_logger().info(f'Moving {group} to "{state}": {joints}')

        if not self._move_client.wait_for_server(timeout_sec=self._server_wait_timeout):
            raise _Failure('MoveGroup action server not available')
        send_future = self._move_client.send_goal_async(self._joint_goal(group, joints))
        self._wait_future(goal_handle, send_future, self._server_wait_timeout,
                          f'sending the MoveGroup goal for {group}')
        move_handle = send_future.result()
        if not move_handle.accepted:
            raise _Failure(f'MoveGroup goal for {group} rejected')

        result_future = move_handle.get_result_async()
        try:
            self._wait_future(goal_handle, result_future, self._move_timeout,
                              f'moving {group} to "{state}"')
        except (_Canceled, _Failure):
            move_handle.cancel_goal_async()
            raise
        error_code = result_future.result().result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            raise _Failure(
                f'Moving {group} to "{state}" failed (MoveIt error code {error_code})')

    def _resolve_named_state(self, goal_handle, group, state):
        deadline = time.monotonic() + self._server_wait_timeout
        while True:
            with self._srdf_lock:
                states = self._named_states
            if states is not None:
                break
            self._check_cancel(goal_handle)
            if time.monotonic() > deadline:
                raise _Failure('No SRDF received on /robot_description_semantic')
            time.sleep(0.1)
        if (group, state) not in states:
            raise _Failure(f'Named state "{state}" of group "{group}" not in SRDF')
        return states[(group, state)]

    def _joint_goal(self, group, positions):
        constraints = Constraints()
        for name, position in positions.items():
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = self._joint_tolerance
            joint.tolerance_below = self._joint_tolerance
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)

        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = group
        request.start_state.is_diff = True
        request.goal_constraints.append(constraints)
        request.num_planning_attempts = self._planning_attempts
        request.allowed_planning_time = self._planning_time
        request.max_velocity_scaling_factor = self._velocity_scaling
        request.max_acceleration_scaling_factor = self._acceleration_scaling
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    # ------------------------------------------------------------ base motion

    def _rotate_away(self, goal_handle, pivot_x, pivot_y):
        """Turn in place until the pivot is straight behind the robot.

        Returns the signed rotated angle [rad] (+ = counter-clockwise).
        """
        px, py, distance, bearing = self._tf(
            lambda: self._pivot_in_base(pivot_x, pivot_y))
        if distance < self._min_pivot_distance:
            raise _Failure(f'Pivot too close to robot ({distance:.3f} m)')

        # Turn away from the pivot: pivot on the left (py > 0) -> clockwise.
        side = 1.0 if py >= 0.0 else -1.0
        start_yaw = self._tf(self._robot_pose)[2]
        self.get_logger().info(
            f'Pivot at ({px:.3f}, {py:.3f}) in {self._base_frame} '
            f'(bearing {math.degrees(bearing):.1f} deg): turning '
            f'{"clockwise" if side > 0 else "counter-clockwise"} by '
            f'{180.0 - abs(math.degrees(bearing)):.1f} deg')

        expected = (math.pi - abs(bearing)) / self._max_angular_speed
        deadline = time.monotonic() + self._timeout_factor * expected + 5.0
        rotated = 0.0
        last_yaw = start_yaw
        while True:
            self._check_cancel(goal_handle)
            if time.monotonic() > deadline:
                raise _Failure(
                    f'Timeout while rotating (rotated {math.degrees(rotated):.1f} deg)')
            _, _, _, bearing = self._tf(lambda: self._pivot_in_base(pivot_x, pivot_y))
            yaw = self._tf(self._robot_pose)[2]
            rotated += _normalize_angle(yaw - last_yaw)
            last_yaw = yaw

            # Target: pivot bearing == +-pi. Turning ccw decreases the bearing,
            # so omega = k * error drives the error to zero.
            error = _normalize_angle(bearing - math.pi)
            if abs(error) > math.pi - 0.1:
                # Pivot (almost) straight ahead: +-pi is ambiguous, use the
                # direction decided at the start.
                error = -side * abs(error)
            self._feedback(goal_handle, 'rotate', math.degrees(abs(error)))
            if abs(error) < self._rotate_threshold:
                break
            omega = self._angular_p * error
            omega = math.copysign(
                min(self._max_angular_speed, max(self._min_angular_speed, abs(omega))),
                omega)
            self._publish_command(0.0, omega)
            time.sleep(self._control_period)

        self._publish_stop()
        return rotated

    def _drive_forward(self, goal_handle, distance):
        """Drive straight forward by distance, holding the current heading."""
        start_x, start_y, start_yaw = self._tf(self._robot_pose)
        expected = distance / self._linear_speed
        deadline = time.monotonic() + self._timeout_factor * expected + 5.0
        while True:
            self._check_cancel(goal_handle)
            if time.monotonic() > deadline:
                raise _Failure('Timeout while driving forward')
            x, y, yaw = self._tf(self._robot_pose)
            # Progress along the start heading.
            driven = (math.cos(start_yaw) * (x - start_x)
                      + math.sin(start_yaw) * (y - start_y))
            remaining = distance - driven
            self._feedback(goal_handle, 'drive', remaining)
            if remaining <= 0.0:
                break
            omega = self._heading_p * _normalize_angle(start_yaw - yaw)
            omega = max(-self._max_angular_speed, min(self._max_angular_speed, omega))
            self._publish_command(self._linear_speed, omega)
            time.sleep(self._control_period)

        self._publish_stop()
        x, y, _ = self._tf(self._robot_pose)
        return math.hypot(x - start_x, y - start_y)

    # --------------------------------------------------------------- helpers

    def _tf(self, func):
        try:
            return func()
        except TransformException as error:
            raise _Failure(f'TF lookup failed: {error}') from error

    def _pivot_in_map(self, pivot):
        frame = pivot.header.frame_id or self._map_frame
        position = pivot.pose.position
        if frame == self._map_frame:
            return position.x, position.y
        transform = self._tf_buffer.lookup_transform(
            self._map_frame, frame, Time()).transform
        x, y, _ = _transform_point(transform, position.x, position.y, position.z)
        return x, y

    def _robot_pose(self):
        transform = self._tf_buffer.lookup_transform(
            self._map_frame, self._base_frame, Time()).transform
        return (transform.translation.x, transform.translation.y,
                _yaw_from_quaternion(transform.rotation))

    def _pivot_in_base(self, pivot_x, pivot_y):
        """Return pivot (x, y, distance, bearing) in the robot (self) frame."""
        robot_x, robot_y, yaw = self._robot_pose()
        dx = pivot_x - robot_x
        dy = pivot_y - robot_y
        px = math.cos(yaw) * dx + math.sin(yaw) * dy
        py = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return px, py, math.hypot(px, py), math.atan2(py, px)

    def _check_cancel(self, goal_handle):
        if goal_handle.is_cancel_requested:
            raise _Canceled()

    def _wait_future(self, goal_handle, future, timeout, what):
        deadline = time.monotonic() + timeout
        while not future.done():
            self._check_cancel(goal_handle)
            if time.monotonic() > deadline:
                raise _Failure(f'Timeout while {what}')
            time.sleep(0.05)

    def _feedback(self, goal_handle, phase, remaining):
        feedback = RetreatFromPivot.Feedback()
        feedback.phase = phase
        feedback.remaining = float(remaining)
        goal_handle.publish_feedback(feedback)

    def _publish_stop(self):
        self._publish_command(0.0, 0.0)

    def _publish_command(self, linear, angular):
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        if self._use_twist_stamped:
            stamped = TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.header.frame_id = self._base_frame
            stamped.twist = command
            self._cmd_vel_publisher.publish(stamped)
        else:
            self._cmd_vel_publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = RetreatFromPivotServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
