"""Action server that slowly drives the robot straight forward.

The robot drives forward along its heading at the start of the goal by the
requested distance (0.0 = default 0.15 m), holding that heading with a P
controller. Progress is measured via TF (map -> arips_wheel_center).
"""

import math
import threading
import time

import rclpy
from arips_action_msgs.action import DriveForward
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class _Failure(Exception):
    """Raised inside the execute callback to abort the goal."""


class _Canceled(Exception):
    """Raised inside the execute callback when the goal was canceled."""



def _ramped_speed(max_speed, min_speed, ramp_gain, remaining):
    """Speed for the remaining distance/angle: max_speed far away, decelerating
    linearly (ramp_gain * remaining) towards the end, but not below min_speed."""
    return max(min_speed, min(max_speed, ramp_gain * max(0.0, remaining)))


class DriveForwardServer(Node):
    def __init__(self):
        super().__init__('drive_forward_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('base_frame', 'arips_wheel_center'),
                ('default_distance', 0.15),
                ('linear_speed', 0.2),
                # Near the end the speed ramps down (ramp_gain * remaining distance) to
                # min_linear_speed, so the faster speed does not overshoot the distance.
                ('min_linear_speed', 0.04),
                ('ramp_gain', 1.5),
                ('max_angular_speed', 0.2),
                ('heading_p_factor', 2.0),
                ('timeout_factor', 3.0),
                ('tf_wait_timeout_s', 5.0),
                ('control_rate_hz', 20.0),
                ('enable_stamped_cmd_vel', True),
                # Input of the Nav2 velocity smoother (shared with controller_server and
                # behavior_server), so commands do not compete with its output on /cmd_vel.
                ('cmd_vel_topic', '/cmd_vel_nav'),
            ],
        )
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._map_frame = p('map_frame')
        self._base_frame = p('base_frame')
        self._default_distance = abs(p('default_distance'))
        self._linear_speed = abs(p('linear_speed'))
        self._max_angular_speed = abs(p('max_angular_speed'))
        self._min_linear_speed = abs(p('min_linear_speed'))
        self._ramp_gain = p('ramp_gain')
        self._heading_p = p('heading_p_factor')
        self._timeout_factor = p('timeout_factor')
        self._tf_wait_timeout = p('tf_wait_timeout_s')
        control_rate_hz = p('control_rate_hz')
        self._use_twist_stamped = p('enable_stamped_cmd_vel')

        if control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be greater than zero')
        if self._linear_speed <= 0.0:
            raise ValueError('linear_speed must be non-zero')
        self._control_period = 1.0 / control_rate_hz

        # Only one goal at a time.
        self._busy_lock = threading.Lock()

        callback_group = ReentrantCallbackGroup()
        command_type = TwistStamped if self._use_twist_stamped else Twist
        self._cmd_vel_publisher = self.create_publisher(command_type, p('cmd_vel_topic'), 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._action_server = ActionServer(
            self,
            DriveForward,
            'drive_forward',
            execute_callback=self.execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=callback_group,
        )
        self.get_logger().info('drive_forward action server ready')

    def _goal_callback(self, _request):
        if self._busy_lock.locked():
            self.get_logger().warn('Rejecting goal: another goal is running')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def execute_callback(self, goal_handle):
        result = DriveForward.Result()
        with self._busy_lock:
            try:
                self._execute(goal_handle, result)
            except _Canceled:
                self._publish_stop()
                result.error_code = DriveForward.Result.FAILURE
                result.error_str = 'DriveForward canceled'
                goal_handle.canceled()
                self.get_logger().info(result.error_str)
                return result
            except _Failure as error:
                self._publish_stop()
                result.error_code = DriveForward.Result.FAILURE
                result.error_str = str(error)
                goal_handle.abort()
                self.get_logger().error(result.error_str)
                return result
            except Exception:
                self._publish_stop()
                raise
        result.error_code = DriveForward.Result.SUCCESS
        goal_handle.succeed()
        return result

    def _execute(self, goal_handle, result):
        distance = goal_handle.request.distance
        distance = abs(distance) if distance != 0.0 else self._default_distance
        self.get_logger().info(f'DriveForward: driving {distance:.3f} m')

        start_x, start_y, start_yaw = self._wait_for_pose(goal_handle)
        expected = distance / self._linear_speed
        deadline = time.monotonic() + self._timeout_factor * expected + 5.0
        while True:
            self._check_cancel(goal_handle)
            if time.monotonic() > deadline:
                raise _Failure('Timeout while driving forward')
            x, y, yaw = self._robot_pose()
            # Progress along the start heading.
            driven = (math.cos(start_yaw) * (x - start_x)
                      + math.sin(start_yaw) * (y - start_y))
            remaining = distance - driven
            self._feedback(goal_handle, remaining)
            if remaining <= 0.0:
                break
            omega = self._heading_p * _normalize_angle(start_yaw - yaw)
            omega = max(-self._max_angular_speed, min(self._max_angular_speed, omega))
            self._publish_command(
                _ramped_speed(self._linear_speed, self._min_linear_speed,
                              self._ramp_gain, remaining),
                omega)
            time.sleep(self._control_period)

        self._publish_stop()
        x, y, _ = self._robot_pose()
        result.driven_distance = math.hypot(x - start_x, y - start_y)
        self.get_logger().info(
            f'DriveForward done: drove {result.driven_distance:.3f} m')

    # --------------------------------------------------------------- helpers

    def _wait_for_pose(self, goal_handle):
        deadline = time.monotonic() + self._tf_wait_timeout
        while True:
            self._check_cancel(goal_handle)
            try:
                return self._lookup_pose()
            except TransformException as error:
                if time.monotonic() > deadline:
                    raise _Failure(f'TF lookup failed: {error}') from error
            time.sleep(0.1)

    def _robot_pose(self):
        try:
            return self._lookup_pose()
        except TransformException as error:
            raise _Failure(f'TF lookup failed: {error}') from error

    def _lookup_pose(self):
        transform = self._tf_buffer.lookup_transform(
            self._map_frame, self._base_frame, Time()).transform
        return (transform.translation.x, transform.translation.y,
                _yaw_from_quaternion(transform.rotation))

    def _check_cancel(self, goal_handle):
        if goal_handle.is_cancel_requested:
            raise _Canceled()

    def _feedback(self, goal_handle, remaining):
        feedback = DriveForward.Feedback()
        feedback.remaining = float(max(0.0, remaining))
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
    node = DriveForwardServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
