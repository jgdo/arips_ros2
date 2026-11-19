"""Action server that drives a diff-drive robot backwards on an arc around a pivot.

The robot's y axis is kept pointing towards the pivot (pivot on the left) or
away from it (pivot on the right), and the distance to the pivot measured at
the start is maintained. Both are controlled with simple P controllers:

* the distance error is converted to a small bearing offset (a diff-drive
  robot can only change its distance to the pivot by tilting its heading),
* the bearing error (incl. that offset) is fed back on the angular velocity
  on top of the feed-forward term v / r that keeps the robot on the circle.
"""

import math
import time

import rclpy
from arips_action_msgs.action import DriveAroundPivot
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.action import ActionServer
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


class DriveAroundPivotServer(Node):
    def __init__(self):
        super().__init__('drive_around_pivot_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('base_frame', 'arips_wheel_center'),
                # 70 deg ran the robot's rear into the wall at the end of the arc in the lu world.
                ('default_angle_deg', 66.0),
                ('linear_speed', 0.1),
                ('bearing_p_factor', 2.0),
                ('distance_p_factor', 3.0),
                ('max_distance_correction_deg', 20.0),
                ('max_angular_speed', 0.5),
                ('initial_align_threshold_deg', 2.0),
                ('min_align_angular_speed', 0.1),
                ('min_pivot_distance', 0.1),
                ('timeout_factor', 3.0),
                ('control_rate_hz', 20.0),
                ('enable_stamped_cmd_vel', True),
            ],
        )
        p = self.get_parameter
        self._map_frame = p('map_frame').value
        self._base_frame = p('base_frame').value
        self._default_angle = math.radians(p('default_angle_deg').value)
        self._linear_speed = abs(p('linear_speed').value)
        self._bearing_p = p('bearing_p_factor').value
        self._distance_p = p('distance_p_factor').value
        self._max_distance_correction = math.radians(
            p('max_distance_correction_deg').value)
        self._max_angular_speed = p('max_angular_speed').value
        self._align_threshold = math.radians(
            p('initial_align_threshold_deg').value)
        self._min_align_angular_speed = p('min_align_angular_speed').value
        self._min_pivot_distance = p('min_pivot_distance').value
        self._timeout_factor = p('timeout_factor').value
        control_rate_hz = p('control_rate_hz').value
        self._use_twist_stamped = p('enable_stamped_cmd_vel').value

        if control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be greater than zero')
        if self._linear_speed <= 0.0:
            raise ValueError('linear_speed must be non-zero')
        self._control_period = 1.0 / control_rate_hz

        command_type = TwistStamped if self._use_twist_stamped else Twist
        self._cmd_vel_publisher = self.create_publisher(
            command_type, '/cmd_vel', 10)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._action_server = ActionServer(
            self,
            DriveAroundPivot,
            'drive_around_pivot',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info('drive_around_pivot action server ready')

    # ------------------------------------------------------------------ main

    def execute_callback(self, goal_handle):
        request = goal_handle.request
        result = DriveAroundPivot.Result()
        self._publish_stop()

        target_angle = (math.radians(abs(request.angle_deg))
                        if request.angle_deg != 0.0 else self._default_angle)

        # Pivot into the (fixed) map frame once, so it stays put while we move.
        try:
            pivot_x, pivot_y = self._pivot_in_map(request.pivot)
            px, py, _, _ = self._pivot_in_base(pivot_x, pivot_y)
        except TransformException as error:
            return self._abort(goal_handle, result, f'TF lookup failed: {error}')

        start_distance = math.hypot(px, py)
        result.start_distance = start_distance
        if start_distance < self._min_pivot_distance:
            return self._abort(
                goal_handle, result,
                f'Pivot too close to robot ({start_distance:.3f} m)')

        # +1: pivot on the left -> +y points towards the pivot.
        # -1: pivot on the right -> +y points away from the pivot.
        side = 1.0 if py >= 0.0 else -1.0
        desired_bearing = side * math.pi / 2.0
        velocity = -self._linear_speed  # backwards
        self.get_logger().info(
            f'DriveAroundPivot: pivot ({pivot_x:.3f}, {pivot_y:.3f}) in '
            f'{self._map_frame}, distance {start_distance:.3f} m, pivot on the '
            f'{"left" if side > 0 else "right"}, '
            f'arc {math.degrees(target_angle):.1f} deg')

        # Phase 1: turn in place until the y axis is aligned with the pivot.
        while True:
            if goal_handle.is_cancel_requested:
                return self._cancel(goal_handle, result)
            try:
                _, _, _, bearing = self._pivot_in_base(pivot_x, pivot_y)
            except TransformException as error:
                return self._abort(
                    goal_handle, result, f'TF lookup failed: {error}')
            bearing_error = _normalize_angle(bearing - desired_bearing)
            if abs(bearing_error) < self._align_threshold:
                break
            omega = self._bearing_p * bearing_error
            omega = math.copysign(
                min(self._max_angular_speed,
                    max(self._min_align_angular_speed, abs(omega))),
                omega)
            self._publish_command(0.0, omega)
            time.sleep(self._control_period)

        # Phase 2: drive backwards on the arc.
        try:
            robot_x, robot_y, _ = self._robot_pose()
        except TransformException as error:
            return self._abort(goal_handle, result, f'TF lookup failed: {error}')
        last_polar = math.atan2(robot_y - pivot_y, robot_x - pivot_x)
        travelled = 0.0
        max_distance_error = 0.0
        max_bearing_error = 0.0
        feed_forward = side * velocity / start_distance
        expected_duration = target_angle * start_distance / self._linear_speed
        deadline = time.monotonic() + self._timeout_factor * expected_duration + 5.0

        while True:
            if goal_handle.is_cancel_requested:
                return self._cancel(goal_handle, result)
            if time.monotonic() > deadline:
                return self._abort(
                    goal_handle, result,
                    f'Timeout after travelling {math.degrees(travelled):.1f} deg')
            try:
                robot_x, robot_y, _ = self._robot_pose()
                px, py, distance, bearing = self._pivot_in_base(
                    pivot_x, pivot_y)
            except TransformException as error:
                return self._abort(
                    goal_handle, result, f'TF lookup failed: {error}')

            polar = math.atan2(robot_y - pivot_y, robot_x - pivot_x)
            travelled += _normalize_angle(polar - last_polar)
            last_polar = polar

            distance_error = distance - start_distance
            bearing_error = _normalize_angle(bearing - desired_bearing)
            max_distance_error = max(max_distance_error, abs(distance_error))
            max_bearing_error = max(max_bearing_error, abs(bearing_error))

            feedback = DriveAroundPivot.Feedback()
            feedback.travelled_angle_deg = math.degrees(abs(travelled))
            feedback.distance_error = distance_error
            feedback.bearing_error_deg = math.degrees(bearing_error)
            goal_handle.publish_feedback(feedback)

            if abs(travelled) >= target_angle:
                self._publish_stop()
                result.error_code = DriveAroundPivot.Result.SUCCESS
                result.final_distance = distance
                result.travelled_angle_deg = math.degrees(abs(travelled))
                result.max_distance_error = max_distance_error
                result.max_bearing_error_deg = math.degrees(max_bearing_error)
                goal_handle.succeed()
                self.get_logger().info(
                    f'DriveAroundPivot done: travelled '
                    f'{result.travelled_angle_deg:.1f} deg, distance '
                    f'{distance:.3f} m (start {start_distance:.3f} m), '
                    f'max distance error {max_distance_error:.3f} m, '
                    f'max bearing error {result.max_bearing_error_deg:.2f} deg')
                return result

            # Distance P: tilt the heading so the radial velocity
            # (dr/dt = v * side * bearing_offset) drives the error to zero.
            correction = -self._distance_p * distance_error * side * math.copysign(
                1.0, velocity)
            correction = max(-self._max_distance_correction,
                             min(self._max_distance_correction, correction))
            heading_error = _normalize_angle(
                bearing - (desired_bearing + correction))

            # Rotating counter-clockwise decreases the pivot's bearing.
            omega = feed_forward + self._bearing_p * heading_error
            omega = max(-self._max_angular_speed,
                        min(self._max_angular_speed, omega))
            self._publish_command(velocity, omega)
            time.sleep(self._control_period)

    # --------------------------------------------------------------- helpers

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

    def _abort(self, goal_handle, result, error_str):
        self._publish_stop()
        result.error_code = DriveAroundPivot.Result.FAILURE
        result.error_str = error_str
        goal_handle.abort()
        self.get_logger().error(error_str)
        return result

    def _cancel(self, goal_handle, result):
        self._publish_stop()
        result.error_code = DriveAroundPivot.Result.FAILURE
        result.error_str = 'DriveAroundPivot canceled'
        goal_handle.canceled()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = DriveAroundPivotServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
