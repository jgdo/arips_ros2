import math
import time

import rclpy
from arips_action_msgs.action import CrossDoorStep
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class CrossDoorStepServer(Node):
    def __init__(self):
        super().__init__('cross_door_step_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('base_frame', 'arips_wheel_center'),
                ('initial_align_angle_threshold_deg', 5.0),
                ('initial_align_veclocity', 0.5),
                ('alignment_p_factor', 3.0),
                ('door_clearing_distance', 0.4),
                ('control_rate_hz', 20.0),
                ('forward_speed', 0.08),
                ('enable_stamped_cmd_vel', True),
            ],
        )
        self._map_frame = self.get_parameter('map_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._initial_align_angle_threshold_rad = math.radians(
            self.get_parameter('initial_align_angle_threshold_deg').value)
        self._alignment_p_factor = self.get_parameter(
            'alignment_p_factor').value
        self._door_clearing_distance = self.get_parameter(
            'door_clearing_distance').value
        control_rate_hz = self.get_parameter('control_rate_hz').value
        self._forward_speed = self.get_parameter('forward_speed').value
        self._initial_align_velocity = self.get_parameter('initial_align_veclocity').value
        self._use_twist_stamped = self.get_parameter(
            'enable_stamped_cmd_vel').value

        if control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz must be greater than zero')
        if self._door_clearing_distance < 0.0:
            raise ValueError('door_clearing_distance cannot be negative')
        self._control_period = 1.0 / control_rate_hz

        command_type = TwistStamped if self._use_twist_stamped else Twist
        self._cmd_vel_publisher = self.create_publisher(command_type, '/cmd_vel', 10)
        self.get_logger().info(
            f'Publishing {"TwistStamped" if self._use_twist_stamped else "Twist"} '
            'commands on /cmd_vel')
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._action_server = ActionServer(
            self,
            CrossDoorStep,
            'cross_door_step',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing CrossDoorStep goal')

        result = CrossDoorStep.Result()
        self._publish_stop()

        try:
            start_x, start_y, _ = self._current_pose()
        except TransformException as error:
            return self._abort(goal_handle, result, f'TF lookup failed: {error}')

        door = goal_handle.request.door
        segment_x = door.extent.x - door.pivot.x
        segment_y = door.extent.y - door.pivot.y
        segment_length_squared = segment_x * segment_x + segment_y * segment_y
        if segment_length_squared < 1e-12:
            return self._abort(
                goal_handle,
                result,
                'Door pivot and extent coincide',
            )

        projection = (
            (start_x - door.pivot.x) * segment_x
            + (start_y - door.pivot.y) * segment_y
        ) / segment_length_squared
        projection = min(1.0, max(0.0, projection))
        closest_x = door.pivot.x + projection * segment_x
        closest_y = door.pivot.y + projection * segment_y
        direction_x = closest_x - start_x
        direction_y = closest_y - start_y
        distance_to_door = math.hypot(direction_x, direction_y)
        if distance_to_door < 1e-6:
            return self._abort(
                goal_handle,
                result,
                'Current robot position coincides with the closest point on the door',
            )

        direction_x /= distance_to_door
        direction_y /= distance_to_door
        line_yaw = math.atan2(direction_y, direction_x)
        target_x = closest_x + direction_x * self._door_clearing_distance
        target_y = closest_y + direction_y * self._door_clearing_distance

        while True:
            if goal_handle.is_cancel_requested:
                return self._cancel(goal_handle, result)
            try:
                current_x, current_y, current_yaw = self._current_pose()
            except TransformException as error:
                return self._abort(
                    goal_handle, result, f'TF lookup failed: {error}')

            distance_left = self._distance_left(
                current_x, current_y, target_x, target_y,
                direction_x, direction_y,
            )
            self._publish_feedback(goal_handle, distance_left)
            alignment_error = self._normalize_angle(line_yaw - current_yaw)
            self.get_logger().info(f'Alignment error: {alignment_error}')
            if abs(alignment_error) < self._initial_align_angle_threshold_rad:
                break

            command = Twist()
            command.angular.z = math.copysign(self._initial_align_velocity, alignment_error)
            self._publish_command(command)
            time.sleep(self._control_period)

        while True:
            if goal_handle.is_cancel_requested:
                return self._cancel(goal_handle, result)
            try:
                current_x, current_y, current_yaw = self._current_pose()
            except TransformException as error:
                return self._abort(
                    goal_handle, result, f'TF lookup failed: {error}')

            distance_left = self._distance_left(
                current_x, current_y, target_x, target_y,
                direction_x, direction_y,
            )
            self._publish_feedback(goal_handle, distance_left)
            if distance_left <= 0.0:
                self._publish_stop()
                result.error_code = CrossDoorStep.Result.SUCCESS
                goal_handle.succeed()
                self.get_logger().info('Successfully crossed the door.')
                return result

            alignment_error = self._normalize_angle(line_yaw - current_yaw)
            self.get_logger().info(f'Alignment error: {alignment_error}')
            command = Twist()
            command.linear.x = self._forward_speed
            command.angular.z = (
                alignment_error * self._alignment_p_factor)
            self._publish_command(command)
            time.sleep(self._control_period)

    def _current_pose(self):
        transform = self._tf_buffer.lookup_transform(
            self._map_frame,
            self._base_frame,
            Time(),
        ).transform
        translation = transform.translation
        rotation = transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        return translation.x, translation.y, yaw

    @staticmethod
    def _distance_left(
        current_x, current_y, target_x, target_y, direction_x, direction_y,
    ):
        remaining = (
            (target_x - current_x) * direction_x
            + (target_y - current_y) * direction_y
        )
        return max(0.0, remaining)

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _publish_feedback(goal_handle, distance_left):
        feedback = CrossDoorStep.Feedback()
        feedback.distance_left = distance_left
        goal_handle.publish_feedback(feedback)

    def _publish_stop(self):
        self._publish_command(Twist())

    def _publish_command(self, command):
        if self._use_twist_stamped:
            stamped_command = TwistStamped()
            stamped_command.header.stamp = self.get_clock().now().to_msg()
            stamped_command.header.frame_id = self._base_frame
            stamped_command.twist = command
            self._cmd_vel_publisher.publish(stamped_command)
        else:
            self._cmd_vel_publisher.publish(command)

    def _abort(self, goal_handle, result, error_str):
        self._publish_stop()
        result.error_code = CrossDoorStep.Result.FAILURE
        result.error_str = error_str
        goal_handle.abort()
        self.get_logger().error(error_str)
        return result

    def _cancel(self, goal_handle, result):
        self._publish_stop()
        result.error_code = CrossDoorStep.Result.FAILURE
        result.error_str = 'CrossDoorStep canceled'
        goal_handle.canceled()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = CrossDoorStepServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass