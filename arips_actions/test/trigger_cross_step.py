#!/usr/bin/env python3

import math

import rclpy
from arips_action_msgs.action import CrossDoorStep
from arips_semantic_map_msgs.msg import SemanticMap
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.task import Future
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class CrossStepTrigger(Node):
    def __init__(self):
        super().__init__('trigger_cross_step')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'arips_wheel_center')
        self._map_frame = self.get_parameter('map_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._semantic_map = None
        self._goal_sent = False
        self.completion_future = Future()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._map_subscription = self.create_subscription(
            SemanticMap,
            '/semantic_map',
            self._map_callback,
            map_qos,
        )
        self._action_client = ActionClient(
            self,
            CrossDoorStep,
            '/cross_door_step',
        )
        self._trigger_timer = self.create_timer(0.1, self._try_send_goal)
        self.get_logger().info('Waiting for semantic map and pose')

    def _map_callback(self, semantic_map):
        self.get_logger().info('Received semantic map update')
        self._semantic_map = semantic_map

    def _try_send_goal(self):
        if self._goal_sent:
            return
        if self._semantic_map is None:
            self.get_logger().info('Waiting for semantic map...')
            return
        if not self._semantic_map.doors:
            self.get_logger().info('No doors found in semantic map...')
            return
        # if not self._action_client.server_is_ready():
        #     self.get_logger().info('Waiting for action server to be ready...')
        #     return

        map_frame = self._semantic_map.header.frame_id or self._map_frame
        try:
            transform = self._tf_buffer.lookup_transform(
                map_frame,
                self._base_frame,
                Time(),
            ).transform
        except TransformException:
            self.get_logger().info(f'Waiting for transform between map and {self._base_frame}..')
            return

        robot_x = transform.translation.x
        robot_y = transform.translation.y
        selected_door = min(
            self._semantic_map.doors,
            key=lambda door: self._midpoint_distance_squared(
                door, robot_x, robot_y),
        )
        midpoint_x = (selected_door.pivot.x + selected_door.extent.x) / 2.0
        midpoint_y = (selected_door.pivot.y + selected_door.extent.y) / 2.0
        distance = math.hypot(midpoint_x - robot_x, midpoint_y - robot_y)

        goal = CrossDoorStep.Goal()
        goal.door = selected_door
        self._goal_sent = True
        self.get_logger().info(
            f'Sending CrossDoorStep for nearest door '
            f'({distance:.2f} m from robot)')
        goal_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback,
        )
        goal_future.add_done_callback(self._goal_response_callback)

    @staticmethod
    def _midpoint_distance_squared(door, robot_x, robot_y):
        midpoint_x = (door.pivot.x + door.extent.x) / 2.0
        midpoint_y = (door.pivot.y + door.extent.y) / 2.0
        return (midpoint_x - robot_x) ** 2 + (midpoint_y - robot_y) ** 2

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f'Failed to send CrossDoorStep: {error}')
            self.completion_future.set_result(None)
            return

        if not goal_handle.accepted:
            self.get_logger().error('CrossDoorStep goal was rejected')
            self.completion_future.set_result(None)
            return

        self.get_logger().info('CrossDoorStep goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback_message):
        distance_left = feedback_message.feedback.distance_left
        self.get_logger().info(f'Distance left: {distance_left:.3f} m')

    def _result_callback(self, future):
        try:
            response = future.result()
            result = response.result
            self.get_logger().info(
                f'CrossDoorStep completed (status={response.status}, '
                f'error_code={result.error_code}): {result.error_str}')
        except Exception as error:
            self.get_logger().error(f'CrossDoorStep failed: {error}')
        self.completion_future.set_result(None)


def main(args=None):
    rclpy.init(args=args)
    node = CrossStepTrigger()
    try:
        rclpy.spin_until_future_complete(node, node.completion_future)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()