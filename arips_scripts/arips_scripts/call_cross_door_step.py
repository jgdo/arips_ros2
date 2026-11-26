#!/usr/bin/env python3
"""Call /cross_door_step for one door from /semantic_map.

Usage: ros2 run arips_scripts call_cross_door_step <door_index>
"""

import argparse
import sys

import rclpy
from action_msgs.msg import GoalStatus
from arips_action_msgs.action import CrossDoorStep
from arips_semantic_map_msgs.msg import SemanticMap
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.task import Future


class CallCrossDoorStep(Node):
    def __init__(self, door_index):
        super().__init__('call_cross_door_step')
        self._door_index = door_index
        self._semantic_map = None
        self._goal_sent = False
        self.completion_future = Future()

        self.create_subscription(
            SemanticMap,
            '/semantic_map',
            self._map_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._action_client = ActionClient(self, CrossDoorStep, '/cross_door_step')
        self.create_timer(0.1, self._try_send_goal)
        self.get_logger().info(
            f'Waiting for semantic map to call CrossDoorStep for door {door_index}')

    def _map_callback(self, semantic_map):
        self._semantic_map = semantic_map

    def _try_send_goal(self):
        if self._goal_sent:
            return
        if self._semantic_map is None:
            self.get_logger().info('Waiting for semantic map...', once=True)
            return

        doors = self._semantic_map.doors
        if not 0 <= self._door_index < len(doors):
            self.get_logger().error(
                f'Door index {self._door_index} out of range '
                f'(map has {len(doors)} doors)')
            self.completion_future.set_result(1)
            self._goal_sent = True
            return

        if not self._action_client.server_is_ready():
            self.get_logger().info(
                'Waiting for /cross_door_step action server...', once=True)
            return

        door = doors[self._door_index]
        goal = CrossDoorStep.Goal()
        goal.door = door
        self._goal_sent = True
        self.get_logger().info(
            f'Sending CrossDoorStep for door {self._door_index} '
            f'pivot=({door.pivot.x:.3f}, {door.pivot.y:.3f}) '
            f'extent=({door.extent.x:.3f}, {door.extent.y:.3f})')
        goal_future = self._action_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback,
        )
        goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f'Failed to send CrossDoorStep: {error}')
            self.completion_future.set_result(1)
            return

        if not goal_handle.accepted:
            self.get_logger().error('CrossDoorStep goal was rejected')
            self.completion_future.set_result(1)
            return

        self.get_logger().info('CrossDoorStep goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback_message):
        distance_left = feedback_message.feedback.distance_left
        self.get_logger().info(
            f'Distance left: {distance_left:.3f} m',
            throttle_duration_sec=1.0,
        )

    def _result_callback(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f'CrossDoorStep failed: {error}')
            self.completion_future.set_result(1)
            return

        result = response.result
        self.get_logger().info(
            f'CrossDoorStep finished (status={response.status}, '
            f'error_code={result.error_code}): {result.error_str}')
        success = (
            response.status == GoalStatus.STATUS_SUCCEEDED
            and result.error_code == CrossDoorStep.Result.SUCCESS
        )
        self.completion_future.set_result(0 if success else 1)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            'Call /cross_door_step with the door at the given index '
            'from /semantic_map.'
        ),
    )
    parser.add_argument(
        'door_index',
        type=int,
        help='index into SemanticMap.doors',
    )
    return parser.parse_known_args(argv)


def main(args=None):
    argv = sys.argv[1:] if args is None else args
    parsed, ros_args = parse_args(argv)
    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = CallCrossDoorStep(parsed.door_index)
    exit_code = 1
    try:
        rclpy.spin_until_future_complete(node, node.completion_future)
        if node.completion_future.done():
            exit_code = node.completion_future.result()
    except KeyboardInterrupt:
        pass
    if exit_code:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
