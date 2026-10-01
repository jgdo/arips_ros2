#!/usr/bin/env python3

from copy import deepcopy

from arips_semantic_map_msgs.srv import ComputeSemanticRoute
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node


class ComputeSemanticRouteClient(Node):

    def __init__(self):
        super().__init__('compute_semantic_route_client')
        self.client = self.create_client(
            ComputeSemanticRoute, '/compute_semantic_route'
        )
        self.previous_goal_pose = None
        self.create_subscription(
            PoseStamped, '/test_goal', self._goal_callback, 10
        )

    def _goal_callback(self, goal_pose):
        if self.previous_goal_pose is None:
            self.previous_goal_pose = deepcopy(goal_pose)
            self.get_logger().info(
                'Received first goal; waiting for a second pose before planning'
            )
            return

        request = ComputeSemanticRoute.Request()
        request.start_pose = deepcopy(self.previous_goal_pose)
        request.goal_pose = deepcopy(goal_pose)
        self.previous_goal_pose = deepcopy(goal_pose)

        future = self.client.call_async(request)
        future.add_done_callback(self._service_response_callback)

    def _service_response_callback(self, future):
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Compute semantic route request failed: {error}'
            )
            return

        self.get_logger().info(
            f'Compute semantic route error code: {response.error_code}'
        )


def main(args=None):
    rclpy.init(args=args)
    client = ComputeSemanticRouteClient()
    try:
        client.client.wait_for_service()
        client.get_logger().info('Waiting for initial two poses')
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()