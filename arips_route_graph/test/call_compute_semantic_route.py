#!/usr/bin/env python3

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

    def call_service(self):
        request = ComputeSemanticRoute.Request()
        request.start_pose = PoseStamped()
        request.start_pose.header.stamp.sec = 1790330280
        request.start_pose.header.stamp.nanosec = 29102212
        request.start_pose.header.frame_id = 'map'
        request.start_pose.pose.position.x = -1.3017265796661377
        request.start_pose.pose.position.y = 0.5074736475944519
        request.start_pose.pose.orientation.z = 0.623683217412804
        request.start_pose.pose.orientation.w = 0.7816771995636134

        request.goal_pose = PoseStamped()
        request.goal_pose.header.stamp.sec = 1790330286
        request.goal_pose.header.stamp.nanosec = 299537245
        request.goal_pose.header.frame_id = 'map'
        request.goal_pose.pose.position.x = 1.7406687
        request.goal_pose.pose.position.y = 0.80640
        request.goal_pose.pose.orientation.z = -0.4796273895693235
        request.goal_pose.pose.orientation.w = 0.8774722600600638

        self.get_logger().info('Waiting for /compute_semantic_route service')
        self.client.wait_for_service()
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is not None:
            print(response.error_code)
            for segment in response.semantic_route.segments:
                print(segment.metadata_json)


def main(args=None):
    rclpy.init(args=args)
    client = ComputeSemanticRouteClient()
    try:
        client.call_service()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
