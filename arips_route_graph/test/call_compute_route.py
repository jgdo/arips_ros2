#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputeRoute
from rclpy.action import ActionClient
from rclpy.node import Node




class ComputeRouteClient(Node):
    def __init__(self):
        super().__init__('compute_route_client')
        self.action_client = ActionClient(self, ComputeRoute, '/compute_route')

    def send_goal(self):
        goal = ComputeRoute.Goal()
        goal.start = PoseStamped()
        goal.start.header.stamp.sec = 1790330280
        goal.start.header.stamp.nanosec = 29102212
        goal.start.header.frame_id = 'map'
        goal.start.pose.position.x = -1.3017265796661377
        goal.start.pose.position.y = 0.5074736475944519
        goal.start.pose.orientation.z = 0.623683217412804
        goal.start.pose.orientation.w = 0.7816771995636134

        goal.goal = PoseStamped()
        goal.goal.header.stamp.sec = 1790330286
        goal.goal.header.stamp.nanosec = 299537245
        goal.goal.header.frame_id = 'map'

        # bedroom
        goal.goal.pose.position.x = 1.7406687
        goal.goal.pose.position.y = 0.80640
        goal.goal.pose.orientation.z = -0.4796273895693235
        goal.goal.pose.orientation.w = 0.8774722600600638

        # office
        # goal.goal.pose.position.x = -1.35153961
        # goal.goal.pose.position.y = 0.76733
        # goal.goal.pose.orientation.z = -0.4796273895693235
        # goal.goal.pose.orientation.w = 0.8774722600600638

        goal.use_poses = True
        goal.use_start = True

        self.get_logger().info('Waiting for /compute_route action server')
        self.action_client.wait_for_server()
        goal_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('ComputeRoute goal was rejected')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        print(result)


def main(args=None):
    rclpy.init(args=args)
    client = ComputeRouteClient()
    try:
        client.send_goal()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
