#!/usr/bin/env python3
"""Send a DriveAroundPivot goal for the pivot of a door from /semantic_map.

Independently of the server, it samples TF at 20 Hz while the goal runs and
prints the start/end distance, the travelled angle around the pivot, and the
maximum distance / bearing errors.

Usage: python3 trigger_drive_around_pivot.py --ros-args -p door_index:=0
"""

import math

import rclpy
from arips_action_msgs.action import DriveAroundPivot
from arips_semantic_map_msgs.msg import SemanticMap
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.task import Future
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class DriveAroundPivotTrigger(Node):
    def __init__(self):
        super().__init__('trigger_drive_around_pivot')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'arips_wheel_center')
        self.declare_parameter('door_index', 0)
        self.declare_parameter('angle_deg', 0.0)
        self._map_frame = self.get_parameter('map_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._door_index = self.get_parameter('door_index').value
        self._angle_deg = self.get_parameter('angle_deg').value

        self._semantic_map = None
        self._goal_sent = False
        self._pivot = None
        self._side = None
        self._start_distance = None
        self._start_polar = None
        self._samples = []
        self.completion_future = Future()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.create_subscription(
            SemanticMap, '/semantic_map', self._map_callback,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._client = ActionClient(
            self, DriveAroundPivot, '/drive_around_pivot')
        self.create_timer(0.1, self._try_send_goal)
        self.create_timer(0.05, self._sample)

    def _map_callback(self, msg):
        self._semantic_map = msg

    def _measure(self):
        """Return (distance, bearing, polar angle) of the robot w.r.t. pivot."""
        t = self._tf_buffer.lookup_transform(
            self._map_frame, self._base_frame, Time()).transform
        q = t.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        dx = self._pivot[0] - t.translation.x
        dy = self._pivot[1] - t.translation.y
        bearing = math.atan2(dy, dx) - yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        polar = math.atan2(-dy, -dx)
        return math.hypot(dx, dy), bearing, polar

    def _try_send_goal(self):
        if self._goal_sent or self._semantic_map is None:
            return
        doors = self._semantic_map.doors
        if self._door_index >= len(doors):
            self.get_logger().error(
                f'Door {self._door_index} not in map ({len(doors)} doors)')
            self.completion_future.set_result(None)
            self._goal_sent = True
            return
        if not self._client.server_is_ready():
            self.get_logger().info('Waiting for action server...', once=True)
            return
        door = doors[self._door_index]
        self._pivot = (door.pivot.x, door.pivot.y)
        try:
            distance, bearing, polar = self._measure()
        except TransformException:
            self.get_logger().info('Waiting for TF...', once=True)
            return
        self._start_distance = distance
        self._start_polar = polar
        self._side = 1.0 if bearing >= 0.0 else -1.0

        goal = DriveAroundPivot.Goal()
        goal.pivot = PoseStamped()
        goal.pivot.header.frame_id = self._semantic_map.header.frame_id or self._map_frame
        goal.pivot.pose.position.x = door.pivot.x
        goal.pivot.pose.position.y = door.pivot.y
        goal.pivot.pose.orientation.w = 1.0
        goal.angle_deg = self._angle_deg
        self._goal_sent = True
        self.get_logger().info(
            f'Door {self._door_index} pivot ({door.pivot.x:.3f}, {door.pivot.y:.3f}); '
            f'start distance {distance:.3f} m, bearing {math.degrees(bearing):.1f} deg')
        future = self._client.send_goal_async(goal, feedback_callback=self._feedback)
        future.add_done_callback(self._goal_response)

    def _sample(self):
        if self._start_distance is None or self.completion_future.done():
            return
        try:
            distance, bearing, polar = self._measure()
        except TransformException:
            return
        bearing_error = bearing - self._side * math.pi / 2.0
        bearing_error = math.atan2(math.sin(bearing_error), math.cos(bearing_error))
        travelled = polar - self._start_polar
        travelled = math.atan2(math.sin(travelled), math.cos(travelled))
        self._samples.append((distance, bearing_error, travelled))

    def _feedback(self, msg):
        f = msg.feedback
        self.get_logger().info(
            f'fb: travelled {f.travelled_angle_deg:5.1f} deg, '
            f'dist err {f.distance_error * 1000:+6.1f} mm, '
            f'bearing err {f.bearing_error_deg:+5.2f} deg',
            throttle_duration_sec=1.0)

    def _goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error('Goal rejected')
            self.completion_future.set_result(None)
            return
        handle.get_result_async().add_done_callback(self._result)

    def _result(self, future):
        response = future.result()
        r = response.result
        self._sample()
        self.get_logger().info(
            f'Server result: status={response.status} error_code={r.error_code} '
            f'"{r.error_str}" start_dist={r.start_distance:.3f} '
            f'final_dist={r.final_distance:.3f} travelled={r.travelled_angle_deg:.1f} deg '
            f'max_dist_err={r.max_distance_error * 1000:.1f} mm '
            f'max_bearing_err={r.max_bearing_error_deg:.2f} deg')
        if self._samples:
            # Ignore the initial in-place alignment when looking at bearing errors.
            final = self._samples[-1]
            max_d = max(abs(s[0] - self._start_distance) for s in self._samples)
            arc = [s for s in self._samples if abs(s[2]) > math.radians(1.0)]
            max_b = max((abs(s[1]) for s in arc), default=float('nan'))
            self.get_logger().info(
                f'Independent TF check ({len(self._samples)} samples): '
                f'start dist {self._start_distance:.3f} m, final dist {final[0]:.3f} m, '
                f'travelled {math.degrees(abs(final[2])):.1f} deg, '
                f'max dist err {max_d * 1000:.1f} mm, '
                f'max bearing err while arcing {math.degrees(max_b):.2f} deg, '
                f'final bearing err {math.degrees(final[1]):+.2f} deg')
        self.completion_future.set_result(None)


def main(args=None):
    rclpy.init(args=args)
    node = DriveAroundPivotTrigger()
    try:
        rclpy.spin_until_future_complete(node, node.completion_future)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
