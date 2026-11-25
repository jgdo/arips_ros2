#!/usr/bin/env python3
"""Remove laser scan points close to door pivots from the semantic map.

Subscribes to ``scan`` and ``semantic_map`` and publishes ``scan_filtered_semantic``.
Every valid scan point within ``radius`` meters of a door pivot (measured in the
x/y plane of the semantic map frame) is replaced by NaN, i.e. marked invalid.
If no semantic map or no transform is available, the scan is forwarded as-is.
"""

from typing import Optional

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from arips_semantic_map_msgs.msg import SemanticMap


def _quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


class DoorLaserFilter(Node):
    def __init__(self) -> None:
        super().__init__('door_laser_filter')
        self.radius = self.declare_parameter('radius', 0.8).value
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.semantic_map: Optional[SemanticMap] = None
        self.pivots = np.zeros((0, 2))

        self.pub = self.create_publisher(LaserScan, 'scan_filtered_semantic', qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self._laser_callback, qos_profile_sensor_data)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(SemanticMap, 'semantic_map', self._semantic_map_callback, map_qos)

        self.get_logger().info(f"Door laser filter ready, radius {self.radius:.2f} m.")

    def _on_set_parameters(self, params):
        for param in params:
            if param.name == 'radius':
                if param.value is None or param.value < 0.0:
                    return SetParametersResult(successful=False, reason='radius must be >= 0')
                self.radius = float(param.value)
        return SetParametersResult(successful=True)

    def _semantic_map_callback(self, msg: SemanticMap) -> None:
        self.semantic_map = msg
        self.pivots = np.array([[d.pivot.x, d.pivot.y] for d in msg.doors]).reshape(-1, 2)
        self.get_logger().info(
            f"Received semantic map with {len(msg.doors)} doors in frame '{msg.header.frame_id}'.")

    def _lookup_transform(self, target: str, scan: LaserScan):
        try:
            return self.tf_buffer.lookup_transform(
                target, scan.header.frame_id, Time.from_msg(scan.header.stamp))
        except TransformException:
            # Scan stamp may be slightly ahead of tf; fall back to the latest transform.
            return self.tf_buffer.lookup_transform(target, scan.header.frame_id, Time())

    def _laser_callback(self, msg: LaserScan) -> None:
        if self.semantic_map is None:
            self.get_logger().info(
                "Received scan but no semantic map yet, forwarding scan as-is.", once=True)
            self.pub.publish(msg)
            return

        if len(self.pivots) == 0 or self.radius <= 0.0:
            self.pub.publish(msg)
            return

        map_frame = self.semantic_map.header.frame_id
        try:
            tf = self._lookup_transform(map_frame, msg)
        except TransformException as ex:
            self.get_logger().warning(
                f"Could not transform {msg.header.frame_id} to {map_frame}: {ex}. "
                "Forwarding scan as-is.",
                throttle_duration_sec=5.0)
            self.pub.publish(msg)
            return

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.logical_and(ranges >= msg.range_min, ranges <= msg.range_max)

        q = tf.transform.rotation
        t = tf.transform.translation
        rot = _quaternion_to_matrix(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z])

        pts_laser = np.stack([ranges * np.cos(angles), ranges * np.sin(angles), np.zeros_like(ranges)], axis=1)
        pts_map = pts_laser @ rot.T + trans

        # (num_points, num_doors) squared distances in the x/y plane
        diff = pts_map[:, None, :2] - self.pivots[None, :, :]
        near_door = np.any(np.einsum('ijk,ijk->ij', diff, diff) < self.radius ** 2, axis=1)
        remove = np.logical_and(valid, near_door)

        if np.any(remove):
            ranges[remove] = np.nan
            msg.ranges = ranges.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DoorLaserFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
