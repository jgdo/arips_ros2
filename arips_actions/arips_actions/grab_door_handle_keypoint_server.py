"""GrabDoorHandle action server that finds the door handle with the keypoint detector.

Serves the same GrabDoorHandle action under the same name ('grab_door_handle') as the other
GrabDoorHandle servers and makes the same arm moves as grab_door_handle_direct_sim_server
(pre-grab with MoveIt, straight forward line, gripper rotation). Only the pre-grab pose comes
from somewhere else: from the 3D keypoints 'handle_start' (lever root at the spindle) and
'handle_end' (lever tip) of arips_keypoint_detection's keypoint_detector_node, e.g.

  ros2 run arips_keypoint_detection keypoint_detector_node --ros-args \\
      -p model:=<ws>/src/arips_data/keypoint_train/pull_right_30_tiny_unet_rgb_s30/model.onnx

On each goal the server waits for num_frames keypoint messages taken after the goal arrived
(robot standing still), transforms both keypoints into the map frame and takes the median.

From the two points:
- Approach direction (where the gripper points): horizontal, perpendicular to the lever
  (the lever is parallel to the door leaf), on the side facing away from the camera. The
  line between the keypoints is not exactly parallel to the leaf (handle_end sits ~7 mm
  closer to the robot than handle_start); this shows up the same with the measured depth
  and with the pixel rays alone, so it comes from where the keypoints sit on the lever, not
  from depth errors. yaw_offset_deg corrects it (mirrored for a left lever).
- Lever side: lever pointing right as seen from the robot -> gripper_rotation, left ->
  -gripper_rotation (like the back handle in the direct sim server).
- Handle point (spindle axis at the outer face of the lever): handle_start plus a fixed
  offset in the handle frame (handle_offset_approach along the approach direction,
  handle_offset_lever along the lever, handle_offset_z up). The depth window hits the lever
  surface behind its outermost face and the keypoint sits slightly above the spindle axis.

The offsets were measured in the Gazebo lu world against the true handle geometry from
grab_door_handle_direct_sim_server, at robot poses within +-5 cm and +-6 deg of the pull
approach pose: handle_start - true point = (+0.7 +- 1.5, +13.5 +- 1.5, +8.6 +- 0.3) mm (along
lever, away from robot, up); lever yaw -3.9 deg (-3.0 .. -5.2).

With the detector's fixed height mode (height_mode:=fixed, default fixed_height 1.053 in
base_link = the measured handle_start height) handle_start lands on the same points, so the
handle offsets stay valid. handle_end is then put at the same height although it really sits
~6 mm lower, which turns the lever further: yaw -5.7 deg (-4.4 .. -6.9). The default
yaw_offset_deg 4.8 is the mean of both modes; the remaining +-0.9 deg do not matter for the
grasp.

The gripper pre-grab pose is then the same as in the direct sim server: gripper axis on the
spindle axis (PRE_GRAB_Z_OFFSET), front edge pre_grab_distance before the lever face.
The pose is published on ~/pre_grab_pose (door gripper origin, x axis along the approach
direction) for RViz.
"""

import math
import threading
import time

import numpy as np
import rclpy
from arips_keypoint_msgs.msg import KeypointArray
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from tf2_ros import TransformException

from .grab_door_handle_direct_sim_server import (
    GRIPPER_REACH, PRE_GRAB_Z_OFFSET, GrabDoorHandleDirectSimServer, gripper_to_link_3_pose)


def _transform_point(tf, p):
    """Apply a geometry_msgs TransformStamped to point p (3 floats), return np.array."""
    q, t = tf.transform.rotation, tf.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return rot @ np.asarray(p, dtype=float) + np.array([t.x, t.y, t.z])


class GrabDoorHandleKeypointServer(GrabDoorHandleDirectSimServer):
    def __init__(self):
        super().__init__('grab_door_handle_keypoint_server')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('keypoints_topic', '/keypoint_detector_node/keypoints'),
                ('start_keypoint', 'handle_start'),
                ('end_keypoint', 'handle_end'),
                # Number of keypoint messages (after the goal arrived) to take the median of
                ('num_frames', 5),
                ('keypoint_timeout_s', 5.0),
                # Handle point (spindle axis, lever face) = handle_start + these offsets [m]
                ('handle_offset_approach', -0.0135),
                ('handle_offset_lever', -0.0007),
                ('handle_offset_z', -0.0086),
                # Added to the approach yaw from the keypoints for a right lever (negated for
                # a left lever) [deg]. Mean of the detector's depth mode (3.9) and fixed
                # height mode (5.7).
                ('yaw_offset_deg', 4.8),
                # Plausible distance between the two keypoints [m]
                ('min_lever_length', 0.05),
                ('max_lever_length', 0.16),
            ],
        )
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._start_name = p('start_keypoint')
        self._end_name = p('end_keypoint')
        self._num_frames = p('num_frames')
        self._keypoint_timeout = p('keypoint_timeout_s')

        self._keypoints_lock = threading.Lock()
        self._keypoint_msgs = []
        self._collect_since = None
        self._keypoints_sub = self.create_subscription(
            KeypointArray, p('keypoints_topic'), self._on_keypoints, 10)
        self._pre_grab_pub = self.create_publisher(PoseStamped, '~/pre_grab_pose', 1)

    def _ready_message(self):
        # Called by the base __init__, before the keypoint parameters are declared
        return 'GrabDoorHandle (keypoint) server ready'

    def _on_keypoints(self, msg):
        with self._keypoints_lock:
            if self._collect_since is None or Time.from_msg(msg.header.stamp) < self._collect_since:
                return
            if len(self._keypoint_msgs) < self._num_frames:
                self._keypoint_msgs.append(msg)

    # ---- keypoints -> pre-grab pose ----------------------------------------------------------

    def _collect_keypoints(self, goal_handle, goal_id):
        """Return num_frames keypoint messages taken after now, or (None, error)."""
        with self._keypoints_lock:
            self._keypoint_msgs = []
            self._collect_since = self.get_clock().now()
        deadline = time.monotonic() + self._keypoint_timeout
        try:
            while True:
                with self._keypoints_lock:
                    if len(self._keypoint_msgs) >= self._num_frames:
                        return list(self._keypoint_msgs), ''
                    received = len(self._keypoint_msgs)
                if goal_handle.is_cancel_requested or self._active_goal_id != goal_id:
                    return None, 'stopped'
                if time.monotonic() > deadline:
                    return None, (f'Received only {received} of {self._num_frames} keypoint '
                                  f'messages within {self._keypoint_timeout:.1f} s; is '
                                  'keypoint_detector_node running?')
                time.sleep(0.02)
        finally:
            with self._keypoints_lock:
                self._collect_since = None

    def _handle_points(self, msgs):
        """Return (start, end, camera position) in the map frame (medians), or (None, error)."""
        starts, ends, cameras = [], [], []
        invisible = 0
        for msg in msgs:
            kps = {kp.name: kp for kp in msg.keypoints}
            start, end = kps.get(self._start_name), kps.get(self._end_name)
            if start is None or end is None:
                return None, (f'Keypoints {self._start_name}/{self._end_name} not in the '
                              f'detector output ({", ".join(kps)})')
            if not (start.visible and end.visible and start.has_position and end.has_position):
                invisible += 1
                continue
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                    timeout=Duration(seconds=self._tf_timeout))
            except TransformException as ex:
                self.get_logger().warning(f'Keypoint frame skipped: {ex}')
                continue
            starts.append(_transform_point(
                tf, (start.position.x, start.position.y, start.position.z)))
            ends.append(_transform_point(tf, (end.position.x, end.position.y, end.position.z)))
            cameras.append(_transform_point(tf, (0.0, 0.0, 0.0)))
        # Majority of the frames must show the handle
        if len(starts) * 2 <= len(msgs):
            return None, (f'Door handle not detected: only {len(starts)} of {len(msgs)} '
                          f'keypoint frames usable ({invisible} without visible 3D keypoints)')
        return (np.median(starts, axis=0), np.median(ends, axis=0),
                np.median(cameras, axis=0)), ''

    def _pre_grab_target(self, goal_handle, goal_id):
        msgs, error = self._collect_keypoints(goal_handle, goal_id)
        if msgs is None:
            return None, False, error
        points, error = self._handle_points(msgs)
        if points is None:
            return None, False, error
        start, end, camera = points

        lever = end - start
        lever_length = float(np.linalg.norm(lever))
        horizontal_length = math.hypot(lever[0], lever[1])
        min_length = self.get_parameter('min_lever_length').value
        max_length = self.get_parameter('max_lever_length').value
        if not min_length <= lever_length <= max_length or horizontal_length < 0.5 * min_length:
            return None, False, (
                f'Implausible door handle: keypoints {lever_length * 100.0:.1f} cm apart '
                f'(expected {min_length * 100.0:.0f}..{max_length * 100.0:.0f} cm), start '
                f'{np.round(start, 3).tolist()}, end {np.round(end, 3).tolist()}')

        # Approach direction: horizontal normal of the lever, pointing away from the camera.
        lever_yaw = math.atan2(lever[1], lever[0])
        approach_yaw = lever_yaw + math.pi / 2.0
        to_handle = start - camera
        if math.cos(approach_yaw) * to_handle[0] + math.sin(approach_yaw) * to_handle[1] < 0.0:
            approach_yaw -= math.pi
        # Lever to the right of the approach direction (seen from the robot)?
        lever_right = math.sin(lever_yaw - approach_yaw) < 0.0
        side = 1.0 if lever_right else -1.0
        approach_yaw += side * math.radians(self.get_parameter('yaw_offset_deg').value)
        approach_yaw = math.atan2(math.sin(approach_yaw), math.cos(approach_yaw))

        # Handle frame: a along the approach direction, l along the lever, z up.
        a = np.array([math.cos(approach_yaw), math.sin(approach_yaw), 0.0])
        lever_dir = side * np.array([a[1], -a[0], 0.0])
        handle = (start
                  + self.get_parameter('handle_offset_approach').value * a
                  + self.get_parameter('handle_offset_lever').value * lever_dir
                  + np.array([0.0, 0.0, self.get_parameter('handle_offset_z').value]))

        # Gripper origin on the spindle axis, front edge pre_grab_distance before the lever face
        gripper = handle - (self._pre_grab_distance + GRIPPER_REACH) * a
        gripper[2] += PRE_GRAB_Z_OFFSET
        self.get_logger().info(
            f'Door handle from {len(msgs)} keypoint frames: lever '
            f'{"right" if lever_right else "left"}, {lever_length * 100.0:.1f} cm, handle point '
            f'{np.round(handle, 4).tolist()}, approach yaw {math.degrees(approach_yaw):.1f} deg')

        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
            float(v) for v in gripper)
        pose.pose.orientation.z = math.sin(approach_yaw / 2.0)
        pose.pose.orientation.w = math.cos(approach_yaw / 2.0)
        self._pre_grab_pub.publish(pose)

        x, y, z, yaw = gripper_to_link_3_pose(*gripper, approach_yaw)
        target = self._map_pose_to_planning(x, y, z, yaw)
        if target is None:
            return None, False, (f'No transform {self._map_frame} -> {self._planning_frame} '
                                 'for the pre-grab pose')
        return target, not lever_right, ''


def main(args=None):
    rclpy.init(args=args)
    node = GrabDoorHandleKeypointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
