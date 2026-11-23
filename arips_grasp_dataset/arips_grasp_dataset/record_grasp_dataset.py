"""Record a dataset of door handle pre-grasp arm movements in the Gazebo lu world.

Every episode is one bag that contains only the arm movement from its (randomized) start
configuration to the pre-grasp pose of the door handle (move 1 of
grab_door_handle_direct_sim_server). Per episode:

  1. fold the arm (if needed), teleport the robot to a random pose in the office
  2. respawn the closed door raised by a random 0..3 cm
  3. arm to door_grasp_folded + joint noise, kinect to a random tilt
  4. drive (Nav2, precise goal checker) to the noisy grasping approach pose
  5. IK for the pre-grasp pose; record the arm movement there until it has settled
  6. validate the bag, keep it as <output>/episode_XXXX

Any failure discards the episode and the next one starts from a new random state.

usage:
  ros2 run arips_grasp_dataset record_grasp_dataset --episodes N --output DIR
      [--seed S] [--start-sim [--gui] [--keep-sim]]
"""

import argparse
import datetime
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from ament_index_python.packages import get_package_prefix
from arips_action_msgs.srv import (
    GetDoorGraspingApproachPoseByIndex,
    StartBagRecording,
    StopBagRecording,
)
from arips_actions.grab_door_handle_direct_sim_server import (
    DOOR_HINGE_Y,
    _cross,
    pre_grab_link_3_pose,
    yaw_of,
)
from component_manager_msgs.srv import StartComponent, StopComponent
from geometry_msgs.msg import PoseStamped, TwistStamped
from lifecycle_msgs.srv import GetState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import Parameter, ParameterType
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter as RclpyParameter
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

from arips_grasp_dataset.sim_stack import SimStack
from arips_grasp_dataset.sim_world import SimWorld

# ---- scene -----------------------------------------------------------------------------------

# Office start pose of the door test (map == Gazebo world frame)
OFFICE_X = -1.2138490676879883
OFFICE_Y = 0.35048535466194153
OFFICE_YAW = 2.0 * math.atan2(0.7812900096663772, 0.6241681831009269)  # ~102.7 deg
DOOR_INDEX = 0

# SRDF states (arips_moveit_config/config/arips.srdf)
ARM_JOINTS = ('joint_slider', 'joint_2', 'joint_3', 'joint_4')
ARM_FOLDED = {'joint_slider': 0.8, 'joint_2': -1.7424, 'joint_3': 1.8494, 'joint_4': -1.4444}
GRIPPER_JOINT = 'joint_door_gripper'
GRIPPER_NEUTRAL = 0.0
SLIDER_LIMITS = (0.0, 0.871)
KINECT_JOINT = 'kinect_joint'
KINECT_LIMITS = (-1.65806, 0.5236)

# ---- ROS interfaces --------------------------------------------------------------------------

JOINT_STATES_TOPIC = '/feetech_joint_state_broadcaster/joint_states'
KINECT_COMMAND_TOPIC = '/kinect_forward_position_controller/commands'
RGB_TOPIC = '/kinect/rgb/image_raw/compressed'
DEPTH_TOPIC = '/kinect/depth/image_raw/compressedDepth'
COMMANDS_TOPIC = '/arm_forward_position_controller/commands'
GRAB_SERVER = '/grab_door_handle_direct_sim_server'

# Expected message rates (sim time) for the dropped message check
EXPECTED_RATES = {
    RGB_TOPIC: 30.0,
    DEPTH_TOPIC: 30.0,
    JOINT_STATES_TOPIC: 120.0,
}


class EpisodeFailure(Exception):
    def __init__(self, step, message):
        super().__init__(f'{step}: {message}')
        self.step = step


def _uniform(rng, half_width):
    return rng.uniform(-half_width, half_width)


def _yaw_quaternion(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


class DatasetRecorder(Node):
    def __init__(self, args):
        super().__init__(
            'grasp_dataset_recorder',
            parameter_overrides=[RclpyParameter('use_sim_time', value=True)])
        self.args = args
        group = ReentrantCallbackGroup()

        self._joint_lock = threading.Lock()
        self._joints = {}       # name -> (position, velocity)
        self._joint_stamp = None
        self._odom_twist = None
        self._map = None

        self.create_subscription(JointState, JOINT_STATES_TOPIC, self._on_joints, 50,
                                 callback_group=group)
        self.create_subscription(Odometry, '/csm_odom', self._on_odom, 10, callback_group=group)
        self.create_subscription(
            OccupancyGrid, '/map', self._on_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE),
            callback_group=group)

        self._kinect_pub = self.create_publisher(Float64MultiArray, KINECT_COMMAND_TOPIC, 10)
        self._cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self._move_group = ActionClient(self, MoveGroup, '/move_action', callback_group=group)
        self._compute_path = ActionClient(self, ComputePathToPose, '/compute_path_to_pose',
                                          callback_group=group)
        self._follow_path = ActionClient(self, FollowPath, '/follow_path', callback_group=group)
        self._ik = self.create_client(GetPositionIK, '/compute_ik', callback_group=group)
        self._approach = self.create_client(
            GetDoorGraspingApproachPoseByIndex, '/get_door_grasping_approach_pose_by_index',
            callback_group=group)
        self._cancel_nav = self.create_client(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal', callback_group=group)
        self._clear_costmaps = [
            self.create_client(ClearEntireCostmap, name, callback_group=group)
            for name in ('/global_costmap/clear_entirely_global_costmap',
                         '/local_costmap/clear_entirely_local_costmap')]
        self._grab_params = self.create_client(
            SetParameters, f'{GRAB_SERVER}/set_parameters', callback_group=group)
        self._rec_start = self.create_client(
            StartBagRecording, '/grasp_bag_recorder/start', callback_group=group)
        self._rec_stop = self.create_client(
            StopBagRecording, '/grasp_bag_recorder/stop', callback_group=group)
        self._start_component = self.create_client(
            StartComponent, '/component_manager/start_component', callback_group=group)
        self._stop_component = self.create_client(
            StopComponent, '/component_manager/stop_component', callback_group=group)

        self.tf_buffer = Buffer()
        self._tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

    # ---- subscriptions ---------------------------------------------------------------------

    def _on_joints(self, msg):
        with self._joint_lock:
            for i, name in enumerate(msg.name):
                velocity = msg.velocity[i] if i < len(msg.velocity) else 0.0
                self._joints[name] = (msg.position[i], velocity)
            self._joint_stamp = self.now_s()

    def _on_odom(self, msg):
        self._odom_twist = msg.twist.twist

    def _on_map(self, msg):
        self._map = msg

    def joints(self):
        with self._joint_lock:
            return dict(self._joints), self._joint_stamp

    # ---- time and futures ------------------------------------------------------------------

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def sim_sleep(self, duration, wall_timeout=None):
        """Sleep duration seconds of sim time."""
        end = self.now_s() + duration
        wall_end = time.monotonic() + (wall_timeout or max(10.0, 20.0 * duration))
        while self.now_s() < end:
            if time.monotonic() > wall_end:
                raise EpisodeFailure('clock', 'sim clock does not advance')
            time.sleep(0.005)

    @staticmethod
    def wait(future, timeout, step, what):
        end = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > end:
                raise EpisodeFailure(step, f'timed out waiting for {what}')
            time.sleep(0.005)
        if future.exception() is not None:
            raise EpisodeFailure(step, f'{what} failed: {future.exception()}')
        return future.result()

    def call(self, client, request, timeout, step):
        if not client.wait_for_service(timeout_sec=5.0):
            raise EpisodeFailure(step, f'service {client.srv_name} not available')
        return self.wait(client.call_async(request), timeout, step, client.srv_name)

    def run_action(self, client, goal, timeout, step, what):
        """Send an action goal and wait for its result; cancels it on timeout."""
        if not client.wait_for_server(timeout_sec=5.0):
            raise EpisodeFailure(step, f'{what}: action server not available')
        handle = self.wait(client.send_goal_async(goal), 10.0, step, f'{what} goal response')
        if not handle.accepted:
            raise EpisodeFailure(step, f'{what}: goal rejected')
        try:
            return self.wait(handle.get_result_async(), timeout, step, f'{what} result')
        except EpisodeFailure:
            cancel = handle.cancel_goal_async()
            end = time.monotonic() + 5.0
            while not cancel.done() and time.monotonic() < end:
                time.sleep(0.01)
            raise

    # ---- robot ----------------------------------------------------------------------------

    def stop_robot(self):
        if self._cancel_nav.service_is_ready():
            future = self._cancel_nav.call_async(CancelGoal.Request())
            end = time.monotonic() + 3.0
            while not future.done() and time.monotonic() < end:
                time.sleep(0.01)
        stop = TwistStamped()
        for _ in range(5):
            stop.header.stamp = self.get_clock().now().to_msg()
            self._cmd_vel_pub.publish(stop)
            time.sleep(0.05)

    def wait_robot_still(self, step, duration=0.5, timeout=10.0):
        still_since = None
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            twist = self._odom_twist
            now = self.now_s()
            if twist is not None and math.hypot(twist.linear.x, twist.linear.y) < 0.005 \
                    and abs(twist.angular.z) < 0.01:
                still_since = now if still_since is None else still_since
                if now - still_since >= duration:
                    return
            else:
                still_since = None
            time.sleep(0.02)
        raise EpisodeFailure(step, 'robot does not come to rest')

    def robot_pose(self, frame='arips_wheel_center'):
        try:
            tf = self.tf_buffer.lookup_transform('map', frame, Time(),
                                                 timeout=Duration(seconds=1.0))
        except TransformException as ex:
            raise EpisodeFailure('tf', str(ex))
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, yaw_of(q)

    def pose_is_free(self, x, y, clearance):
        """True if no occupied /map cell is within clearance of (x, y)."""
        grid = self._map
        if grid is None:
            return True
        info = grid.info
        res = info.resolution
        cx = int((x - info.origin.position.x) / res)
        cy = int((y - info.origin.position.y) / res)
        r = int(math.ceil(clearance / res))
        for j in range(cy - r, cy + r + 1):
            for i in range(cx - r, cx + r + 1):
                if (i - cx) ** 2 + (j - cy) ** 2 > r * r:
                    continue
                if not (0 <= i < info.width and 0 <= j < info.height):
                    return False
                if grid.data[j * info.width + i] >= 50:
                    return False
        return True

    # ---- arm -------------------------------------------------------------------------------

    def joint_goal(self, group, positions, tolerance):
        constraints = Constraints()
        for name, position in positions.items():
            joint = JointConstraint()
            joint.joint_name = name
            joint.position = position
            joint.tolerance_above = tolerance
            joint.tolerance_below = tolerance
            joint.weight = 1.0
            constraints.joint_constraints.append(joint)
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = group
        request.start_state.is_diff = True
        request.goal_constraints.append(constraints)
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = self.args.velocity_scaling
        request.max_acceleration_scaling_factor = self.args.velocity_scaling
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    def move_joints(self, group, positions, step, tolerance=0.0005):
        result = self.run_action(self._move_group, self.joint_goal(group, positions, tolerance),
                                 60.0, step, f'moving {group}')
        code = result.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            raise EpisodeFailure(step, f'moving {group} failed, MoveIt error code {code}')

    def arm_error(self, goal):
        joints, _ = self.joints()
        missing = [n for n in goal if n not in joints]
        if missing:
            return math.inf, math.inf
        pos_err = max(abs(joints[n][0] - goal[n]) for n in goal)
        vel = max(abs(joints[n][1]) for n in goal)
        return pos_err, vel

    def wait_arm_settled(self, goal, step, timeout=15.0):
        """Wait until all goal joints stay within tolerance and at rest for settle_s.

        Returns (t_settle_start, t_settled) in sim time.
        """
        a = self.args
        start = None
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            pos_err, vel = self.arm_error(goal)
            now = self.now_s()
            if pos_err < a.settle_position_tolerance and vel < a.settle_velocity_tolerance:
                start = now if start is None else start
                if now - start >= a.settle_s:
                    return start, now
            else:
                start = None
            time.sleep(0.005)
        pos_err, vel = self.arm_error(goal)
        raise EpisodeFailure(step, f'arm does not settle (error {pos_err:.4f}, velocity '
                                   f'{vel:.4f})')

    def set_kinect(self, angle, step, timeout=10.0):
        msg = Float64MultiArray()
        msg.data = [angle]
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self._kinect_pub.publish(msg)
            joints, _ = self.joints()
            if KINECT_JOINT in joints and abs(joints[KINECT_JOINT][0] - angle) \
                    < math.radians(0.5) and abs(joints[KINECT_JOINT][1]) < 0.01:
                return
            time.sleep(0.05)
        raise EpisodeFailure(step, f'kinect did not reach {math.degrees(angle):.1f} deg')

    # ---- navigation ------------------------------------------------------------------------

    def clear_costmaps(self):
        for client in self._clear_costmaps:
            if client.service_is_ready():
                future = client.call_async(ClearEntireCostmap.Request())
                end = time.monotonic() + 3.0
                while not future.done() and time.monotonic() < end:
                    time.sleep(0.01)

    def drive_to(self, pose, step):
        goal = ComputePathToPose.Goal()
        goal.goal = pose
        goal.planner_id = 'GridBased'
        goal.use_start = False
        result = self.run_action(self._compute_path, goal, 15.0, step, 'path planning')
        if result.status != GoalStatus.STATUS_SUCCEEDED or not result.result.path.poses:
            raise EpisodeFailure(step, f'no path to the approach pose (status {result.status}, '
                                       f'error {result.result.error_code})')
        follow = FollowPath.Goal()
        follow.path = result.result.path
        follow.controller_id = 'FollowPath'
        follow.goal_checker_id = 'precise_goal_checker'
        follow.progress_checker_id = 'progress_checker'
        result = self.run_action(self._follow_path, follow, self.args.drive_timeout, step,
                                 'following the path')
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            raise EpisodeFailure(step, f'following the path failed (status {result.status}, '
                                       f'error {result.result.error_code})')

    def pre_grab_target(self, z_offset, step):
        """Pre-grasp pose of link_3 in base_link, as grab_door_handle_direct_sim_server."""
        try:
            tf = self.tf_buffer.lookup_transform('base_link', 'map', Time(),
                                                 timeout=Duration(seconds=1.0))
            robot = self.tf_buffer.lookup_transform('map', 'base_link', Time(),
                                                    timeout=Duration(seconds=1.0))
        except TransformException as ex:
            raise EpisodeFailure(step, str(ex))
        back_side = robot.transform.translation.y > DOOR_HINGE_Y
        x, y, z, yaw = pre_grab_link_3_pose(self.args.pre_grab_distance, back_side, z_offset)
        t, q = tf.transform.translation, tf.transform.rotation
        qv = (q.x, q.y, q.z)
        uv = _cross(qv, (x, y, z))
        uuv = _cross(qv, uv)
        target = PoseStamped()
        target.header.frame_id = 'base_link'
        target.pose.position.x = x + 2.0 * (q.w * uv[0] + uuv[0]) + t.x
        target.pose.position.y = y + 2.0 * (q.w * uv[1] + uuv[1]) + t.y
        target.pose.position.z = z + 2.0 * (q.w * uv[2] + uuv[2]) + t.z
        local_yaw = yaw + yaw_of(q)
        target.pose.orientation.z = math.sin(local_yaw / 2.0)
        target.pose.orientation.w = math.cos(local_yaw / 2.0)
        return target

    def solve_ik(self, target, step, attempts=5):
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'arm'
        request.ik_request.ik_link_name = 'link_3'
        request.ik_request.pose_stamped = target
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout = Duration(seconds=1.0).to_msg()
        request.ik_request.robot_state.is_diff = True
        # KDL's random restarts sometimes return the mirrored elbow configuration (sign of
        # joint_3 flipped), which swings the whole arm around. Only the elbow branch of the
        # start configuration is accepted, so all episodes show the same kind of movement.
        joints, _ = self.joints()
        elbow_sign = math.copysign(1.0, joints['joint_3'][0])
        error = None
        for _ in range(attempts):
            response = self.call(self._ik, request, 10.0, step)
            if response.error_code.val != MoveItErrorCodes.SUCCESS:
                p = target.pose.position
                error = (f'pre-grasp pose not reachable (IK error {response.error_code.val}, '
                         f'target {p.x:.3f} {p.y:.3f} {p.z:.3f})')
                continue
            solution = dict(zip(response.solution.joint_state.name,
                                response.solution.joint_state.position))
            arm = {n: solution[n] for n in ARM_JOINTS if n in solution}
            if len(arm) != len(ARM_JOINTS):
                raise EpisodeFailure(step, 'IK solution misses arm joints')
            if math.copysign(1.0, arm['joint_3']) == elbow_sign:
                return arm
            error = f'IK only found the mirrored elbow configuration (joint_3 {arm["joint_3"]:.2f})'
        raise EpisodeFailure(step, error)

    def set_grab_server_z_offset(self, z_offset):
        """Keep the sim grab server consistent with the raised door (best effort)."""
        if not self._grab_params.service_is_ready():
            return
        param = Parameter()
        param.name = 'door_z_offset'
        param.value.type = ParameterType.PARAMETER_DOUBLE
        param.value.double_value = z_offset
        request = SetParameters.Request()
        request.parameters = [param]
        future = self._grab_params.call_async(request)
        end = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < end:
            time.sleep(0.01)

    # ---- recorder --------------------------------------------------------------------------

    def start_recording(self, uri, step, timeout=20.0):
        end = time.monotonic() + timeout
        while True:
            request = StartBagRecording.Request()
            request.uri = uri
            response = self.call(self._rec_start, request, 10.0, step)
            if response.success:
                return
            if time.monotonic() > end or 'Not subscribed' not in response.message:
                raise EpisodeFailure(step, response.message)
            time.sleep(0.5)

    def stop_recording(self, step):
        response = self.call(self._rec_stop, StopBagRecording.Request(), 30.0, step)
        stats = {}
        for i, name in enumerate(response.topics):
            stats[name] = {
                'count': response.counts[i],
                'first': Time.from_msg(response.first_stamps[i]).nanoseconds * 1e-9,
                'last': Time.from_msg(response.last_stamps[i]).nanoseconds * 1e-9,
            }
        if not response.success:
            raise EpisodeFailure(step, response.message)
        return stats

    # ---- system ----------------------------------------------------------------------------

    def wait_for_stack(self, timeout):
        """Wait until everything the episodes use is up."""
        end = time.monotonic() + timeout
        checks = [
            ('/clock', lambda: self.now_s() > 0.0),
            ('joint states', lambda: self.joints()[1] is not None),
            ('/map', lambda: self._map is not None),
            ('/compute_ik', self._ik.service_is_ready),
            ('/move_action', self._move_group.server_is_ready),
            ('/compute_path_to_pose', self._compute_path.server_is_ready),
            ('/follow_path', self._follow_path.server_is_ready),
            ('approach pose service', self._approach.service_is_ready),
            ('recorder', lambda: self._rec_start.service_is_ready()
                and self._rec_stop.service_is_ready()),
            ('Nav2 active', self._nav2_active),
        ]
        pending = list(checks)
        last_log = 0.0
        while pending:
            pending = [(name, check) for name, check in pending if not check()]
            if not pending:
                break
            if time.monotonic() > end:
                raise RuntimeError('Not ready: ' + ', '.join(n for n, _ in pending))
            if time.monotonic() - last_log > 10.0:
                last_log = time.monotonic()
                self.get_logger().info('Waiting for: ' + ', '.join(n for n, _ in pending))
            time.sleep(0.5)
        self.get_logger().info('System ready')

    def _nav2_active(self):
        # velocity_smoother and collision_monitor forward /cmd_vel: when the Nav2 bringup
        # aborts (e.g. bt_navigator fails to load its BT) they stay inactive and the robot
        # never moves, although the planner and controller servers are active.
        for name in ('controller_server', 'planner_server', 'velocity_smoother',
                     'collision_monitor'):
            client = self.create_client(GetState, f'/{name}/get_state')
            try:
                if not client.wait_for_service(timeout_sec=1.0):
                    return False
                future = client.call_async(GetState.Request())
                end = time.monotonic() + 3.0
                while not future.done() and time.monotonic() < end:
                    time.sleep(0.01)
                if not future.done() or future.result().current_state.label != 'active':
                    return False
            finally:
                self.destroy_client(client)
        return True

    def component(self, name, start, timeout=30.0):
        client = self._start_component if start else self._stop_component
        request = (StartComponent if start else StopComponent).Request()
        request.name = name
        response = self.call(client, request, timeout, 'component_manager')
        if not response.success:
            raise RuntimeError(f'{"Starting" if start else "Stopping"} {name} failed: '
                               f'{response.message}')


class Episode:
    """One randomized episode; run() raises EpisodeFailure on any problem."""

    def __init__(self, node, world, rng, index, tmp_uri):
        self.node = node
        self.world = world
        self.rng = rng
        self.index = index
        self.tmp_uri = tmp_uri
        self.meta = {'index': index}
        self.recording = False

    def run(self):
        node, world, rng, a = self.node, self.world, self.rng, self.node.args
        log = node.get_logger()

        # 1. Clean start: no navigation, gripper neutral, arm folded (before teleporting).
        node.stop_robot()
        joints, _ = node.joints()
        if GRIPPER_JOINT in joints and abs(joints[GRIPPER_JOINT][0] - GRIPPER_NEUTRAL) > 0.02:
            node.move_joints('door_gripper_group', {GRIPPER_JOINT: GRIPPER_NEUTRAL}, 'reset_arm',
                             tolerance=0.01)
        pos_err, _ = node.arm_error(ARM_FOLDED)
        if pos_err > 0.15:
            node.move_joints('arm', ARM_FOLDED, 'reset_arm', tolerance=0.01)

        # 2. Random start pose in the office, free in the static map.
        for _ in range(100):
            sx = OFFICE_X + _uniform(rng, a.start_xy_noise)
            sy = OFFICE_Y + _uniform(rng, a.start_xy_noise)
            if node.pose_is_free(sx, sy, a.start_clearance):
                break
        else:
            raise EpisodeFailure('start_pose', 'no free start pose found')
        syaw = OFFICE_YAW + math.radians(_uniform(rng, a.start_yaw_noise_deg))
        self.meta['start_pose'] = {'x': sx, 'y': sy, 'yaw': syaw}
        if not world.teleport(sx, sy, syaw):
            raise EpisodeFailure('teleport', 'teleporting the robot failed')
        node.sim_sleep(1.0)
        if not world.teleport(sx, sy, syaw):  # settle any bounce from the first teleport
            raise EpisodeFailure('teleport', 'teleporting the robot failed')
        node.wait_robot_still('teleport')

        # 3. Door height
        dz = rng.uniform(0.0, a.door_z_max)
        self.meta['door_z_offset'] = dz
        error = world.respawn_door(dz)
        if error is not None:
            raise EpisodeFailure('door', error)
        node.set_grab_server_z_offset(dz)
        node.clear_costmaps()

        # 4. Noisy folded arm start configuration
        start_arm = dict(ARM_FOLDED)
        for name in ('joint_2', 'joint_3', 'joint_4'):
            start_arm[name] += math.radians(_uniform(rng, a.arm_noise_deg))
        start_arm['joint_slider'] = min(max(
            start_arm['joint_slider'] + _uniform(rng, a.slider_noise),
            SLIDER_LIMITS[0] + 0.005), SLIDER_LIMITS[1] - 0.005)
        self.meta['start_arm_goal'] = start_arm
        node.move_joints('arm', start_arm, 'start_arm', tolerance=0.002)

        # 5. Kinect tilt
        kinect = math.radians(a.kinect_deg + _uniform(rng, a.kinect_noise_deg))
        kinect = min(max(kinect, KINECT_LIMITS[0]), KINECT_LIMITS[1])
        self.meta['kinect_angle'] = kinect
        node.set_kinect(kinect, 'kinect')

        # 6. Drive to the noisy approach pose
        request = GetDoorGraspingApproachPoseByIndex.Request()
        request.door_index = DOOR_INDEX
        response = node.call(node._approach, request, 10.0, 'approach_pose')
        if not response.success:
            raise EpisodeFailure('approach_pose', response.error_message)
        pose = response.pose
        nominal_yaw = yaw_of(pose.pose.orientation)
        noise = {'dx': _uniform(rng, a.approach_xy_noise),
                 'dy': _uniform(rng, a.approach_xy_noise),
                 'dyaw': math.radians(_uniform(rng, a.approach_yaw_noise_deg))}
        self.meta['approach_pose_nominal'] = {'x': pose.pose.position.x,
                                              'y': pose.pose.position.y, 'yaw': nominal_yaw}
        self.meta['approach_noise'] = noise
        pose.pose.position.x += noise['dx']
        pose.pose.position.y += noise['dy']
        qz, qw = _yaw_quaternion(nominal_yaw + noise['dyaw'])
        pose.pose.orientation.x = pose.pose.orientation.y = 0.0
        pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
        pose.header.stamp = node.get_clock().now().to_msg()
        node.drive_to(pose, 'drive')
        node.stop_robot()
        node.wait_robot_still('drive')
        x, y, yaw = node.robot_pose()
        self.meta['base_pose'] = {'x': x, 'y': y, 'yaw': yaw}
        goal_err = math.hypot(x - pose.pose.position.x, y - pose.pose.position.y)
        yaw_err = abs(_angle_diff(yaw, nominal_yaw + noise['dyaw']))
        self.meta['base_pose_error'] = {'xy': goal_err, 'yaw': yaw_err}
        if goal_err > a.max_base_error or yaw_err > math.radians(a.max_base_yaw_error_deg):
            raise EpisodeFailure('drive', f'approach pose missed by {goal_err:.3f} m, '
                                          f'{math.degrees(yaw_err):.1f} deg')

        # 7. IK for the pre-grasp pose (before recording: unreachable -> no bag)
        target = node.pre_grab_target(dz, 'ik')
        arm_goal = node.solve_ik(target, 'ik', a.ik_attempts)
        p = target.pose.position
        self.meta['pre_grasp_target_base_link'] = {
            'x': p.x, 'y': p.y, 'z': p.z, 'yaw': yaw_of(target.pose.orientation)}
        self.meta['arm_goal'] = arm_goal
        joints, _ = node.joints()
        self.meta['arm_start'] = {n: joints[n][0] for n in ARM_JOINTS}
        self.meta['kinect_actual'] = joints.get(KINECT_JOINT, (None,))[0]

        # 8. Record the arm movement
        if os.path.exists(self.tmp_uri):
            shutil.rmtree(self.tmp_uri)
        node.start_recording(self.tmp_uri, 'record')
        self.recording = True
        t_start = node.now_s()
        node.sim_sleep(a.pre_roll_s)
        t_move = node.now_s()
        node.move_joints('arm', arm_goal, 'move', tolerance=a.joint_goal_tolerance)
        t_done = node.now_s()
        _, t_settled = node.wait_arm_settled(arm_goal, 'settle')
        node.sim_sleep(a.post_roll_s)
        pos_err, vel = node.arm_error(arm_goal)
        if pos_err > a.settle_position_tolerance or vel > a.settle_velocity_tolerance:
            raise EpisodeFailure('settle', f'arm moved again after settling (error '
                                           f'{pos_err:.4f}, velocity {vel:.4f})')
        stats = node.stop_recording('record')
        self.recording = False
        t_stop = node.now_s()
        self.meta['times'] = {'record_start': t_start, 'move_start': t_move,
                              'move_done': t_done, 'settled': t_settled, 'record_stop': t_stop}
        self.meta['topics'] = stats

        # 9. Validate
        self.validate(stats, arm_goal, t_move, t_settled)
        log.info(f'Episode {self.index}: recorded {t_stop - t_start:.1f} s, move '
                 f'{t_settled - t_move:.1f} s')

    def validate(self, stats, arm_goal, t_move, t_settled):
        a = self.node.args
        for name, s in stats.items():
            if s['count'] == 0:
                raise EpisodeFailure('validate', f'no messages on {name}')
        for name, rate in EXPECTED_RATES.items():
            s = stats[name]
            duration = s['last'] - s['first']
            expected = rate * duration
            if s['count'] < a.min_rate_fraction * expected:
                raise EpisodeFailure('validate', f'{name}: {s["count"]} messages in '
                                                 f'{duration:.2f} s, expected ~{expected:.0f}')
            if s['first'] > t_move:
                raise EpisodeFailure('validate', f'{name}: first message after move start')
            if s['last'] < t_settled:
                raise EpisodeFailure('validate', f'{name}: last message before the arm settled')
        if stats[COMMANDS_TOPIC]['count'] < 2:
            raise EpisodeFailure('validate', 'no arm commands recorded')
        self.validate_bag(arm_goal)

    def validate_bag(self, arm_goal):
        """Check the last recorded joint state really is the settled pre-grasp pose."""
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        a = self.node.args
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=self.tmp_uri, storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        reader.set_filter(rosbag2_py.StorageFilter(topics=[JOINT_STATES_TOPIC]))
        last = None
        while reader.has_next():
            _, data, _ = reader.read_next()
            last = data
        del reader
        if last is None:
            raise EpisodeFailure('validate', 'bag contains no joint states')
        msg = deserialize_message(last, JointState)
        state = dict(zip(msg.name, zip(msg.position, msg.velocity or [0.0] * len(msg.name))))
        for name, goal in arm_goal.items():
            position, velocity = state.get(name, (math.inf, math.inf))
            if abs(position - goal) > a.settle_position_tolerance \
                    or abs(velocity) > a.settle_velocity_tolerance:
                raise EpisodeFailure('validate', f'last joint state of {name} not at the goal '
                                                 f'({position:.4f} vs {goal:.4f}, velocity '
                                                 f'{velocity:.4f})')


# ---- dataset ---------------------------------------------------------------------------------

def existing_episodes(output):
    indices = []
    for name in os.listdir(output):
        if name.startswith('episode_') and os.path.isdir(os.path.join(output, name)):
            try:
                indices.append(int(name[len('episode_'):]))
            except ValueError:
                pass
    return sorted(indices)


def load_summary(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {'episodes': [], 'failures': {}, 'runs': []}


def write_json(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def start_recorder_process(log_dir):
    exe = os.path.join(get_package_prefix('arips_grasp_dataset'), 'lib', 'arips_grasp_dataset',
                       'grasp_bag_recorder')
    log = open(os.path.join(log_dir, 'grasp_bag_recorder.log'), 'w')
    proc = subprocess.Popen([exe, '--ros-args', '-p', 'use_sim_time:=true'],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
    log.close()
    return proc


def stop_process(proc, timeout=10.0):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def parse_args(argv):
    p = argparse.ArgumentParser(
        description='Record door handle pre-grasp arm movements in the Gazebo lu world, '
                    'one bag per episode.')
    p.add_argument('--episodes', type=int, required=True, help='number of valid episodes')
    p.add_argument('--output', required=True, help='dataset folder (resumes if it exists)')
    p.add_argument('--seed', type=int, default=None, help='random seed (default: random)')
    p.add_argument('--start-sim', action='store_true',
                   help='launch the simulation (headless unless --gui) and navigation, stop them at the end')
    p.add_argument('--keep-sim', action='store_true',
                   help='with --start-sim: leave the simulation running at the end')
    p.add_argument('--gui', action='store_true',
                   help='with --start-sim: show the Gazebo GUI instead of running headless')
    p.add_argument('--max-consecutive-failures', type=int, default=10)
    g = p.add_argument_group('randomization')
    g.add_argument('--start-xy-noise', type=float, default=0.30, help='[m], +-')
    g.add_argument('--start-yaw-noise-deg', type=float, default=0.0, help='[deg], +-')
    g.add_argument('--start-clearance', type=float, default=0.30,
                   help='min. distance of the start position to static obstacles [m]')
    g.add_argument('--door-z-max', type=float, default=0.03, help='door raised 0..this [m]')
    g.add_argument('--arm-noise-deg', type=float, default=5.0, help='j2..j4 [deg], +-')
    g.add_argument('--slider-noise', type=float, default=0.05, help='[m], +-')
    g.add_argument('--kinect-deg', type=float, default=-60.0)
    g.add_argument('--kinect-noise-deg', type=float, default=10.0, help='[deg], +-')
    g.add_argument('--approach-xy-noise', type=float, default=0.05, help='[m], +-')
    g.add_argument('--approach-yaw-noise-deg', type=float, default=5.0, help='[deg], +-')
    g = p.add_argument_group('execution and validation')
    g.add_argument('--pre-grab-distance', type=float, default=0.05)
    g.add_argument('--ik-attempts', type=int, default=5,
                   help='IK calls until a solution on the start elbow branch is found')
    g.add_argument('--velocity-scaling', type=float, default=0.5)
    g.add_argument('--joint-goal-tolerance', type=float, default=0.0005)
    g.add_argument('--drive-timeout', type=float, default=90.0, help='[s] wall time')
    g.add_argument('--max-base-error', type=float, default=0.05, help='[m]')
    g.add_argument('--max-base-yaw-error-deg', type=float, default=4.0)
    g.add_argument('--pre-roll-s', type=float, default=0.5, help='recorded before the move')
    g.add_argument('--settle-s', type=float, default=0.5,
                   help='arm must stay at the goal this long before stopping')
    g.add_argument('--post-roll-s', type=float, default=0.3, help='recorded after settling')
    g.add_argument('--settle-position-tolerance', type=float, default=0.01, help='[rad/m]')
    g.add_argument('--settle-velocity-tolerance', type=float, default=0.01)
    g.add_argument('--min-rate-fraction', type=float, default=0.9,
                   help='min. fraction of the expected image/joint state messages')
    return p.parse_known_args(argv)


def main(argv=None):
    args, ros_args = parse_args(sys.argv[1:] if argv is None else argv)
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)
    tmp_dir = os.path.join(output, '.tmp')
    log_dir = os.path.join(output, 'logs')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2 ** 31)

    # No rclpy signal handler: Ctrl-C raises KeyboardInterrupt and the context stays
    # usable for the cleanup (stopping the recorder and the simulation).
    rclpy.init(args=[sys.argv[0]] + ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = DatasetRecorder(args)
    log = node.get_logger()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    sim = SimStack(log_dir, log, gui=args.gui) if args.start_sim else None
    recorder = None
    summary_path = os.path.join(output, 'dataset.json')
    summary = load_summary(summary_path)
    run_info = {'started': datetime.datetime.now().isoformat(timespec='seconds'),
                'seed': seed, 'args': vars(args), 'recorded': 0, 'failed': 0}
    summary['runs'].append(run_info)
    exit_code = 1
    try:
        if sim is not None:
            # Besides the Gazebo GUI-only race (handled in SimStack.start), a controller
            # spawner sometimes never reaches the controller manager's services, so the arm
            # controllers and joint state broadcaster are missing: restart in that case.
            for attempt in range(1, 4):
                if not sim.start():
                    raise RuntimeError('Could not start the simulation')
                end = time.monotonic() + 60.0
                while node.joints()[1] is None and time.monotonic() < end:
                    time.sleep(0.5)
                if node.joints()[1] is not None:
                    break
                log.warning('No joint states (controllers not spawned), restarting the '
                            'simulation')
                sim.stop()
            else:
                raise RuntimeError('Simulation started without its arm controllers')
            # wait for the component manager, then start the navigation stack
            if not node._start_component.wait_for_service(timeout_sec=60.0):
                raise RuntimeError('component_manager not available')
            node.component('full_navigation', True, timeout=60.0)
        recorder = start_recorder_process(log_dir)
        if sim is None:
            node.wait_for_stack(timeout=60.0)
        else:
            # The Nav2 bringup aborts when bt_navigator cannot load its BT, e.g. because
            # /get_door_state was not up within its 10 s wait while everything starts.
            for attempt in range(1, 4):
                try:
                    node.wait_for_stack(timeout=120.0)
                    break
                except RuntimeError as error:
                    if attempt == 3:
                        raise
                    log.warning(f'{error}, restarting full_navigation')
                    node.component('full_navigation', False, timeout=60.0)
                    node.component('full_navigation', True, timeout=60.0)
        world = SimWorld()

        rng = random.Random(seed)
        done = len(existing_episodes(output))
        index = (existing_episodes(output) or [-1])[-1] + 1
        target = done + args.episodes
        consecutive = 0
        attempt = 0
        log.info(f'Recording {args.episodes} episodes into {output} (seed {seed}, '
                 f'{done} already there)')
        while done < target:
            attempt += 1
            name = f'episode_{index:04d}'
            tmp_uri = os.path.join(tmp_dir, name)
            episode = Episode(node, world, rng, index, tmp_uri)
            episode.meta['seed'] = seed
            episode.meta['attempt'] = attempt
            t0 = time.monotonic()
            try:
                episode.run()
            except EpisodeFailure as failure:
                if episode.recording:
                    try:
                        node.stop_recording('record')
                    except EpisodeFailure:
                        pass
                shutil.rmtree(tmp_uri, ignore_errors=True)
                node.stop_robot()
                consecutive += 1
                run_info['failed'] += 1
                summary['failures'][failure.step] = summary['failures'].get(failure.step, 0) + 1
                write_json(summary_path, summary)
                log.warning(f'Episode attempt {attempt} discarded ({failure}), '
                            f'{consecutive} consecutive failures')
                if consecutive >= args.max_consecutive_failures:
                    raise RuntimeError(f'{consecutive} consecutive failures, giving up')
                continue
            consecutive = 0
            episode.meta['wall_time_s'] = time.monotonic() - t0
            final = os.path.join(output, name)
            shutil.move(tmp_uri, final)
            write_json(os.path.join(final, 'metadata.json'), episode.meta)
            summary['episodes'].append({
                'name': name,
                'duration_s': episode.meta['times']['record_stop']
                - episode.meta['times']['record_start'],
                'door_z_offset': episode.meta['door_z_offset'],
                'kinect_angle': episode.meta['kinect_angle'],
            })
            run_info['recorded'] += 1
            write_json(summary_path, summary)
            done += 1
            index += 1
            log.info(f'Saved {name} ({done}/{target})')
        exit_code = 0
        log.info(f'Done: {run_info["recorded"]} episodes recorded, {run_info["failed"]} '
                 f'attempts discarded')
    except KeyboardInterrupt:
        log.warning('Interrupted')
    except RuntimeError as error:
        log.error(str(error))
    finally:
        run_info['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
        write_json(summary_path, summary)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        stop_process(recorder)
        if sim is not None and not args.keep_sim:
            try:
                node.component('full_navigation', False, timeout=60.0)
            except (RuntimeError, EpisodeFailure) as error:
                log.warning(f'Stopping full_navigation: {error}')
            sim.stop()
        executor.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
