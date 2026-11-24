"""GrabDoorHandle action server that reaches the pre-grasp pose with the learned doorhandle model.

Serves the same GrabDoorHandle action under the same name ('grab_door_handle') as
grab_door_handle_server and grab_door_handle_direct_sim_server, so the behavior tree does not
change. Instead of the handle pose being known (direct sim server), the doorhandle model
(lerobot plugin lerobot_policy_doorhandle, pip package in the workspace venv) estimates the
pre-grasp joint configuration from the Kinect RGB-D frame and the current joints.

The arm makes these moves:
1. Pre-grasp (model, closed loop): an inference thread runs the model on the newest RGB-D frame
   at most inference_rate_hz (<= 10 Hz) while a goal is active. The goal thread smooths the
   confident predictions (all sigma <= max_sigma) into a target and publishes it at
   control_rate_hz on target_topic. arips_lerobot's command_rate_limiter moves the arm towards
   it with limited speed and publishes target_reached. The pose is reached when that signal
   has held for reached_hold_time_s and the last converge_window predictions agree within
   converge_tol. The target is then frozen until the measured joints have settled, and
   publishing stops so the limiter goes idle and MoveIt owns the arm command topic again.
2. Forward: a straight Cartesian line (compute_cartesian_path, then execute_trajectory) of
   forward_distance along the gripper's pointing direction, as in the direct sim server.
3. Rotate: the door gripper joint turns to gripper_rotation (MoveIt, door gripper group).

The goal's prompt is ignored; max_duration_s > 0 limits the whole goal instead of
approach_timeout_s / execution_timeout_s.
"""

import collections
import math
import threading
import time

import numpy as np
import rclpy
import torch
from arips_action_msgs.action import GrabDoorHandle
from arips_lerobot.pregrasp_bags_to_lerobot import (
    decode_compressed_depth, decode_rgb, resize_frame)
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

import lerobot_policy_doorhandle  # noqa: F401  registers the "doorhandle" policy type
from lerobot.policies.factory import make_pre_post_processors
from lerobot_policy_doorhandle.kinematics import link_3_pose
from lerobot_policy_doorhandle.modeling_doorhandle import DoorhandlePolicy

from arips_actions.grab_door_handle_direct_sim_server import (
    DOOR_HINGE_Y, _cross, pre_grab_link_3_pose, yaw_of)

MAX_INFERENCE_RATE_HZ = 10.0
ARM_JOINTS = ['joint_slider', 'joint_2', 'joint_3', 'joint_4', 'joint_door_gripper']


class _Stop(Exception):
    """Carries the action result to return (failure, cancel or preemption)."""

    def __init__(self, result):
        super().__init__()
        self.result = result


class GrabDoorHandlePolicyServer(Node):
    def __init__(self):
        super().__init__('grab_door_handle_policy_server')

        self.declare_parameters(
            namespace='',
            parameters=[
                # pretrained_model folder of a doorhandle checkpoint
                ('policy_path', ''),
                ('device', 'cuda'),
                ('rgb_topic', '/kinect/rgb/image_raw/compressed'),
                ('depth_topic', '/kinect/depth/image_raw/compressedDepth'),
                # Must contain all state joints of the model (incl. kinect_joint); the training
                # data came from this topic.
                ('joint_states_topic', '/feetech_joint_state_broadcaster/joint_states'),
                # command_rate_limiter input and reached signal
                ('target_topic', '/arm_forward_position_controller/dummy_commands'),
                ('target_reached_topic', '/arm_forward_position_controller/target_reached'),
                # Inference thread rate, capped at 10 Hz
                ('inference_rate_hz', 10.0),
                # Rate of the goal loop (target publishing, reached / convergence checks)
                ('control_rate_hz', 20.0),
                # RGB and depth frames are paired when their stamps differ less than this
                ('max_image_skew_s', 0.05),
                # Observations (images, joint states) older than this are not used
                ('max_obs_age_s', 0.5),
                # Predictions with any std above this (per action joint, m / rad) are ignored
                ('max_sigma', [0.02, 0.1, 0.1, 0.1, 0.1]),
                # Abort if no confident prediction arrives for this long
                ('uncertain_timeout_s', 5.0),
                # Weight of a new prediction in the smoothed target (1 = no smoothing)
                ('target_smoothing', 0.5),
                # Joint limits (URDF) the target is clamped to
                ('joint_lower', [0.0, -2.26893, -2.44346, -2.44346, -1.65806]),
                ('joint_upper', [0.871, 2.26893, 2.44346, 2.44346, 0.5236]),
                # The reached signal is ignored this long after the approach started
                ('min_approach_s', 1.0),
                ('reached_hold_time_s', 0.5),
                ('reached_signal_timeout_s', 0.5),
                # The last converge_window predictions must agree within converge_tol
                ('converge_window', 5),
                ('converge_tol', [0.001, 0.005, 0.005, 0.005, 0.005]),
                # If the predictions do not converge this long after reached held, the mean of
                # the last converge_window predictions is used anyway
                ('converge_timeout_s', 3.0),
                # Frozen target: wait until the measured joints are this close (or timeout)
                ('settle_tol', [0.001, 0.005, 0.005, 0.005, 0.005]),
                ('settle_timeout_s', 3.0),
                # command_rate_limiter's 'timeout' parameter: it stops commanding after this
                ('limiter_timeout_s', 0.5),
                ('approach_timeout_s', 60.0),
                # Log the analytic pre-grab pose of the Gazebo lu door next to the model target
                ('compare_with_geometry', False),
                # Debug only: added to every prediction (action joint order)
                ('target_offset', [0.0, 0.0, 0.0, 0.0, 0.0]),
                # Log every prediction (and its geometry error with compare_with_geometry)
                ('log_predictions', False),
                ('pre_grab_distance', 0.05),
                ('door_z_offset', 0.0),
                ('map_frame', 'map'),
                # ---- forward move and gripper rotation (as in the direct sim server) ----
                ('planning_frame', 'base_link'),
                ('planning_group', 'arm'),
                ('ik_link', 'link_3'),
                ('move_group_action', '/move_action'),
                ('forward_distance', 0.045),
                ('forward_velocity_scaling', 0.1),
                ('forward_acceleration_scaling', 0.1),
                ('forward_max_step', 0.002),
                ('forward_revolute_jump_threshold', 0.3),
                ('forward_prismatic_jump_threshold', 0.05),
                ('compute_cartesian_path_service', '/compute_cartesian_path'),
                ('gripper_group', 'door_gripper_group'),
                ('gripper_joint', 'joint_door_gripper'),
                # Final door gripper joint position after the forward move [rad]
                ('gripper_rotation', 0.165),
                ('execute_trajectory_action', '/execute_trajectory'),
                ('avoid_collisions', True),
                ('planning_time_s', 5.0),
                ('planning_attempts', 5),
                ('max_velocity_scaling', 0.5),
                ('max_acceleration_scaling', 0.5),
                ('joint_goal_tolerance', 0.0005),
                ('execution_timeout_s', 60.0),
                # Long: right after a (re)start, discovering move_group took > 5 s.
                ('server_wait_timeout_s', 15.0),
                ('server_recreate_after_s', 5.0),
                ('tf_timeout_s', 1.0),
                ('cancel_timeout_s', 5.0),
            ],
        )
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self._p = p
        n = len(ARM_JOINTS)
        for name in ('max_sigma', 'joint_lower', 'joint_upper', 'converge_tol', 'settle_tol'):
            if len(p(name)) != n:
                raise ValueError(f'{name} needs {n} values ({ARM_JOINTS})')
        self._inference_period = 1.0 / min(p('inference_rate_hz'), MAX_INFERENCE_RATE_HZ)
        self._control_period = 1.0 / p('control_rate_hz')
        self._max_sigma = np.array(p('max_sigma'))
        self._lower = np.array(p('joint_lower'))
        self._upper = np.array(p('joint_upper'))
        self._converge_tol = np.array(p('converge_tol'))
        self._settle_tol = np.array(p('settle_tol'))

        self._load_policy(p('policy_path'), p('device'))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Latest sensor data, written by the subscription callbacks (decoded in the worker).
        self._obs_lock = threading.Lock()
        self._rgb_msgs = collections.deque(maxlen=5)
        self._depth_msgs = collections.deque(maxlen=5)
        self._joint_msg = None
        # Reached signal (monotonic timestamps)
        self._reached = False
        self._reached_since = None
        self._reached_stamp = None

        # Inference worker: runs while _infer_active is set, newest result in _result.
        self._infer_active = threading.Event()
        self._result_lock = threading.Lock()
        self._result = None  # (seq, target, sigma, obs_stamp_s, latency_s)
        self._result_seq = 0
        self._shutdown = False

        # Only one goal executes at a time; a new goal preempts the running one.
        self._execute_lock = threading.Lock()
        self._active_goal_id = None

        callback_group = ReentrantCallbackGroup()
        self.create_subscription(CompressedImage, p('rgb_topic'), self._on_rgb,
                                 qos_profile_sensor_data, callback_group=callback_group)
        self.create_subscription(CompressedImage, p('depth_topic'), self._on_depth,
                                 qos_profile_sensor_data, callback_group=callback_group)
        self.create_subscription(JointState, p('joint_states_topic'), self._on_joints, 10,
                                 callback_group=callback_group)
        self.create_subscription(Bool, p('target_reached_topic'), self._on_reached, 10,
                                 callback_group=callback_group)
        self._target_pub = self.create_publisher(Float64MultiArray, p('target_topic'), 10)
        self._published_targets = 0

        self._callback_group = callback_group
        self._move_client = self._make_client('move')
        self._cartesian_client = self._make_client('cartesian')
        self._execute_client = self._make_client('execute')

        self._worker = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker.start()

        self._action_server = ActionServer(
            self,
            GrabDoorHandle,
            'grab_door_handle',
            execute_callback=self.execute_callback,
            goal_callback=self._goal_callback,
            handle_accepted_callback=self._handle_accepted,
            cancel_callback=self._cancel_callback,
            callback_group=callback_group,
        )
        self.get_logger().info(
            f'GrabDoorHandle (policy) server ready, model {p("policy_path")} on {self._device}, '
            f'inference <= {1.0 / self._inference_period:.1f} Hz')

    # ---- model ---------------------------------------------------------------------------

    def _load_policy(self, path, device):
        if not path:
            raise ValueError('policy_path parameter is not set')
        if device.startswith('cuda') and not torch.cuda.is_available():
            self.get_logger().warning('CUDA not available, running the model on the CPU')
            device = 'cpu'
        self._device = device
        policy = DoorhandlePolicy.from_pretrained(path)
        policy.config.device = device
        self._policy = policy.to(device).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy.config, pretrained_path=path,
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        cfg = policy.config
        self._rgb_key, self._depth_key = cfg.rgb_and_depth_keys()
        self._image_hw = tuple(cfg.image_size)
        self._state_joints = list(cfg.state_joint_names)
        if list(cfg.action_joint_names) != ARM_JOINTS:
            raise ValueError(f'model action joints {cfg.action_joint_names} != {ARM_JOINTS}')
        self.get_logger().info(
            f'Loaded doorhandle model ({cfg.vision_backbone}, image {self._image_hw}, state '
            f'{self._state_joints})')

    def _infer(self, rgb, depth_m, joints):
        """rgb (H, W, 3) uint8, depth (H, W, 1) metres, joints in state order -> (target, sigma)."""
        obs = {
            self._rgb_key: torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0,
            self._depth_key: torch.from_numpy(depth_m).permute(2, 0, 1).float() * 1000.0,
            'observation.state': torch.tensor(joints, dtype=torch.float32),
        }
        with torch.inference_mode():
            obs = self._preprocessor(obs)
            target, sigma = self._policy.predict_with_uncertainty(obs)
            target = self._postprocessor(target)
        return target[0].cpu().numpy().astype(float), sigma[0].float().cpu().numpy().astype(float)

    def _inference_loop(self):
        h, w = self._image_hw
        try:
            # Warm-up (cuDNN autotuning, lazy CUDA init) before the first goal
            self._infer(np.zeros((h, w, 3), np.uint8), np.zeros((h, w, 1), np.float32),
                        [0.65] + [0.0] * (len(self._state_joints) - 1))
        except Exception as ex:  # noqa: BLE001
            self.get_logger().error(f'Model warm-up failed: {ex}')
        while not self._shutdown and rclpy.ok():
            if not self._infer_active.wait(timeout=0.2):
                continue
            start = time.monotonic()
            obs = self._take_observation()
            if obs is None:
                time.sleep(0.01)  # no inference happened, so the rate cap is unaffected
                continue
            rgb_msg, depth_msg, joints, stamp = obs
            try:
                frame = resize_frame({
                    'observation.images.kinect': decode_rgb(rgb_msg),
                    'observation.images.kinect_depth': decode_compressed_depth(depth_msg),
                }, h, w)
                target, sigma = self._infer(
                    frame['observation.images.kinect'],
                    frame['observation.images.kinect_depth'], joints)
            except Exception as ex:  # noqa: BLE001
                self.get_logger().error(f'Inference failed: {ex}', throttle_duration_sec=2.0)
                target = None
            if target is not None and self._infer_active.is_set():
                with self._result_lock:
                    self._result_seq += 1
                    self._result = (self._result_seq, target, sigma, stamp,
                                    time.monotonic() - start)
            remaining = self._inference_period - (time.monotonic() - start)
            if remaining > 0.0:
                time.sleep(remaining)

    def _take_observation(self):
        """Newest RGB frame with a depth frame within max_image_skew_s, plus the joints."""
        now = self.get_clock().now().nanoseconds * 1e-9
        max_age = self._p('max_obs_age_s')
        max_skew = self._p('max_image_skew_s')
        with self._obs_lock:
            rgbs, depths, joint_msg = list(self._rgb_msgs), list(self._depth_msgs), self._joint_msg
        if not rgbs or not depths or joint_msg is None:
            self.get_logger().warning('Waiting for RGB, depth and joint state messages',
                                      throttle_duration_sec=2.0)
            return None
        for rgb_stamp, rgb_msg in reversed(rgbs):
            if now - rgb_stamp > max_age:
                break
            depth_stamp, depth_msg = min(depths, key=lambda d: abs(d[0] - rgb_stamp))
            if abs(depth_stamp - rgb_stamp) <= max_skew:
                break
        else:
            rgb_msg = None
        if rgb_msg is None or now - rgb_stamp > max_age:
            self.get_logger().warning('No recent RGB-D frame pair', throttle_duration_sec=2.0)
            return None
        joint_stamp = _stamp_s(joint_msg.header)
        if now - joint_stamp > max_age:
            self.get_logger().warning('Joint states are stale', throttle_duration_sec=2.0)
            return None
        positions = dict(zip(joint_msg.name, joint_msg.position))
        missing = [j for j in self._state_joints if j not in positions]
        if missing:
            self.get_logger().warning(f'Joint states lack {missing}', throttle_duration_sec=2.0)
            return None
        return rgb_msg, depth_msg, [positions[j] for j in self._state_joints], rgb_stamp

    def _latest_result(self):
        with self._result_lock:
            return self._result

    # ---- subscriptions -------------------------------------------------------------------

    def _on_rgb(self, msg):
        with self._obs_lock:
            self._rgb_msgs.append((_stamp_s(msg.header), msg))

    def _on_depth(self, msg):
        with self._obs_lock:
            self._depth_msgs.append((_stamp_s(msg.header), msg))

    def _on_joints(self, msg):
        with self._obs_lock:
            self._joint_msg = msg

    def _on_reached(self, msg):
        now = time.monotonic()
        with self._obs_lock:
            if msg.data and not self._reached:
                self._reached_since = now
            elif not msg.data:
                self._reached_since = None
            self._reached = msg.data
            self._reached_stamp = now

    def _reached_now(self, now):
        """Current limiter target_reached value, False when the signal is stale."""
        with self._obs_lock:
            return bool(self._reached and self._reached_stamp is not None
                        and now - self._reached_stamp <= self._p('reached_signal_timeout_s'))

    def _reached_held(self, now):
        with self._obs_lock:
            fresh = (self._reached_stamp is not None
                     and now - self._reached_stamp <= self._p('reached_signal_timeout_s'))
            return (fresh and self._reached and self._reached_since is not None
                    and now - self._reached_since >= self._p('reached_hold_time_s'))

    def _measured_arm_joints(self):
        with self._obs_lock:
            msg = self._joint_msg
        if msg is None:
            return None
        positions = dict(zip(msg.name, msg.position))
        if not all(j in positions for j in ARM_JOINTS):
            return None
        return np.array([positions[j] for j in ARM_JOINTS])

    # ---- action callbacks ----------------------------------------------------------------

    def _goal_callback(self, _goal_request):
        return GoalResponse.ACCEPT

    def _handle_accepted(self, goal_handle):
        # Mark the newest goal as active; a running execute loop sees this and stops.
        self._active_goal_id = bytes(goal_handle.goal_id.uuid)
        goal_handle.execute()

    def _cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        goal_id = bytes(goal_handle.goal_id.uuid)
        with self._execute_lock:
            if self._active_goal_id != goal_id:
                return self._abort(goal_handle, 'Preempted by a newer goal before starting')
            try:
                return self._execute(goal_handle, goal_id)
            except _Stop as stop:
                return stop.result

    # ---- execution -----------------------------------------------------------------------

    def _execute(self, goal_handle, goal_id):
        self.get_logger().info('Executing GrabDoorHandle goal (policy)')
        self._published_targets = 0
        self._publish_feedback(goal_handle, False)
        deadline = None
        if goal_handle.request.max_duration_s > 0.0:
            deadline = time.monotonic() + goal_handle.request.max_duration_s

        # Check the MoveIt servers before the arm moves, not only when they are needed.
        self._wait_for_servers(goal_handle, goal_id)

        with self._result_lock:
            self._result = None
        self._infer_active.set()
        try:
            target = self._approach(goal_handle, goal_id, deadline)
        finally:
            self._infer_active.clear()
        self._settle(goal_handle, goal_id, target)
        self.get_logger().info('Arm is at the pre-grab pose')

        trajectory = self._forward_path(goal_handle)
        forward = self._p('forward_distance')
        self._run_move(trajectory, goal_handle, goal_id, deadline,
                       f'the pose {forward:.3f} m forward', self._execute_client)
        self.get_logger().info(f'Moved {forward:.3f} m forward')

        rotation = self._p('gripper_rotation')
        self._run_move(
            self._joint_goal(self._p('gripper_group'), {self._p('gripper_joint'): rotation}),
            goal_handle, goal_id, deadline, f'gripper rotation {rotation:.3f} rad')
        self.get_logger().info(f'Rotated the gripper to {rotation:.3f} rad')

        self._publish_feedback(goal_handle, True)
        result = GrabDoorHandle.Result()
        result.error_code = GrabDoorHandle.Result.SUCCESS
        goal_handle.succeed()
        return result

    def _approach(self, goal_handle, goal_id, deadline):
        """Closed-loop move to the predicted pre-grasp pose; returns the final target."""
        start = time.monotonic()
        approach_deadline = start + self._p('approach_timeout_s')
        if deadline is not None:
            approach_deadline = min(approach_deadline, deadline)
        uncertain_timeout = self._p('uncertain_timeout_s')
        alpha = self._p('target_smoothing')
        history = collections.deque(maxlen=max(1, self._p('converge_window')))
        target = None
        seen_seq = 0
        last_confident = start
        held_since = None
        n_results, latencies, n_uncertain = 0, [], 0
        with self._obs_lock:
            self._reached_since = None
        log_ref = None
        if self._p('log_predictions'):
            log_ref = self._geometry_reference()

        while True:
            now = time.monotonic()
            self._check_active(goal_handle, goal_id)
            if now > approach_deadline:
                raise _Stop(self._abort(goal_handle, 'Pre-grasp pose not reached in time'))

            result = self._latest_result()
            if result is not None and result[0] != seen_seq:
                seen_seq, prediction, sigma, _, latency = result
                n_results += 1
                self._publish_feedback(goal_handle, self._reached_now(now))
                latencies.append(latency)
                if self._p('log_predictions'):
                    measured = self._measured_arm_joints()
                    self.get_logger().info(
                        f'#{seen_seq} t={now - start:.1f}s prediction ' + _fmt(prediction)
                        + ' sigma ' + _fmt(sigma)
                        + (' measured ' + _fmt(measured) if measured is not None else '')
                        + (' | pred ' + self._geometry_error(prediction, log_ref)
                           if log_ref is not None else '')
                        + (' | meas ' + self._geometry_error(measured, log_ref)
                           if log_ref is not None and measured is not None else ''))
                # non-finite predictions count as uncertain (NaN sigma already fails the comparison)
                if np.all(np.isfinite(prediction)) and np.all(sigma <= self._max_sigma):
                    last_confident = now
                    prediction = np.clip(prediction + np.array(self._p('target_offset')),
                                         self._lower, self._upper)
                    history.append(prediction)
                    target = prediction if target is None else target + alpha * (prediction - target)
                else:
                    n_uncertain += 1
                    self.get_logger().warning(
                        'Model uncertain, sigma ' + _fmt(sigma), throttle_duration_sec=1.0)
            if now - last_confident > uncertain_timeout:
                raise _Stop(self._abort(
                    goal_handle, f'No confident model prediction for {uncertain_timeout:.1f} s'
                    + (' (no prediction at all)' if n_results == 0 else '')))
            if target is not None:
                self._publish_target(target)

            if now - start >= self._p('min_approach_s') and self._reached_held(now):
                held_since = held_since or now
                spread = (np.ptp(np.array(history), axis=0) if len(history) == history.maxlen
                          else None)
                if spread is not None and np.all(spread <= self._converge_tol):
                    break
                if now - held_since > self._p('converge_timeout_s'):
                    self.get_logger().warning(
                        'Predictions did not converge (spread '
                        + (_fmt(spread) if spread is not None else 'n/a')
                        + '), using the mean of the last ones')
                    break
            else:
                held_since = None
            time.sleep(self._control_period)

        final = np.clip(np.mean(np.array(history), axis=0), self._lower, self._upper)
        elapsed = time.monotonic() - start
        self.get_logger().info(
            f'Pre-grasp converged after {elapsed:.1f} s: {n_results} predictions '
            f'({n_results / elapsed:.1f} Hz, latency {1000 * np.mean(latencies):.0f} ms, '
            f'{n_uncertain} uncertain), target ' + _fmt(final) + ', sigma ' + _fmt(sigma))
        if self._p('compare_with_geometry'):
            self._log_geometry_error(final, 'model target')
        return final

    def _settle(self, goal_handle, goal_id, target):
        """Hold the frozen target until the measured joints settle, then release the limiter."""
        end = time.monotonic() + self._p('settle_timeout_s')
        error = None
        while time.monotonic() < end:
            self._check_active(goal_handle, goal_id)
            self._publish_target(target)
            measured = self._measured_arm_joints()
            if measured is not None:
                error = measured - target
                if np.all(np.abs(error) <= self._settle_tol):
                    break
            time.sleep(self._control_period)
        else:
            self.get_logger().warning(
                'Joints did not settle within settle_timeout_s, error '
                + (_fmt(error) if error is not None else 'unknown'))
        # Stop publishing; after its timeout the limiter stops commanding the arm.
        time.sleep(self._p('limiter_timeout_s') + 0.2)
        measured = self._measured_arm_joints()
        if measured is not None:
            self.get_logger().info('Measured - target joints: ' + _fmt(measured - target))
            if self._p('compare_with_geometry'):
                self._log_geometry_error(measured, 'measured joints')

    def _make_client(self, kind):
        group = self._callback_group
        if kind == 'move':
            return ActionClient(self, MoveGroup, self._p('move_group_action'),
                                callback_group=group)
        if kind == 'execute':
            return ActionClient(self, ExecuteTrajectory, self._p('execute_trajectory_action'),
                                callback_group=group)
        return self.create_client(GetCartesianPath, self._p('compute_cartesian_path_service'),
                                  callback_group=group)

    def _wait_for_servers(self, goal_handle, goal_id):
        """Wait for the MoveIt servers, honoring cancel and preemption.

        Sometimes a client of this process never matched move_group's endpoints after a
        restart (a fresh process saw them within a second). A client that is still not
        ready after server_recreate_after_s is destroyed and created again once, which makes
        DDS announce and match its endpoints anew.
        """
        names = {'execute': 'execute_trajectory action server',
                 'move': 'move_group action server',
                 'cartesian': 'compute_cartesian_path service'}
        start = time.monotonic()
        recreated = set()
        while True:
            self._check_active(goal_handle, goal_id)
            missing = [kind for kind in names if not self._client_ready(kind)]
            if not missing:
                if recreated:
                    self.get_logger().info(
                        'MoveIt servers available after recreating the '
                        + ', '.join(names[k] for k in recreated) + ' client(s)')
                return
            waited = time.monotonic() - start
            if waited > self._p('server_wait_timeout_s'):
                raise _Stop(self._abort(
                    goal_handle, ', '.join(names[k] for k in missing) + ' not available'))
            for kind in missing:
                if kind not in recreated and waited > self._p('server_recreate_after_s'):
                    self.get_logger().warning(
                        f'{names[kind]} not seen after {waited:.1f} s, recreating the client')
                    self._recreate_client(kind)
                    recreated.add(kind)
            time.sleep(0.1)

    def _client_ready(self, kind):
        if kind == 'move':
            return self._move_client.server_is_ready()
        if kind == 'execute':
            return self._execute_client.server_is_ready()
        return self._cartesian_client.service_is_ready()

    def _recreate_client(self, kind):
        if kind == 'move':
            self._move_client.destroy()
            self._move_client = self._make_client(kind)
        elif kind == 'execute':
            self._execute_client.destroy()
            self._execute_client = self._make_client(kind)
        else:
            self.destroy_client(self._cartesian_client)
            self._cartesian_client = self._make_client(kind)

    def _check_active(self, goal_handle, goal_id):
        if goal_handle.is_cancel_requested:
            raise _Stop(self._canceled(goal_handle))
        if self._active_goal_id != goal_id:
            raise _Stop(self._abort(goal_handle, 'Preempted by a newer goal'))

    def _publish_target(self, target):
        self._target_pub.publish(Float64MultiArray(data=[float(v) for v in target]))
        self._published_targets += 1

    def _geometry_reference(self):
        """Analytic pre-grab link_3 pose (Gazebo lu door) in the planning frame, or None."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._p('planning_frame'), self._p('map_frame'), Time(),
                timeout=Duration(seconds=self._p('tf_timeout_s')))
            robot = self._tf_buffer.lookup_transform(
                self._p('map_frame'), self._p('planning_frame'), Time(),
                timeout=Duration(seconds=self._p('tf_timeout_s'))).transform.translation
        except TransformException as ex:
            self.get_logger().warning(f'Geometry comparison: {ex}')
            return None
        x, y, z, yaw = pre_grab_link_3_pose(
            self._p('pre_grab_distance'), robot.y > DOOR_HINGE_Y, self._p('door_z_offset'))
        t, q = tf.transform.translation, tf.transform.rotation
        qv = (q.x, q.y, q.z)
        uv = _cross(qv, (x, y, z))
        uuv = _cross(qv, uv)
        return (x + 2.0 * (q.w * uv[0] + uuv[0]) + t.x,
                y + 2.0 * (q.w * uv[1] + uuv[1]) + t.y,
                z + 2.0 * (q.w * uv[2] + uuv[2]) + t.z,
                yaw + yaw_of(q))

    @staticmethod
    def _geometry_error(joints, ref):
        """link_3 FK of joints minus ref -> 'dx=.. dy=.. dz=.. mm, dyaw=.. deg'."""
        fx, fy, fz, fyaw = (v.item() for v in link_3_pose(*torch.tensor(joints[:4]).double()))
        dyaw = math.degrees(math.atan2(math.sin(fyaw - ref[3]), math.cos(fyaw - ref[3])))
        return (f'dx={1000 * (fx - ref[0]):+.1f} dy={1000 * (fy - ref[1]):+.1f} '
                f'dz={1000 * (fz - ref[2]):+.1f} mm, dyaw={dyaw:+.2f} deg')

    def _log_geometry_error(self, joints, what):
        """Compare link_3 from FK of joints with the analytic pre-grab pose (Gazebo lu door)."""
        ref = self._geometry_reference()
        if ref is not None:
            self.get_logger().info(
                f'Geometry check ({what}): link_3 error vs analytic pre-grab pose '
                + self._geometry_error(joints, ref))

    # ---- MoveIt moves (as in grab_door_handle_direct_sim_server) -------------------------

    def _run_move(self, move_goal, goal_handle, goal_id, deadline, what, client=None):
        """Send a MoveGroup (or ExecuteTrajectory) goal and wait for it; raise _Stop on failure."""
        client = client or self._move_client
        send_future = client.send_goal_async(move_goal)
        if not self._wait_for(send_future, goal_handle, goal_id, self._p('server_wait_timeout_s')):
            if send_future.done() and send_future.result().accepted:
                self._cancel_move(send_future.result())
            raise _Stop(self._stopped_or_abort(goal_handle, goal_id,
                                               'Timed out sending the MoveIt goal'))
        move_handle = send_future.result()
        if not move_handle.accepted:
            raise _Stop(self._abort(goal_handle, 'MoveIt goal rejected'))

        timeout = (self._p('execution_timeout_s') if deadline is None
                   else deadline - time.monotonic())
        result_future = move_handle.get_result_async()
        if not self._wait_for(result_future, goal_handle, goal_id, max(timeout, 0.0)):
            self._cancel_move(move_handle)
            raise _Stop(self._stopped_or_abort(goal_handle, goal_id,
                                               f'Arm did not reach {what} in time'))
        error_code = result_future.result().result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            raise _Stop(self._abort(
                goal_handle, f'Moving to {what} failed (MoveIt error code {error_code})'))

    def _forward_path(self, goal_handle):
        """ExecuteTrajectory goal for the straight move of the IK link along the gripper axis."""
        planning_frame, ik_link = self._p('planning_frame'), self._p('ik_link')
        forward = self._p('forward_distance')
        try:
            tf = self._tf_buffer.lookup_transform(
                planning_frame, ik_link, Time(),
                timeout=Duration(seconds=self._p('tf_timeout_s')))
        except TransformException as ex:
            raise _Stop(self._abort(
                goal_handle,
                f'No transform {planning_frame} -> {ik_link} for the forward move: {ex}'))
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = yaw_of(q)
        # -y axis of a frame with this yaw (roll/pitch are zero for this arm)
        dx, dy = math.sin(yaw), -math.cos(yaw)

        waypoint = PoseStamped().pose
        waypoint.position.x = t.x + forward * dx
        waypoint.position.y = t.y + forward * dy
        waypoint.position.z = t.z
        waypoint.orientation.z = math.sin(yaw / 2.0)
        waypoint.orientation.w = math.cos(yaw / 2.0)

        if not self._cartesian_client.wait_for_service(
                timeout_sec=self._p('server_wait_timeout_s')):
            raise _Stop(self._abort(goal_handle, 'compute_cartesian_path service not available'))
        request = GetCartesianPath.Request()
        request.header.frame_id = planning_frame
        request.start_state.is_diff = True
        request.group_name = self._p('planning_group')
        request.link_name = ik_link
        request.waypoints = [waypoint]
        request.max_step = self._p('forward_max_step')
        request.jump_threshold = 0.0
        request.revolute_jump_threshold = self._p('forward_revolute_jump_threshold')
        request.prismatic_jump_threshold = self._p('forward_prismatic_jump_threshold')
        request.avoid_collisions = self._p('avoid_collisions')
        request.max_velocity_scaling_factor = self._p('forward_velocity_scaling')
        request.max_acceleration_scaling_factor = self._p('forward_acceleration_scaling')
        future = self._cartesian_client.call_async(request)
        goal_id = bytes(goal_handle.goal_id.uuid)
        if not self._wait_for(future, goal_handle, goal_id,
                              self._p('planning_time_s') + self._p('server_wait_timeout_s')):
            raise _Stop(self._stopped_or_abort(goal_handle, goal_id,
                                               'Timed out planning the forward move'))
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS or response.fraction < 0.999:
            raise _Stop(self._abort(
                goal_handle,
                f'Forward move of {forward:.3f} m not feasible: only '
                f'{100.0 * response.fraction:.0f}% of the line planned '
                f'(MoveIt error code {response.error_code.val})'))
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        return goal

    def _joint_goal(self, group, positions):
        """MoveGroup goal moving the joints of group to positions (name -> value)."""
        tolerance = self._p('joint_goal_tolerance')
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
        request.num_planning_attempts = self._p('planning_attempts')
        request.allowed_planning_time = self._p('planning_time_s')
        request.max_velocity_scaling_factor = self._p('max_velocity_scaling')
        request.max_acceleration_scaling_factor = self._p('max_acceleration_scaling')
        goal.planning_options.plan_only = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        return goal

    # ---- helpers -------------------------------------------------------------------------

    def _publish_feedback(self, goal_handle, reached):
        feedback = GrabDoorHandle.Feedback()
        feedback.handle_reached = reached
        feedback.published_actions = self._published_targets
        goal_handle.publish_feedback(feedback)

    def _wait_for(self, future, goal_handle, goal_id, timeout):
        """Wait for a future; stop early if the goal is cancelled or preempted."""
        deadline = time.monotonic() + timeout
        while not future.done():
            if (goal_handle.is_cancel_requested or self._active_goal_id != goal_id
                    or time.monotonic() > deadline):
                return False
            time.sleep(0.01)
        return True

    def _cancel_move(self, move_handle):
        cancel_future = move_handle.cancel_goal_async()
        deadline = time.monotonic() + self._p('cancel_timeout_s')
        while not cancel_future.done():
            if time.monotonic() > deadline:
                self.get_logger().warning('Timed out waiting for the MoveIt cancel response')
                return
            time.sleep(0.01)

    def _stopped_or_abort(self, goal_handle, goal_id, error_str):
        if goal_handle.is_cancel_requested:
            return self._canceled(goal_handle)
        if self._active_goal_id != goal_id:
            return self._abort(goal_handle, 'Preempted by a newer goal')
        return self._abort(goal_handle, error_str)

    def _abort(self, goal_handle, error_str):
        result = GrabDoorHandle.Result()
        result.error_code = GrabDoorHandle.Result.FAILURE
        result.error_str = error_str
        goal_handle.abort()
        self.get_logger().error(error_str)
        return result

    def _canceled(self, goal_handle):
        result = GrabDoorHandle.Result()
        result.error_code = GrabDoorHandle.Result.FAILURE
        result.error_str = 'GrabDoorHandle canceled'
        goal_handle.canceled()
        self.get_logger().info(result.error_str)
        return result


def _stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _fmt(values):
    return '[' + ', '.join(f'{v:.4f}' for v in values) + ']'


def main(args=None):
    rclpy.init(args=args)
    node = GrabDoorHandlePolicyServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
