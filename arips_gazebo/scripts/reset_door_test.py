#!/usr/bin/env python3
"""Reset the Gazebo lu world for door opening tests.

Steps:
  1. Cancel all running NavigateToPose goals and stop the robot.
  2. Move the door gripper to door_gripper_neutral and the arm to door_grasp_folded
     (MoveIt, /move_action), so the gripper lets go of the handle.
  3. Close the door: apply a persistent closing torque to the door leaf.
  4. Teleport the robot to the start pose in front of the door.
  5. Wait until the leaf is closed, then remove the torque.

usage:
  ros2 run arips_gazebo reset_door_test.py [--x X --y Y --yaw YAW_DEG] [--skip-arm]
      [--world lu] [--robot arips] [--door door]

Requires a running simulation (arips_gazebo startup.launch.py world:=lu) and move_group.
"""

import argparse
import math
import re
import subprocess
import sys
import time

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import TwistStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from rclpy.action import ActionClient
from rclpy.node import Node

# Start pose of the door opening test in the lu world (map == world frame).
START_X = -1.2138490676879883
START_Y = 0.35048535466194153
START_YAW_DEG = math.degrees(2.0 * math.atan2(0.7812900096663772, 0.6241681831009269))

# SRDF states (arips_moveit_config/config/arips.srdf), as joint goals for MoveGroup.
GRIPPER_NEUTRAL = {'joint_door_gripper': 0.0}
ARM_FOLDED = {'joint_slider': 0.8, 'joint_2': -1.7424, 'joint_3': 1.8494, 'joint_4': -1.4444}

# Torque on the leaf about the hinge axis; negative closes the door.
CLOSING_TORQUE = -4.0
CLOSED_TOLERANCE_DEG = 1.0
CLOSE_TIMEOUT_S = 20.0


def gz(args, timeout=10.0):
    """Run a gz CLI command, return stdout ('' on failure)."""
    try:
        return subprocess.run(['gz'] + args, capture_output=True, text=True,
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ''


class ResetDoorTest(Node):
    def __init__(self, args):
        super().__init__('reset_door_test')
        self.args = args
        self._move_group = ActionClient(self, MoveGroup, '/move_action')
        self._cancel_nav = self.create_client(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        self._cmd_vel = self.create_publisher(TwistStamped, '/cmd_vel', 10)

    # ------------------------------------------------------------------ steps

    def cancel_navigation(self):
        if self._cancel_nav.wait_for_service(timeout_sec=2.0):
            # A zero goal id and zero stamp cancel all goals.
            future = self._cancel_nav.call_async(CancelGoal.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            n = len(future.result().goals_canceling) if future.result() else 0
            self.get_logger().info(f'Canceled {n} navigation goal(s)')
        else:
            self.get_logger().info('No navigate_to_pose server, nothing to cancel')
        stop = TwistStamped()
        for _ in range(5):
            stop.header.stamp = self.get_clock().now().to_msg()
            self._cmd_vel.publish(stop)
            time.sleep(0.05)

    def move_joints(self, group, joints):
        if not self._move_group.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('MoveGroup action /move_action not available')
            return False
        goal = MoveGroup.Goal()
        goal.request.group_name = group
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5
        constraints = Constraints()
        for name, position in joints.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        send = self._move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error(f'MoveGroup goal for {group} rejected')
            return False
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=60.0)
        if not result.done():
            self.get_logger().error(f'MoveGroup goal for {group} timed out')
            return False
        code = result.result().result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f'Moving {group} failed, MoveIt error code {code}')
            return False
        self.get_logger().info(f'{group} moved')
        return True

    def set_door_torque(self, torque):
        a = self.args
        gz(['topic', '-t', f'/world/{a.world}/wrench/persistent', '-m', 'gz.msgs.EntityWrench',
            '-p', f'entity: {{name: "{a.door}::leaf", type: LINK}}, '
                  f'wrench: {{torque: {{z: {torque}}}}}'])

    def clear_door_torque(self):
        a = self.args
        gz(['topic', '-t', f'/world/{a.world}/wrench/clear', '-m', 'gz.msgs.Entity',
            '-p', f'name: "{a.door}::leaf", type: LINK'])

    def teleport_robot(self, x, y, yaw):
        a = self.args
        req = (f'name: "{a.robot}", position: {{x: {x}, y: {y}, z: 0.01}}, '
               f'orientation: {{x: 0, y: 0, z: {math.sin(yaw / 2.0)}, w: {math.cos(yaw / 2.0)}}}')
        out = gz(['service', '-s', f'/world/{a.world}/set_pose', '--reqtype', 'gz.msgs.Pose',
                  '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req])
        return 'true' in out

    def leaf_angle_deg(self):
        """Hinge angle of the door leaf from the world pose info, None if unavailable."""
        out = gz(['topic', '-e', '-t', f'/world/{self.args.world}/pose/info', '-n', '1'],
                 timeout=5.0)
        i = out.find('name: "leaf"')
        if i < 0:
            return None
        # The leaf pose is relative to the door model: a pure rotation about z by the hinge angle.
        orientation = out[i:].split('orientation', 1)
        if len(orientation) < 2:
            return None
        z = re.search(r'\bz:\s*(-?[\d.e+-]+)', orientation[1].split('}', 1)[0])
        qz = float(z.group(1)) if z else 0.0  # gz omits zero fields
        return math.degrees(2.0 * math.asin(max(-1.0, min(1.0, qz))))

    # ------------------------------------------------------------------- main

    def run(self):
        a = self.args
        self.cancel_navigation()

        if not a.skip_arm:
            # Gripper first, so it turns out of the handle lever before the arm pulls back.
            ok = self.move_joints('door_gripper_group', GRIPPER_NEUTRAL)
            ok = self.move_joints('arm', ARM_FOLDED) and ok
            if not ok:
                self.get_logger().warning('Arm reset failed, continuing with robot and door')

        self.set_door_torque(CLOSING_TORQUE)
        try:
            if not self.teleport_robot(a.x, a.y, math.radians(a.yaw)):
                self.get_logger().error('Teleporting the robot failed')
                return False
            # Wait for the leaf to swing shut (the door may be up to ~110 deg open).
            deadline = time.monotonic() + CLOSE_TIMEOUT_S
            angle = None
            while time.monotonic() < deadline:
                time.sleep(1.0)
                angle = self.leaf_angle_deg()
                if angle is not None and abs(angle) < CLOSED_TOLERANCE_DEG:
                    break
            # Let it settle against the frame.
            time.sleep(1.0)
        finally:
            self.clear_door_torque()
        # Teleport again: the robot may have been nudged while the door swung shut.
        self.teleport_robot(a.x, a.y, math.radians(a.yaw))

        angle = self.leaf_angle_deg()
        if angle is None or abs(angle) >= CLOSED_TOLERANCE_DEG:
            self.get_logger().error(f'Door not closed, leaf angle {angle}')
            return False
        self.get_logger().info(
            f'Reset done: robot at ({a.x:.3f}, {a.y:.3f}, {a.yaw:.1f} deg), '
            f'door closed ({angle:.1f} deg)')
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--x', type=float, default=START_X)
    parser.add_argument('--y', type=float, default=START_Y)
    parser.add_argument('--yaw', type=float, default=START_YAW_DEG, help='yaw [deg]')
    parser.add_argument('--skip-arm', action='store_true', help='do not move the arm')
    parser.add_argument('--world', default='lu')
    parser.add_argument('--robot', default='arips')
    parser.add_argument('--door', default='door')
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = ResetDoorTest(args)
    try:
        ok = node.run()
    except KeyboardInterrupt:
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
