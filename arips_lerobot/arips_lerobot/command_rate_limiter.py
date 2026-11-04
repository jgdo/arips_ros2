"""Moves the arm towards dummy command targets with limited speed.

Steps are applied to cached working positions instead of the (noisy, delayed) joint states.
The working positions are re-synced to the joint states after a break in the target stream
or when the joint states deviate too far from them.
"""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

DEFAULT_JOINTS = ['joint_slider', 'joint_2', 'joint_3', 'joint_4', 'joint_door_gripper']
DEFAULT_RESYNC_THRESHOLDS = [0.01, 0.05, 0.05, 0.05, 0.05]


class CommandRateLimiter(Node):
    def __init__(self):
        super().__init__('command_rate_limiter')
        self.joint_names = list(self.declare_parameter('joint_names', DEFAULT_JOINTS).value)
        self.max_speed = float(self.declare_parameter('max_speed', 0.2).value)
        self.rate = float(self.declare_parameter('rate', 25.0).value)
        self.timeout = Duration(seconds=float(self.declare_parameter('timeout', 0.5).value))
        self.resync_break = Duration(
            seconds=float(self.declare_parameter('resync_break', 1.0).value))
        self.resync_thresholds = [float(v) for v in self.declare_parameter(
            'resync_thresholds', DEFAULT_RESYNC_THRESHOLDS).value]
        if len(self.resync_thresholds) != len(self.joint_names):
            raise RuntimeError(
                f'resync_thresholds has {len(self.resync_thresholds)} values, '
                f'expected {len(self.joint_names)}')
        target_topic = self.declare_parameter(
            'target_topic', '/arm_forward_position_controller/dummy_commands').value
        command_topic = self.declare_parameter(
            'command_topic', '/arm_forward_position_controller/commands').value
        joint_states_topic = self.declare_parameter('joint_states_topic', '/joint_states').value

        self.current = None
        self.working = None
        self.target = None
        self.target_stamp = None
        self.timed_out = True

        self.pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.create_subscription(JointState, joint_states_topic, self.on_joint_state, 10)
        self.create_subscription(Float64MultiArray, target_topic, self.on_target, 10)
        self.create_timer(1.0 / self.rate, self.on_timer)
        self.get_logger().info(
            f'Ready: joints={self.joint_names} max_speed={self.max_speed} rate={self.rate} '
            f'resync_thresholds={self.resync_thresholds}')

    def on_joint_state(self, msg):
        received = dict(zip(msg.name, msg.position))
        if all(j in received for j in self.joint_names):
            self.current = [received[j] for j in self.joint_names]

    def on_target(self, msg):
        if len(msg.data) != len(self.joint_names):
            self.get_logger().warning(
                f'Ignoring target with {len(msg.data)} values, expected {len(self.joint_names)}',
                throttle_duration_sec=1.0)
            return
        now = self.get_clock().now()
        if self.target_stamp is None or now - self.target_stamp > self.resync_break:
            # start from the measured joint positions on the next timer tick
            self.working = None
        self.target = list(msg.data)
        self.target_stamp = now

    def on_timer(self):
        if self.target_stamp is None or self.current is None:
            return
        if self.get_clock().now() - self.target_stamp > self.timeout:
            if not self.timed_out:
                self.get_logger().info('Target timeout, stopped publishing commands')
                self.timed_out = True
            return
        if self.timed_out:
            self.get_logger().info('Receiving targets, publishing commands')
            self.timed_out = False

        if self.working is None:
            self.working = list(self.current)
        else:
            deviated = [
                name for name, w, c, th in zip(
                    self.joint_names, self.working, self.current, self.resync_thresholds)
                if abs(w - c) > th]
            if deviated:
                self.get_logger().info(
                    f'Joint states deviate from working positions ({deviated}), re-syncing',
                    throttle_duration_sec=1.0)
                self.working = list(self.current)

        diffs = [t - w for t, w in zip(self.target, self.working)]
        max_diff = max(abs(d) for d in diffs)
        max_step = self.max_speed / self.rate
        scale = 1.0 if max_diff <= max_step else max_step / max_diff
        self.working = [w + scale * d for w, d in zip(self.working, diffs)]

        out = Float64MultiArray()
        out.data = [float(w) for w in self.working]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CommandRateLimiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
