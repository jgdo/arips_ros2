from math import copysign, cos, sin

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float64MultiArray

from arips_teleop.scara_kinematics import IKError, ScaraKinematics


class TeleopNode(Node):
    def __init__(self):
        super().__init__(
            'arips_teleop',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.link_name = self._param('link_name', 'link_3')
        self.command_rate = float(self._param('command_rate', 50.0))
        self.joy_timeout = float(self._param('joy_timeout', 0.25))
        if self.command_rate <= 0:
            raise RuntimeError('command_rate must be positive')
        self.dt = 1.0 / self.command_rate
        self.last_joy = None
        self.last_joy_time = None
        self._param('enable_button', 4)
        self._param('linear_axes', [4, 3, 1])
        self._param('angular_axes', [-1, -1, 0])
        self._param('linear_scale', 0.05)
        self._param('angular_scale', 0.05)
        command_topic = self._param('command_topic', '/arm_forward_position_controller/commands')
        joint_states_topic = self._param('joint_states_topic', '/joint_states')

        robot_description = self._param('robot_description', '')
        if not robot_description:
            raise RuntimeError('parameter robot_description is required')
        # Raises KinematicsError (aborting the node) if the URDF is not the expected SCARA arm
        self.kinematics = ScaraKinematics(robot_description, tip_link=self.link_name)
        self.group_joints = list(self.kinematics.joint_names)
        self.joint_names = list(self._param('joint_names', []))
        missing = [j for j in self.group_joints if j not in self.joint_names]
        if missing:
            raise RuntimeError(f'joint_names must contain all arm joints: missing {missing}')
        self.latest = {}
        # Commanded joint targets, cached when the enable button is pressed
        self.cached = None
        self.have_joint_state = False

        self.pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.create_subscription(JointState, joint_states_topic, self.on_joint_state, 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.create_timer(self.dt, self.update)
        self.get_logger().info(
            f'Ready: link={self.link_name} joints={self.group_joints}')

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    @staticmethod
    def _axis(msg, idx):
        return float(msg.axes[abs(idx)]) * copysign(1, idx) if 0 <= abs(idx) < len(msg.axes) else 0.0

    def on_joint_state(self, msg):
        received = dict(zip(msg.name, msg.position))
        self.latest.update((j, received[j]) for j in self.joint_names if j in received)
        if all(j in received for j in self.group_joints):
            self.have_joint_state = True

    def joy_callback(self, msg):
        self.last_joy = msg
        self.last_joy_time = self.get_clock().now()
        if not self._enabled(msg):
            self.cached = None

    def _enabled(self, msg):
        button = self.get_parameter('enable_button').value
        return 0 <= button < len(msg.buttons) and bool(msg.buttons[button])

    def update(self):
        msg = self.last_joy
        if msg is None or not self._enabled(msg):
            return
        age = (self.get_clock().now() - self.last_joy_time).nanoseconds * 1e-9
        if age > self.joy_timeout:
            self.get_logger().warning('Joy timeout; arm stopped', throttle_duration_sec=2.0)
            # Re-sync from measured positions when joy messages resume
            self.cached = None
            return
        if not self.have_joint_state:
            self.get_logger().warning('No joint state received yet', throttle_duration_sec=2.0)
            return
        if any(j not in self.latest for j in self.joint_names):
            self.get_logger().warning(
                'Waiting for joint states of all joint_names', throttle_duration_sec=2.0)
            return
        if self.cached is None:
            # Button just pressed: start from the measured position, then integrate commands
            # on top of the cached target instead of the noisy measurements
            self.cached = dict(self.latest)

        lin = self.get_parameter('linear_axes').value
        ang = self.get_parameter('angular_axes').value
        ls = self.get_parameter('linear_scale').value
        a_s = self.get_parameter('angular_scale').value
        linear = [ls * self._axis(msg, i) * self.dt for i in lin]
        # Only rotation about z (yaw) is possible for the SCARA arm
        yaw_delta = a_s * self._axis(msg, ang[2]) * self.dt
        pose_commanded = bool(any(linear) or yaw_delta)
        gripper_delta = (msg.axes[2]-msg.axes[5]) * a_s * self.dt
        if not pose_commanded and not gripper_delta:
            return

        if pose_commanded:
            self.get_logger().info(
                f'Linear: {linear}, Yaw: {yaw_delta}', throttle_duration_sec=0.5)

            x, y, z, yaw = self.kinematics.fk(self.cached)
            # Linear delta is expressed in the tip link frame, z is parallel to base z
            dx = linear[0] * cos(yaw) - linear[1] * sin(yaw)
            dy = linear[0] * sin(yaw) + linear[1] * cos(yaw)
            try:
                positions = self.kinematics.ik(
                    x + dx, y + dy, z + linear[2], yaw + yaw_delta)
            except IKError as exc:
                self.get_logger().warning(f'IK failed ({exc}); pose command not sent',
                                          throttle_duration_sec=1.0)
                pose_commanded = False
            else:
                self.cached.update(positions)

        if gripper_delta:
            self.get_logger().info(
                                f'gripper moving  {self.cached["joint_door_gripper"]} with delta {gripper_delta}', throttle_duration_sec=0.5)
            self.cached['joint_door_gripper'] += gripper_delta

        if not pose_commanded and not gripper_delta:
            return

        out = Float64MultiArray()
        # Joints outside the arm keep their position from button press
        out.data = [float(self.cached[j]) for j in self.joint_names]

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
