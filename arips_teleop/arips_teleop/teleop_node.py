from math import copysign

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.node import Node
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float64MultiArray

OWN_PARAMS = (
    'enable_button', 'command_topic', 'joint_states_topic', 'group_name', 'link_name',
    'ik_timeout', 'dt', 'joint_names', 'linear_axes', 'angular_axes', 'linear_scale', 'angular_scale',
    'use_sim_time',
)


def rotation_from_vector(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3)
    k = rotvec / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def quaternion_from_rotation(r):
    """Return (x, y, z, w)."""
    t = np.trace(r)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        return ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    q[3] = (r[k, j] - r[j, k]) / s
    return tuple(q)


class TeleopNode(Node):
    def __init__(self):
        super().__init__(
            'teleop_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.group_name = self._param('group_name', 'arm')
        self.link_name = self._param('link_name', 'link_3')
        self.ik_timeout = float(self._param('ik_timeout', 0.05))
        self.dt = float(self._param('dt', 0.02))
        self._param('enable_button', 4)
        self._param('linear_axes', [4, 3, 1])
        self._param('angular_axes', [-1, -1, 0])
        self._param('linear_scale', 0.05)
        self._param('angular_scale', 0.05)
        command_topic = self._param('command_topic', '/arm_forward_position_controller/commands')
        joint_states_topic = self._param('joint_states_topic', '/joint_states')

        # MoveIt settings (robot_description*, kinematics, ...) are passed as node parameters
        config = {
            name: param.value
            for name, param in self.get_parameters_by_prefix('').items()
            if name not in OWN_PARAMS
        }
        if not config.get('robot_description'):
            raise RuntimeError('parameter robot_description is required')

        self.moveit_py = MoveItPy(node_name='teleop_moveit_py', config_dict=config)
        robot_model = self.moveit_py.get_robot_model()
        group = robot_model.get_joint_model_group(self.group_name)
        if group is None:
            raise RuntimeError(f'unknown joint group {self.group_name}')
        self.group_joints = list(group.active_joint_model_names)
        self.joint_names = list(self._param('joint_names', self.group_joints))
        missing = [j for j in self.group_joints if j not in self.joint_names]
        if missing:
            raise RuntimeError(f'joint_names must contain all joints of the group: missing {missing}')
        self.latest = {}
        # Commanded joint targets, cached when the enable button is pressed
        self.cached = None
        self.state = RobotState(robot_model)
        self.have_joint_state = False

        self.pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.create_subscription(JointState, joint_states_topic, self.on_joint_state, 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.get_logger().info(
            f'Ready: group={self.group_name} link={self.link_name} joints={self.group_joints}')

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
        if self.cached is not None or not all(j in received for j in self.group_joints):
            return
        self.state.set_joint_group_positions(
            self.group_name, [received[j] for j in self.group_joints])
        self.state.update()
        self.have_joint_state = True

    def joy_callback(self, msg):
        button = self.get_parameter('enable_button').value
        if not (0 <= button < len(msg.buttons)) or not msg.buttons[button]:
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
            self.state.set_joint_group_positions(
                self.group_name, [self.cached[j] for j in self.group_joints])
            self.state.update()

        lin = self.get_parameter('linear_axes').value
        ang = self.get_parameter('angular_axes').value
        ls = self.get_parameter('linear_scale').value
        a_s = self.get_parameter('angular_scale').value
        linear = np.array([ls * self._axis(msg, i) for i in lin]) * self.dt
        angular = np.array([a_s * self._axis(msg, i) for i in ang]) * self.dt
        if not (linear.any() or angular.any()):
            return
        
        self.get_logger().info(f'Linear: {linear}, Angular: {angular}', throttle_duration_sec=0.5)

        # Delta expressed in the end effector frame: T_target = T_current * T_delta
        delta = np.eye(4)
        delta[:3, :3] = rotation_from_vector(angular)
        delta[:3, 3] = linear
        target = np.array(self.state.get_global_link_transform(self.link_name)) @ delta

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (float(v) for v in target[:3, 3])
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = (
            float(v) for v in quaternion_from_rotation(target[:3, :3]))
        if not self.state.set_from_ik(self.group_name, pose, self.link_name, self.ik_timeout):
            self.get_logger().warning('IK failed; no command sent', throttle_duration_sec=1.0)
            self.state.set_joint_group_positions(
                self.group_name, [self.cached[j] for j in self.group_joints])
            self.state.update()
            return
        self.state.update()
        

        out = Float64MultiArray()
        positions = dict(zip(
            self.group_joints, self.state.get_joint_group_positions(self.group_name)))
        self.get_logger().info(f'Joint4 moving from {self.cached["joint_4"]} to {positions["joint_4"]}', throttle_duration_sec=0.5)
        self.cached.update(positions)
        # Joints outside the planning group keep their position from button press
        out.data = [float(self.cached[j]) for j in self.joint_names]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
