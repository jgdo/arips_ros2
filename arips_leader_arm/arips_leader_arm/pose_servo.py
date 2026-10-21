"""Pose servo: solves IK (moveit_py) for each received PoseStamped and commands joint positions."""

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

OWN_PARAMS = (
    'pose_topic', 'command_topic', 'joint_states_topic', 'group_name', 'link_name',
    'ik_timeout', 'joint_names', 'use_sim_time',
)


class PoseServo(Node):
    def __init__(self):
        super().__init__(
            'pose_servo',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.group_name = self._param('group_name', 'arm')
        self.link_name = self._param('link_name', 'door_gripper')
        self.ik_timeout = float(self._param('ik_timeout', 0.05))
        pose_topic = self._param('pose_topic', '~/pose_target_cmds')
        command_topic = self._param('command_topic', '/arm_forward_position_controller/commands')
        joint_states_topic = self._param('joint_states_topic', '/joint_states')
        joint_names = list(self._param('joint_names', ['']))

        # MoveIt settings (robot_description*, kinematics, planning pipelines) are passed
        # as node parameters, same as for move_group
        config = {
            name: param.value
            for name, param in self.get_parameters_by_prefix('').items()
            if name not in OWN_PARAMS
        }
        if not config.get('robot_description'):
            raise RuntimeError('parameter robot_description is required')

        self.moveit_py = MoveItPy(node_name='pose_servo_moveit_py', config_dict=config)
        self.robot_model = self.moveit_py.get_robot_model()
        self.root_frame = self.robot_model.model_frame
        group = self.robot_model.get_joint_model_group(self.group_name)
        if group is None:
            raise RuntimeError(f'unknown joint group {self.group_name}')
        self.joint_names = [j for j in joint_names if j] or list(group.active_joint_model_names)

        self.group_joints = list(group.active_joint_model_names)
        self.state = RobotState(self.robot_model)
        self.have_joint_state = False

        self.pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.create_subscription(JointState, joint_states_topic, self.on_joint_state, 10)
        self.create_subscription(PoseStamped, pose_topic, self.on_pose, 10)
        self.get_logger().info(
            f'Ready: group={self.group_name} link={self.link_name} root={self.root_frame} '
            f'joints={self.joint_names}')

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def on_joint_state(self, msg):
        received = dict(zip(msg.name, msg.position))
        if not all(j in received for j in self.group_joints):
            return
        self.state.set_joint_group_positions(
            self.group_name, [received[j] for j in self.group_joints])
        self.state.update()
        self.have_joint_state = True

    def on_pose(self, msg):
        frame = msg.header.frame_id.lstrip('/')
        if frame != self.root_frame:
            self.get_logger().warning(
                f"Rejecting pose: frame_id '{frame}' is not the robot root link "
                f"'{self.root_frame}'")
            return
        if not self.have_joint_state:
            self.get_logger().warning('Rejecting pose: no joint state received yet')
            return

        pose = Pose()
        pose.position = msg.pose.position
        pose.orientation = msg.pose.orientation
        if not self.state.set_from_ik(self.group_name, pose, self.link_name, self.ik_timeout):
            self.get_logger().warning('IK failed for received pose; no command sent')
            return
        self.state.update()
        positions = dict(zip(
            self.group_joints, self.state.get_joint_group_positions(self.group_name)))
        out = Float64MultiArray()
        out.data = [float(positions[j]) for j in self.joint_names]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PoseServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
