"""Leader/follower teleoperation: maps leader EE motion to follower EE pose targets."""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q."""
    x, y, z, w = quat_mul(quat_mul(q, (v[0], v[1], v[2], 0.0)), quat_conj(q))
    return (x, y, z)


class Pose:
    """Rigid transform: position tuple and unit quaternion (x, y, z, w)."""

    def __init__(self, p, q):
        self.p = p
        self.q = q

    @staticmethod
    def from_transform(t):
        tr, r = t.transform.translation, t.transform.rotation
        return Pose((tr.x, tr.y, tr.z), (r.x, r.y, r.z, r.w))

    def inverse(self):
        qi = quat_conj(self.q)
        pr = quat_rotate(qi, self.p)
        return Pose((-pr[0], -pr[1], -pr[2]), qi)

    def __mul__(self, other):
        pr = quat_rotate(self.q, other.p)
        p = (self.p[0] + pr[0], self.p[1] + pr[1], self.p[2] + pr[2])
        return Pose(p, quat_mul(self.q, other.q))


class LeaderTeleop(Node):
    def __init__(self):
        super().__init__('leader_teleop')

        self.declare_parameter('follower_base_frame', 'base_link')
        self.declare_parameter('follower_ee_frame', 'door_gripper')
        self.declare_parameter('leader_base_frame', 'leader_base_link')
        self.declare_parameter('leader_ee_frame', 'leader_ee_link')
        self.declare_parameter('scale', 1.0)
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('activate_topic', 'activate_leader_teleop')
        self.declare_parameter('pose_topic', 'leader_teleop/target_pose')

        self.follower_base = self.get_parameter('follower_base_frame').value
        self.follower_ee = self.get_parameter('follower_ee_frame').value
        self.leader_base = self.get_parameter('leader_base_frame').value
        self.leader_ee = self.get_parameter('leader_ee_frame').value

        self.active = False
        self.deadline = None
        self.follower_offset = None
        self.leader_offset = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, 10)
        self.create_subscription(
            Float32, self.get_parameter('activate_topic').value, self.on_activate, 10)
        self.create_timer(1.0 / self.get_parameter('publish_rate').value, self.on_timer)

    def on_activate(self, msg):
        if msg.data <= 0.0:
            self.deadline = None
            return
        self.deadline = self.get_clock().now() + Duration(seconds=float(msg.data))

    def lookup(self, base, ee):
        try:
            return Pose.from_transform(self.tf_buffer.lookup_transform(base, ee, Time()))
        except TransformException as ex:
            self.get_logger().warning(
                f'No transform {base} -> {ee}: {ex}', throttle_duration_sec=2.0)
            return None

    def set_idle(self):
        if self.active:
            self.get_logger().info('Teleoperation idle')
        self.active = False
        self.deadline = None

    def on_timer(self):
        if self.deadline is None or self.get_clock().now() >= self.deadline:
            self.set_idle()
            return

        follower = self.lookup(self.follower_base, self.follower_ee)
        leader = self.lookup(self.leader_base, self.leader_ee)
        if follower is None or leader is None:
            return

        if not self.active:
            self.follower_offset = follower
            self.leader_offset = leader
            self.active = True
            self.get_logger().info('Teleoperation active')

        # Leader motion since activation, expressed in the leader base frame
        scale = self.get_parameter('scale').value
        lo = self.leader_offset
        dp = tuple((leader.p[i] - lo.p[i]) * scale for i in range(3))
        dq = quat_mul(leader.q, quat_conj(lo.q))
        fo = self.follower_offset
        target = Pose(
            (fo.p[0] + dp[0], fo.p[1] + dp[1], fo.p[2] + dp[2]),
            quat_mul(dq, fo.q))

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.follower_base
        out.pose.position.x, out.pose.position.y, out.pose.position.z = target.p
        (out.pose.orientation.x, out.pose.orientation.y,
         out.pose.orientation.z, out.pose.orientation.w) = target.q
        self.pose_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LeaderTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
