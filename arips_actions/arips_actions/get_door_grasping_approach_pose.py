"""Service that computes the approach pose for grasping a door handle.

Geometry (all in the map frame):

    extent                                   pivot
      E ------------------------------------- P
                 |  door_dist
                 x  <- approach position
      |<-------->|<-------------------------->|
                          pivot_dist

The approach position is located pivot_dist from the pivot along the closed door
line (towards the extent point) and door_dist perpendicular to that line, on the side
of the door where the robot (base frame) currently is. The orientation is chosen so
that the pose's x axis always faces towards the door line, while the y axis lies
exactly on the line through the pivot point: it points away from the pivot on one
side of the door and towards the pivot on the mirrored side.

Services:
  get_door_grasping_approach_pose          - door given explicitly in the request
  get_door_grasping_approach_pose_by_index - door looked up by index in the last
                                             received semantic map
Every computed pose is also published on the 'pose_topic' topic.
"""

import math

import rclpy
from arips_action_msgs.srv import (
    GetDoorGraspingApproachPose,
    GetDoorGraspingApproachPoseByIndex,
)
from arips_semantic_map_msgs.msg import SemanticMap
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def compute_approach_pose(pivot, extent, robot, door_dist, pivot_dist):
    """Return (x, y, yaw) of the approach pose, or None if the door is degenerate.

    pivot, extent and robot are (x, y) tuples in the same frame.
    """
    dx = extent[0] - pivot[0]
    dy = extent[1] - pivot[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None

    # Unit vector along the door line, from pivot towards extent, and its left normal.
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    # Mirror the approach position to the side of the door the robot is on.
    side = (robot[0] - pivot[0]) * nx + (robot[1] - pivot[1]) * ny
    sign = -1.0 if side < 0.0 else 1.0

    x = pivot[0] + pivot_dist * ux + sign * door_dist * nx
    y = pivot[1] + pivot_dist * uy + sign * door_dist * ny

    # The y axis lies on the line through pivot and approach position. Start with y
    # pointing away from the pivot (x = y rotated by -90 deg) ...
    y_axis_angle = math.atan2(y - pivot[1], x - pivot[0])
    yaw = y_axis_angle - math.pi / 2.0
    # ... and flip by 180 deg (y then points to the pivot) if x does not face the door.
    # Direction from the approach position towards the door line is -sign * n.
    to_door = -sign * (math.cos(yaw) * nx + math.sin(yaw) * ny)
    if to_door < 0.0:
        yaw += math.pi
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return x, y, yaw


class GetDoorGraspingApproachPoseNode(Node):
    def __init__(self):
        super().__init__('get_door_grasping_approach_pose')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('base_frame', 'base_link'),
                # Perpendicular distance of the approach position from the door line.
                ('door_dist', 0.25),
                # Distance from the pivot along the door line (towards the extent point).
                ('pivot_dist', 0.7),
                ('tf_timeout_s', 0.5),
                ('pose_topic', 'door_grasping_approach_pose'),
                ('semantic_map_topic', 'semantic_map'),
            ],
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Every successfully computed approach pose (from either service) is published here.
        self._pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, 10)

        # The semantic map server publishes latched (transient local), so match that.
        self._semantic_map = None
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            SemanticMap,
            self.get_parameter('semantic_map_topic').value,
            self._on_semantic_map,
            map_qos,
        )

        self.create_service(
            GetDoorGraspingApproachPose,
            'get_door_grasping_approach_pose',
            self._handle_request,
        )
        self.create_service(
            GetDoorGraspingApproachPoseByIndex,
            'get_door_grasping_approach_pose_by_index',
            self._handle_index_request,
        )
        self.get_logger().info('get_door_grasping_approach_pose services ready')

    def _on_semantic_map(self, msg):
        self._semantic_map = msg

    def _handle_request(self, request, response):
        return self._compute(request.door, response)

    def _handle_index_request(self, request, response):
        semantic_map = self._semantic_map
        if semantic_map is None:
            return self._fail(response, 'No semantic map received yet')
        num_doors = len(semantic_map.doors)
        if not 0 <= request.door_index < num_doors:
            return self._fail(
                response,
                f'Door index {request.door_index} out of range (map has {num_doors} doors)')
        return self._compute(semantic_map.doors[request.door_index], response)

    def _compute(self, door, response):
        map_frame = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value
        door_dist = self.get_parameter('door_dist').value
        pivot_dist = self.get_parameter('pivot_dist').value
        tf_timeout = self.get_parameter('tf_timeout_s').value

        try:
            transform = self._tf_buffer.lookup_transform(
                map_frame, base_frame, Time(), timeout=Duration(seconds=tf_timeout))
        except TransformException as error:
            return self._fail(response, f'TF lookup {map_frame} -> {base_frame} failed: {error}')

        robot = (transform.transform.translation.x, transform.transform.translation.y)
        result = compute_approach_pose(
            (door.pivot.x, door.pivot.y), (door.extent.x, door.extent.y),
            robot, door_dist, pivot_dist)
        if result is None:
            return self._fail(response, 'Door pivot and extent coincide')
        x, y, yaw = result

        pose = PoseStamped()
        pose.header.frame_id = map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        response.success = True
        response.error_message = ''
        response.pose = pose
        self._pose_pub.publish(pose)
        self.get_logger().info(
            f'Approach pose: x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f} deg '
            f'(robot at {robot[0]:.3f}, {robot[1]:.3f})')
        return response

    def _fail(self, response, error_message):
        response.success = False
        response.error_message = error_message
        self.get_logger().error(error_message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GetDoorGraspingApproachPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
