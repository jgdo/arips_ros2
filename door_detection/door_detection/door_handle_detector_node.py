import math

import message_filters
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .detector import DoorHandleDetector, installed_model_path
from .projection import CameraProjector, project_ray_to_plane


class DoorHandleDetectorNode(Node):
    BASE_FRAME = 'arips_base'
    HANDLE_HEIGHT = 1.04
    IMAGE_TOPIC = '/kinect/rgb/image_color'
    CAMERA_INFO_TOPIC = '/kinect/rgb/camera_info'
    ANNOTATED_IMAGE_TOPIC = '/kinect/rgb/door_handle_image'

    def __init__(self):
        super().__init__('door_handle_detector')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('base_frame', self.BASE_FRAME),
                ('image_topic', self.IMAGE_TOPIC),
                ('camera_info_topic', self.CAMERA_INFO_TOPIC),
                ('annotated_image_topic', self.ANNOTATED_IMAGE_TOPIC),
            ],
        )
        self._base_frame = self.get_parameter('base_frame').value
        self._image_topic = self.get_parameter('image_topic').value
        self._camera_info_topic = self.get_parameter('camera_info_topic').value
        self._annotated_image_topic = self.get_parameter(
            'annotated_image_topic').value
        self._bridge = CvBridge()
        self._camera_projector = CameraProjector()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._detector = DoorHandleDetector(
            installed_model_path('door_handle.onnx'))

        self._image_publisher = self.create_publisher(
            Image,
            self._annotated_image_topic,
            1,
        )
        self._pose_publisher = self.create_publisher(
            PoseStamped,
            'door_handle/pose',
            1,
        )
        image_subscriber = message_filters.Subscriber(
            self,
            Image,
            self._image_topic,
            qos_profile=qos_profile_sensor_data,
        )
        camera_info_subscriber = message_filters.Subscriber(
            self,
            CameraInfo,
            self._camera_info_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_subscriber, camera_info_subscriber],
            queue_size=30,
            slop=0.1,
        )
        self._synchronizer.registerCallback(self.image_callback)

    def image_callback(self, image_msg: Image, info_msg: CameraInfo) -> None:
        if image_msg.width != 320 or image_msg.height != 240:
            self.get_logger().warning(
                'Wrong image size for door handle detection, only 320x240 '
                'supported')
            return

        self._camera_projector.from_camera_info(info_msg)
        try:
            image = self._bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Failed to convert image: {error}')
            return

        door_handle = self._detector.detect(image)
        DoorHandleDetector.annotate_detected(image, door_handle)
        self._publish_image(image, image_msg)

        if door_handle is None:
            return

        handle_start = self._calc_3d_pose(
            door_handle.handle_start.x,
            door_handle.handle_start.y,
            image_msg.header.frame_id,
        )
        handle_end = self._calc_3d_pose(
            door_handle.handle_end.x,
            door_handle.handle_end.y,
            image_msg.header.frame_id,
        )
        if handle_start is None or handle_end is None:
            return

        handle_direction = handle_end - handle_start
        handle_angle = math.atan2(-handle_direction[0], handle_direction[1])

        pose = PoseStamped()
        pose.header.frame_id = self._base_frame
        pose.header.stamp = image_msg.header.stamp
        pose.pose.position.x = float(handle_end[0])
        pose.pose.position.y = float(handle_end[1])
        pose.pose.position.z = float(handle_end[2])
        pose.pose.orientation.z = math.sin(handle_angle / 2.0)
        pose.pose.orientation.w = 1.0
        self._pose_publisher.publish(pose)

    def _calc_3d_pose(self, x: int, y: int, image_frame_id: str):
        ray_cam = self._camera_projector.project_pixel_to_3d_ray(x, y)
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                image_frame_id,
                Time(),
            )
        except TransformException as error:
            self.get_logger().warning(f'Door handle detector TF error: {error}')
            return None

        point = project_ray_to_plane(
            ray_cam,
            transform,
            self.HANDLE_HEIGHT,
            min_ray_z=0.3,
        )
        if point is None:
            self.get_logger().warning(
                'Camera seems not to point up, could not compute door handle '
                'pose')
        return point

    def _publish_image(self, image, source_message: Image) -> None:
        output_message = self._bridge.cv2_to_imgmsg(image, encoding='bgr8')
        output_message.header = source_message.header
        self._image_publisher.publish(output_message)


def main(args=None):
    rclpy.init(args=args)
    node = DoorHandleDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()