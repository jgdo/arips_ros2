import cv2

import message_filters
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32, PolygonStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .detector import DoorHandleDetector, installed_model_path
from .projection import CameraProjector, project_ray_to_plane


class StepDetectorNode(Node):
    BASE_FRAME = 'arips_wheel_center'
    FLOOR_HEIGHT = 0.0
    MODEL_WIDTH = 384
    MODEL_HEIGHT = 288
    IMAGE_TOPIC = '/kinect/rgb/image_raw'
    CAMERA_INFO_TOPIC = '/kinect/rgb/camera_info'
    ANNOTATED_IMAGE_TOPIC = '/kinect/rgb/step_detector_image'

    def __init__(self):
        super().__init__('step_detector')
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
            installed_model_path('floor_door_edge.onnx'))

        self._image_publisher = self.create_publisher(
            Image,
            self._annotated_image_topic,
            1,
        )
        self._step_publisher = self.create_publisher(
            PolygonStamped,
            'door_step_polygon',
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
        self._camera_projector.from_camera_info(info_msg)
        x_factor = image_msg.width / float(self.MODEL_WIDTH)
        y_factor = image_msg.height / float(self.MODEL_HEIGHT)

        try:
            source_image = self._bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            image = cv2.resize(
                source_image,
                (self.MODEL_WIDTH, self.MODEL_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
        except CvBridgeError as error:
            self.get_logger().error(f'Failed to convert image: {error}')
            return

        floor_edge = self._detector.detect(image)
        DoorHandleDetector.annotate_detected(image, floor_edge)
        self._publish_image(image, image_msg)

        margin = 5
        points = [
            self._calc_3d_pose(
                margin,
                margin,
                image_msg.header.frame_id,
            ),
            self._calc_3d_pose(
                margin,
                image_msg.height - margin,
                image_msg.header.frame_id,
            ),
            self._calc_3d_pose(
                image_msg.width - margin,
                image_msg.height - margin,
                image_msg.header.frame_id,
            ),
            self._calc_3d_pose(
                image_msg.width - margin,
                margin,
                image_msg.header.frame_id,
            ),
        ]
        if any(point is None for point in points):
            return

        if floor_edge is not None:
            points.extend([
                self._calc_3d_pose(
                    int(floor_edge.handle_start.x * x_factor),
                    int(floor_edge.handle_start.y * y_factor),
                    image_msg.header.frame_id,
                ),
                self._calc_3d_pose(
                    int(floor_edge.handle_end.x * x_factor),
                    int(floor_edge.handle_end.y * y_factor),
                    image_msg.header.frame_id,
                ),
            ])
        if any(point is None for point in points):
            return

        polygon = PolygonStamped()
        polygon.header.frame_id = self._base_frame
        polygon.header.stamp = image_msg.header.stamp
        polygon.polygon.points = [
            Point32(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            for point in points
        ]
        self._step_publisher.publish(polygon)

    def _calc_3d_pose(self, x: int, y: int, image_frame_id: str):
        ray_cam = self._camera_projector.project_pixel_to_3d_ray(x, y)
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame,
                image_frame_id,
                Time(),
            )
        except TransformException as error:
            self.get_logger().warning(f'Door detector TF error: {error}')
            return None

        point = project_ray_to_plane(
            ray_cam,
            transform,
            self.FLOOR_HEIGHT,
            max_ray_z=-0.3,
        )
        if point is None:
            self.get_logger().warning(
                'Camera seems not to point down, could not compute floor step '
                'pose')
        return point

    def _publish_image(self, image, source_message: Image) -> None:
        output_message = self._bridge.cv2_to_imgmsg(image, encoding='bgr8')
        output_message.header = source_message.header
        self._image_publisher.publish(output_message)


def main(args=None):
    rclpy.init(args=args)
    node = StepDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass