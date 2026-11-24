"""Generic keypoint detector node for models trained with train_keypoints / exported with
export_keypoints_onnx.

Parameters:
  model        path to the exported .onnx (metadata .json next to it)
  rgb_topic    sensor_msgs/Image, or CompressedImage if the topic ends with /compressed
  depth_topic  sensor_msgs/Image (16UC1 mm / 32FC1 m), or CompressedImage if it ends with /compressedDepth
  threshold    visibility threshold on the heatmap peak (< 0: use the value from the model metadata)
  max_rate     maximum processing rate in Hz (0 = every frame)
  sync_slop    max rgb/depth stamp difference in seconds when rgb and depth are both used
  use_depth    also subscribe to depth for RGB-only models, for Keypoint.depth / position (default true);
               ignored in the fixed height mode
  height_mode  how the 3D position of a keypoint is found: 'depth' (default) = depth image at the keypoint,
               'fixed' = the keypoint is assumed to lie at height fixed_height (z) in fixed_height_frame;
               its position is where the camera ray through the keypoint pixel crosses that plane (needs
               camera_info and TF, no depth image). Keypoint.depth is then the z of that point in the
               camera optical frame.
  fixed_height       keypoint z in fixed_height_frame for height_mode 'fixed' [m]. Default 1.053 = the
               door handle keypoints of the Gazebo lu world in base_link (handle_start as measured
               with depth, 8.5 mm above the spindle axis)
  fixed_height_frame reference frame of fixed_height, z up (default base_link)
  depth_radius depth = median of the valid pixels in a (2 * depth_radius + 1)^2 window (default 3 -> 7x7)
  camera_info_topic  sensor_msgs/CameraInfo of the image the keypoints are in (rgb, or depth for
               depth-only models); empty = <image topic namespace>/camera_info, e.g. /kinect/rgb/camera_info
  marker_size  diameter of the rviz marker spheres in metres
  marker_latest_tf  stamp markers with time 0 so rviz uses the latest TF instead of the image stamp, which
               avoids TF extrapolation errors in rviz (its MarkerArray display doesn't wait for TF; default true)

The depth image must be registered to the rgb image (same pixel = same point). Publishes:
  ~/keypoints        arips_keypoint_msgs/KeypointArray (all keypoints, see Keypoint.visible)
  ~/markers          visualization_msgs/MarkerArray: one sphere per keypoint with a 3D position, colored
                     per keypoint, color saturation = confidence
  ~/image_annotated  sensor_msgs/Image, only computed while somebody subscribes
"""

import colorsys
import time

import cv2
import message_filters
import numpy as np
import rclpy
from arips_keypoint_msgs.msg import Keypoint, KeypointArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point
from rclpy.time import Time as RclpyTime
from tf2_ros import Buffer, TransformException, TransformListener
from builtin_interfaces.msg import Time
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray

from . import dataset_io as dio
from .learning import viz
from .learning.inference import KeypointDetector


def depth_at(depth_m, x, y, ref_size, radius=3):
    """Median of the valid depth in a small window around reference-frame point (x, y)."""
    h, w = depth_m.shape
    u = int(x * w / ref_size[0])
    v = int(y * h / ref_size[1])
    if not (0 <= u < w and 0 <= v < h):
        return None
    win = depth_m[max(v - radius, 0):v + radius + 1, max(u - radius, 0):u + radius + 1]
    valid = win[win > 0]
    return float(np.median(valid)) if valid.size else None


def info_topic_for(image_topic: str) -> str:
    """image_transport convention: /ns/image_raw/compressed -> /ns/camera_info."""
    t = image_topic.rstrip('/')
    for suffix in ('/compressed', '/compressedDepth'):
        if t.endswith(suffix):
            t = t[:-len(suffix)]
    return t.rsplit('/', 1)[0] + '/camera_info'


def project_to_3d(info: CameraInfo, x, y, z, ref_size):
    """Reference-frame corner pixel (x, y) + depth z -> 3D point in the camera optical frame.

    Uses the projection matrix P (rectified image), or K if P is not set.
    """
    w, h = (info.width, info.height) if info.width and info.height else ref_size
    # corner convention in the reference frame -> pixel-center convention of the camera_info image
    u = x * w / ref_size[0] - 0.5
    v = y * h / ref_size[1] - 0.5
    p = info.p
    if p[0] > 0 and p[5] > 0:
        fx, fy, cx, cy, tx, ty = p[0], p[5], p[2], p[6], p[3], p[7]
    else:
        k = info.k
        fx, fy, cx, cy, tx, ty = k[0], k[4], k[2], k[5], 0.0, 0.0
    if fx <= 0 or fy <= 0:
        return None
    return Point(x=float(((u - cx) * z - tx) / fx), y=float(((v - cy) * z - ty) / fy), z=float(z))


def _rotation_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pixel_ray(info: CameraInfo, x, y, ref_size):
    """Reference-frame corner pixel (x, y) -> ray direction (z = 1) in the camera optical frame."""
    p = project_to_3d(info, x, y, 1.0, ref_size)
    return None if p is None else np.array([p.x, p.y, p.z])


def intersect_height(tf, ray, height):
    """Intersect the camera ray (optical frame, from the camera origin) with the plane z = height of
    the reference frame. tf transforms optical frame -> reference frame. Returns the point in the
    optical frame, or None if the ray does not hit the plane in front of the camera."""
    rot = _rotation_matrix(tf.transform.rotation)
    t = tf.transform.translation
    origin_z = t.z
    direction = rot @ ray
    if abs(direction[2]) < 1e-9:
        return None
    s = (height - origin_z) / direction[2]
    if s <= 0.0:
        return None
    return s * ray


def confidence_color(k: int, confidence: float):
    """Keypoint color (same as the annotated image) with saturation = confidence, as RGBA floats."""
    r, g, b = (c / 255.0 for c in viz.color(k))
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(hue, sat * float(np.clip(confidence, 0.0, 1.0)), val)


class KeypointDetectorNode(Node):

    def __init__(self):
        super().__init__('keypoint_detector_node')
        self.declare_parameter('model', '')
        self.declare_parameter('rgb_topic', '/kinect/rgb/image_raw/compressed')
        self.declare_parameter('depth_topic', '/kinect/depth/image_raw/compressedDepth')
        self.declare_parameter('threshold', -1.0)
        self.declare_parameter('max_rate', 10.0)
        self.declare_parameter('sync_slop', 0.05)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('depth_radius', 3)
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('marker_size', 0.03)
        self.declare_parameter('marker_latest_tf', True)
        self.declare_parameter('height_mode', 'depth')
        self.declare_parameter('fixed_height', 1.053)
        self.declare_parameter('fixed_height_frame', 'base_link')

        model_path = self.get_parameter('model').value
        if not model_path:
            raise RuntimeError('parameter "model" (path to .onnx) is required')
        self.detector = KeypointDetector(model_path)
        threshold = self.get_parameter('threshold').value
        self.threshold = threshold if threshold >= 0 else self.detector.threshold
        max_rate = self.get_parameter('max_rate').value
        self.min_period = 1.0 / max_rate if max_rate > 0 else 0.0
        self.last_time = 0.0
        self.meta = self.detector.meta
        self.ref_size = tuple(self.meta['ref_size'])
        self.depth_radius = int(self.get_parameter('depth_radius').value)
        self.marker_size = float(self.get_parameter('marker_size').value)
        self.marker_latest_tf = bool(self.get_parameter('marker_latest_tf').value)
        self.camera_info = None
        self.height_mode = self.get_parameter('height_mode').value
        if self.height_mode not in ('depth', 'fixed'):
            raise RuntimeError(f'parameter "height_mode" must be depth or fixed, not {self.height_mode!r}')
        self.fixed_height = float(self.get_parameter('fixed_height').value)
        self.fixed_height_frame = self.get_parameter('fixed_height_frame').value
        if self.height_mode == 'fixed':
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        self.kp_pub = self.create_publisher(KeypointArray, '~/keypoints', 10)
        self.image_pub = self.create_publisher(Image, '~/image_annotated', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', 1)

        inputs = self.detector.inputs
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.rgb_compressed = rgb_topic.endswith('/compressed')
        self.depth_compressed = depth_topic.endswith('/compressedDepth')
        rgb_type = CompressedImage if self.rgb_compressed else Image
        depth_type = CompressedImage if self.depth_compressed else Image

        use_rgb = 'rgb' in inputs
        use_depth = 'depth' in inputs or (
            self.height_mode == 'depth' and bool(self.get_parameter('use_depth').value))
        if use_rgb and use_depth:
            subs = [message_filters.Subscriber(self, rgb_type, rgb_topic, qos_profile=qos_profile_sensor_data),
                    message_filters.Subscriber(self, depth_type, depth_topic, qos_profile=qos_profile_sensor_data)]
            self.sync = message_filters.ApproximateTimeSynchronizer(
                subs, queue_size=10, slop=self.get_parameter('sync_slop').value)
            self.sync.registerCallback(self.on_images)
            self._subs = subs
            topics = f'{rgb_topic} + {depth_topic}'
        elif use_rgb:
            self._subs = [self.create_subscription(rgb_type, rgb_topic, lambda m: self.on_images(m, None),
                                                   qos_profile_sensor_data)]
            topics = rgb_topic
        else:
            self._subs = [self.create_subscription(depth_type, depth_topic, lambda m: self.on_images(None, m),
                                                   qos_profile_sensor_data)]
            topics = depth_topic

        info_topic = self.get_parameter('camera_info_topic').value or \
            info_topic_for(rgb_topic if use_rgb else depth_topic)
        self._info_sub = self.create_subscription(CameraInfo, info_topic, self.on_camera_info,
                                                  qos_profile_sensor_data)
        topics += f', {info_topic}'
        self.get_logger().info(
            f'model {model_path} ({self.meta["model"]}, tag {self.meta["tag"]}, keypoints '
            f'{self.detector.names}, inputs {"+".join(inputs)}, backend {self.detector.backend}, '
            f'threshold {self.threshold}), listening on {topics}, 3D from '
            + ('depth' if self.height_mode == 'depth'
               else f'fixed height {self.fixed_height:.3f} m in {self.fixed_height_frame}'))

    def on_camera_info(self, msg):
        if self.camera_info is None:
            self.get_logger().info(f'got camera_info ({msg.width}x{msg.height}, frame {msg.header.frame_id})')
        self.camera_info = msg

    def decode_rgb(self, msg):
        if self.rgb_compressed:
            bgr = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('could not decode compressed rgb image')
        else:
            bgr = dio.decode_raw_rgb_bgr(msg)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def decode_depth(self, msg):
        return dio.decode_compressed_depth(msg) if self.depth_compressed else dio.decode_raw_depth(msg)

    def on_images(self, rgb_msg, depth_msg):
        now = time.monotonic()
        if now - self.last_time < self.min_period:
            return
        self.last_time = now
        try:
            rgb = self.decode_rgb(rgb_msg) if rgb_msg is not None else None
            depth = self.decode_depth(depth_msg) if depth_msg is not None else None
            pts, conf, hm = self.detector.detect(rgb, depth)
        except Exception as e:  # noqa: BLE001 - keep running on single bad frames
            self.get_logger().warning(f'detection failed: {e}', throttle_duration_sec=5.0)
            return

        header = (rgb_msg or depth_msg).header
        out = KeypointArray(header=header, tag=self.meta['tag'])
        height_tf = self.lookup_height_tf(header) if self.height_mode == 'fixed' else None
        for k, name in enumerate(self.detector.names):
            kp = Keypoint(index=k, name=name, x=float(pts[k, 0]), y=float(pts[k, 1]),
                          confidence=float(conf[k]), visible=bool(conf[k] >= self.threshold))
            if self.height_mode == 'fixed':
                ray = pixel_ray(self.camera_info, pts[k, 0], pts[k, 1], self.ref_size) \
                    if self.camera_info is not None and height_tf is not None else None
                pos = intersect_height(height_tf, ray, self.fixed_height) if ray is not None else None
                if pos is not None:
                    kp.has_depth, kp.depth = True, float(pos[2])
                    kp.has_position = True
                    kp.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                out.keypoints.append(kp)
                continue
            d = depth_at(depth, pts[k, 0], pts[k, 1], self.ref_size, self.depth_radius) \
                if depth is not None else None
            if d is not None:
                kp.has_depth, kp.depth = True, d
                pos = project_to_3d(self.camera_info, pts[k, 0], pts[k, 1], d, self.ref_size) \
                    if self.camera_info is not None else None
                if pos is not None:
                    kp.has_position, kp.position = True, pos
            out.keypoints.append(kp)
        if (depth is not None or self.height_mode == 'fixed') and self.camera_info is None:
            self.get_logger().warning('no camera_info received yet, no 3D positions', throttle_duration_sec=5.0)
        self.kp_pub.publish(out)
        self.publish_markers(out)

        if self.image_pub.get_subscription_count() > 0:
            if rgb is not None:
                base = cv2.resize(rgb, self.ref_size) if (rgb.shape[1], rgb.shape[0]) != self.ref_size else rgb
            else:
                base = cv2.resize(viz.colorize_depth(depth), self.ref_size, interpolation=cv2.INTER_NEAREST)
            img = viz.draw_keypoints(base, pts, conf, self.threshold, self.detector.names, heatmaps=hm)
            self.image_pub.publish(Image(header=header, height=img.shape[0], width=img.shape[1], encoding='rgb8',
                                         is_bigendian=0, step=img.shape[1] * 3, data=img.tobytes()))

    def lookup_height_tf(self, header):
        """Transform image frame -> fixed_height_frame at the image stamp, or the latest one if the
        stamp is not (yet) in the buffer; None without a transform."""
        frame = header.frame_id or (self.camera_info.header.frame_id if self.camera_info else '')
        for stamp in (RclpyTime.from_msg(header.stamp), RclpyTime()):
            try:
                return self.tf_buffer.lookup_transform(self.fixed_height_frame, frame, stamp)
            except TransformException as e:
                error = e
        self.get_logger().warning(f'no transform {frame} -> {self.fixed_height_frame}: {error}',
                                  throttle_duration_sec=5.0)
        return None

    def publish_markers(self, kps: KeypointArray):
        if self.marker_pub.get_subscription_count() == 0:
            return
        frame = kps.header.frame_id or (self.camera_info.header.frame_id if self.camera_info else '')
        markers = MarkerArray()
        for kp in kps.keypoints:
            m = Marker(header=kps.header, ns=f'{kps.tag}/{kp.name}', id=kp.index)
            m.header.frame_id = frame
            if self.marker_latest_tf:
                m.header.stamp = Time()
            if not kp.has_position:
                m.action = Marker.DELETE
            else:
                m.type, m.action = Marker.SPHERE, Marker.ADD
                m.pose.position = kp.position
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = self.marker_size
                m.color.r, m.color.g, m.color.b = confidence_color(kp.index, kp.confidence)
                m.color.a = 1.0
                m.lifetime = Duration(seconds=1.0).to_msg()
            markers.markers.append(m)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = KeypointDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
