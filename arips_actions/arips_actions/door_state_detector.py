"""Service that detects whether a door is open or closed from laser scans.

Only the door geometry from the semantic map is used (pivot, extent and opening angle,
all in the map frame), so it works for any mapped door, not just in simulation.

Method (all in the map frame):
  The last few laser scans are transformed into the map frame (each with the TF at its
  own stamp). For a hypothetical leaf position, i.e. a line segment starting at the pivot
  at a given angle, every beam whose ray crosses that segment is classified as
    hit      - the beam ends on the segment (within range_tolerance),
    through  - the beam ends clearly behind the segment, so there is no leaf there,
    occluded - the beam ends clearly in front of it (no information).
  The closed hypothesis (angle 0) is evaluated with small angle offsets and pivot shifts
  perpendicular to the door to tolerate errors of the semantic map, the localization and
  the laser; the variant with the most hits is used. If most crossing beams hit the closed
  leaf, the door is closed; if most of them pass through, it is open. Only the inner part
  of the leaf (segment_start..segment_end of its length) is used, so the door frame and the
  wall next to it do not count.
  For information, the leaf angle is estimated by sweeping the hypothesis from the closed
  to the open angle (and a bit beyond) and taking the angle with the most hits.

Service:
  get_door_state (arips_action_msgs/srv/GetDoorState)
Debug output:
  door_state_markers (visualization_msgs/MarkerArray): closed door line, estimated leaf
  line and the beam end points classified as hit/through for the closed hypothesis.
"""

import collections
import math

import numpy as np
import rclpy
from arips_action_msgs.srv import GetDoorState
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def scan_to_beams(scan, transform):
    """Return beam origins (K, 2), unit directions (K, 2) and ranges (K,) in the map frame.

    Beams without a return (inf or beyond range_max) get range inf, so they count as passing
    through any segment they cross. Invalid beams (NaN, below range_min) are dropped.
    """
    ranges = np.asarray(scan.ranges, dtype=float)
    angles = scan.angle_min + scan.angle_increment * np.arange(len(ranges))
    ranges = np.where(np.isposinf(ranges) | (ranges >= scan.range_max), np.inf, ranges)
    valid = ~np.isnan(ranges) & (ranges > scan.range_min)

    t = transform.transform.translation
    q = transform.transform.rotation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    # The laser may be mounted upside down (roll = pi): then the scan angles are mirrored.
    up_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    sign = 1.0 if up_z >= 0.0 else -1.0

    beam_angles = yaw + sign * angles[valid]
    directions = np.stack([np.cos(beam_angles), np.sin(beam_angles)], axis=1)
    origins = np.tile([t.x, t.y], (len(beam_angles), 1))
    return origins, directions, ranges[valid]


def classify_beams(origins, directions, ranges, seg_a, seg_b, range_tolerance,
                   min_sin_incidence):
    """Classify beams against H segments.

    seg_a, seg_b: (H, 2) segment end points. Returns boolean arrays hit, through of shape
    (H, K). Beams that do not cross a segment, or cross it at a grazing angle, are neither.
    """
    seg = seg_b - seg_a                                   # (H, 2)
    seg_len = np.linalg.norm(seg, axis=1)                 # (H,)
    # 2D cross products, broadcast to (H, K).
    denom = directions[None, :, 0] * seg[:, None, 1] - directions[None, :, 1] * seg[:, None, 0]
    ao = seg_a[:, None, :] - origins[None, :, :]          # (H, K, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_int = (ao[..., 0] * seg[:, None, 1] - ao[..., 1] * seg[:, None, 0]) / denom
        tau = (ao[..., 0] * directions[None, :, 1] - ao[..., 1] * directions[None, :, 0]) / denom
    crossing = ((np.abs(denom) > min_sin_incidence * seg_len[:, None])
                & (tau >= 0.0) & (tau <= 1.0) & (r_int > 0.0))
    r = ranges[None, :]
    hit = crossing & (np.abs(r - r_int) <= range_tolerance)
    through = crossing & (r > r_int + range_tolerance)
    return hit, through


class DoorStateDetector(Node):
    def __init__(self):
        super().__init__('door_state_detector')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_frame', 'map'),
                ('scan_topic', 'scan'),
                # Number of most recent scans that are combined (noise robustness).
                ('num_scans', 5),
                # Scans older than this (relative to the newest one) are not used.
                ('max_scan_age_s', 2.0),
                # The newest scan must not be older than this (relative to now).
                ('max_scan_delay_s', 2.0),
                # Part of the leaf (fraction of its length from the pivot) that is evaluated.
                ('segment_start', 0.15),
                ('segment_end', 0.9),
                # Search window around each hypothesis for map/localization/laser errors.
                ('max_angle_offset_deg', 12.0),
                ('angle_step_deg', 1.0),
                ('max_shift', 0.08),
                ('shift_step', 0.02),
                # A beam hits the leaf if it ends within this distance of the leaf (along the beam).
                ('range_tolerance', 0.07),
                # Beams crossing the leaf line flatter than this are ignored.
                ('min_incidence_deg', 15.0),
                # Minimum number of beams crossing the closed door line for a decision.
                ('min_beams', 10),
                # The door is closed if at least this fraction of the crossing beams hit it.
                ('closed_ratio_threshold', 0.5),
                ('marker_topic', 'door_state_markers'),
            ],
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._scans = collections.deque(maxlen=max(1, self.get_parameter('num_scans').value))
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value, self._on_scan,
            qos_profile_sensor_data)
        self._marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter('marker_topic').value, 1)
        self.create_service(GetDoorState, 'get_door_state', self._handle_request)
        self.get_logger().info('get_door_state service ready')

    def _on_scan(self, msg):
        self._scans.append(msg)

    # ------------------------------------------------------------------ helpers

    def _param(self, name):
        return self.get_parameter(name).value

    def _collect_beams(self):
        """Return (origins, directions, ranges) of the recent scans in the map frame, or an error."""
        if not self._scans:
            return None, 'No laser scan received yet'
        map_frame = self._param('map_frame')
        newest = Time.from_msg(self._scans[-1].header.stamp)
        delay = (self.get_clock().now() - newest).nanoseconds * 1e-9
        if delay > self._param('max_scan_delay_s'):
            return None, f'Newest laser scan is {delay:.1f} s old'

        parts = []
        last_error = ''
        for scan in self._scans:
            stamp = Time.from_msg(scan.header.stamp)
            if (newest - stamp).nanoseconds * 1e-9 > self._param('max_scan_age_s'):
                continue
            try:
                transform = self._tf_buffer.lookup_transform(
                    map_frame, scan.header.frame_id, stamp)
            except TransformException as error:
                last_error = str(error)
                continue
            parts.append(scan_to_beams(scan, transform))
        if not parts:
            return None, f'No laser scan could be transformed to {map_frame}: {last_error}'
        return tuple(np.concatenate(p) for p in zip(*parts)), len(parts)

    def _segments(self, pivot, closed_angle, length, angles_deg, shifts):
        """End points (H, 2) of leaf segments for all angle/shift combinations."""
        s0 = self._param('segment_start') * length
        s1 = self._param('segment_end') * length
        a = closed_angle + np.radians(np.repeat(angles_deg, len(shifts)))
        shift = np.tile(shifts, len(angles_deg))
        # Shift perpendicular to the closed door line.
        normal = np.array([-math.sin(closed_angle), math.cos(closed_angle)])
        base = pivot[None, :] + shift[:, None] * normal[None, :]
        u = np.stack([np.cos(a), np.sin(a)], axis=1)
        return base + s0 * u, base + s1 * u, np.repeat(angles_deg, len(shifts)), shift

    # ------------------------------------------------------------------ service

    def _handle_request(self, request, response):
        door = request.door
        pivot = np.array([door.pivot.x, door.pivot.y])
        extent = np.array([door.extent.x, door.extent.y])
        length = float(np.linalg.norm(extent - pivot))
        if length < 0.1:
            return self._fail(response, 'Door pivot and extent (almost) coincide')
        closed_angle = math.atan2(extent[1] - pivot[1], extent[0] - pivot[0])

        beams, info = self._collect_beams()
        if beams is None:
            return self._fail(response, info)
        origins, directions, ranges = beams
        tol = self._param('range_tolerance')
        min_sin = math.sin(math.radians(self._param('min_incidence_deg')))

        max_shift = self._param('max_shift')
        shift_step = max(1e-3, self._param('shift_step'))
        shifts = np.arange(-max_shift, max_shift + 0.5 * shift_step, shift_step)
        step = max(0.1, self._param('angle_step_deg'))
        max_off = self._param('max_angle_offset_deg')

        # Closed hypothesis with small angle offsets and shifts.
        offsets = np.arange(-max_off, max_off + 0.5 * step, step)
        seg_a, seg_b, seg_angles, seg_shifts = self._segments(
            pivot, closed_angle, length, offsets, shifts)
        hit, through = classify_beams(origins, directions, ranges, seg_a, seg_b, tol, min_sin)
        hits = hit.sum(axis=1)
        throughs = through.sum(axis=1)
        # Most hits; among equal hits the fewest pass-through beams, then the smallest offset.
        order = np.lexsort((np.abs(seg_angles), throughs, -hits))
        best = order[0]
        num_hits = int(hits[best])
        num_beams = num_hits + int(throughs[best])
        if num_beams < self._param('min_beams'):
            self._publish_markers(pivot, extent, None, origins, directions, ranges,
                                  hit[best], through[best])
            return self._fail(
                response,
                f'Door not observed well enough: only {num_beams} laser beams cross the '
                f'closed door line (need {self._param("min_beams")})')
        closed_ratio = num_hits / num_beams
        is_open = closed_ratio < self._param('closed_ratio_threshold')

        # Leaf angle estimate: sweep from the closed to the open angle (and a bit beyond).
        open_deg = door.open_angle_deg
        lo, hi = min(0.0, open_deg) - max_off, max(0.0, open_deg) + max_off
        sweep = np.arange(lo, hi + 0.5 * step, step)
        # Exclude the closed region when open: hits there are the door frame or noise.
        sweep = sweep[np.abs(sweep) > 2.0 * max_off]
        angle_deg = float(seg_angles[best])
        if is_open and len(sweep) > 0:
            sa, sb, s_angles, _ = self._segments(pivot, closed_angle, length, sweep, shifts)
            s_hit, _ = classify_beams(origins, directions, ranges, sa, sb, tol, min_sin)
            s_hits = s_hit.sum(axis=1)
            if s_hits.max() > 0:
                angle_deg = float(s_angles[int(np.argmax(s_hits))])

        response.success = True
        response.error_message = ''
        response.is_open = bool(is_open)
        response.angle_deg = angle_deg
        response.closed_ratio = float(closed_ratio)
        response.num_beams = int(num_beams)
        leaf_angle = closed_angle + math.radians(angle_deg)
        leaf_end = pivot + length * np.array([math.cos(leaf_angle), math.sin(leaf_angle)])
        self._publish_markers(pivot, extent, leaf_end, origins, directions, ranges,
                              hit[best], through[best])
        self.get_logger().info(
            f'Door {"OPEN" if is_open else "CLOSED"}: {num_hits}/{num_beams} beams hit the '
            f'closed leaf (ratio {closed_ratio:.2f}, best offset {seg_angles[best]:.0f} deg, '
            f'shift {seg_shifts[best]:+.2f} m), estimated angle {angle_deg:.0f} deg, '
            f'{info} scans')
        return response

    def _fail(self, response, error_message):
        response.success = False
        response.error_message = error_message
        self.get_logger().warning(error_message)
        return response

    # ------------------------------------------------------------------ markers

    def _publish_markers(self, pivot, extent, leaf_end, origins, directions, ranges,
                         hit, through):
        stamp = self.get_clock().now().to_msg()
        frame = self._param('map_frame')

        def marker(ns, mtype, scale, color):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = ns
            m.id = 0
            m.type = mtype
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
            return m

        def pt(xy, z=0.05):
            return Point(x=float(xy[0]), y=float(xy[1]), z=z)

        markers = MarkerArray()
        closed = marker('closed_door', Marker.LINE_LIST, 0.02, (0.2, 0.4, 1.0))
        closed.points = [pt(pivot), pt(extent)]
        markers.markers.append(closed)
        leaf = marker('estimated_leaf', Marker.LINE_LIST, 0.04, (1.0, 0.6, 0.0))
        if leaf_end is None:
            leaf.action = Marker.DELETE
        else:
            leaf.points = [pt(pivot, 0.1), pt(leaf_end, 0.1)]
        markers.markers.append(leaf)
        ends = origins + directions * np.minimum(ranges, 5.0)[:, None]
        for ns, mask, color in (('hit', hit, (0.0, 1.0, 0.0)), ('through', through, (1.0, 0.0, 0.0))):
            m = marker(ns, Marker.POINTS, 0.04, color)
            m.points = [pt(p) for p in ends[mask]]
            markers.markers.append(m)
        self._marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DoorStateDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
