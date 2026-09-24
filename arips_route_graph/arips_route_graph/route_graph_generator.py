from copy import deepcopy
import math
from typing import Iterable, Optional, Sequence, Tuple

from arips_semantic_map_msgs.msg import SemanticMap
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


PaletteColor = Tuple[float, float, float]

PALETTE: Tuple[PaletteColor, ...] = (
    (0.90, 0.16, 0.16),
    (0.16, 0.48, 0.90),
    (0.16, 0.72, 0.34),
    (0.90, 0.58, 0.12),
    (0.62, 0.25, 0.84),
    (0.10, 0.72, 0.72),
    (0.90, 0.26, 0.58),
    (0.50, 0.66, 0.18),
)


def occupancy_array(grid_map: OccupancyGrid) -> np.ndarray:
    """Copy an OccupancyGrid into a signed int8 row-major array."""
    height = grid_map.info.height
    width = grid_map.info.width
    expected_size = height * width
    if len(grid_map.data) != expected_size:
        raise ValueError(
            f'Grid data has {len(grid_map.data)} cells, expected {expected_size}'
        )
    return np.asarray(grid_map.data, dtype=np.int8).reshape((height, width)).copy()


def _quaternion_yaw(quaternion: Quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def world_to_grid_coordinates(
    grid_map: OccupancyGrid, x: float, y: float
) -> Tuple[float, float]:
    """Convert a world point to continuous map-cell coordinates."""
    origin = grid_map.info.origin
    yaw = _quaternion_yaw(origin.orientation)
    dx = x - origin.position.x
    dy = y - origin.position.y
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    resolution = grid_map.info.resolution
    if resolution <= 0.0:
        raise ValueError('Grid resolution must be positive')
    return local_x / resolution, local_y / resolution


def _clip_segment(
    start: Tuple[float, float],
    end: Tuple[float, float],
    width: int,
    height: int,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clip a line segment to the inclusive cell-coordinate bounds."""
    if width == 0 or height == 0:
        return None

    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    lower = 0.0
    upper = 1.0
    limits = (
        (-dx, x0),
        (dx, width - 1.0 - x0),
        (-dy, y0),
        (dy, height - 1.0 - y0),
    )
    for coefficient, value in limits:
        if coefficient == 0.0:
            if value < 0.0:
                return None
            continue
        parameter = value / coefficient
        if coefficient < 0.0:
            lower = max(lower, parameter)
        else:
            upper = min(upper, parameter)
        if lower > upper:
            return None

    return (
        (x0 + lower * dx, y0 + lower * dy),
        (x0 + upper * dx, y0 + upper * dy),
    )


def _bresenham_cells(
    start: Tuple[int, int], end: Tuple[int, int]
) -> Iterable[Tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    delta_x = abs(x1 - x0)
    step_x = 1 if x0 < x1 else -1
    delta_y = -abs(y1 - y0)
    step_y = 1 if y0 < y1 else -1
    error = delta_x + delta_y

    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        doubled_error = 2 * error
        if doubled_error >= delta_y:
            error += delta_y
            x0 += step_x
        if doubled_error <= delta_x:
            error += delta_x
            y0 += step_y


def rasterize_line(
    occupancy: np.ndarray,
    start: Tuple[float, float],
    end: Tuple[float, float],
    value: int = 100,
) -> None:
    """Rasterize a clipped line into an occupancy array."""
    height, width = occupancy.shape
    clipped = _clip_segment(start, end, width, height)
    if clipped is None:
        return
    clipped_start, clipped_end = clipped
    integer_start = (round(clipped_start[0]), round(clipped_start[1]))
    integer_end = (round(clipped_end[0]), round(clipped_end[1]))
    for x, y in _bresenham_cells(integer_start, integer_end):
        if 0 <= y < height and 0 <= x < width:
            occupancy[y, x] = value


def segment_free_space(
    occupancy: np.ndarray, minimum_segment_size: int
) -> np.ndarray:
    """Label 4-connected free regions and discard small regions."""
    if occupancy.ndim != 2:
        raise ValueError('Occupancy data must be a two-dimensional array')
    if minimum_segment_size < 0:
        raise ValueError('minimum_segment_size must not be negative')

    labels = np.zeros(occupancy.shape, dtype=np.uint16)
    next_label = 1
    maximum_label = np.iinfo(np.uint16).max
    height, width = occupancy.shape

    for row in range(height):
        for column in range(width):
            if occupancy[row, column] != 0 or labels[row, column] != 0:
                continue
            if next_label > maximum_label:
                raise OverflowError('Too many free-space segments for uint16 labels')

            cells = []
            stack = [(row, column)]
            labels[row, column] = next_label
            while stack:
                current_row, current_column = stack.pop()
                cells.append((current_row, current_column))
                for neighbor_row, neighbor_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if not (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_column < width
                    ):
                        continue
                    if occupancy[neighbor_row, neighbor_column] != 0:
                        continue
                    if labels[neighbor_row, neighbor_column] != 0:
                        continue
                    labels[neighbor_row, neighbor_column] = next_label
                    stack.append((neighbor_row, neighbor_column))

            if len(cells) < minimum_segment_size:
                for cell_row, cell_column in cells:
                    labels[cell_row, cell_column] = 0
            next_label += 1

    return labels


def _grid_to_world(grid_map: OccupancyGrid, column: int, row: int) -> Point:
    origin = grid_map.info.origin
    yaw = _quaternion_yaw(origin.orientation)
    local_x = (column + 0.5) * grid_map.info.resolution
    local_y = (row + 0.5) * grid_map.info.resolution
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return Point(
        x=origin.position.x + cos_yaw * local_x - sin_yaw * local_y,
        y=origin.position.y + sin_yaw * local_x + cos_yaw * local_y,
        z=0.0,
    )


def make_segmentation_marker(
    grid_map: OccupancyGrid,
    labels: np.ndarray,
    marker_height: float,
) -> MarkerArray:
    """Create one colored CUBE_LIST marker for retained segments."""
    marker = Marker()
    marker.header = deepcopy(grid_map.header)
    marker.ns = 'map_segmentation'
    marker.id = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.scale.x = grid_map.info.resolution
    marker.scale.y = grid_map.info.resolution
    marker.scale.z = marker_height
    marker.color.a = 1.0

    for (row, column), segment_number in np.ndenumerate(labels):
        if segment_number == 0:
            continue
        marker.points.append(_grid_to_world(grid_map, column, row))
        red, green, blue = PALETTE[(int(segment_number) - 1) % len(PALETTE)]
        marker.colors.append(
            ColorRGBA(r=red, g=green, b=blue, a=1.0)
        )

    return MarkerArray(markers=[marker])


class RouteGraphGenerator(Node):
    """Create a corrected map and connected free-space visualization."""

    def __init__(self) -> None:
        super().__init__('route_graph_generator')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('semantic_map_topic', '/semantic_map')
        self.declare_parameter('corrected_map_topic', '/corrected_map')
        self.declare_parameter('segmentation_topic', '/map_segmentation')
        self.declare_parameter('minimum_segment_size', 500)
        self.declare_parameter('marker_height', 0.02)

        map_topic = self.get_parameter('map_topic').value
        semantic_map_topic = self.get_parameter('semantic_map_topic').value
        corrected_map_topic = self.get_parameter('corrected_map_topic').value
        segmentation_topic = self.get_parameter('segmentation_topic').value
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self._minimum_segment_size = int(
            self.get_parameter('minimum_segment_size').value
        )
        self._marker_height = float(self.get_parameter('marker_height').value)
        self._grid_map: Optional[OccupancyGrid] = None
        self._semantic_map: Optional[SemanticMap] = None
        self._corrected_map_publisher = self.create_publisher(
            OccupancyGrid, corrected_map_topic, qos
        )
        self._segmentation_publisher = self.create_publisher(
            MarkerArray, segmentation_topic, qos
        )
        self.create_subscription(
            OccupancyGrid, map_topic, self._grid_map_callback, qos
        )
        self.create_subscription(
            SemanticMap,
            semantic_map_topic,
            self._semantic_map_callback,
            qos,
        )

    def _grid_map_callback(self, message: OccupancyGrid) -> None:
        self.get_logger().info('Received new grid map.')
        self._grid_map = message
        self._rebuild()

    def _semantic_map_callback(self, message: SemanticMap) -> None:
        self.get_logger().info('Received new semantic map.')
        self._semantic_map = message
        self._rebuild()

    def _rebuild(self) -> None:
        if self._grid_map is None:
            return

        try:
            occupancy = occupancy_array(self._grid_map)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        if (
            self._semantic_map is not None
            and self._grid_map.header.frame_id
            != self._semantic_map.header.frame_id
        ):
            self.get_logger().error(
                'Grid map frame and semantic map frame differ; '
                'discarding segmentation'
            )
            self._publish_corrected_map(occupancy)
            return

        if self._semantic_map is not None:
            for door in self._semantic_map.doors:
                start = world_to_grid_coordinates(
                    self._grid_map, door.pivot.x, door.pivot.y
                )
                end = world_to_grid_coordinates(
                    self._grid_map, door.extent.x, door.extent.y
                )
                rasterize_line(occupancy, start, end)

        self._publish_corrected_map(occupancy)
        labels = segment_free_space(occupancy, self._minimum_segment_size)

        marker_msg = make_segmentation_marker(
            self._grid_map, labels, self._marker_height
        )
        self._segmentation_publisher.publish(marker_msg)
        self.get_logger().info(f'Rebuilt route graph and published segmentation with {len(marker_msg.markers[0].points)} points.')

    def _publish_corrected_map(self, occupancy: np.ndarray) -> None:
        corrected_map = OccupancyGrid()
        corrected_map.header = deepcopy(self._grid_map.header)
        corrected_map.info = deepcopy(self._grid_map.info)
        corrected_map.data = [int(value) for value in occupancy.flat]
        self._corrected_map_publisher.publish(corrected_map)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = RouteGraphGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
