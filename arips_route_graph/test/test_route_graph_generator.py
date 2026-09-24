import math

from arips_route_graph.route_graph_generator import (
    make_segmentation_marker,
    occupancy_array,
    rasterize_line,
    segment_free_space,
    world_to_grid_coordinates,
)
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest


def _grid(width=5, height=4, resolution=1.0):
    message = OccupancyGrid()
    message.header.frame_id = 'map'
    message.info.width = width
    message.info.height = height
    message.info.resolution = resolution
    message.info.origin.orientation.w = 1.0
    message.data = [0] * (width * height)
    return message


def test_occupancy_array_preserves_signed_unknown_value():
    message = _grid(width=2, height=2)
    message.data = [0, -1, 100, 0]

    result = occupancy_array(message)

    assert result.dtype == np.int8
    assert result.tolist() == [[0, -1], [100, 0]]


def test_world_to_grid_coordinates_accounts_for_origin_yaw():
    message = _grid()
    message.info.origin.orientation = Quaternion(
        z=math.sin(math.pi / 4.0), w=math.cos(math.pi / 4.0)
    )

    assert world_to_grid_coordinates(message, 0.0, 1.0) == pytest.approx(
        (1.0, 0.0)
    )


def test_rasterize_line_marks_diagonal_and_clips_outside_map():
    occupancy = np.zeros((4, 5), dtype=np.int8)

    rasterize_line(occupancy, (-2.0, -2.0), (3.0, 3.0))

    assert occupancy[0, 0] == 100
    assert occupancy[1, 1] == 100
    assert occupancy[2, 2] == 100
    assert occupancy[3, 3] == 100
    assert occupancy[:, 4].tolist() == [0, 0, 0, 0]


def test_segment_free_space_uses_four_connectivity_and_filters_small_regions():
    occupancy = np.array(
        [
            [0, 100, 0],
            [100, 0, 100],
            [0, 100, 0],
        ],
        dtype=np.int8,
    )

    labels = segment_free_space(occupancy, minimum_segment_size=2)

    assert labels.dtype == np.uint16
    assert np.count_nonzero(labels) == 0


def test_segment_free_space_keeps_large_region():
    occupancy = np.zeros((2, 3), dtype=np.int8)
    occupancy[0, 0] = 100

    labels = segment_free_space(occupancy, minimum_segment_size=4)

    assert set(labels.flat) == {0, 1}
    assert np.count_nonzero(labels == 1) == 5


def test_segmentation_marker_contains_colored_cell_centers():
    message = _grid(width=2, height=1, resolution=0.5)
    labels = np.array([[1, 2]], dtype=np.uint16)

    marker_array = make_segmentation_marker(message, labels, 0.02)

    marker = marker_array.markers[0]
    assert marker.type == marker.CUBE_LIST
    assert marker.scale.x == 0.5
    assert marker.scale.y == 0.5
    assert marker.scale.z == 0.02
    assert [(point.x, point.y, point.z) for point in marker.points] == [
        (0.25, 0.25, 0.0),
        (0.75, 0.25, 0.0),
    ]
    assert len(marker.colors) == 2
