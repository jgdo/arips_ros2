import json

from arips_route_graph.route_graph import (
    build_route_graph,
    compute_approach_points,
    make_route_graph_marker_array,
    point_to_segment_index,
    write_route_graph_geojson,
)
from arips_semantic_map_msgs.msg import Door, Point2D, SemanticMap
from nav_msgs.msg import OccupancyGrid
import numpy as np
import pytest


def _grid(width=6, height=4, resolution=1.0):
    message = OccupancyGrid()
    message.header.frame_id = 'map'
    message.info.width = width
    message.info.height = height
    message.info.resolution = resolution
    message.info.origin.orientation.w = 1.0
    message.data = [0] * (width * height)
    return message


def _door(pivot, extent, angle=90.0):
    return Door(
        pivot=Point2D(x=pivot[0], y=pivot[1]),
        extent=Point2D(x=extent[0], y=extent[1]),
        open_angle_deg=angle,
    )


def _semantic_map(*doors):
    message = SemanticMap()
    message.header.frame_id = 'map'
    message.doors = list(doors)
    return message


def test_compute_approach_points_uses_signed_open_side_and_distance():
    door = _door((1.0, 1.0), (3.0, 1.0), angle=90.0)
    negative_door = _door((1.0, 1.0), (3.0, 1.0), angle=-90.0)

    point_a, point_b = compute_approach_points(door, 0.5)
    negative_a, negative_b = compute_approach_points(negative_door, 0.5)

    assert point_a == pytest.approx((2.0, 1.5))
    assert point_b == pytest.approx((2.0, 0.5))
    assert negative_a == pytest.approx((2.0, 0.5))
    assert negative_b == pytest.approx((2.0, 1.5))


def test_compute_approach_points_rejects_degenerate_doors():
    with pytest.raises(ValueError):
        compute_approach_points(_door((1.0, 1.0), (1.0, 1.0)), 0.5)


def test_point_to_segment_index_uses_containing_cell_and_zero_outside():
    grid_map = _grid(width=2, height=2)
    segmentation = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    assert point_to_segment_index(grid_map, segmentation, (0.1, 0.1)) == 1
    assert point_to_segment_index(grid_map, segmentation, (1.9, 1.9)) == 4
    assert point_to_segment_index(grid_map, segmentation, (-0.1, 0.1)) == 0


def test_build_route_graph_adds_one_door_edge_and_cross_door_segment_edges():
    grid_map = _grid()
    semantic_map = _semantic_map(
        _door((1.0, 1.0), (3.0, 1.0)),
        _door((1.0, 2.0), (3.0, 2.0)),
    )
    segmentation = np.ones((4, 6), dtype=np.uint16)

    graph = build_route_graph(semantic_map, grid_map, segmentation, 0.5)

    assert [node.node_id for node in graph.nodes] == [
        'door_0_A', 'door_0_B', 'door_1_A', 'door_1_B'
    ]
    assert len(graph.edges) == 12
    assert sum(edge.kind == 'door' for edge in graph.edges) == 4
    assert sum(edge.kind == 'segment' for edge in graph.edges) == 8
    assert len(graph.segment_points[1]) == 4


def test_build_route_graph_ignores_segment_zero_for_cross_door_edges():
    grid_map = _grid()
    semantic_map = _semantic_map(
        _door((1.0, 1.0), (3.0, 1.0)),
        _door((1.0, 2.0), (3.0, 2.0)),
    )
    segmentation = np.zeros((4, 6), dtype=np.uint16)

    graph = build_route_graph(semantic_map, grid_map, segmentation, 0.5)

    assert len(graph.edges) == 4
    assert set(graph.segment_points) == {0}


def test_build_route_graph_skips_invalid_door():
    grid_map = _grid()
    semantic_map = _semantic_map(
        _door((1.0, 1.0), (1.0, 1.0)),
        _door((1.0, 1.0), (3.0, 1.0)),
    )
    segmentation = np.ones((4, 6), dtype=np.uint16)

    graph = build_route_graph(semantic_map, grid_map, segmentation, 0.5)

    assert graph.skipped_door_indices == [0]
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 2


def test_write_route_graph_geojson_contains_nodes_and_edges(tmp_path):
    grid_map = _grid()
    semantic_map = _semantic_map(_door((1.0, 1.0), (3.0, 1.0)))
    segmentation = np.ones((4, 6), dtype=np.uint16)
    graph = build_route_graph(semantic_map, grid_map, segmentation, 0.5)
    path = tmp_path / 'route_graph.geojson'

    written_path = write_route_graph_geojson(graph, 'map', path)
    document = json.loads(written_path.read_text())

    assert document['type'] == 'FeatureCollection'
    assert document['properties']['frame_id'] == 'map'
    assert len(document['features']) == 4
    assert {feature['geometry']['type'] for feature in document['features']} == {
        'Point', 'LineString'
    }
    node_features = [
        feature
        for feature in document['features']
        if feature['properties']['feature_type'] == 'node'
    ]
    edge_features = [
        feature
        for feature in document['features']
        if feature['properties']['feature_type'] == 'edge'
    ]
    assert all(isinstance(feature['properties']['id'], int)
               for feature in node_features)
    assert all(isinstance(feature['properties']['id'], int)
               for feature in edge_features)
    assert all(isinstance(feature['properties'][key], int)
               for feature in edge_features for key in ('startid', 'endid'))


def test_route_graph_markers_use_green_spheres_and_lines():
    grid_map = _grid()
    semantic_map = _semantic_map(
        _door((1.0, 1.0), (3.0, 1.0)),
        _door((1.0, 2.0), (3.0, 2.0)),
    )
    segmentation = np.ones((4, 6), dtype=np.uint16)
    graph = build_route_graph(semantic_map, grid_map, segmentation, 0.5)

    marker_array = make_route_graph_marker_array(graph, 'map', 0.12, 0.03)

    assert len(marker_array.markers) == 2
    node_marker, edge_marker = marker_array.markers
    assert node_marker.type == node_marker.SPHERE_LIST
    assert edge_marker.type == edge_marker.LINE_LIST
    assert len(node_marker.points) == len(graph.nodes)
    assert len(edge_marker.points) == 2 * len(graph.edges)
    assert node_marker.color.g == 1.0
    assert node_marker.color.r == 0.0
    assert node_marker.color.b == 0.0
    assert edge_marker.color.g == 1.0
    assert edge_marker.color.r == 0.0
    assert edge_marker.color.b == 0.0
    assert all(point.z == 0.0 for point in node_marker.points)
    assert all(point.z == 0.0 for point in edge_marker.points)


def test_compute_approach_points_rejects_negative_distance():
    with pytest.raises(ValueError):
        compute_approach_points(_door((1.0, 1.0), (3.0, 1.0)), -0.1)
