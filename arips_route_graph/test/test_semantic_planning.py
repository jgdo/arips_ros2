import json
import math

from arips_route_graph.route_graph import RouteEdge, RouteGraph, RouteNode
from arips_route_graph.semantic_planning import (
    format_route_summary,
    plan_semantic_route,
    semantic_route_to_path,
)
from arips_semantic_map_msgs.msg import Segment, SemanticRoute
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np


def _grid(width=5):
    grid_map = OccupancyGrid()
    grid_map.header.frame_id = 'map'
    grid_map.info.width = width
    grid_map.info.height = 1
    grid_map.info.resolution = 1.0
    grid_map.info.origin.orientation.w = 1.0
    grid_map.data = [0] * width
    return grid_map


def _pose(x, frame_id='map'):
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = 0.5
    pose.pose.orientation.w = 1.0
    return pose


def _node(node_id, x, segment_index, door_index=-1, y=0.5):
    return RouteNode(
        node_id=node_id,
        coordinate=(x, y),
        door_index=door_index,
        side='',
        segment_index=segment_index,
    )


def _edge(edge_id, from_node, to_node, kind, segment_index=None):
    return RouteEdge(
        edge_id=edge_id,
        from_node=from_node,
        to_node=to_node,
        kind=kind,
        segment_index=segment_index,
    )


def _graph(nodes, edges):
    return RouteGraph(
        nodes=nodes,
        edges=edges,
        segment_points={},
        skipped_door_indices=[],
    )


def test_same_segment_returns_one_room_with_empty_metadata():
    start = _pose(0.1)
    goal = _pose(1.1)

    response = plan_semantic_route(
        start,
        goal,
        _graph([], []),
        _grid(),
        np.array([[1, 1, 0, 2, 2]], dtype=np.uint16),
    )

    assert response.error_code == response.SUCCESS
    assert len(response.semantic_route.segments) == 1
    segment = response.semantic_route.segments[0]
    assert segment.segment_type == 'room'
    assert segment.metadata_json == ''
    assert segment.start_pose == start
    assert segment.end_pose == goal
    assert format_route_summary(response.semantic_route, 1) == (
        'Computed semantic route with 1 segments: [room_1]'
    )


def test_semantic_route_path_uses_first_start_then_each_segment_end():
    semantic_route = SemanticRoute()
    for start_x, end_x in ((1.0, 2.0), (2.0, 3.0), (3.0, 4.0)):
        segment = Segment()
        segment.start_pose.header.frame_id = 'map'
        segment.start_pose.pose.position.x = start_x
        segment.end_pose.header.frame_id = 'map'
        segment.end_pose.pose.position.x = end_x
        semantic_route.segments.append(segment)

    path = semantic_route_to_path(semantic_route)

    assert path.header.frame_id == 'map'
    assert [pose.pose.position.x for pose in path.poses] == [1.0, 2.0, 3.0, 4.0]


def test_invalid_start_and_goal_return_their_not_found_statuses():
    graph = _graph([], [])
    labels = np.array([[0, 1, 0, 2, 0]], dtype=np.uint16)

    start_response = plan_semantic_route(
        _pose(0.1), _pose(3.1), graph, _grid(), labels
    )
    goal_response = plan_semantic_route(
        _pose(1.1), _pose(4.1), graph, _grid(), labels
    )

    assert start_response.error_code == start_response.START_NOT_FOUND
    assert goal_response.error_code == goal_response.GOAL_NOT_FOUND


def test_cross_segment_route_serializes_room_and_door_edges():
    graph = _graph(
        [
            _node('door_4_A', 1.5, 1, door_index=4),
            _node('door_4_B', 3.5, 2, door_index=4, y=2.5),
        ],
        [_edge('door_edge', 'door_4_A', 'door_4_B', 'door')],
    )
    start = _pose(0.1)
    start.pose.orientation.z = 0.25
    start.pose.orientation.w = 0.9682458365518543
    goal = _pose(4.1)
    goal.pose.orientation.z = -0.3
    goal.pose.orientation.w = 0.9539392014169457

    response = plan_semantic_route(
        start,
        goal,
        graph,
        _grid(),
        np.array([[1, 1, 0, 2, 2]], dtype=np.uint16),
    )

    assert response.error_code == response.SUCCESS
    segments = response.semantic_route.segments
    assert [segment.segment_type for segment in segments] == [
        'room', 'door', 'room'
    ]
    assert json.loads(segments[0].metadata_json) == {'segment_id': 1}
    assert json.loads(segments[1].metadata_json) == {'door_id': 4}
    assert json.loads(segments[2].metadata_json) == {'segment_id': 2}
    assert segments[0].start_pose == start
    assert segments[-1].end_pose == goal
    assert segments[0].end_pose.header.frame_id == 'map'
    expected_yaw = math.atan2(2.0, 2.0)
    expected_z = math.sin(expected_yaw * 0.5)
    expected_w = math.cos(expected_yaw * 0.5)
    assert segments[0].end_pose.pose.orientation.z == expected_z
    assert segments[0].end_pose.pose.orientation.w == expected_w
    assert segments[1].start_pose == segments[0].end_pose
    assert segments[1].end_pose.pose.orientation.z == expected_z
    assert segments[1].end_pose.pose.orientation.w == expected_w
    assert format_route_summary(response.semantic_route, 1) == (
        'Computed semantic route with 3 segments: [room_1, door_4, room_2]'
    )
    assert graph.nodes == [
        _node('door_4_A', 1.5, 1, door_index=4),
        _node('door_4_B', 3.5, 2, door_index=4, y=2.5),
    ]


def test_dijkstra_uses_configured_door_cost():
    nodes = [
        _node('single_start', 100.5, 1, door_index=0),
        _node('single_goal', 101.5, 4, door_index=0),
        _node('multi_start', 1.5, 1, door_index=1),
        _node('middle_a', 2.5, 2, door_index=1),
        _node('middle_b', 3.5, 2, door_index=2),
        _node('multi_goal', 199.5, 4, door_index=2),
    ]
    graph = _graph(
        nodes,
        [
            _edge('single_door', 'single_start', 'single_goal', 'door'),
            _edge('first_door', 'multi_start', 'middle_a', 'door'),
            _edge('middle_room', 'middle_a', 'middle_b', 'room', 2),
            _edge('second_door', 'middle_b', 'multi_goal', 'door'),
        ],
    )
    labels = np.ones((1, 201), dtype=np.uint16)
    labels[0, 2:4] = 2
    labels[0, 101:] = 4

    cheap_doors = plan_semantic_route(
        _pose(0.1), _pose(200.1), graph, _grid(201), labels, 3.0
    )
    expensive_doors = plan_semantic_route(
        _pose(0.1), _pose(200.1), graph, _grid(201), labels, 250.0
    )

    assert sum(
        segment.segment_type == 'door'
        for segment in cheap_doors.semantic_route.segments
    ) == 2
    assert sum(
        segment.segment_type == 'door'
        for segment in expensive_doors.semantic_route.segments
    ) == 1


def test_disconnected_graph_and_unsupported_edges_fail():
    nodes = [_node('start_side', 1.5, 1), _node('goal_side', 3.5, 2)]
    labels = np.array([[1, 1, 0, 2, 2]], dtype=np.uint16)
    disconnected = plan_semantic_route(
        _pose(0.1), _pose(4.1), _graph(nodes, []), _grid(), labels
    )
    unsupported = plan_semantic_route(
        _pose(0.1),
        _pose(4.1),
        _graph(nodes, [_edge('stairs', 'start_side', 'goal_side', 'stairs')]),
        _grid(),
        labels,
    )

    assert disconnected.error_code == disconnected.NO_ROUTE_FOUND
    assert unsupported.error_code == unsupported.UNKNOWN_ERROR
    assert 'Unsupported route edge kind' in unsupported.error_msg
