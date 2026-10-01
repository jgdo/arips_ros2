from copy import deepcopy
import heapq
import json
import math
from typing import Dict, List, Optional, Tuple

from arips_route_graph.route_graph import (
    point_to_segment_index,
    RouteEdge,
    RouteGraph,
    RouteNode,
)
from arips_semantic_map_msgs.msg import Segment, SemanticRoute
from arips_semantic_map_msgs.srv import ComputeSemanticRoute
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np


SUCCESS = int(getattr(ComputeSemanticRoute.Response, 'SUCCESS'))
START_NOT_FOUND = int(getattr(ComputeSemanticRoute.Response, 'START_NOT_FOUND'))
GOAL_NOT_FOUND = int(getattr(ComputeSemanticRoute.Response, 'GOAL_NOT_FOUND'))
NO_ROUTE_FOUND = int(getattr(ComputeSemanticRoute.Response, 'NO_ROUTE_FOUND'))
UNKNOWN_ERROR = int(getattr(ComputeSemanticRoute.Response, 'UNKNOWN_ERROR'))


def _error_response(
    response: ComputeSemanticRoute.Response, error_code: int, error_msg: str
) -> ComputeSemanticRoute.Response:
    response.semantic_route = SemanticRoute()
    response.error_code = error_code
    response.error_msg = error_msg
    return response


def _pose_segment_index(
    pose: PoseStamped, grid_map: OccupancyGrid, segmentation: np.ndarray
) -> int:
    if pose.header.frame_id != grid_map.header.frame_id:
        return 0
    if not math.isfinite(pose.pose.position.x) or not math.isfinite(
        pose.pose.position.y
    ):
        return 0
    return point_to_segment_index(
        grid_map,
        segmentation,
        (pose.pose.position.x, pose.pose.position.y),
    )


def _unique_id(prefix: str, existing_ids: set) -> str:
    index = 0
    candidate = prefix
    while candidate in existing_ids:
        index += 1
        candidate = f'{prefix}_{index}'
    return candidate


def _add_planning_node(
    graph: RouteGraph,
    pose: PoseStamped,
    node_id: str,
    side: str,
    segment_index: int,
) -> RouteNode:
    node = RouteNode(
        node_id=node_id,
        coordinate=(pose.pose.position.x, pose.pose.position.y),
        door_index=-1,
        side=side,
        segment_index=segment_index,
    )
    graph.nodes.append(node)
    return node


def _node_pose(node: RouteNode, grid_map: OccupancyGrid) -> PoseStamped:
    pose = PoseStamped()
    pose.header = deepcopy(grid_map.header)
    pose.pose.position.x = node.coordinate[0]
    pose.pose.position.y = node.coordinate[1]
    pose.pose.orientation.w = 1.0
    return pose


def _edge_costs(
    graph: RouteGraph, door_edge_cost: float
) -> Tuple[Dict[str, RouteNode], Dict[str, List[Tuple[RouteEdge, float]]]]:
    if not math.isfinite(door_edge_cost) or door_edge_cost < 0.0:
        raise ValueError('door_edge_cost must be finite and nonnegative')

    node_by_id = {node.node_id: node for node in graph.nodes}
    adjacency: Dict[str, List[Tuple[RouteEdge, float]]] = {
        node_id: [] for node_id in node_by_id
    }
    for edge in graph.edges:
        if edge.from_node not in node_by_id or edge.to_node not in node_by_id:
            raise ValueError(f'Edge {edge.edge_id} references a missing node')
        if edge.kind == 'room':
            from_coordinate = node_by_id[edge.from_node].coordinate
            to_coordinate = node_by_id[edge.to_node].coordinate
            cost = math.dist(from_coordinate, to_coordinate)
        elif edge.kind == 'door':
            cost = door_edge_cost
        else:
            raise ValueError(f'Unsupported route edge kind: {edge.kind}')
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f'Edge {edge.edge_id} has an invalid cost')
        adjacency[edge.from_node].append((edge, cost))
    return node_by_id, adjacency


def _shortest_path(
    adjacency: Dict[str, List[Tuple[RouteEdge, float]]],
    start_node_id: str,
    goal_node_id: str,
) -> Optional[List[RouteEdge]]:
    distances = {start_node_id: 0.0}
    previous_edges: Dict[str, RouteEdge] = {}
    queue = [(0.0, start_node_id)]

    while queue:
        distance, node_id = heapq.heappop(queue)
        if distance > distances.get(node_id, math.inf):
            continue
        if node_id == goal_node_id:
            break
        for edge, cost in adjacency[node_id]:
            next_distance = distance + cost
            if next_distance < distances.get(edge.to_node, math.inf):
                distances[edge.to_node] = next_distance
                previous_edges[edge.to_node] = edge
                heapq.heappush(queue, (next_distance, edge.to_node))

    if goal_node_id not in previous_edges:
        return None

    path = []
    current_node_id = goal_node_id
    while current_node_id != start_node_id:
        edge = previous_edges[current_node_id]
        path.append(edge)
        current_node_id = edge.from_node
    path.reverse()
    return path


def _route_segments(
    path: List[RouteEdge],
    node_by_id: Dict[str, RouteNode],
    grid_map: OccupancyGrid,
    start_pose: PoseStamped,
    goal_pose: PoseStamped,
    start_node_id: str,
    goal_node_id: str,
) -> SemanticRoute:
    semantic_route = SemanticRoute()
    for edge in path:
        segment = Segment()
        segment.start_pose = (
            deepcopy(start_pose)
            if edge.from_node == start_node_id
            else _node_pose(node_by_id[edge.from_node], grid_map)
        )
        segment.end_pose = (
            deepcopy(goal_pose)
            if edge.to_node == goal_node_id
            else _node_pose(node_by_id[edge.to_node], grid_map)
        )
        segment.segment_type = edge.kind
        if edge.kind == 'room':
            if edge.segment_index is None:
                raise ValueError(f'Room edge {edge.edge_id} has no segment id')
            segment.metadata_json = json.dumps(
                {'segment_id': edge.segment_index}
            )
        elif edge.kind == 'door':
            from_door = node_by_id[edge.from_node].door_index
            to_door = node_by_id[edge.to_node].door_index
            if from_door < 0 or from_door != to_door:
                raise ValueError(f'Door edge {edge.edge_id} has invalid door nodes')
            segment.metadata_json = json.dumps({'door_id': from_door})
        semantic_route.segments.append(segment)
    return semantic_route


def plan_semantic_route(
    start_pose: PoseStamped,
    goal_pose: PoseStamped,
    graph: RouteGraph,
    grid_map: OccupancyGrid,
    segmentation: np.ndarray,
    door_edge_cost: float = 3.0,
) -> ComputeSemanticRoute.Response:
    response = ComputeSemanticRoute.Response()
    response.error_code = SUCCESS
    response.error_msg = ''
    response.semantic_route = SemanticRoute()

    try:
        start_segment = _pose_segment_index(start_pose, grid_map, segmentation)
        if start_segment <= 0:
            return _error_response(
                response, START_NOT_FOUND,
                'Start pose is not on a valid segment',
            )
        goal_segment = _pose_segment_index(goal_pose, grid_map, segmentation)
        if goal_segment <= 0:
            return _error_response(
                response, GOAL_NOT_FOUND,
                'Goal pose is not on a valid segment',
            )

        if start_segment == goal_segment:
            segment = Segment()
            segment.start_pose = deepcopy(start_pose)
            segment.end_pose = deepcopy(goal_pose)
            segment.segment_type = 'room'
            segment.metadata_json = ''
            response.semantic_route.segments.append(segment)
            return response

        planning_graph = deepcopy(graph)
        node_ids = {node.node_id for node in planning_graph.nodes}
        start_node_id = _unique_id('__semantic_start__', node_ids)
        node_ids.add(start_node_id)
        goal_node_id = _unique_id('__semantic_goal__', node_ids)
        start_node = _add_planning_node(
            planning_graph, start_pose, start_node_id, 'start', start_segment
        )
        goal_node = _add_planning_node(
            planning_graph, goal_pose, goal_node_id, 'goal', goal_segment
        )

        existing_edge_ids = {edge.edge_id for edge in planning_graph.edges}

        def add_room_connections(planning_node: RouteNode) -> None:
            for node in planning_graph.nodes:
                if node.node_id == planning_node.node_id:
                    continue
                if node.segment_index != planning_node.segment_index:
                    continue
                edge_id = _unique_id('__semantic_edge__', existing_edge_ids)
                existing_edge_ids.add(edge_id)
                planning_graph.edges.append(
                    RouteEdge(
                        edge_id=edge_id,
                        from_node=planning_node.node_id,
                        to_node=node.node_id,
                        kind='room',
                        segment_index=planning_node.segment_index,
                    )
                )
                edge_id = _unique_id('__semantic_edge__', existing_edge_ids)
                existing_edge_ids.add(edge_id)
                planning_graph.edges.append(
                    RouteEdge(
                        edge_id=edge_id,
                        from_node=node.node_id,
                        to_node=planning_node.node_id,
                        kind='room',
                        segment_index=planning_node.segment_index,
                    )
                )

        add_room_connections(start_node)
        add_room_connections(goal_node)
        node_by_id, adjacency = _edge_costs(planning_graph, door_edge_cost)
        path = _shortest_path(adjacency, start_node_id, goal_node_id)
        if path is None:
            return _error_response(
                response, NO_ROUTE_FOUND,
                'No route connects the start and goal segments',
            )

        response.semantic_route = _route_segments(
            path,
            node_by_id,
            grid_map,
            start_pose,
            goal_pose,
            start_node_id,
            goal_node_id,
        )
        return response
    except Exception as error:
        return _error_response(response, UNKNOWN_ERROR, str(error))


def format_route_summary(
    semantic_route: SemanticRoute, default_room_segment_id: int
) -> str:
    segment_names = []
    for segment in semantic_route.segments:
        metadata = json.loads(segment.metadata_json) if segment.metadata_json else {}
        if segment.segment_type == 'room':
            segment_id = metadata.get('segment_id', default_room_segment_id)
            segment_names.append(f'room_{segment_id}')
        elif segment.segment_type == 'door':
            segment_names.append(f"door_{metadata['door_id']}")
        else:
            segment_names.append(segment.segment_type)
    return (
        f'Computed semantic route with {len(semantic_route.segments)} segments: '
        f"[{', '.join(segment_names)}]"
    )


def semantic_route_to_path(semantic_route: SemanticRoute) -> Path:
    path = Path()
    if not semantic_route.segments:
        return path

    first_segment = semantic_route.segments[0]
    path.header = deepcopy(first_segment.start_pose.header)
    path.poses = [deepcopy(first_segment.start_pose)]
    path.poses.extend(
        deepcopy(segment.end_pose) for segment in semantic_route.segments
    )
    return path
