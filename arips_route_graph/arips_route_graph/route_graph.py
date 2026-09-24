from dataclasses import dataclass
from itertools import combinations
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple, Union

from arips_semantic_map_msgs.msg import Door, SemanticMap
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import OccupancyGrid
import numpy as np
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


Coordinate = Tuple[float, float]
SegmentPoint = Tuple[Coordinate, int]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class RouteNode:
    node_id: str
    coordinate: Coordinate
    door_index: int
    side: str
    segment_index: int


@dataclass(frozen=True)
class RouteEdge:
    edge_id: str
    from_node: str
    to_node: str
    kind: str
    segment_index: Optional[int] = None


@dataclass
class RouteGraph:
    nodes: List[RouteNode]
    edges: List[RouteEdge]
    segment_points: Dict[int, List[SegmentPoint]]
    skipped_door_indices: List[int]


def _quaternion_yaw(quaternion: Quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def world_to_grid_coordinates(
    grid_map: OccupancyGrid, x: float, y: float
) -> Coordinate:
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


def point_to_segment_index(
    grid_map: OccupancyGrid, segmentation: np.ndarray, point: Coordinate
) -> int:
    """Return the segment containing a world point, or zero when unassigned."""
    if segmentation.ndim != 2:
        raise ValueError('Segmentation data must be two-dimensional')
    grid_x, grid_y = world_to_grid_coordinates(grid_map, point[0], point[1])
    column = math.floor(grid_x)
    row = math.floor(grid_y)
    height, width = segmentation.shape
    if not (0 <= row < height and 0 <= column < width):
        return 0
    return int(segmentation[row, column])


def compute_approach_points(
    door: Door, approach_distance: float
) -> Tuple[Coordinate, Coordinate]:
    """Compute the open-side and opposite approach points for a door."""
    if not math.isfinite(approach_distance) or approach_distance < 0.0:
        raise ValueError('door_approach_distance must be finite and nonnegative')

    direction_x = door.extent.x - door.pivot.x
    direction_y = door.extent.y - door.pivot.y
    length = math.hypot(direction_x, direction_y)
    if not math.isfinite(length) or length == 0.0:
        raise ValueError('Door pivot and extent must be distinct finite points')

    center_x = (door.pivot.x + door.extent.x) * 0.5
    center_y = (door.pivot.y + door.extent.y) * 0.5
    side = 1.0 if door.open_angle_deg >= 0.0 else -1.0
    normal_x = side * -direction_y / length
    normal_y = side * direction_x / length
    return (
        (
            center_x + approach_distance * normal_x,
            center_y + approach_distance * normal_y,
        ),
        (
            center_x - approach_distance * normal_x,
            center_y - approach_distance * normal_y,
        ),
    )


def build_route_graph(
    semantic_map: SemanticMap,
    grid_map: OccupancyGrid,
    segmentation: np.ndarray,
    approach_distance: float,
) -> RouteGraph:
    """Build door and same-segment connectivity from map state."""
    nodes: List[RouteNode] = []
    segment_entries: Dict[int, List[Tuple[Coordinate, int, str]]] = {}
    door_nodes: Dict[int, Dict[str, str]] = {}
    skipped_door_indices: List[int] = []

    for door_index, door in enumerate(semantic_map.doors):
        try:
            approach_points = compute_approach_points(door, approach_distance)
        except ValueError:
            skipped_door_indices.append(door_index)
            continue

        door_nodes[door_index] = {}
        for side, coordinate in zip(('A', 'B'), approach_points):
            node_id = f'door_{door_index}_{side}'
            segment_index = point_to_segment_index(
                grid_map, segmentation, coordinate
            )
            nodes.append(
                RouteNode(
                    node_id=node_id,
                    coordinate=coordinate,
                    door_index=door_index,
                    side=side,
                    segment_index=segment_index,
                )
            )
            door_nodes[door_index][side] = node_id
            segment_entries.setdefault(segment_index, []).append(
                (coordinate, door_index, node_id)
            )

    edges: List[RouteEdge] = []
    edge_keys = set()

    def add_edge(
        from_node: str,
        to_node: str,
        kind: str,
        segment_index: Optional[int] = None,
    ) -> None:
        edge_key = tuple(sorted((from_node, to_node)))
        if edge_key in edge_keys:
            return
        edge_keys.add(edge_key)
        edges.append(
            RouteEdge(
                edge_id=f'edge_{len(edges)}',
                from_node=from_node,
                to_node=to_node,
                kind=kind,
                segment_index=segment_index,
            )
        )

    for door_index in sorted(door_nodes):
        add_edge(
            door_nodes[door_index]['A'],
            door_nodes[door_index]['B'],
            'door',
        )

    for segment_index, entries in segment_entries.items():
        if segment_index == 0:
            continue
        for first, second in combinations(entries, 2):
            if first[1] == second[1]:
                continue
            add_edge(first[2], second[2], 'segment', segment_index)

    segment_points = {
        segment_index: [
            (coordinate, door_index)
            for coordinate, door_index, _ in entries
        ]
        for segment_index, entries in segment_entries.items()
    }
    return RouteGraph(
        nodes=nodes,
        edges=edges,
        segment_points=segment_points,
        skipped_door_indices=skipped_door_indices,
    )


def make_route_graph_marker_array(
    graph: RouteGraph,
    frame_id: str,
    node_diameter: float,
    edge_width: float,
) -> MarkerArray:
    """Create green sphere and line markers for a route graph."""
    node_marker = Marker()
    node_marker.header.frame_id = frame_id
    node_marker.ns = 'route_graph_nodes'
    node_marker.id = 0
    node_marker.type = Marker.SPHERE_LIST
    node_marker.action = Marker.ADD
    node_marker.pose.orientation.w = 1.0
    node_marker.scale.x = node_diameter
    node_marker.scale.y = node_diameter
    node_marker.scale.z = node_diameter
    node_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
    node_marker.points = [
        Point(x=node.coordinate[0], y=node.coordinate[1], z=0.005)
        for node in graph.nodes
    ]

    node_by_id = {node.node_id: node for node in graph.nodes}
    edge_marker = Marker()
    edge_marker.header.frame_id = frame_id
    edge_marker.ns = 'route_graph_edges'
    edge_marker.id = 0
    edge_marker.type = Marker.LINE_LIST
    edge_marker.action = Marker.ADD
    edge_marker.pose.orientation.w = 1.0
    edge_marker.scale.x = edge_width
    edge_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
    for edge in graph.edges:
        from_coordinate = node_by_id[edge.from_node].coordinate
        to_coordinate = node_by_id[edge.to_node].coordinate
        edge_marker.points.extend([
            Point(x=from_coordinate[0], y=from_coordinate[1], z=0.01),
            Point(x=to_coordinate[0], y=to_coordinate[1], z=0.01),
        ])

    return MarkerArray(markers=[node_marker, edge_marker])


def route_graph_to_geojson(graph: RouteGraph, frame_id: str) -> dict:
    """Serialize a route graph as a local-map GeoJSON FeatureCollection."""
    node_by_id = {node.node_id: node for node in graph.nodes}
    features = []
    for node in graph.nodes:
        features.append(
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': list(node.coordinate),
                },
                'properties': {
                    'feature_type': 'node',
                    'node_id': node.node_id,
                    'door_index': node.door_index,
                    'side': node.side,
                    'segment_index': node.segment_index,
                },
            }
        )

    for edge in graph.edges:
        from_node = node_by_id[edge.from_node]
        to_node = node_by_id[edge.to_node]
        features.append(
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [
                        list(from_node.coordinate),
                        list(to_node.coordinate),
                    ],
                },
                'properties': {
                    'feature_type': 'edge',
                    'edge_id': edge.edge_id,
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'bidirectional': True,
                    'kind': edge.kind,
                    'segment_index': edge.segment_index,
                },
            }
        )

    return {
        'type': 'FeatureCollection',
        'properties': {
            'format': 'arips_route_graph_v1',
            'frame_id': frame_id,
            'coordinate_system': 'local_map_xy',
        },
        'features': features,
    }


def write_route_graph_geojson(
    graph: RouteGraph, frame_id: str, path: PathLike
) -> Path:
    """Atomically overwrite a GeoJSON route graph file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=target.parent,
            prefix=f'.{target.name}.',
            suffix='.tmp',
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                route_graph_to_geojson(graph, frame_id),
                temporary_file,
                indent=2,
            )
            temporary_file.write('\n')
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target
