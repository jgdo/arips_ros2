from pathlib import Path
from typing import List, Optional, Tuple
import arips_semantic_map_msgs.msg as smm
import yaml
import math
from rosidl_runtime_py.convert import message_to_yaml
from rosidl_runtime_py.utilities import get_message
from interactive_markers.interactive_marker_server import InteractiveMarkerServer, InteractiveMarker
from visualization_msgs.msg import Marker, InteractiveMarkerControl, InteractiveMarkerFeedback
import geometry_msgs.msg
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np


def _message_from_yaml(values, msg) -> None:
    """Populate a ROS 2 message from the YAML mapping used by message_to_yaml."""
    field_types = msg.get_fields_and_field_types()
    for field_name, value in values.items():
        field_type = field_types[field_name]
        if field_type.startswith('sequence<'):
            element_type = field_type[len('sequence<'):-1]
            element_class = get_message(element_type)
            setattr(msg, field_name, [
                _message_from_value(element, element_class)
                for element in value
            ])
        elif hasattr(getattr(msg, field_name), 'get_fields_and_field_types'):
            _message_from_yaml(value, getattr(msg, field_name))
        else:
            setattr(msg, field_name, value)


def _message_from_value(value, message_class):
    message = message_class()
    _message_from_yaml(value, message)
    return message


def point_to_numpy(p: smm.Point2D) -> np.array:
    return np.array([p.x, p.y])


def get_door_open_extent(door: smm.Door, angle_rad=None) -> smm.Point2D:
    if angle_rad is None:
        angle_rad = math.radians(door.open_angle_deg)
    rot_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

    pivot = point_to_numpy(door.pivot)
    extent = point_to_numpy(door.extent)

    p = rot_matrix @ (extent - pivot) + pivot
    return smm.Point2D(x=p[0], y=p[1])


class SemanticMap:
    def __init__(self, node):
        self.node = node
        self._map = smm.SemanticMap()
        self._map.header.frame_id = "map"
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = node.create_publisher(smm.SemanticMap, "semantic_map", qos)
        self.marker_server = InteractiveMarkerServer(node, "semantic_map_markers")
        self.marker_server.applyChanges()

    @property
    def map(self) -> smm.SemanticMap:
        return self._map

    @map.setter
    def map(self, new_map: smm.SemanticMap):
        if not isinstance(new_map, smm.SemanticMap):
            raise ValueError("Semantic map must be a SemanticMap")
        self._map = new_map
        self.publish_map()

    def save(self, filename: Path):
        with open(filename, "w") as file:
            file.write(message_to_yaml(self.map))

    def load(self, filename: Path):
        with open(filename, "r") as file:
            yaml_msg = yaml.safe_load(file)
            map = smm.SemanticMap()
            _message_from_yaml(yaml_msg, map)
            self.map = map

    def publish_map(self):
        self.map_pub.publish(self.map)

        all_markers: List[Tuple] = []

        for index, door in enumerate(self.map.doors):
            all_markers.append(self._create_door_point_control(index, door.pivot, "pivot"))
            all_markers.append(self._create_door_point_control(index, door.extent, "extent"))
            all_markers.append(self._create_door_polygon(index))

        self.marker_server.clear()
        for marker, feedback in all_markers:
            self.marker_server.insert(
                marker,
                feedback_callback=feedback,
                feedback_type=InteractiveMarkerFeedback.MOUSE_UP,
            )
        self.marker_server.applyChanges()

    def _create_door_point_control(self, index, point, field_name):
        # create an interactive marker for our server
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = f"door_{index}_{field_name}"

        int_marker.pose.position.x = point.x
        int_marker.pose.position.y = point.y
        int_marker.pose.position.z = 0.001  # to stand out in rviz
        int_marker.pose.orientation.w = 1.0
        int_marker.pose.orientation.y = 1.0

        # create a grey box marker
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = 0.1
        sphere.scale.y = 0.1
        sphere.scale.z = 0.1
        sphere.color.r = 0.0
        sphere.color.g = 0.5
        sphere.color.b = 0.5
        sphere.color.a = 1.0

        # create a control which will move the sphere
        move_control = InteractiveMarkerControl()
        move_control.name = "move_xy"
        move_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE
        move_control.markers.append(sphere)

        # add the control to the interactive marker
        int_marker.controls.append(move_control)

        return int_marker, self._create_door_callback(index, point)

    def _create_door_polygon(self, index):
        def to_point(p):
            return geometry_msgs.msg.Point(x=p.x, y=p.y)

        door: smm.Door = self.map.doors[index]

        # create an interactive marker for our server
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "map"
        int_marker.name = f"door_{index}_outline"

        int_marker.pose.position.z = 0.001  # to stand out in rviz

        # create a control which will move the sphere
        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.NONE

        # create a grey box marker
        line = Marker()
        line.type = Marker.LINE_STRIP
        line.scale.x = 0.03  # line width
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.0
        line.color.a = 1.0
        line.points = [to_point(door.pivot), to_point(door.extent)]
        control.markers.append(line)

        line = Marker()
        line.type = Marker.LINE_STRIP
        line.scale.x = 0.01  # line width
        line.color.r = 0.5
        line.color.g = 0.7
        line.color.b = 0.5
        line.color.a = 1.0

        angle_rad = math.radians(door.open_angle_deg)
        num_steps = int(abs(angle_rad) / math.radians(15))
        if num_steps:
            stepsize = angle_rad / num_steps
            line.points = [
                to_point(get_door_open_extent(door, i * stepsize))
                for i in range(num_steps + 1)
            ]
        else:
            line.points = [to_point(door.extent)]
        line.points.append(to_point(door.pivot))
        control.markers.append(line)

        # add the control to the interactive marker
        int_marker.controls.append(control)

        return int_marker, None

    def _create_door_callback(self, door_index: int, point: Optional[smm.Point2D]):
        return lambda msg: self._interactive_marker_door_callback(msg, door_index, point)

    def _interactive_marker_door_callback(
        self, feedback: InteractiveMarkerFeedback, door_index: int, point: Optional[smm.Point2D]
    ):
        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return

        assert feedback.header.frame_id == "map"

        print(f"------------- setting pose for door {door_index} ----------------")
        print(feedback)
        point.x = feedback.pose.position.x
        point.y = feedback.pose.position.y

        self.publish_map()
