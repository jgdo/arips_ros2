from pathlib import Path

import arips_semantic_map_msgs.msg as smm
from arips_semantic_map_msgs.srv import AddDoor, HandleMapFile
from arips_semantic_map.semantic_map import SemanticMap
from arips_semantic_map.semantic_map_server import (
    _handle_add_door,
    _handle_load,
    _handle_save,
)


class _Logger:
    def error(self, message):
        pass


class _Node:
    def get_logger(self):
        return _Logger()


class _MapServer:
    def __init__(self):
        self.node = _Node()
        self.loaded_path = None
        self.saved_path = None
        self.added_door = None

    def load(self, filename):
        self.loaded_path = Path(filename)

    def save(self, filename):
        self.saved_path = Path(filename)

    def add_door(self, door):
        self.added_door = door
        return "door_7"


def _map_with_door():
    semantic_map = smm.SemanticMap()
    semantic_map.header.frame_id = "map"
    semantic_map.doors.append(
        smm.Door(
            pivot=smm.Point2D(x=1.0, y=2.0),
            extent=smm.Point2D(x=1.8, y=2.0),
            open_angle_deg=105.0,
        )
    )
    return semantic_map


def _server_for_map(semantic_map):
    server = SemanticMap.__new__(SemanticMap)
    server._map = semantic_map
    server.publish_map = lambda: None
    return server


def test_map_yaml_round_trip(tmp_path):
    source = _server_for_map(_map_with_door())
    path = tmp_path / "map.yaml"

    source.save(path)

    loaded = _server_for_map(smm.SemanticMap())
    loaded.load(path)

    assert loaded.map.header.frame_id == "map"
    assert len(loaded.map.doors) == 1
    assert loaded.map.doors[0].pivot.x == 1.0
    assert loaded.map.doors[0].extent.x == 1.8
    assert loaded.map.doors[0].open_angle_deg == 105.0


def test_add_door_copies_and_publishes():
    server = _server_for_map(smm.SemanticMap())
    door = smm.Door(
        pivot=smm.Point2D(x=1.0, y=2.0),
        extent=smm.Point2D(x=1.8, y=2.0),
        open_angle_deg=105.0,
    )

    name = server.add_door(door)
    door.pivot.x = 10.0

    assert name == "door_0"
    assert server.map.doors[0].pivot.x == 1.0


def test_map_file_callbacks():
    server = _MapServer()
    request = HandleMapFile.Request(file_path="/tmp/map.yaml")

    load_response = _handle_load(request, HandleMapFile.Response(), server)
    save_response = _handle_save(request, HandleMapFile.Response(), server)

    assert load_response.success
    assert save_response.success
    assert server.loaded_path == Path("/tmp/map.yaml")
    assert server.saved_path == Path("/tmp/map.yaml")


def test_map_file_callback_rejects_empty_path():
    server = _MapServer()
    response = _handle_load(HandleMapFile.Request(), HandleMapFile.Response(), server)

    assert not response.success
    assert "file_path" in response.error_message


def test_add_door_callback():
    server = _MapServer()
    request = AddDoor.Request(door=smm.Door(open_angle_deg=105.0))

    response = _handle_add_door(request, AddDoor.Response(), server)

    assert response.success
    assert response.new_door_name == "door_7"
    assert server.added_door.open_angle_deg == 105.0
