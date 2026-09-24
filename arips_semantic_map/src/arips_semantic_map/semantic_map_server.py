#!/usr/bin/env python3

import rclpy
from arips_semantic_map_msgs.srv import AddDoor, HandleMapFile

from arips_semantic_map.semantic_map import SemanticMap


def _map_file_error(response, error: Exception):
    response.success = False
    response.error_message = str(error)
    return response


def _handle_load(request, response, map_server: SemanticMap):
    try:
        if not request.file_path.strip():
            raise ValueError("file_path must not be empty")
        map_server.load(request.file_path)
    except Exception as error:
        map_server.node.get_logger().error(f"Could not load semantic map: {error}")
        return _map_file_error(response, error)

    response.success = True
    response.error_message = ""
    return response


def _handle_save(request, response, map_server: SemanticMap):
    try:
        if not request.file_path.strip():
            raise ValueError("file_path must not be empty")
        map_server.save(request.file_path)
    except Exception as error:
        map_server.node.get_logger().error(f"Could not save semantic map: {error}")
        return _map_file_error(response, error)

    response.success = True
    response.error_message = ""
    return response


def _handle_add_door(request, response, map_server: SemanticMap):
    try:
        response.new_door_name = map_server.add_door(request.door)
    except Exception as error:
        map_server.node.get_logger().error(f"Could not add semantic map door: {error}")
        response.success = False
        response.new_door_name = ""
        response.error_message = str(error)
        return response

    response.success = True
    response.error_message = ""
    return response


def main():
    rclpy.init()
    node = rclpy.create_node('semantic_map_server')
    node.declare_parameter('map_file', '')
    map_file = node.get_parameter('map_file').get_parameter_value().string_value

    map_server = SemanticMap(node)
    node.create_service(
        HandleMapFile,
        '~/load_map',
        lambda request, response: _handle_load(request, response, map_server),
    )
    node.create_service(
        HandleMapFile,
        '~/save_map',
        lambda request, response: _handle_save(request, response, map_server),
    )
    node.create_service(
        AddDoor,
        '~/add_door',
        lambda request, response: _handle_add_door(request, response, map_server),
    )
    if map_file:
        node.get_logger().info(f"Using map file: {map_file}")
        map_server.load(map_file)
    else:
        node.get_logger().info("No map_file provided; starting with an empty map.")
        map_server.publish_map()
    node.get_logger().info("Semantic map server ready.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
