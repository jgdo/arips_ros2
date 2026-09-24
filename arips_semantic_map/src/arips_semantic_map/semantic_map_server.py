#!/usr/bin/env python3

import rclpy

from arips_semantic_map.semantic_map import SemanticMap


def main():
    rclpy.init()
    node = rclpy.create_node('semantic_map_server')
    node.declare_parameter('map_file', '')
    map_file = node.get_parameter('map_file').get_parameter_value().string_value

    map_server = SemanticMap(node)
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
