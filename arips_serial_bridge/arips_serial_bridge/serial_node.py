#!/usr/bin/env python3
"""
ROS2 node that bridges a custom serial protocol to ROS2 topics.

Serial protocol format (newline-terminated):
  pub <topic_name> <message_type> <json>\n

Example:
  pub imu sensor_msgs/msg/Imu {"header": {...}, "orientation": {...}, ...}

Usage:
  python3 serial_bridge_node.py
  python3 serial_bridge_node.py --ros-args -p port:=/dev/ttyACM0 -p baudrate:=115200
"""

import json
import importlib
import threading
import math

import rclpy
from rclpy.node import Node
import serial
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.convert import message_to_ordereddict


def resolve_msg_class(type_str: str):
    """Resolve 'sensor_msgs/msg/Imu' -> sensor_msgs.msg.Imu class."""
    parts = type_str.split("/")
    if len(parts) != 3:
        return None
    pkg, interface, name = parts
    try:
        module = importlib.import_module(f"{pkg}.{interface}")
    except ImportError:
        return None
    return getattr(module, name, None)


def msg_to_json(msg) -> str:
    """Serialize a ROS2 message to a JSON string."""
    d = message_to_ordereddict(msg)
    return json.dumps(d)


def null_to_nan(obj):
    """Recursively replace JSON null values with float('nan')."""
    if isinstance(obj, dict):
        return {k: null_to_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [null_to_nan(v) for v in obj]
    if obj is None:
        return float("nan")
    return obj


class SerialBridgeNode(Node):
    def __init__(self):
        super().__init__("serial_bridge")

        # Parameters
        self.declare_parameter("port", "/dev/arduino_base")
        self.declare_parameter("baudrate", 1000000)

        port = self.get_parameter("port").value
        baudrate = self.get_parameter("baudrate").value

        self.get_logger().info(f"Opening serial port {port} at {baudrate} baud")
        self.ser = serial.Serial(port, baudrate, timeout=0.1)

        # Send two newlines to synchronize the serial connection
        self.ser.write(b"\n\n")

        # Send current ROS timestamp to the device upon connection
        now = self.get_clock().now().to_msg()
        ts_line = f"timestamp {now.sec} {now.nanosec}\n"
        self.ser.write(ts_line.encode("utf-8"))
        self.get_logger().info(f"Sent timestamp: {now.sec} {now.nanosec}")

        # Request subscription list from Arduino
        self.ser.write(b"list_subs\n")

        # Dynamic publishers: topic_name -> (publisher, msg_class)
        self._serial_publishers: dict[str, tuple] = {}

        # Lock for serial writes from multiple subscription callbacks
        self._serial_lock = threading.Lock()

        # ROS2 subscriptions (populated dynamically from Arduino response)
        self._ros_subs = []

        # Start background serial reader
        self._running = True
        self._reader_thread = threading.Thread(target=self._serial_reader, daemon=True)
        self._reader_thread.start()

    # ------------------------------------------------------------------
    # ROS2 -> Serial (subscriptions)
    # ------------------------------------------------------------------

    def _send_to_serial(self, topic_name: str, type_str: str, msg):
        line = f"pub {topic_name} {type_str} {msg_to_json(msg)}\n"
        with self._serial_lock:
            try:
                self.ser.write(line.encode("utf-8"))
            except serial.SerialException as exc:
                self.get_logger().error(f"Serial write error: {exc}")

    # ------------------------------------------------------------------
    # Serial -> ROS2 (publishers, created on demand)
    # ------------------------------------------------------------------

    def _get_or_create_publisher(self, topic_name: str, type_str: str):
        if topic_name in self._serial_publishers:
            return self._serial_publishers[topic_name]

        msg_class = resolve_msg_class(type_str)
        if msg_class is None:
            self.get_logger().warn(f"Could not resolve message type: {type_str}")
            return None

        pub = self.create_publisher(msg_class, topic_name, 10)
        entry = (pub, msg_class)
        self._serial_publishers[topic_name] = entry
        self.get_logger().info(f"Created publisher: {topic_name} [{type_str}]")
        return entry

    def _create_subscription(self, topic_name: str, type_str: str):
        msg_class = resolve_msg_class(type_str)
        if msg_class is None:
            self.get_logger().warn(f"Could not resolve message type: {type_str}")
            return

        def _make_cb(tn, ts):
            def _cb(msg):
                self._send_to_serial(tn, ts, msg)
            return _cb

        sub = self.create_subscription(
            msg_class, topic_name, _make_cb(topic_name, type_str), 10
        )
        self._ros_subs.append(sub)
        self.get_logger().info(
            f"Subscribed to ROS2 topic: {topic_name} [{type_str}]"
        )

    def _handle_subscriptions_response(self, json_str: str):
        try:
            entries = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to parse subscriptions: {exc}")
            return
        if not isinstance(entries, list):
            self.get_logger().error("subscriptions response is not a list")
            return
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for topic_name, type_str in entry.items():
                self._create_subscription(topic_name, type_str)

    def _process_line(self, line: str):
        # Format: "subscriptions <json>"
        if line.startswith("subscriptions "):
            self._handle_subscriptions_response(line[len("subscriptions "):])
            return

        # Format: "pub <topic> <type> <json>"
        if not line.startswith("pub "):
            return
        line = line[4:]  # strip "pub " prefix

        first_space = line.find(" ")
        if first_space < 0:
            return
        topic_name = line[:first_space]

        rest = line[first_space + 1 :]
        second_space = rest.find(" ")
        if second_space < 0:
            return
        type_str = rest[:second_space]
        json_str = rest[second_space + 1 :]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self.get_logger().debug(f"JSON parse error on topic {topic_name}: {exc}")
            return

        data = null_to_nan(data)

        result = self._get_or_create_publisher(topic_name, type_str)
        if result is None:
            return

        pub, msg_class = result
        try:
            msg = msg_class()
            set_message_fields(msg, data)
            pub.publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish {topic_name}: {exc}")

    # ------------------------------------------------------------------
    # Background serial reader thread
    # ------------------------------------------------------------------

    def _serial_reader(self):
        buf = b""
        while self._running:
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if line:
                        self._process_line(line)
            except serial.SerialException as exc:
                self.get_logger().error(f"Serial read error: {exc}")
                break
            except Exception as exc:
                self.get_logger().warn(f"Reader error: {exc}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._running = False
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
