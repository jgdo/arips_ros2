#!/usr/bin/env python3
"""Reads raw ADC values from a serial port and publishes them as JointState."""

import math

import rclpy
import serial
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt16MultiArray

JOINT_FIELDS = (
    'index', 'low_adc', 'low_angle_deg', 'high_adc', 'high_angle_deg',
    'min_angle_deg', 'max_angle_deg')


def adc_to_angle(
        adc, low_adc, low_angle_deg, high_adc, high_angle_deg,
        min_angle_deg, max_angle_deg):
    """Linearly map adc to an angle and clamp it to the configured limits."""
    if high_adc == low_adc:
        angle_deg = low_angle_deg
    else:
        ratio = (adc - low_adc) / (high_adc - low_adc)
        angle_deg = low_angle_deg + ratio * (high_angle_deg - low_angle_deg)
    return min(max(angle_deg, min_angle_deg), max_angle_deg)


class AdcJointStateNode(Node):
    def __init__(self):
        super().__init__(
            'adc_joint_state',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True)

        port = self._param('port', '/dev/ttyACM0')
        baudrate = int(self._param('baudrate', 115200))
        topic = self._param('topic', 'joint_states')
        self._frame_id = self._param('frame_id', '')
        self._num_values = int(self._param('num_values', 0))

        self._joints = self._load_joints()
        self._validate()

        self._pub = self.create_publisher(JointState, topic, 10)
        self._raw_pub = self.create_publisher(UInt16MultiArray, '~/raw_values', 10)

        self.get_logger().info(f'Opening serial port {port} at {baudrate} baud')
        self._ser = serial.Serial(port, baudrate, timeout=0.1)

        self._timer = self.create_timer(0.001, self._read_once)

    def _param(self, name, default):
        if not self.has_parameter(name):
            self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _load_joints(self):
        params = self.get_parameters_by_prefix('joints')
        joints = {}
        for key, param in params.items():
            name, _, field = key.rpartition('.')
            if not name or field not in JOINT_FIELDS:
                raise ValueError(f"Invalid joint parameter 'joints.{key}'")
            joints.setdefault(name, {})[field] = param.value
        return joints

    def _validate(self):
        if self._num_values <= 0:
            raise ValueError('num_values must be a positive integer')
        if not self._joints:
            raise ValueError("No joints configured under 'joints'")
        used = {}
        for name, cfg in self._joints.items():
            missing = [f for f in JOINT_FIELDS if f not in cfg]
            if missing:
                raise ValueError(f"Joint '{name}' is missing fields: {missing}")
            if cfg['min_angle_deg'] > cfg['max_angle_deg']:
                raise ValueError(
                    f"Joint '{name}' min_angle_deg must not exceed max_angle_deg")
            idx = int(cfg['index'])
            cfg['index'] = idx
            if idx < 0 or idx >= self._num_values:
                raise ValueError(
                    f"Joint '{name}' index {idx} is out of range "
                    f'for num_values={self._num_values}')
            if idx in used:
                raise ValueError(
                    f"Joints '{used[idx]}' and '{name}' share index {idx}")
            used[idx] = name

    def _read_once(self):
        try:
            raw = self._ser.readline()
        except serial.SerialException as exc:
            self.get_logger().error(f'Serial read error: {exc}')
            raise
        if not raw.endswith(b'\n'):
            return  # timeout or partial line

        try:
            values = [int(tok) for tok in raw.decode('ascii').split()]
        except (ValueError, UnicodeDecodeError):
            self.get_logger().warning(
                f'Discarding unparsable line: {raw!r}', throttle_duration_sec=2.0)
            return

        if len(values) != self._num_values:
            self.get_logger().warning(
                f'Discarding reading with {len(values)} values, '
                f'expected {self._num_values}', throttle_duration_sec=2.0)
            return

        if any(v < 0 or v > 0xFFFF for v in values):
            self.get_logger().warning(
                f'Discarding reading with values outside uint16 range: {values}',
                throttle_duration_sec=2.0)
            return

        raw_msg = UInt16MultiArray()
        raw_msg.data = values
        self._raw_pub.publish(raw_msg)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        for name, cfg in self._joints.items():
            angle_deg = adc_to_angle(
                values[cfg['index']], cfg['low_adc'], cfg['low_angle_deg'],
                cfg['high_adc'], cfg['high_angle_deg'],
                cfg['min_angle_deg'], cfg['max_angle_deg'])
            msg.name.append(name)
            msg.position.append(math.radians(angle_deg))
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AdcJointStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
