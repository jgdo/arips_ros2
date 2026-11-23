"""Gazebo helpers for the lu world (gz CLI): teleport the robot, respawn the door raised."""

import math
import os
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory


def gz(args, timeout=10.0):
    """Run a gz CLI command, return stdout ('' on failure or timeout)."""
    try:
        return subprocess.run(['gz'] + args, capture_output=True, text=True,
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ''


def _quat_z(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class SimWorld:
    def __init__(self, world='lu', robot='arips', door='door'):
        self.world = world
        self.robot = robot
        self.door = door
        self._door_model = self._load_door_model()
        self._tmp_dir = tempfile.mkdtemp(prefix='grasp_dataset_door_')

    # ---- robot ------------------------------------------------------------------------------

    def teleport(self, x, y, yaw):
        qz, qw = _quat_z(yaw)
        req = (f'name: "{self.robot}", position: {{x: {x}, y: {y}, z: 0.01}}, '
               f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}')
        out = gz(['service', '-s', f'/world/{self.world}/set_pose', '--reqtype', 'gz.msgs.Pose',
                  '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req])
        return 'true' in out

    # ---- door -------------------------------------------------------------------------------

    def _load_door_model(self):
        """The door <model> element of the installed world file."""
        path = os.path.join(get_package_share_directory('arips_gazebo'), 'worlds',
                            f'{self.world}.sdf')
        root = ET.parse(path).getroot()
        self._sdf_version = root.get('version', '1.9')
        for model in root.iter('model'):
            if model.get('name') == self.door:
                return model
        raise RuntimeError(f'No model "{self.door}" in {path}')

    def _door_sdf_file(self, z_offset):
        """Write the door model with its pose raised by z_offset, return the file path."""
        model = ET.fromstring(ET.tostring(self._door_model))
        pose = model.find('pose')
        values = [float(v) for v in pose.text.split()]
        values[2] += z_offset
        pose.text = ' '.join(f'{v:.6f}' for v in values)
        sdf = ET.Element('sdf', version=self._sdf_version)
        sdf.append(model)
        path = os.path.join(self._tmp_dir, 'door.sdf')
        ET.ElementTree(sdf).write(path, xml_declaration=True, encoding='utf-8')
        return path

    def pose_info(self):
        return gz(['topic', '-e', '-t', f'/world/{self.world}/pose/info', '-n', '1'],
                  timeout=5.0)

    def door_present(self, info=None):
        info = self.pose_info() if info is None else info
        return f'name: "{self.door}"' in info

    def door_z(self, info=None):
        """z of the door model from the world pose info, None if not found."""
        info = self.pose_info() if info is None else info
        i = info.find(f'name: "{self.door}"')
        if i < 0:
            return None
        position = info[i:].split('position', 1)
        if len(position) < 2:
            return None
        block = position[1].split('}', 1)[0]
        z = re.search(r'\bz:\s*(-?[\d.e+-]+)', block)
        return float(z.group(1)) if z else 0.0  # gz omits zero fields

    def leaf_angle_deg(self, info=None):
        """Hinge angle of the door leaf, None if unavailable (leaf pose is model relative)."""
        info = self.pose_info() if info is None else info
        i = info.find('name: "leaf"')
        if i < 0:
            return None
        orientation = info[i:].split('orientation', 1)
        if len(orientation) < 2:
            return None
        z = re.search(r'\bz:\s*(-?[\d.e+-]+)', orientation[1].split('}', 1)[0])
        qz = float(z.group(1)) if z else 0.0
        return math.degrees(2.0 * math.asin(max(-1.0, min(1.0, qz))))

    def respawn_door(self, z_offset, timeout=15.0):
        """Remove the door and create it again (closed) raised by z_offset.

        The door frame is fixed to the world, so it cannot be moved with set_pose.
        Returns None on success, else an error string.
        """
        if self.door_present():
            out = gz(['service', '-s', f'/world/{self.world}/remove',
                      '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
                      '--timeout', '5000', '--req', f'name: "{self.door}", type: MODEL'])
            if 'true' not in out:
                return 'removing the door failed'
            deadline = time.monotonic() + timeout
            while self.door_present():
                if time.monotonic() > deadline:
                    return 'door still present after remove'
                time.sleep(0.2)

        path = self._door_sdf_file(z_offset)
        out = gz(['service', '-s', f'/world/{self.world}/create',
                  '--reqtype', 'gz.msgs.EntityFactory', '--reptype', 'gz.msgs.Boolean',
                  '--timeout', '5000', '--req', f'sdf_filename: "{path}"'])
        if 'true' not in out:
            return 'creating the door failed'
        expected_z = float(self._door_model.find('pose').text.split()[2]) + z_offset
        deadline = time.monotonic() + timeout
        while True:
            info = self.pose_info()
            z = self.door_z(info)
            angle = self.leaf_angle_deg(info)
            if z is not None and abs(z - expected_z) < 0.002 and angle is not None \
                    and abs(angle) < 1.0:
                return None
            if time.monotonic() > deadline:
                return f'respawned door not where expected (z {z}, leaf angle {angle})'
            time.sleep(0.2)
