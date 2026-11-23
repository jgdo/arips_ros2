"""Differentiable forward kinematics of the arips SCARA arm (arips_description/urdf/arips_onshape.urdf.xacro).

Chain in base_link: joint_slider (prismatic z) -> joint_2 -> joint_3 -> joint_4 (inverted axis), all
revolute joints about the vertical axis. Returns poses in base_link, metres / radians.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# Fixed joint origins from the URDF (xyz, yaw); every rpy roll/pitch is zero.
SLIDER_ORIGIN = (0.14, 0.135, 0.292, math.pi / 2)  # base_link -> link_0 (at slider = 0)
J2_ORIGIN = (-0.0962, 0.0, 0.06555)  # link_0 -> link_1
J3_ORIGIN = (-0.17, 0.0, 0.0)  # link_1 -> link_2
J4_ORIGIN = (-0.15, 0.0, -0.0516, -math.pi / 2)  # link_2 -> link_3, axis -z
GRIPPER_IN_LINK_3 = (0.0, -0.0533, -0.0248)  # link_3 -> door_gripper origin


def _rot(x: Tensor, y: Tensor, yaw: Tensor) -> tuple[Tensor, Tensor]:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * x - s * y, s * x + c * y


def link_3_pose(slider: Tensor, j2: Tensor, j3: Tensor, j4: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(x, y, z, yaw) of link_3 in base_link."""
    zero = torch.zeros_like(slider)
    yaw = zero + SLIDER_ORIGIN[3]
    x, y = zero + SLIDER_ORIGIN[0], zero + SLIDER_ORIGIN[1]
    z = slider + SLIDER_ORIGIN[2]

    dx, dy = _rot(zero + J2_ORIGIN[0], zero + J2_ORIGIN[1], yaw)
    x, y, z, yaw = x + dx, y + dy, z + J2_ORIGIN[2], yaw + j2

    dx, dy = _rot(zero + J3_ORIGIN[0], zero + J3_ORIGIN[1], yaw)
    x, y, z, yaw = x + dx, y + dy, z + J3_ORIGIN[2], yaw + j3

    dx, dy = _rot(zero + J4_ORIGIN[0], zero + J4_ORIGIN[1], yaw)
    x, y, z = x + dx, y + dy, z + J4_ORIGIN[2]
    yaw = yaw + J4_ORIGIN[3] - j4
    return x, y, z, torch.atan2(torch.sin(yaw), torch.cos(yaw))


def gripper_position(slider: Tensor, j2: Tensor, j3: Tensor, j4: Tensor) -> Tensor:
    """(..., 3) position of the door gripper origin in base_link."""
    x, y, z, yaw = link_3_pose(slider, j2, j3, j4)
    dx, dy = _rot(torch.zeros_like(x) + GRIPPER_IN_LINK_3[0], torch.zeros_like(x) + GRIPPER_IN_LINK_3[1], yaw)
    return torch.stack([x + dx, y + dy, z + GRIPPER_IN_LINK_3[2]], dim=-1)
