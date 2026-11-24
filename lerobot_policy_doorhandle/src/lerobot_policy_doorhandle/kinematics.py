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


# z of the door gripper origin in base_link minus the slider position (all joints rotate about z)
GRIPPER_Z_ABOVE_SLIDER = SLIDER_ORIGIN[2] + J2_ORIGIN[2] + J3_ORIGIN[2] + J4_ORIGIN[2] + GRIPPER_IN_LINK_3[2]


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


def arm_ik(gx: Tensor, gy: Tensor, yaw: Tensor, elbow: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    """(joint_2, joint_3, joint_4) that put the door gripper origin at (gx, gy) in base_link with link_3
    yaw ``yaw`` (inverse of link_3_pose / gripper_position in the horizontal plane; the slider sets z).

    elbow: sign of joint_3 (the pre-grasp labels all have joint_3 > 0). Unreachable targets are clamped
    to the stretched / folded arm; the result stays differentiable.
    """
    zero = torch.zeros_like(gx)
    dx, dy = _rot(zero + GRIPPER_IN_LINK_3[0], zero + GRIPPER_IN_LINK_3[1], yaw)
    x3, y3 = gx - dx, gy - dy  # link_3 origin
    ox, oy = _rot(zero + J2_ORIGIN[0], zero + J2_ORIGIN[1], zero + SLIDER_ORIGIN[3])
    px, py = x3 - (SLIDER_ORIGIN[0] + ox), y3 - (SLIDER_ORIGIN[1] + oy)  # from joint_2 axis
    a1, a2 = -J3_ORIGIN[0], -J4_ORIGIN[0]  # links point along -x of their frames
    c3 = ((px * px + py * py - a1 * a1 - a2 * a2) / (2 * a1 * a2)).clamp(-1 + 1e-4, 1 - 1e-4)
    j3 = elbow * torch.acos(c3)
    # direction of link_1 (-x of the joint_2 frame) is yaw0 + j2 + pi with yaw0 = SLIDER_ORIGIN yaw
    th1 = torch.atan2(py, px) - torch.atan2(a2 * torch.sin(j3), a1 + a2 * torch.cos(j3))
    j2 = th1 - SLIDER_ORIGIN[3] - math.pi
    j2 = torch.atan2(torch.sin(j2), torch.cos(j2))
    j4 = SLIDER_ORIGIN[3] + j2 + j3 + J4_ORIGIN[3] - yaw
    j4 = torch.atan2(torch.sin(j4), torch.cos(j4))
    return j2, j3, j4


def camera_points(depth_m: Tensor, tilt: Tensor, hfov: float, tilt_origin: tuple[float, float, float],
                  camera_in_tilt_link: tuple[float, float, float]) -> Tensor:
    """Back-projects a depth image to points in base_link.

    depth_m (B, 1, H, W) z-depth in metres along the optical axis (0 = invalid), tilt (B,) angle of the
    camera tilt joint (arips kinect_joint: origin ``tilt_origin`` in base_link, axis -y). The optical
    centre sits at ``camera_in_tilt_link`` in the tilted link and looks along its +x (optical frame
    x right, y down, z forward). Pinhole with square pixels: fx = fy = (W / 2) / tan(hfov / 2), the
    principal point in the image centre. Returns (B, 3, H, W) xyz [m]; invalid pixels get garbage
    values (mask them with depth > 0).
    """
    b, _, h, w = depth_m.shape
    f = (w / 2) / math.tan(hfov / 2)
    u = (torch.arange(w, device=depth_m.device, dtype=depth_m.dtype) + 0.5 - w / 2) / f
    v = (torch.arange(h, device=depth_m.device, dtype=depth_m.dtype) + 0.5 - h / 2) / f
    z = depth_m[:, 0]
    # optical (x right, y down, z forward) -> tilted link (x forward, y left, z up)
    lx = camera_in_tilt_link[0] + z
    ly = camera_in_tilt_link[1] - u.view(1, 1, w) * z
    lz = camera_in_tilt_link[2] - v.view(1, h, 1) * z
    # rotation about -y by tilt = rotation about +y by -tilt
    a = (-tilt).view(b, 1, 1).to(depth_m.dtype)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([tilt_origin[0] + c * lx + s * lz,
                        tilt_origin[1] + ly,
                        tilt_origin[2] - s * lx + c * lz], 1)


def project_points(p: Tensor, tilt: Tensor, hfov: float, image_hw: tuple[int, int],
                   tilt_origin: tuple[float, float, float],
                   camera_in_tilt_link: tuple[float, float, float]) -> tuple[Tensor, Tensor, Tensor]:
    """Inverse of camera_points for single points: p (B, 3) in base_link -> (u, v, z).

    u, v are normalized image coordinates in [-1, 1] across the image (the convention of the model's
    soft-argmax and of grid_sample with align_corners=False), z the depth along the optical axis [m].
    """
    h, w = image_hw
    t = math.tan(hfov / 2)
    dx, dy, dz = p[:, 0] - tilt_origin[0], p[:, 1] - tilt_origin[1], p[:, 2] - tilt_origin[2]
    a = -tilt.to(p.dtype)
    c, s = torch.cos(a), torch.sin(a)
    lx = c * dx - s * dz - camera_in_tilt_link[0]
    ly = dy - camera_in_tilt_link[1]
    lz = s * dx + c * dz - camera_in_tilt_link[2]
    z = lx
    zs = z.clamp_min(1e-3)
    return (-ly / zs) / t, (-lz / zs) * (w / h) / t, z


def surface_normals(points: Tensor, valid: Tensor, step: int = 2, smooth: int = 5,
                    max_jump: float = 0.03) -> Tensor:
    """Unit surface normals (B, 3, H, W) of a point map (B, 3, H, W), pointing towards the camera side
    of the surface is not enforced (sign follows the image axes: cross(right, down)).

    Central differences over ``step`` pixels; pixels with an invalid neighbour or a depth jump above
    ``max_jump`` metres between the neighbours get a zero normal. Box-smoothed over ``smooth`` pixels.
    """
    import torch.nn.functional as F  # noqa: N812

    p = F.pad(points, (step, step, step, step), mode="replicate")
    v = F.pad(valid.float(), (step, step, step, step))
    h, w = points.shape[-2:]
    s2 = 2 * step
    du = p[..., step:step + h, s2:s2 + w] - p[..., step:step + h, 0:w]
    dv = p[..., s2:s2 + h, step:step + w] - p[..., 0:h, step:step + w]
    ok = (v[..., step:step + h, s2:s2 + w] * v[..., step:step + h, 0:w] * v[..., s2:s2 + h, step:step + w]
          * v[..., 0:h, step:step + w] * valid.float())
    ok = ok * (du.norm(dim=1, keepdim=True) < max_jump).float() * (dv.norm(dim=1, keepdim=True) < max_jump).float()
    n = torch.cross(du, dv, dim=1) * ok
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-9) * ok
    if smooth > 1:
        n = F.avg_pool2d(n, smooth, stride=1, padding=smooth // 2, count_include_pad=False)
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-6) * (n.norm(dim=1, keepdim=True) > 1e-3).float()
    return n
