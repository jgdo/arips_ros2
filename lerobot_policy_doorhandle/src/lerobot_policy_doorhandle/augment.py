"""Batched GPU augmentation for the doorhandle policy (training only, runs on the resized images).

- color: LeRobot's default color transforms (brightness, contrast, saturation, hue, sharpness), each
  sample draws ``max_num_transforms`` of them with its own parameters (same as lerobot_policy_target_reach).
- geometric: one random rigid image transform per sample (rotation about the image centre plus a
  shift), applied identically to RGB (bilinear) and depth (nearest, so no invalid / edge depths get
  blended). No zoom and no perspective tilt: the apparent size stays consistent with the depth values.
- depth: additive uniform noise of fixed size (+-depth_noise_mm, independent of the distance), random
  rectangular holes and pixel dropout (holes = 0 = invalid).
Joint noise is added in the model (it needs the raw state).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

GRAY_WEIGHTS = (0.2989, 0.587, 0.114)  # torchvision's rgb_to_grayscale

# name -> range of the random factor, same as LeRobot's ImageTransformsConfig defaults.
COLOR_RANGES = {
    "brightness": (0.8, 1.2),
    "contrast": (0.8, 1.2),
    "saturation": (0.5, 1.5),
    "hue": (-0.05, 0.05),
    "sharpness": (0.5, 1.5),
}


# Constants are applied as Python scalars: building small tensors from Python lists on the GPU
# (new_tensor / torch.tensor(..., device=cuda)) is a blocking copy that stalls the training step.


def _gray(x: Tensor) -> Tensor:
    wr, wg, wb = GRAY_WEIGHTS
    return wr * x[:, 0:1] + wg * x[:, 1:2] + wb * x[:, 2:3]


def _blend(x: Tensor, other: Tensor, f: Tensor) -> Tensor:
    return (f * x + (1 - f) * other).clamp(0, 1)


def _brightness(x: Tensor, f: Tensor) -> Tensor:
    return (x * f).clamp(0, 1)


def _contrast(x: Tensor, f: Tensor) -> Tensor:
    return _blend(x, _gray(x).mean((-2, -1), keepdim=True), f)


def _saturation(x: Tensor, f: Tensor) -> Tensor:
    return _blend(x, _gray(x), f)


def _sharpness(x: Tensor, f: Tensor) -> Tensor:
    # torchvision's smoothing kernel; the 1-pixel border keeps the original pixels.
    # kernel [[1, 1, 1], [1, 5, 1], [1, 1, 1]] / 13 = (3x3 box sum + 4 * centre) / 13
    blurred = (9.0 * F.avg_pool2d(x, 3, stride=1) + 4.0 * x[..., 1:-1, 1:-1]) / 13.0
    smooth = x.clone()
    smooth[..., 1:-1, 1:-1] = blurred
    return _blend(x, smooth, f)


def _hue(x: Tensor, shift: Tensor) -> Tensor:
    """Shifts the HSV hue of (B, 3, H, W) by ``shift`` (B,) turns (exact RGB -> HSV -> RGB)."""
    r, g, b = x.unbind(1)
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    d = delta.clamp_min(1e-8)
    rc, gc, bc = (maxc - r) / d, (maxc - g) / d, (maxc - b) / d
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    h6 = ((h + shift.view(-1, 1, 1)) % 1.0).unsqueeze(1) * 6.0  # (B, 1, H, W)
    # closed-form HSV -> RGB: c_n = v - v s clamp(min(k, 4 - k), 0, 1), k = (n + 6h) mod 6, n = 5, 3, 1
    k = torch.cat([h6 + 5.0, h6 + 3.0, h6 + 1.0], 1) % 6.0
    v = maxc.unsqueeze(1)
    return v - v * s.unsqueeze(1) * torch.minimum(k, 4.0 - k).clamp(0, 1)


def color_augment(x: Tensor, max_num_transforms: int) -> Tensor:
    """(B, 3, H, W) in [0, 1]; each sample gets ``max_num_transforms`` distinct color transforms.

    Free of host syncs: every transform runs on the whole batch, samples that did not draw it get the
    identity parameter (factor 1, hue shift 0). Fixed order, like LeRobot's ImageTransforms.
    """
    names = list(COLOR_RANGES)
    n = x.shape[0]
    k = min(max_num_transforms, len(names))
    picked = torch.rand(n, len(names), device=x.device).argsort(1)[:, :k]
    chosen = torch.zeros(n, len(names), dtype=torch.bool, device=x.device)
    chosen.scatter_(1, picked, True)
    for j, name in enumerate(names):
        lo, hi = COLOR_RANGES[name]
        f = torch.where(chosen[:, j], torch.empty(n, device=x.device).uniform_(lo, hi), 1.0 if name != "hue" else 0.0)
        if name == "hue":
            x = _hue(x, f)
        else:
            x = {"brightness": _brightness, "contrast": _contrast, "saturation": _saturation,
                 "sharpness": _sharpness}[name](x, f.view(n, 1, 1, 1))
    return x


def random_homographies(n: int, height: int, width: int, rotation_deg: float, shift: float,
                        device) -> Tensor:
    """(n, 3, 3) rigid maps (rotation about the centre + shift) from output to input, in normalized
    grid coordinates ([-1, 1] spans the image). No scaling, so the image is neither zoomed nor tilted.

    Built in an isotropic frame (x scaled by the aspect ratio) so rotations stay rotations.
    """
    aspect = width / height
    a = torch.empty(n, device=device).uniform_(-rotation_deg, rotation_deg) * (math.pi / 180)
    tx = torch.empty(n, device=device).uniform_(-shift, shift) * 2 * aspect
    ty = torch.empty(n, device=device).uniform_(-shift, shift) * 2
    cos, sin = torch.cos(a), torch.sin(a)
    zero, one = torch.zeros_like(a), torch.ones_like(a)
    # m in the isotropic frame, returned as from_iso @ m @ to_iso with to_iso = diag(aspect, 1, 1)
    # written out element-wise: row 0 / aspect, column 0 * aspect
    return torch.stack([
        torch.stack([cos, -sin / aspect, tx / aspect], 1),
        torch.stack([sin * aspect, cos, ty], 1),
        torch.stack([zero, zero, one], 1),
    ], 1)


def warp(x: Tensor, homographies: Tensor, mode: str) -> Tensor:
    """Warps (B, C, H, W) with output->input homographies; outside pixels become 0."""
    b, _, h, w = x.shape
    ys = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) / w * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], -1).view(1, h * w, 3)
    mapped = pts @ homographies.to(x.dtype).transpose(1, 2)  # (B, HW, 3)
    grid = mapped[..., :2] / mapped[..., 2:].clamp_min(1e-6)
    return F.grid_sample(x, grid.view(b, h, w, 2), mode=mode, padding_mode="zeros", align_corners=False)


def depth_augment(depth: Tensor, noise_mm: float, max_holes: int, hole_size: float, dropout: float) -> Tensor:
    """(B, 1, H, W) depth in metres, 0 = invalid: uniform noise in [-noise_mm, noise_mm] millimetres
    (the same at every distance), rectangular holes, pixel dropout."""
    b, _, h, w = depth.shape
    valid = depth > 0
    depth = depth + (torch.rand_like(depth) * 2 - 1) * (noise_mm / 1000.0)
    invalid = torch.rand_like(depth) < dropout
    if max_holes > 0:
        n_holes = torch.randint(0, max_holes + 1, (b, 1), device=depth.device)
        k = torch.arange(max_holes, device=depth.device).view(1, -1)
        active = (k < n_holes).view(b, max_holes, 1, 1)
        size_h = torch.rand(b, max_holes, 1, 1, device=depth.device) * hole_size * h
        size_w = torch.rand(b, max_holes, 1, 1, device=depth.device) * hole_size * w
        top = torch.rand(b, max_holes, 1, 1, device=depth.device) * (h - size_h)
        left = torch.rand(b, max_holes, 1, 1, device=depth.device) * (w - size_w)
        yy = torch.arange(h, device=depth.device).view(1, 1, h, 1)
        xx = torch.arange(w, device=depth.device).view(1, 1, 1, w)
        holes = active & (yy >= top) & (yy < top + size_h) & (xx >= left) & (xx < left + size_w)
        invalid = invalid | holes.any(1, keepdim=True)
    return torch.where(valid & ~invalid, depth, torch.zeros_like(depth))


class DoorhandleAugment(nn.Module):
    """Augments (rgb (B,3,H,W) in [0,1], depth (B,1,H,W) metres) according to a DoorhandleConfig."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.last_homographies = None  # (B, 3, 3) output -> input of the last geometric warp, or None

    @property
    def enabled(self) -> bool:
        c = self.config
        return c.color_augmentation or c.geometric_augmentation or c.depth_augmentation

    @torch.no_grad()
    def forward(self, rgb: Tensor, depth: Tensor, points_fn=None) -> tuple[Tensor, Tensor, Tensor | None]:
        """-> (rgb, depth, points). ``points_fn(depth) -> (B, C, H, W)`` (optional) maps the noisy depth
        to a per-pixel point map before the geometric warp, so the points move with their pixels."""
        # always float32, also when training under bf16 / fp16 autocast
        with torch.autocast(rgb.device.type, enabled=False):
            return self._augment(rgb.float(), depth.float(), points_fn)

    def _augment(self, rgb: Tensor, depth: Tensor, points_fn=None) -> tuple[Tensor, Tensor, Tensor | None]:
        c = self.config
        n = rgb.shape[0]
        if c.color_augmentation:
            rgb = color_augment(rgb, c.color_max_num_transforms)
        # depth noise / holes are per pixel, so applying them before the warp is equivalent
        if c.depth_augmentation:
            aug = depth_augment(depth, c.depth_noise_mm, c.depth_max_holes, c.depth_hole_size, c.depth_dropout)
            apply = (torch.rand(n, 1, 1, 1, device=depth.device) < c.depth_prob)
            depth = torch.where(apply, aug, depth)
        points = points_fn(depth) if points_fn is not None else None
        self.last_homographies = None
        if c.geometric_augmentation:
            hom = random_homographies(n, rgb.shape[-2], rgb.shape[-1], c.geometric_rotation_deg,
                                      c.geometric_shift, rgb.device)
            keep = torch.rand(n, 1, 1, device=rgb.device) >= c.geometric_prob
            hom = torch.where(keep, torch.eye(3, device=rgb.device), hom)
            self.last_homographies = hom
            rgb = warp(rgb, hom, "bilinear")
            depth = warp(depth, hom, "nearest")
            if points is not None:
                points = warp(points, hom, "nearest")
        return rgb, depth, points
