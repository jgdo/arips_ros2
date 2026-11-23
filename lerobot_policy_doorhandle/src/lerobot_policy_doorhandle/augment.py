"""Batched GPU augmentation for the doorhandle policy (training only, runs on the resized images).

- color: LeRobot's default color transforms (brightness, contrast, saturation, hue, sharpness), each
  sample draws ``max_num_transforms`` of them with its own parameters (same as lerobot_policy_target_reach).
- geometric: one random homography per sample (zoom, rotation, shift, perspective tilt), applied
  identically to RGB (bilinear) and depth (nearest, so no invalid / edge depths get blended).
- depth: multiplicative noise, random rectangular holes and pixel dropout (holes = 0 = invalid).
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


def _uniform(n: int, lo: float, hi: float, device) -> Tensor:
    return torch.empty(n, 1, 1, 1, device=device).uniform_(lo, hi)


def _gray(x: Tensor) -> Tensor:
    w = x.new_tensor(GRAY_WEIGHTS).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


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
    k = x.new_tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]]) / 13
    blurred = F.conv2d(x, k.expand(3, 1, 3, 3), groups=3)
    smooth = x.clone()
    smooth[..., 1:-1, 1:-1] = blurred
    return _blend(x, smooth, f)


def _hue(x: Tensor, shift: Tensor) -> Tensor:
    r, g, b = x.unbind(1)
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    delta = maxc - minc
    v = maxc
    s = torch.where(maxc > 0, delta / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    d = delta.clamp_min(1e-8)
    rc, gc, bc = (maxc - r) / d, (maxc - g) / d, (maxc - b) / d
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(delta > 0, (h / 6.0) % 1.0, torch.zeros_like(h))
    h = (h + shift.view(-1, 1, 1)) % 1.0

    i = torch.floor(h * 6.0)
    fr = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - s * fr)
    t = v * (1 - s * (1 - fr))
    i = i.long() % 6
    choices = torch.stack(
        [torch.stack(c, 1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))], 0
    )  # (6, B, 3, H, W)
    return choices.gather(0, i.unsqueeze(0).unsqueeze(2).expand(1, -1, 3, -1, -1)).squeeze(0)




def color_augment(x: Tensor, max_num_transforms: int) -> Tensor:
    """(B, 3, H, W) in [0, 1]; each sample gets ``max_num_transforms`` distinct color transforms."""
    names = list(COLOR_RANGES)
    n = x.shape[0]
    k = min(max_num_transforms, len(names))
    picked = torch.rand(n, len(names), device=x.device).argsort(1)[:, :k]
    chosen = torch.zeros(n, len(names), dtype=torch.bool, device=x.device)
    chosen.scatter_(1, picked, True)
    x = x.clone()
    for j, name in enumerate(names):
        mask = chosen[:, j]
        if not mask.any():
            continue
        sel = x[mask]
        if name == "hue":
            out = _hue(sel, torch.empty(len(sel), device=x.device).uniform_(*COLOR_RANGES["hue"]))
        else:
            f = _uniform(len(sel), *COLOR_RANGES[name], x.device)
            out = {"brightness": _brightness, "contrast": _contrast, "saturation": _saturation,
                   "sharpness": _sharpness}[name](sel, f)
        x[mask] = out
    return x


def random_homographies(n: int, height: int, width: int, zoom: tuple[float, float], rotation_deg: float,
                        shift: float, perspective: float, device) -> Tensor:
    """(n, 3, 3) maps from output to input in normalized grid coordinates ([-1, 1] spans the image).

    Built in an isotropic frame (x scaled by the aspect ratio) so rotations stay rotations.
    """
    aspect = width / height
    s = torch.empty(n, device=device).uniform_(*zoom)
    a = torch.empty(n, device=device).uniform_(-rotation_deg, rotation_deg) * (math.pi / 180)
    tx = torch.empty(n, device=device).uniform_(-shift, shift) * 2 * aspect
    ty = torch.empty(n, device=device).uniform_(-shift, shift) * 2
    px = torch.empty(n, device=device).uniform_(-perspective, perspective) / aspect
    py = torch.empty(n, device=device).uniform_(-perspective, perspective)
    cos, sin = torch.cos(a), torch.sin(a)
    one = torch.ones_like(s)
    # output -> input: dividing by the zoom factor makes zoom > 1 magnify the image.
    m = torch.stack([
        torch.stack([cos / s, -sin / s, tx], 1),
        torch.stack([sin / s, cos / s, ty], 1),
        torch.stack([px, py, one], 1),
    ], 1)
    to_iso = torch.diag(torch.tensor([aspect, 1.0, 1.0], device=device))
    from_iso = torch.diag(torch.tensor([1.0 / aspect, 1.0, 1.0], device=device))
    return from_iso @ m @ to_iso


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


def depth_augment(depth: Tensor, noise_std: float, max_holes: int, hole_size: float, dropout: float) -> Tensor:
    """(B, 1, H, W) depth in metres, 0 = invalid: relative noise, rectangular holes, pixel dropout."""
    b, _, h, w = depth.shape
    valid = depth > 0
    depth = depth * (1 + noise_std * torch.randn_like(depth))
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

    @property
    def enabled(self) -> bool:
        c = self.config
        return c.color_augmentation or c.geometric_augmentation or c.depth_augmentation

    @torch.no_grad()
    def forward(self, rgb: Tensor, depth: Tensor) -> tuple[Tensor, Tensor]:
        c = self.config
        n = rgb.shape[0]
        if c.color_augmentation:
            rgb = color_augment(rgb, c.color_max_num_transforms)
        if c.geometric_augmentation:
            hom = random_homographies(n, rgb.shape[-2], rgb.shape[-1], tuple(c.geometric_zoom),
                                      c.geometric_rotation_deg, c.geometric_shift, c.geometric_perspective,
                                      rgb.device)
            keep = torch.rand(n, device=rgb.device) >= c.geometric_prob
            hom[keep] = torch.eye(3, device=rgb.device)
            rgb = warp(rgb, hom, "bilinear")
            depth = warp(depth, hom, "nearest")
        if c.depth_augmentation:
            aug = depth_augment(depth, c.depth_noise_std, c.depth_max_holes, c.depth_hole_size, c.depth_dropout)
            apply = (torch.rand(n, 1, 1, 1, device=depth.device) < c.depth_prob)
            depth = torch.where(apply, aug, depth)
        return rgb, depth
