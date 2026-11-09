"""Batched GPU image augmentation with LeRobot's default transforms.

Same transforms, ranges and selection as ``lerobot.transforms.ImageTransforms`` with the
visualizer's settings (each image gets ``max_num_transforms`` of the enabled transforms, applied
in a fixed order), but every sample of the batch draws its own transforms and parameters in one
vectorized pass. It runs on the already resized images, so it is cheap compared to the CPU
version that works on full-size frames inside the data loader workers.
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
AFFINE_DEGREES = 5.0
AFFINE_TRANSLATE = 0.05  # fraction of width / height


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


def _affine(x: Tensor) -> Tensor:
    n, _, h, w = x.shape
    angle = torch.empty(n, device=x.device).uniform_(-AFFINE_DEGREES, AFFINE_DEGREES) * (math.pi / 180)
    # Translation in normalized coordinates ([-1, 1] spans the image, so 2 * fraction).
    tx = torch.empty(n, device=x.device).uniform_(-AFFINE_TRANSLATE, AFFINE_TRANSLATE) * 2
    ty = torch.empty(n, device=x.device).uniform_(-AFFINE_TRANSLATE, AFFINE_TRANSLATE) * 2
    cos, sin = torch.cos(angle), torch.sin(angle)
    aspect = w / h
    # Rotation about the centre in pixel space, expressed for normalized grid coordinates.
    theta = torch.stack(
        [
            torch.stack([cos, -sin / aspect, -tx], 1),
            torch.stack([sin * aspect, cos, -ty], 1),
        ],
        1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class GpuAugment(nn.Module):
    """Augments a float image batch (B, 3, H, W) in [0, 1]; a no-op when nothing is enabled."""

    def __init__(self, color: bool, position: bool, max_num_transforms: int = 3):
        super().__init__()
        self.names = (list(COLOR_RANGES) if color else []) + (["affine"] if position else [])
        self.max_num_transforms = max_num_transforms

    @property
    def enabled(self) -> bool:
        return bool(self.names)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if not self.names:
            return x
        n = x.shape[0]
        k = min(self.max_num_transforms, len(self.names))
        # Each sample picks k distinct transforms with equal probability.
        picked = torch.rand(n, len(self.names), device=x.device).argsort(1)[:, :k]
        chosen = torch.zeros(n, len(self.names), dtype=torch.bool, device=x.device)
        chosen.scatter_(1, picked, True)

        x = x.clone()
        for j, name in enumerate(self.names):
            mask = chosen[:, j]
            if not mask.any():
                continue
            sel = x[mask]
            if name == "affine":
                out = _affine(sel)
            elif name == "hue":
                out = _hue(sel, torch.empty(len(sel), device=x.device).uniform_(*COLOR_RANGES["hue"]))
            else:
                f = _uniform(len(sel), *COLOR_RANGES[name], x.device)
                out = {"brightness": _brightness, "contrast": _contrast, "saturation": _saturation,
                       "sharpness": _sharpness}[name](sel, f)
            x[mask] = out
        return x
