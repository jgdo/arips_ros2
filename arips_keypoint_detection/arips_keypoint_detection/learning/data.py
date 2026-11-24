"""Keypoint dataset with augmentation for heatmap training."""

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .. import dataset_io as dio
from . import preprocess as pp


@dataclass
class Sample:
    rgb_path: Path
    depth_path: Path | None
    bag: str
    keypoints: np.ndarray      # K x 2, corner coordinates in the reference (RGB) frame
    visible: np.ndarray        # K bool


@dataclass
class AugmentConfig:
    pan: float = 0.15            # fraction of the image size
    rotate_deg: float = 15.0
    zoom: tuple = (0.75, 1.3)
    shear_deg: float = 5.0
    hflip: bool = False          # changes left/right semantics of some keypoints, off by default
    brightness: float = 0.15
    contrast: tuple = (0.6, 1.4)
    gamma: tuple = (0.6, 1.6)
    saturation: tuple = (0.6, 1.4)
    hue_deg: float = 8.0
    noise_std: float = 0.04
    blur_prob: float = 0.2
    jpeg_prob: float = 0.3
    depth_scale: float = 0.03
    depth_noise: float = 0.01    # sigma = U(0, depth_noise) * d^2
    depth_holes: int = 5         # max random invalid patches
    depth_dropout: float = 0.02  # max fraction of randomly invalid pixels
    modality_dropout: float = 0.1


def _make_samples(dataset_dir: Path, rgb_paths, tag: str, num_keypoints: int, annotated_only: bool):
    depth_index = dio.DepthIndex(dataset_dir)
    samples, skipped = [], []
    for rgb in rgb_paths:
        ann = dio.load_annotation(rgb, tag) if rgb.is_file() else None
        if ann is None and (annotated_only or not rgb.is_file()):
            skipped.append(rgb.name)
            continue
        ann = ann or {}
        kps = np.zeros((num_keypoints, 2), np.float64)
        vis = np.zeros(num_keypoints, bool)
        for k, (x, y) in ann.items():
            if k < num_keypoints:
                kps[k] = (x, y)
                vis[k] = True
        parsed = dio.parse_frame_name(rgb)
        bag = parsed[0] if parsed else rgb.stem
        samples.append(Sample(rgb, depth_index.find(rgb), f'{dataset_dir.name}/{bag}', kps, vis))
    return samples, skipped


def load_samples(dataset_dirs, tag: str, num_keypoints: int, annotated_only: bool = True) -> list[Sample]:
    """All (annotated) images of the given dataset folders."""
    samples = []
    for d in dataset_dirs:
        d = Path(d)
        samples += _make_samples(d, dio.list_rgb_images(d), tag, num_keypoints, annotated_only)[0]
    return samples


def load_split_samples(split_files, tag: str, num_keypoints: int) -> list[Sample]:
    """Images listed in split files (split_<tag>_..._<kind>.json); names are relative to each file's folder.

    Images that are missing or not annotated for the tag are skipped with a warning.
    """
    samples = []
    for split in split_files:
        split = Path(split)
        found, skipped = _make_samples(split.parent, dio.load_split(split), tag, num_keypoints, True)
        if skipped:
            print(f'warning: {split.name}: {len(skipped)} image(s) missing or not annotated for {tag!r}, '
                  f'skipped (e.g. {skipped[0]})')
        samples += found
    return samples


def split_by_bag(samples: list[Sample], val_bags: int = 1, val_fraction: float | None = None):
    """Validation = the last bags (sorted by name), never frames of the same bag in train and val."""
    bags = sorted({s.bag for s in samples})
    if len(bags) < 2:
        raise ValueError(f'need at least 2 bags for a bag-wise split, got {bags}')
    n_val = max(1, round(len(bags) * val_fraction)) if val_fraction else val_bags
    n_val = min(n_val, len(bags) - 1)
    val = set(bags[-n_val:])
    return [s for s in samples if s.bag not in val], [s for s in samples if s.bag in val], sorted(val)


def random_affine(ref_size, input_size, cfg: AugmentConfig, rng: random.Random) -> np.ndarray:
    """Resize + random pan/rotation/zoom/shear(/flip), as reference -> input center coords."""
    w0, h0 = ref_size
    w, h = input_size
    c_ref = np.array([(w0 - 1) / 2, (h0 - 1) / 2])
    c_in = np.array([(w - 1) / 2, (h - 1) / 2])
    s = w / w0
    zoom = math.exp(rng.uniform(math.log(cfg.zoom[0]), math.log(cfg.zoom[1])))
    ang = math.radians(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    sh = math.tan(math.radians(rng.uniform(-cfg.shear_deg, cfg.shear_deg)))
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    a = rot @ np.array([[1.0, sh], [0.0, 1.0]]) * (s * zoom)
    a[:, 1] *= (h / h0) / s   # keep exact aspect handling if ref/input aspect ratios differ
    if cfg.hflip and rng.random() < 0.5:
        a = a @ np.diag([-1.0, 1.0])
    pan = np.array([rng.uniform(-cfg.pan, cfg.pan) * w, rng.uniform(-cfg.pan, cfg.pan) * h])
    t = c_in + pan - a @ c_ref
    return np.hstack([a, t[:, None]])


def augment_rgb(rgb_u8: np.ndarray, cfg: AugmentConfig, rng: random.Random) -> np.ndarray:
    """Photometric augmentation, uint8 RGB -> float32 RGB in [0, 1]."""
    if rng.random() < cfg.jpeg_prob:
        q = rng.randint(30, 90)
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            rgb_u8 = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if rng.random() < cfg.blur_prob:
        k = rng.choice([3, 5])
        rgb_u8 = cv2.GaussianBlur(rgb_u8, (k, k), 0)
    if cfg.hue_deg > 0 or cfg.saturation != (1.0, 1.0):
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg.hue_deg, cfg.hue_deg) / 2) % 180   # OpenCV hue: 0..180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*cfg.saturation), 0, 255)
        rgb_u8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    x = rgb_u8.astype(np.float32) / 255.0
    x = np.clip(x, 0, 1) ** rng.uniform(*cfg.gamma)
    mean = x.mean()
    x = (x - mean) * rng.uniform(*cfg.contrast) + mean + rng.uniform(-cfg.brightness, cfg.brightness)
    sigma = rng.uniform(0, cfg.noise_std)
    if sigma > 0:
        x = x + np.random.default_rng(rng.getrandbits(32)).normal(0, sigma, x.shape).astype(np.float32)
    return np.clip(x, 0, 1)


def augment_depth(depth_m: np.ndarray, cfg: AugmentConfig, rng: random.Random) -> np.ndarray:
    nrng = np.random.default_rng(rng.getrandbits(32))
    d = depth_m.copy()
    valid = d > 0
    d *= 1.0 + rng.uniform(-cfg.depth_scale, cfg.depth_scale)
    k = rng.uniform(0, cfg.depth_noise)
    if k > 0:
        d += (nrng.normal(0, 1, d.shape) * k * d ** 2).astype(np.float32)
    h, w = d.shape
    for _ in range(rng.randint(0, cfg.depth_holes)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax, ay = rng.uniform(2, w * 0.08), rng.uniform(2, h * 0.08)
        cv2.ellipse(d, (int(cx), int(cy)), (int(ax), int(ay)), rng.uniform(0, 180), 0, 360, 0.0, -1)
    p = rng.uniform(0, cfg.depth_dropout)
    if p > 0:
        d[nrng.random(d.shape) < p] = 0.0
    d[~valid] = 0.0
    d[d < 0] = 0.0
    return d


class KeypointDataset(Dataset):

    def __init__(self, samples: list[Sample], inputs, input_size, ref_size, num_keypoints: int,
                 stride: int = 4, sigma: float = 2.0, augment: AugmentConfig | None = None,
                 depth_range=pp.DEPTH_RANGE):
        self.samples = samples
        self.inputs = pp.check_inputs(inputs)
        self.input_size = tuple(input_size)
        self.ref_size = tuple(ref_size)
        self.num_keypoints = num_keypoints
        self.stride = stride
        self.sigma = sigma
        self.augment = augment
        self.depth_range = depth_range
        self.hm_size = (self.input_size[0] // stride, self.input_size[1] // stride)

    def __len__(self):
        return len(self.samples)

    def load_raw(self, s: Sample):
        # rgb is always loaded: it defines the reference frame and is used for previews
        bgr = cv2.imread(str(s.rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f'could not read {s.rgb_path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != self.ref_size:
            rgb = cv2.resize(rgb, self.ref_size, interpolation=cv2.INTER_LINEAR)
        depth = None
        if 'depth' in self.inputs:
            depth = dio.load_depth_m(s.depth_path) if s.depth_path is not None else \
                np.zeros(self.ref_size[::-1], np.float32)
        return rgb, depth

    def __getitem__(self, index):
        s = self.samples[index]
        rng = random.Random()
        rgb, depth = self.load_raw(s)
        if self.augment is not None:
            m = random_affine(self.ref_size, self.input_size, self.augment, rng)
        else:
            m = pp.resize_matrix(self.ref_size, self.input_size)

        out = {}
        rgb_warped = pp.warp_rgb(rgb, m, self.input_size)
        if 'rgb' in self.inputs:
            x = augment_rgb(rgb_warped, self.augment, rng) if self.augment else rgb_warped / 255.0
            out['rgb'] = torch.from_numpy(pp.normalize_rgb(x))
        if 'depth' in self.inputs:
            d = pp.warp_depth(depth, m, self.ref_size, self.input_size)
            if self.augment:
                d = augment_depth(d, self.augment, rng)
            out['depth'] = torch.from_numpy(pp.normalize_depth(d, self.depth_range))
        if self.augment and len(self.inputs) > 1 and rng.random() < self.augment.modality_dropout:
            out[rng.choice(self.inputs)].zero_()

        # keypoints: reference corner -> input center -> heatmap
        kps_in = pp.apply_affine(m, s.keypoints - 0.5)
        w, h = self.input_size
        vis = s.visible & (kps_in[:, 0] >= -0.5) & (kps_in[:, 0] < w - 0.5) & \
            (kps_in[:, 1] >= -0.5) & (kps_in[:, 1] < h - 0.5)
        kps_hm = pp.input_to_heatmap(kps_in, self.stride)
        out['heatmap'] = torch.from_numpy(pp.gaussian_heatmaps(kps_hm, vis, self.hm_size, self.sigma))
        # ground truth in the reference frame for metrics (only meaningful without augmentation)
        out['keypoints'] = torch.from_numpy(s.keypoints.astype(np.float32))
        out['visible'] = torch.from_numpy(vis if self.augment else s.visible.copy())
        out['m_inv'] = torch.from_numpy(pp.invert_affine(m).astype(np.float32))
        out['preview'] = torch.from_numpy(rgb_warped)
        out['index'] = index
        return out
