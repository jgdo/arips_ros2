"""Input preprocessing and heatmap decoding shared by training, evaluation and the ROS node (NumPy only).

Coordinate conventions:
- Annotations / published keypoints use "corner" pixel coordinates (0 = left edge of the first pixel, as
  the annotation tool stores clicks); internally all warps use OpenCV "center" coordinates (pixel i at i).
- The reference frame is the original RGB image (ref_size = (W0, H0)). Depth images of a different
  resolution are mapped into it by scaling.
- An affine M (2x3) maps reference center coordinates to network input center coordinates.
"""

import json
from pathlib import Path

import cv2
import numpy as np

INPUT_CHANNELS = {'rgb': 3, 'depth': 2}
VALID_INPUTS = ('rgb', 'depth')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DEPTH_RANGE = (0.2, 5.0)


def check_inputs(inputs) -> list[str]:
    inputs = [i for i in VALID_INPUTS if i in inputs]   # canonical order rgb, depth
    if not inputs:
        raise ValueError(f'inputs must be a non-empty subset of {VALID_INPUTS}')
    return inputs


def inputs_name(inputs) -> str:
    return {('rgb',): 'rgb', ('depth',): 'd', ('rgb', 'depth'): 'rgbd'}[tuple(check_inputs(inputs))]


# ---------------------------------------------------------------------------------------------
# affine helpers
# ---------------------------------------------------------------------------------------------

def to3x3(m: np.ndarray) -> np.ndarray:
    return np.vstack([m, [0.0, 0.0, 1.0]])


def resize_matrix(ref_size, input_size) -> np.ndarray:
    """Plain resize of the reference image to the network input size (center convention)."""
    sx = input_size[0] / ref_size[0]
    sy = input_size[1] / ref_size[1]
    return np.array([[sx, 0.0, 0.5 * sx - 0.5],
                     [0.0, sy, 0.5 * sy - 0.5]], np.float64)


def image_to_ref_matrix(image_size, ref_size) -> np.ndarray:
    """Maps pixel centers of an image of image_size (e.g. depth) into the reference frame."""
    sx = ref_size[0] / image_size[0]
    sy = ref_size[1] / image_size[1]
    return np.array([[sx, 0.0, 0.5 * sx - 0.5],
                     [0.0, sy, 0.5 * sy - 0.5]], np.float64)


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ m[:, :2].T + m[:, 2]


def invert_affine(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(np.asarray(m, np.float64))


# ---------------------------------------------------------------------------------------------
# input tensors
# ---------------------------------------------------------------------------------------------

def warp_rgb(rgb_u8: np.ndarray, m: np.ndarray, input_size) -> np.ndarray:
    """RGB uint8 (reference frame) -> warped uint8 at input size."""
    return cv2.warpAffine(rgb_u8, m, tuple(input_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_depth(depth_m: np.ndarray, m: np.ndarray, ref_size, input_size) -> np.ndarray:
    """Depth (any resolution) -> warped depth in metres at input size (nearest, 0 = invalid)."""
    h, w = depth_m.shape[:2]
    md = (to3x3(m) @ to3x3(image_to_ref_matrix((w, h), ref_size)))[:2]
    return cv2.warpAffine(depth_m.astype(np.float32), md, tuple(input_size), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def normalize_rgb(rgb_float01: np.ndarray) -> np.ndarray:
    """HxWx3 float RGB in [0, 1] -> 3xHxW normalized float32."""
    x = (rgb_float01.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def normalize_depth(depth_m: np.ndarray, depth_range=DEPTH_RANGE) -> np.ndarray:
    """HxW depth in metres -> 2xHxW float32: [scaled depth (0..1, 0 = invalid), valid mask]."""
    lo, hi = depth_range
    valid = depth_m > 0
    scaled = np.where(valid, np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0), 0.0)
    return np.stack([scaled, valid]).astype(np.float32)


def prepare_inputs(meta: dict, rgb_u8: np.ndarray | None = None, depth_m: np.ndarray | None = None):
    """Inference preprocessing (no augmentation). rgb_u8 is RGB (not BGR).

    Returns ({input name: 1xCxHxW float32}, M_inv) where M_inv maps input center coords to reference
    center coords.
    """
    ref_size, input_size = meta['ref_size'], meta['input_size']
    m = resize_matrix(ref_size, input_size)
    out = {}
    if 'rgb' in meta['inputs']:
        if rgb_u8 is None:
            raise ValueError('model needs an rgb image')
        if (rgb_u8.shape[1], rgb_u8.shape[0]) != tuple(ref_size):
            # different camera resolution than in training: map it onto the reference frame
            rgb_u8 = cv2.resize(rgb_u8, tuple(ref_size), interpolation=cv2.INTER_LINEAR)
        out['rgb'] = normalize_rgb(warp_rgb(rgb_u8, m, input_size) / 255.0)[None]
    if 'depth' in meta['inputs']:
        if depth_m is None:
            raise ValueError('model needs a depth image')
        out['depth'] = normalize_depth(warp_depth(depth_m, m, ref_size, input_size),
                                       meta.get('depth_range', DEPTH_RANGE))[None]
    return out, invert_affine(m)


# ---------------------------------------------------------------------------------------------
# targets / decoding
# ---------------------------------------------------------------------------------------------

def input_to_heatmap(pts, stride: int) -> np.ndarray:
    return (np.asarray(pts, np.float64) + 0.5) / stride - 0.5


def heatmap_to_input(pts, stride: int) -> np.ndarray:
    return (np.asarray(pts, np.float64) + 0.5) * stride - 0.5


def gaussian_heatmaps(kps_hm: np.ndarray, visible: np.ndarray, hm_size, sigma: float) -> np.ndarray:
    """K Gaussian target heatmaps (K x Hh x Wh, float32, peak ~1). Invisible keypoints -> all zero."""
    wh, hh = hm_size
    xs = np.arange(wh, dtype=np.float32)
    ys = np.arange(hh, dtype=np.float32)
    out = np.zeros((len(kps_hm), hh, wh), np.float32)
    for k, ((x, y), vis) in enumerate(zip(kps_hm, visible)):
        if vis:
            gx = np.exp(-((xs - x) ** 2) / (2 * sigma ** 2))
            gy = np.exp(-((ys - y) ** 2) / (2 * sigma ** 2))
            out[k] = gy[:, None] * gx[None, :]
    return out


def decode_heatmaps(heatmaps: np.ndarray, stride: int, m_inv: np.ndarray | None = None):
    """K x Hh x Wh probabilities -> (points K x 2 in corner coords of the reference frame (or of the
    network input if m_inv is None), confidences K).

    Peak = argmax, refined per axis by a parabola fit through the log of the peak and its two
    neighbours (exact for a Gaussian peak).
    """
    k, hh, wh = heatmaps.shape
    pts = np.zeros((k, 2), np.float64)
    conf = np.zeros(k, np.float32)

    def offset(left, center, right):
        if min(left, center, right) <= 1e-6:
            return 0.0
        ll, lc, lr = np.log(left), np.log(center), np.log(right)
        denom = ll - 2 * lc + lr
        if denom >= -1e-9:
            return 0.0
        return float(np.clip(0.5 * (ll - lr) / denom, -1.0, 1.0))

    for i in range(k):
        hm = heatmaps[i].astype(np.float64)
        py, px = divmod(int(np.argmax(hm)), wh)
        conf[i] = hm[py, px]
        dx = offset(hm[py, px - 1], hm[py, px], hm[py, px + 1]) if 0 < px < wh - 1 else 0.0
        dy = offset(hm[py - 1, px], hm[py, px], hm[py + 1, px]) if 0 < py < hh - 1 else 0.0
        pts[i] = [px + dx, py + dy]
    pts = heatmap_to_input(pts, stride)
    if m_inv is not None:
        pts = apply_affine(m_inv, pts)
    return pts + 0.5, conf


# ---------------------------------------------------------------------------------------------
# model metadata (json next to the onnx / checkpoint)
# ---------------------------------------------------------------------------------------------

def load_meta(model_path) -> dict:
    path = Path(model_path)
    meta_path = path.with_suffix('.json')
    with open(meta_path) as f:
        return json.load(f)


def save_meta(model_path, meta: dict) -> Path:
    path = Path(model_path).with_suffix('.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')
    return path
