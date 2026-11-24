"""Drawing of keypoint predictions (NumPy / OpenCV)."""

import cv2
import numpy as np

DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 5.0
COLORS_RGB = [(255, 40, 40), (40, 220, 40), (60, 120, 255), (255, 200, 0), (255, 0, 255), (0, 230, 230)]


def color(k):
    return COLORS_RGB[k % len(COLORS_RGB)]


def draw_keypoints(rgb_u8: np.ndarray, pts, conf=None, threshold=0.3, names=None, gt=None, gt_vis=None,
                   heatmaps=None) -> np.ndarray:
    """Draws on an RGB image. pts / gt in pixel (corner) coordinates of that image.

    Prediction: filled circle + name + confidence (hollow/gray if below threshold);
    ground truth: white ring.
    """
    img = rgb_u8.copy()
    h, w = img.shape[:2]
    if heatmaps is not None and len(heatmaps):
        hm = cv2.resize(np.max(heatmaps, axis=0), (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.applyColorMap((np.clip(hm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        a = (np.clip(hm, 0, 1) * 0.6)[..., None]
        img = (img * (1 - a) + overlay * a).astype(np.uint8)
    scale = max(1, round(w / 320))
    if gt is not None:
        for k, (x, y) in enumerate(np.asarray(gt)):
            if gt_vis is None or gt_vis[k]:
                cv2.circle(img, (int(round(x)), int(round(y))), 7 * scale, (255, 255, 255), max(1, scale),
                           cv2.LINE_AA)
    for k, (x, y) in enumerate(np.asarray(pts)):
        c = conf[k] if conf is not None else 1.0
        visible = c >= threshold
        col = color(k) if visible else (150, 150, 150)
        p = (int(round(x)), int(round(y)))
        cv2.circle(img, p, 3 * scale, col, -1 if visible else max(1, scale), cv2.LINE_AA)
        label = (names[k] if names and k < len(names) else str(k)) + (f' {c:.2f}' if conf is not None else '')
        cv2.putText(img, label, (p[0] + 5 * scale, p[1] - 5 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35 * scale, col, max(1, scale // 2), cv2.LINE_AA)
    return img


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Depth in metres -> RGB uint8 (turbo colormap, near = red, invalid = black)."""
    valid = depth_m > 0
    norm = np.clip((depth_m - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M), 0.0, 1.0)
    img = cv2.applyColorMap((255 * (1.0 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def grid(images, cols=4) -> np.ndarray:
    if not images:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = cv2.resize(im, (w, h))
    return out


def save_rgb(path, rgb_u8):
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
