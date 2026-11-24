"""Keypoint metrics (NumPy). All coordinates in pixels of the reference (original RGB) image."""

import numpy as np

CLIP_PX = 50.0


def keypoint_metrics(pred, conf, gt, gt_vis, threshold: float, names=None, clip=CLIP_PX) -> dict:
    """pred/gt: N x K x 2, conf: N x K, gt_vis: N x K bool."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    gt_vis = np.asarray(gt_vis, bool)
    pred_vis = np.asarray(conf) >= threshold
    err = np.linalg.norm(pred - gt, axis=-1)
    both = gt_vis & pred_vis
    tp, fp, fn = both.sum(), (pred_vis & ~gt_vis).sum(), (~pred_vis & gt_vis).sum()
    # score: mean over all keypoint slots, clipped error; misses and false positives cost `clip`
    per_slot = np.where(gt_vis, np.where(pred_vis, np.minimum(err, clip), clip), np.where(pred_vis, clip, 0.0))
    out = {
        'score': float(per_slot.mean()) if per_slot.size else float('nan'),
        'mean_err': float(err[both].mean()) if both.any() else float('nan'),
        'median_err': float(np.median(err[both])) if both.any() else float('nan'),
        'pck10': float((both & (err <= 10)).sum() / max(1, gt_vis.sum())),
        'pck20': float((both & (err <= 20)).sum() / max(1, gt_vis.sum())),
        'vis_precision': float(tp / max(1, tp + fp)),
        'vis_recall': float(tp / max(1, tp + fn)),
        'n_images': int(len(gt)),
        'n_visible': int(gt_vis.sum()),
    }
    for k in range(gt.shape[1] if gt.ndim == 3 else 0):
        name = names[k] if names and k < len(names) else str(k)
        mk = both[:, k]
        out[f'err_{name}'] = float(err[mk, k].mean()) if mk.any() else float('nan')
    return out


def format_metrics(m: dict) -> str:
    return (f"score {m['score']:.2f}  err {m['mean_err']:.2f}px (median {m['median_err']:.2f})  "
            f"PCK@10 {m['pck10']:.3f}  PCK@20 {m['pck20']:.3f}  "
            f"vis P/R {m['vis_precision']:.2f}/{m['vis_recall']:.2f}")
