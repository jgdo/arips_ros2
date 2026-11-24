"""Heatmap losses. Targets are soft Gaussians (peak ~1), predictions are logits."""

import torch
import torch.nn.functional as F


def quality_focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE for soft targets (Li et al., Generalized Focal Loss): BCE * |target - p|^gamma.

    Handles the huge background / foreground imbalance like the CenterNet focal loss but
    regresses the full Gaussian shape, which keeps the peak refinement meaningful.
    Normalised by the number of visible keypoints in the batch.
    """
    logits = logits.float()
    p = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    loss = bce * (target - p).abs().pow(gamma)
    num_visible = (target.flatten(2).amax(dim=2) > 0.5).sum().clamp(min=1)
    return loss.sum() / num_visible


def weighted_mse_loss(logits: torch.Tensor, target: torch.Tensor, fg_weight: float = 10.0) -> torch.Tensor:
    p = logits.float().sigmoid()
    loss = (1.0 + fg_weight * target) * (p - target) ** 2
    num_visible = (target.flatten(2).amax(dim=2) > 0.5).sum().clamp(min=1)
    return loss.sum() / num_visible


LOSSES = {'qfl': quality_focal_loss, 'mse': weighted_mse_loss}
