"""Frozen ResNet18 + MLP that predicts the joint pose at the episode's annotated target frame.

image -> ResNet18 (frozen, eval) -> avgpool 512 ─┐
state (normalized) ──────────────────────────────┴─> MLP -> target pose (action-normalized)

The output lives in the ``action`` feature slot, so the standard postprocessor unnormalizes it
with the action statistics and evaluation tools can chart it next to the recorded actions.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .augment import GpuAugment
from .configuration_target_reach import TargetReachConfig
from .targets import load_tag_frames

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TargetReachPolicy(PreTrainedPolicy):
    config_class = TargetReachConfig
    name = "target_reach"

    def __init__(self, config: TargetReachConfig, dataset_stats=None, dataset_meta=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        weights = config.pretrained_backbone_weights
        backbone = getattr(torchvision.models, config.vision_backbone)(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        feat_dim = backbone.fc.in_features
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        state_dim = config.robot_state_feature.shape[0]
        action_dim = config.action_feature.shape[0]
        layers: list[nn.Module] = []
        in_dim = feat_dim * len(config.image_features) + state_dim
        for i, hidden in enumerate(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            if i == 0:
                layers += [nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(config.dropout)]
            else:
                layers.append(nn.GELU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.head = nn.Sequential(*layers)
        self.augment = GpuAugment(config.color_augmentation, config.position_augmentation)

        # Training-only lookup (non-persistent, rebuilt from the annotations each run):
        # target_last[e] = dataset index of episode e's target frame (-1: episode skipped),
        # target_pose[e] = pose at that frame, normalized like the action.
        self.register_buffer("target_last", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("target_pose", torch.empty(0, action_dim), persistent=False)
        if config.annotation_path and config.target_tag and dataset_meta is not None:
            self._build_targets(dataset_meta, dataset_stats)
        self._warned_masking = False
        self.reset()

    # -- targets -------------------------------------------------------------

    def _build_targets(self, dataset_meta, dataset_stats) -> None:
        import pyarrow.parquet as pq

        cfg = self.config
        frames = load_tag_frames(cfg.annotation_path, cfg.target_tag)
        key = cfg.target_key
        rows: dict[tuple[int, int], tuple[int, list[float]]] = {}
        wanted = set(frames.items())
        for file in sorted((Path(dataset_meta.root) / "data").glob("*/*.parquet")):
            table = pq.read_table(file, columns=["index", "episode_index", "frame_index", key]).to_pydict()
            for idx, ep, fr, pose in zip(
                table["index"], table["episode_index"], table["frame_index"], table[key], strict=True
            ):
                if (ep, fr) in wanted:
                    rows[(ep, fr)] = (idx, pose)
        missing = wanted - rows.keys()
        if missing:
            raise ValueError(f"target frames not found in the dataset: {sorted(missing)[:5]}")

        num_episodes = max(dataset_meta.total_episodes, max(frames) + 1)
        last = torch.full((num_episodes,), -1, dtype=torch.long)
        pose = torch.zeros(num_episodes, self.target_pose.shape[1])
        for (ep, fr), (idx, values) in rows.items():
            last[ep] = idx
            pose[ep] = torch.tensor(values, dtype=torch.float32)

        stats = (dataset_stats or {}).get(ACTION)
        if stats is None:
            raise ValueError("target_reach needs the dataset's action statistics")
        mean = torch.as_tensor(np.asarray(stats["mean"]), dtype=torch.float32)
        std = torch.as_tensor(np.asarray(stats["std"]), dtype=torch.float32)
        pose = (pose - mean) / (std + 1e-8)

        self.target_last = last
        self.target_pose = pose
        logger.info(
            "target_reach: tag %r found in %d/%d episodes (target_key=%s)",
            cfg.target_tag, len(rows), dataset_meta.total_episodes, key,
        )

    # -- model ---------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # frozen: BatchNorm keeps the ImageNet statistics
        return self

    def _predict(self, batch: dict[str, Tensor]) -> Tensor:
        features = []
        for key in self.config.image_features:
            img = batch[key]
            if self.config.image_size is not None and tuple(img.shape[-2:]) != tuple(self.config.image_size):
                img = F.interpolate(img, size=tuple(self.config.image_size), mode="bilinear", align_corners=False)
            if self.training:
                img = self.augment(img)
            img = (img - self.img_mean) / self.img_std
            with torch.no_grad():
                features.append(self.backbone(img).flatten(1))
        features.append(batch[OBS_STATE])
        return self.head(torch.cat(features, dim=1))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.target_last.numel() == 0:
            raise RuntimeError("target_reach training needs policy.annotation_path and policy.target_tag")
        pred = self._predict(batch)
        episode = batch["episode_index"].long().view(-1)
        last = self.target_last[episode]
        valid = last >= 0
        n_valid = int(valid.sum())
        if n_valid < len(valid) and not self._warned_masking:
            logger.warning(
                "target_reach: %d/%d samples have no target and are masked (filtered sampler not active?)",
                len(valid) - n_valid, len(valid),
            )
            self._warned_masking = True
        if n_valid == 0:
            return pred.sum() * 0.0, {"valid_samples": 0}
        loss = F.l1_loss(pred[valid], self.target_pose[episode[valid]])
        return loss, {"valid_samples": n_valid}

    # -- inference -----------------------------------------------------------

    def get_optim_params(self) -> dict:
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self):
        self._queue = deque([], maxlen=1)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        return self._predict(batch).unsqueeze(1)  # (B, 1, action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.predict_action_chunk(batch)[:, 0]
