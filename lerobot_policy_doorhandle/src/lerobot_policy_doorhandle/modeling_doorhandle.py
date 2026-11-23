"""RGB-D keypoint regressor for the door handle pre-grasp joint pose.

rgb (3) + depth (1), resized, normalized ─> 4-channel ConvNeXt-Tiny (depth stem filter = mean of the
RGB filters) ─> FPN, fused top-down to stride 8 ─> K heatmaps ─> per keypoint: soft-argmax (u, v),
expected depth, pooled descriptor; + global pooled feature ─┐
q_current (normalized) ─────────────────────────────────────┴─> MLP ─> Δq, log σ² per target joint

q_target = q_current[target joints] + Δq in normalized joint space, un-normalized to m / rad.
Loss: Gaussian NLL (normalized joint space) + Huber on the door gripper position from SCARA forward
kinematics, in mm. The prediction lives in the ``action`` slot (IDENTITY normalization, raw units).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .augment import DoorhandleAugment
from .configuration_doorhandle import DoorhandleConfig
from .kinematics import gripper_position

logger = logging.getLogger(__name__)

FK_JOINTS = ("joint_slider", "joint_2", "joint_3", "joint_4")


def _widen_stem(conv: nn.Conv2d) -> nn.Conv2d:
    """3 -> 4 input channels; the depth filter starts as the mean of the RGB filters."""
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = conv.weight.mean(1, keepdim=True)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class Backbone(nn.Module):
    """4-channel backbone returning the stride 8, 16 and 32 feature maps."""

    def __init__(self, name: str, weights: str | None):
        super().__init__()
        if name == "convnext_tiny":
            net = torchvision.models.convnext_tiny(weights=weights)
            net.features[0][0] = _widen_stem(net.features[0][0])
            f = net.features
            # stem+stage1 (s4) | down+stage2 (s8) | down+stage3 (s16) | down+stage4 (s32)
            self.stages = nn.ModuleList([f[0:2], f[2:4], f[4:6], f[6:8]])
            self.channels = [192, 384, 768]
        else:
            net = torchvision.models.resnet18(weights=weights)
            net.conv1 = _widen_stem(net.conv1)
            self.stages = nn.ModuleList([
                nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1),
                net.layer2, net.layer3, net.layer4,
            ])
            self.channels = [128, 256, 512]

    def forward(self, x: Tensor) -> list[Tensor]:
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i > 0:
                feats.append(x)
        return feats


class DoorhandlePolicy(PreTrainedPolicy):
    config_class = DoorhandleConfig
    name = "doorhandle"

    # Set by from_pretrained: the checkpoint has all weights, so skip the ImageNet download.
    _loading_checkpoint = False

    def __init__(self, config: DoorhandleConfig, dataset_stats=None, dataset_meta=None, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.rgb_key, self.depth_key = config.rgb_and_depth_keys()

        # -- normalization constants (all from the config) --
        s_off, s_scale = config.joint_offset_scale(config.state_joint_names)
        a_off, a_scale = config.joint_offset_scale(config.action_joint_names)
        self.register_buffer("state_offset", torch.tensor(s_off), persistent=False)
        self.register_buffer("state_scale", torch.tensor(s_scale), persistent=False)
        self.register_buffer("action_offset", torch.tensor(a_off), persistent=False)
        self.register_buffer("action_scale", torch.tensor(a_scale), persistent=False)
        self.register_buffer("rgb_mean", torch.tensor(config.rgb_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor(config.rgb_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("state_noise_std", torch.tensor(config.state_noise_std), persistent=False)
        self.register_buffer(
            "base_index",
            torch.tensor([config.state_joint_names.index(j) for j in config.action_joint_names]),
            persistent=False,
        )
        self.fk_index = (
            [config.action_joint_names.index(j) for j in FK_JOINTS]
            if all(j in config.action_joint_names for j in FK_JOINTS) else None
        )

        # -- network --
        weights = None if self._loading_checkpoint else config.pretrained_backbone_weights
        self.backbone = Backbone(config.vision_backbone, weights)
        c = config.fpn_channels
        self.fpn = FeaturePyramidNetwork(self.backbone.channels, c)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, config.num_keypoints, 1),
        )
        self.descriptor_head = nn.Conv2d(c, config.keypoint_feature_dim, 1)
        k, d = config.num_keypoints, config.keypoint_feature_dim
        n_state, n_action = len(config.state_joint_names), len(config.action_joint_names)
        dims = [k * (3 + d) + c + n_state, *config.hidden_dims]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(config.dropout)]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 2 * n_action)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

        self.augment = DoorhandleAugment(config)
        self.reset()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls._loading_checkpoint = True
        try:
            return super().from_pretrained(*args, **kwargs)
        finally:
            cls._loading_checkpoint = False

    # -- normalization helpers -------------------------------------------------------------------

    def normalize_state(self, q: Tensor) -> Tensor:
        return (q - self.state_offset) / self.state_scale

    def normalize_action(self, q: Tensor) -> Tensor:
        return (q - self.action_offset) / self.action_scale

    def unnormalize_action(self, q_n: Tensor) -> Tensor:
        return q_n * self.action_scale + self.action_offset

    def normalize_depth(self, depth_m: Tensor) -> Tensor:
        lo, hi = self.config.depth_norm_range
        valid = torch.isfinite(depth_m) & (depth_m > 0)
        d = ((depth_m - (lo + hi) / 2) / ((hi - lo) / 2)).clamp(-1, 1)
        return torch.where(valid, d, torch.full_like(d, self.config.depth_invalid_value))

    def _images(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Resized rgb (B,3,H,W) in [0,1] and depth (B,1,H,W) in metres (0 = invalid)."""
        rgb = batch[self.rgb_key].float()
        depth = batch[self.depth_key].float()
        if rgb.ndim == 5:  # (B, T, C, H, W) with n_obs_steps = 1
            rgb, depth = rgb[:, -1], depth[:, -1]
        if self.config.depth_input_unit == "mm":
            depth = depth / 1000.0
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        size = tuple(self.config.image_size)
        if tuple(rgb.shape[-2:]) != size:
            rgb = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False, antialias=True)
        if tuple(depth.shape[-2:]) != size:
            depth = F.interpolate(depth, size=size, mode="nearest")
        return rgb, depth

    # -- network ---------------------------------------------------------------------------------

    def _predict(self, batch: dict[str, Tensor], augment: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """-> (mean_n, log_var) in normalized joint space and the normalized base state."""
        rgb, depth = self._images(batch)
        state = batch[OBS_STATE].float()
        if state.ndim == 3:
            state = state[:, -1]
        if augment:
            if self.augment.enabled:
                rgb, depth = self.augment(rgb, depth)
            if self.config.state_noise:
                state = state + torch.randn_like(state) * self.state_noise_std
        state_n = self.normalize_state(state)

        depth_n = self.normalize_depth(depth)
        x = torch.cat([(rgb - self.rgb_mean) / self.rgb_std, depth_n], 1)
        feats = self.backbone(x)
        pyramid = self.fpn({str(i): f for i, f in enumerate(feats)})
        p8 = pyramid["0"]

        b, _, h, w = p8.shape
        logits = self.heatmap_head(p8).float().flatten(2)  # (B, K, HW), float32 for the softmax
        attn = logits.softmax(-1)
        ys = (torch.arange(h, device=x.device, dtype=torch.float32) + 0.5) / h * 2 - 1
        xs = (torch.arange(w, device=x.device, dtype=torch.float32) + 0.5) / w * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        u = (attn * gx.flatten()).sum(-1)
        v = (attn * gy.flatten()).sum(-1)
        depth_small = F.adaptive_avg_pool2d(depth_n.float(), (h, w)).flatten(2)  # (B, 1, HW)
        kp_depth = (attn * depth_small).sum(-1)
        desc = self.descriptor_head(p8).float().flatten(2)  # (B, D, HW)
        kp_desc = torch.einsum("bkn,bdn->bkd", attn, desc)
        # global context: average pooled coarser levels (stride 16 and 32)
        glob = sum(pyramid[str(i)].float().mean((-2, -1)) for i in range(1, len(feats)))

        z = torch.cat([u, v, kp_depth, kp_desc.flatten(1), glob, state_n], 1)
        out = self.out(self.mlp(z))
        delta, log_var = out.chunk(2, -1)
        lo, hi = self.config.log_var_range
        base_n = self.normalize_action(state[:, self.base_index])
        return base_n + delta, log_var.clamp(lo, hi), base_n

    # -- training --------------------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        mean_n, log_var, _ = self._predict(batch, augment=self.training)
        target = batch[ACTION].float()
        if target.ndim == 3:
            target = target[:, 0]
        target_n = self.normalize_action(target)
        nll = 0.5 * (log_var + (target_n - mean_n) ** 2 * torch.exp(-log_var))
        loss = self.config.nll_weight * nll.mean()

        mean = self.unnormalize_action(mean_n)
        info = {"loss_nll": nll.mean().item()}
        if self.fk_index is not None:
            p_pred = gripper_position(*mean[:, self.fk_index].unbind(-1)) * 1000.0
            p_true = gripper_position(*target[:, self.fk_index].unbind(-1)) * 1000.0
            fk = F.huber_loss(p_pred, p_true, delta=self.config.fk_huber_delta_mm)
            loss = loss + self.config.fk_weight * fk
            info["loss_fk"] = fk.item()
            info["fk_mm"] = (p_pred - p_true).norm(dim=-1).mean().item()
        err = (mean - target).abs().mean(0)
        for name, e in zip(self.config.action_joint_names, err.tolist()):
            info[f"mae_{name}"] = e
        info["sigma_mean"] = torch.exp(0.5 * log_var).mean().item()
        return loss, info

    def get_optim_params(self) -> list[dict]:
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return [
            {"params": backbone, "lr": self.config.optimizer_lr * self.config.backbone_lr_scale},
            {"params": rest},
        ]

    # -- inference -------------------------------------------------------------------------------

    def reset(self):
        pass

    @torch.no_grad()
    def predict_with_uncertainty(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """-> (target joints (B, n_action) in m / rad, standard deviation (B, n_action) in m / rad)."""
        mean_n, log_var, _ = self._predict(batch)
        return self.unnormalize_action(mean_n), torch.exp(0.5 * log_var) * self.action_scale

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        mean_n, _, _ = self._predict(batch)
        return self.unnormalize_action(mean_n).unsqueeze(1)  # (B, 1, n_action)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.predict_action_chunk(batch)[:, 0]
