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
import math

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .augment import DoorhandleAugment
from .configuration_doorhandle import DoorhandleConfig
from .kinematics import GRIPPER_Z_ABOVE_SLIDER, arm_ik, camera_points, gripper_position, link_3_pose, project_points, surface_normals

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

    @property
    def stem(self) -> nn.Conv2d:
        """The widened 4-channel input conv."""
        first = self.stages[0][0]
        return first if isinstance(first, nn.Conv2d) else first[0]  # resnet18 | convnext stem block

    def freeze(self, include_stem: bool) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        if not include_stem:
            for p in self.stem.parameters():
                p.requires_grad_(True)

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
        self.tilt_index = (config.state_joint_names.index(config.camera_tilt_joint)
                           if config.camera_tilt_joint in config.state_joint_names else None)
        self.register_buffer("xyz_center", torch.tensor(config.xyz_center).view(1, 1, 3), persistent=False)
        self.fk_index = (
            [config.action_joint_names.index(j) for j in FK_JOINTS]
            if all(j in config.action_joint_names for j in FK_JOINTS) else None
        )

        # -- network --
        weights = None if self._loading_checkpoint else config.pretrained_backbone_weights
        self.backbone = Backbone(config.vision_backbone, weights)
        if config.freeze_backbone:
            self.backbone.freeze(include_stem=config.freeze_backbone_stem)
        c = config.fpn_channels
        self.fpn = FeaturePyramidNetwork(self.backbone.channels, c)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, config.num_keypoints, 1),
        )
        self.descriptor_head = nn.Conv2d(c, config.keypoint_feature_dim, 1)
        k, d = config.num_keypoints, config.keypoint_feature_dim
        n_state, n_action = len(config.state_joint_names), len(config.action_joint_names)
        # per keypoint: (u, v), expected depth, descriptor, (xyz, valid fraction)
        per_kp = ((2 if config.keypoint_uv else 0) + (1 if config.keypoint_depth else 0)
                  + (d if config.keypoint_descriptors else 0) + (4 if config.xyz_keypoints else 0)
                  + (3 if config.xyz_normals else 0))
        dims = [k * per_kp + (c if config.global_feature else 0) + n_state, *config.hidden_dims]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.Dropout(config.dropout)]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 2 * n_action)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)
        if config.xyz_refine:
            r = config.xyz_refine_channels
            self.refine_reduce = nn.Conv2d(c, 16, 1)
            self.refine = nn.Sequential(
                nn.Conv2d(4 + 16, r, 3, padding=1), nn.GELU(),
                nn.Conv2d(r, r, 3, padding=1), nn.GELU(),
                nn.Conv2d(r, config.num_keypoints, 3, padding=1),
            )
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
        if config.xyz_skip:
            self.xyz_skip = nn.Linear(k * 3 + n_state, n_action)
            nn.init.zeros_(self.xyz_skip.weight)
            nn.init.zeros_(self.xyz_skip.bias)
        if config.xyz_aux_weight > 0:
            self.xyz_aux_offset = nn.Parameter(torch.zeros(1 if config.xyz_aux_dims == "z" else 3))
        self._kp_xyz = None  # last keypoint xyz [m] (B, K, 3), for the auxiliary loss
        self._homographies = None  # geometric augmentation of the last training batch, for the uv aux labels
        self._kp_uv = None

        self.augment = DoorhandleAugment(config)
        self.reset()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_backbone:
            # frozen backbone = fixed pretrained feature extractor: BatchNorm statistics (resnet18) stay
            # fixed and ConvNeXt's stochastic depth is off (the trainable stem conv has no train mode)
            self.backbone.eval()
        return self

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
        points_fn = None
        if self.config.xyz_keypoints:
            # the measured tilt (before the state noise below): the point map is exact geometry
            tilt = state[:, self.tilt_index]
            cfg = self.config
            def points_fn(d):
                pts = camera_points(d, tilt, cfg.camera_hfov, cfg.camera_tilt_origin, cfg.camera_in_tilt_link)
                if cfg.xyz_normals:  # computed before the warp, moved with the pixels like the points
                    pts = torch.cat([pts, surface_normals(pts, d > 0)], 1)
                return pts
        points = None
        self._homographies = None
        if augment and self.augment.enabled:
            rgb, depth, points = self.augment(rgb, depth, points_fn)
            self._homographies = self.augment.last_homographies
        elif points_fn is not None:
            points = points_fn(depth)
        if augment:
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
        self._kp_uv = torch.stack([u, v], -1)  # (B, K, 2), for the training-only uv auxiliary loss
        depth_small = F.adaptive_avg_pool2d(depth_n.float(), (h, w)).flatten(2)  # (B, 1, HW)
        kp_depth = (attn * depth_small).sum(-1)
        desc = self.descriptor_head(p8).float().flatten(2)  # (B, D, HW)
        kp_desc = torch.einsum("bkn,bdn->bkd", attn, desc)
        # global context: average pooled coarser levels (stride 16 and 32)
        glob = sum(pyramid[str(i)].float().mean((-2, -1)) for i in range(1, len(feats)))

        parts = [u, v] if self.config.keypoint_uv else []
        if self.config.keypoint_depth:
            parts.append(kp_depth)
        kp_xyz_n = None
        normals = None
        if points is not None and self.config.xyz_normals:
            points, normals = points[:, :3], points[:, 3:]
        if points is not None:
            valid = (depth > 0).float()
            if self.config.xyz_fullres:
                full = F.interpolate(logits.view(b, -1, h, w), size=depth.shape[-2:], mode="bilinear",
                                     align_corners=False)
                if self.config.xyz_refine:
                    feat = F.interpolate(self.refine_reduce(p8), size=depth.shape[-2:], mode="bilinear",
                                         align_corners=False)
                    full = full + self.refine(torch.cat([x, feat.to(x.dtype)], 1)).float()
                full = full.flatten(2).softmax(-1)  # (B, K, HW full)
                vf = valid.flatten(2)
                kp_valid = (full * vf).sum(-1)
                kp_xyz = torch.einsum("bkn,bcn->bkc", full * vf, points.float().flatten(2))
                if normals is not None:
                    kp_nrm = torch.einsum("bkn,bcn->bkc", full, normals.float().flatten(2))
            else:
                num = F.adaptive_avg_pool2d(points.float() * valid, (h, w)).flatten(2)  # (B, 3, HW)
                den = F.adaptive_avg_pool2d(valid, (h, w)).flatten(2)  # (B, 1, HW)
                kp_valid = (attn * den).sum(-1)  # (B, K)
                kp_xyz = torch.einsum("bkn,bcn->bkc", attn, num)
                if normals is not None:
                    nrm = F.adaptive_avg_pool2d(normals.float(), (h, w)).flatten(2)
                    kp_nrm = torch.einsum("bkn,bcn->bkc", attn, nrm)
            kp_xyz = kp_xyz / kp_valid.unsqueeze(-1).clamp_min(1e-4)
            self._kp_xyz = kp_xyz
            kp_xyz_n = ((kp_xyz - self.xyz_center) / self.config.xyz_scale).clamp(-4, 4)
            parts += [kp_xyz_n.flatten(1), kp_valid]
            if normals is not None:
                kp_nrm = kp_nrm / kp_nrm.norm(dim=-1, keepdim=True).clamp_min(1e-3)
                self._kp_nrm = kp_nrm
                parts.append(kp_nrm.flatten(1))
        if self.config.keypoint_descriptors:
            parts.append(kp_desc.flatten(1))
        if self.config.global_feature:
            parts.append(glob)
        z = torch.cat([*parts, state_n], 1)
        with torch.autocast(x.device.type, enabled=False):  # tiny head: keep log_var / loss in float32
            out = self.out(self.mlp(z.float()))
        delta, log_var = out.chunk(2, -1)
        if self.config.xyz_skip:
            delta = delta + self.xyz_skip(torch.cat([kp_xyz_n.flatten(1), state_n], 1).float())
        lo, hi = self.config.log_var_range
        base_n = self.normalize_action(state[:, self.base_index])
        if self.config.ik_from_keypoints:
            ik_n = self._ik_base_n(base_n)
            mean_n = (ik_n.detach() if self.config.ik_detach_base else ik_n) + delta
        else:
            mean_n = base_n + delta
        if self.config.slider_from_keypoint:
            j = self.config.action_joint_names.index("joint_slider")
            slider = self._kp_xyz[:, 0, 2] - self.xyz_aux_offset[-1] - GRIPPER_Z_ABOVE_SLIDER
            slider_n = (slider - self.action_offset[j]) / self.action_scale[j]
            mean_n = torch.cat([mean_n[:, :j], slider_n.unsqueeze(1), mean_n[:, j + 1:]], 1)
        return mean_n, log_var.clamp(lo, hi), base_n

    def _ik_base_n(self, base_n: Tensor) -> Tensor:
        """Normalized joints from keypoint 0 (handle point) + keypoint 1 normal via analytic IK; joints
        outside the arm chain keep their base (current) value."""
        n = self._kp_nrm[:, 1]
        # horizontal direction with an epsilon: atan2's gradient is unbounded when the attended normal
        # is (nearly) vertical (floor / ceiling), which made training diverge
        r = torch.sqrt(n[:, 0] ** 2 + n[:, 1] ** 2 + 1e-4)
        yaw = torch.atan2(n[:, 1] / r, n[:, 0] / r) - math.radians(self.config.normal_aux_yaw_offset_deg)
        c, s = torch.cos(yaw), torch.sin(yaw)
        o = self.xyz_aux_offset
        kp = self._kp_xyz[:, 0]
        gx = kp[:, 0] - (c * o[0] - s * o[1])
        gy = kp[:, 1] - (s * o[0] + c * o[1])
        slider = kp[:, 2] - o[2] - GRIPPER_Z_ABOVE_SLIDER
        j2, j3, j4 = arm_ik(gx, gy, yaw)
        q = torch.stack([slider, j2, j3, j4], 1)
        out = base_n.clone()
        out[:, self.fk_index] = (q - self.action_offset[self.fk_index]) / self.action_scale[self.fk_index]
        return out

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
        logged = {"loss_nll": nll.mean()}
        if self.fk_index is not None:
            p_pred = gripper_position(*mean[:, self.fk_index].unbind(-1)) * 1000.0
            p_true = gripper_position(*target[:, self.fk_index].unbind(-1)) * 1000.0
            fk = F.huber_loss(p_pred, p_true, delta=self.config.fk_huber_delta_mm)
            loss = loss + self.config.fk_weight * fk
            logged["loss_fk"] = fk
            logged["fk_mm"] = (p_pred - p_true).norm(dim=-1).mean()
        if self.config.xyz_aux_weight > 0 and self.fk_index is not None:
            kp_true = self.aux_target(target)
            kp = self._kp_xyz[:, 0] if self.config.xyz_aux_dims == "xyz" else self._kp_xyz[:, 0, 2:]
            aux = F.huber_loss(kp * 1000.0, kp_true * 1000.0, delta=self.config.xyz_aux_huber_delta_mm)
            loss = loss + self.config.xyz_aux_weight * aux
            logged["loss_xyz_aux"] = aux
            logged["xyz_aux_mm"] = (kp - kp_true).norm(dim=-1).mean() * 1000.0
        if self.config.normal_aux_weight > 0 and self.fk_index is not None:
            yaw = link_3_pose(*target[:, self.fk_index].unbind(-1))[3]
            a = yaw + math.radians(self.config.normal_aux_yaw_offset_deg)
            n_true = torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], -1)
            deg = 180.0 / math.pi
            aux_n = F.huber_loss(self._kp_nrm[:, 1] * deg, n_true * deg, delta=2.0)
            loss = loss + self.config.normal_aux_weight * aux_n
            logged["loss_normal_aux"] = aux_n
            cos = (self._kp_nrm[:, 1] * n_true).sum(-1).clamp(-1, 1)
            logged["normal_aux_deg"] = torch.rad2deg(torch.acos(cos)).mean()
        if self.config.uv_aux_weight > 0 and self.fk_index is not None:
            aux_uv, uv_px = self._uv_aux_loss(batch, target)
            loss = loss + self.config.uv_aux_weight * aux_uv
            logged["loss_uv_aux"] = aux_uv
            logged["uv_aux_px"] = uv_px
        err = (mean - target).abs().mean(0)
        for name, e in zip(self.config.action_joint_names, err):
            logged[f"mae_{name}"] = e
        logged["sigma_mean"] = torch.exp(0.5 * log_var).mean()
        # one device -> host copy for all logged values (each .item() would be a separate sync)
        info = dict(zip(logged, torch.stack([v.detach().float() for v in logged.values()]).tolist()))
        return loss, info

    def uv_aux_target(self, state: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        """Training labels for keypoints 0 / 1: (B, 2, 2) normalized (u, v) of the target and the current
        door gripper point in the (augmented) image, and (B, 2) validity (inside the image, in front)."""
        cfg = self.config
        tilt = state[:, self.tilt_index]
        q_cur = state[:, self.base_index][:, self.fk_index]
        pts = [gripper_position(*target[:, self.fk_index].unbind(-1)), gripper_position(*q_cur.unbind(-1))]
        uvs, valid = [], []
        for p in pts:
            u, v, z = project_points(p, tilt, cfg.camera_hfov, tuple(cfg.image_size), cfg.camera_tilt_origin,
                                     cfg.camera_in_tilt_link)
            uvs.append(torch.stack([u, v], -1))
            valid.append(z > 0.05)
        uv = torch.stack(uvs, 1)  # (B, 2, 2)
        ok = torch.stack(valid, 1)
        if self._homographies is not None:  # the warp maps output -> input grid coordinates
            inv = torch.linalg.inv(self._homographies.float())
            h = torch.cat([uv, torch.ones_like(uv[..., :1])], -1) @ inv.transpose(1, 2)
            uv = h[..., :2] / h[..., 2:].clamp_min(1e-6)
        ok = ok & (uv.abs() < 0.95).all(-1)
        return uv, ok

    def _uv_aux_loss(self, batch: dict[str, Tensor], target: Tensor) -> tuple[Tensor, Tensor]:
        state = batch[OBS_STATE].float()
        if state.ndim == 3:
            state = state[:, -1]
        uv_true, ok = self.uv_aux_target(state, target)
        hh, ww = self.config.image_size
        px = torch.tensor([ww / 2, hh / 2], device=uv_true.device)
        pred = self._kp_uv[:, :2] * px
        true = uv_true * px
        per = F.huber_loss(pred, true, delta=self.config.uv_aux_huber_delta_px, reduction="none").sum(-1)
        w = ok.float()
        n = w.sum().clamp_min(1.0)
        dist = (pred - true).norm(dim=-1)
        return (per * w).sum() / n, (dist * w).sum() / n

    def aux_target(self, target: Tensor) -> Tensor:
        """Where keypoint 0 should be for the target joints (B, 5): (B, 1) z or (B, 3) xyz in base_link."""
        q = target[:, self.fk_index].unbind(-1)
        p = gripper_position(*q)
        if self.config.xyz_aux_dims == "z":
            return p[:, 2:] + self.xyz_aux_offset
        yaw = link_3_pose(*q)[3]
        c, s = torch.cos(yaw), torch.sin(yaw)
        o = self.xyz_aux_offset
        return p + torch.stack([c * o[0] - s * o[1], s * o[0] + c * o[1], o[2].expand_as(c)], -1)

    def get_optim_params(self) -> list[dict]:
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        ids = {id(p) for p in backbone}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        groups = [{"params": rest}]
        if backbone:  # empty when the whole backbone is frozen
            groups.insert(0, {"params": backbone, "lr": self.config.optimizer_lr * self.config.backbone_lr_scale})
        return groups

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
