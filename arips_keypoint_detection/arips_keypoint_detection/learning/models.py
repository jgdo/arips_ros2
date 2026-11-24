"""Small heatmap keypoint models (< 1M parameters) for rgb, depth or rgb+depth input.

Each enabled input has its own stride-2 stem; the stem outputs are summed and fed into a shared
encoder, so the trunk is identical for all input combinations. Output: K heatmap logits at 1/4 input
resolution (apply sigmoid for probabilities).

- mobilenet: ImageNet-pretrained MobileNetV3-Small encoder cut at stride 16 + light U-Net decoder
- tiny_unet: from-scratch inverted-residual U-Net down to stride 32
"""

import torch
import torch.nn.functional as F
from torch import nn

from . import preprocess as pp

MAX_PARAMS = 1_000_000
HEAD_BIAS = -4.0   # sigmoid(-4) ~ 0.018: most target pixels are ~0


def conv_bn_act(cin, cout, k=3, stride=1, groups=1, act=nn.ReLU):
    return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), act(inplace=True))


class SepConv(nn.Sequential):
    """Depthwise kxk + pointwise 1x1."""

    def __init__(self, cin, cout, k=3):
        super().__init__(conv_bn_act(cin, cin, k, groups=cin), conv_bn_act(cin, cout, 1))


class InvertedResidual(nn.Module):

    def __init__(self, cin, cout, stride=1, expand=4, k=3):
        super().__init__()
        mid = cin * expand
        self.block = nn.Sequential(
            conv_bn_act(cin, mid, 1),
            conv_bn_act(mid, mid, k, stride, groups=mid),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return x + y if self.residual else y


class FuseBlock(nn.Module):
    """Upsample the coarse feature to the skip resolution, concat, mix."""

    def __init__(self, c_coarse, c_skip, cout):
        super().__init__()
        self.mix = nn.Sequential(conv_bn_act(c_coarse + c_skip, cout, 1),
                                 SepConv(cout, cout, 3), SepConv(cout, cout, 3))

    def forward(self, coarse, skip):
        up = F.interpolate(coarse, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.mix(torch.cat([up, skip], dim=1))


class HeatmapModelBase(nn.Module):
    """Stems per input, encoder stages (stride 4, 8, ...), top-down decoder to stride 4, heatmap head."""

    stride = 4

    def __init__(self, inputs, num_keypoints):
        super().__init__()
        self.inputs = pp.check_inputs(inputs)
        self.num_keypoints = num_keypoints

    def _build_head(self, cin):
        self.head = nn.Sequential(SepConv(cin, cin, 3), nn.Conv2d(cin, self.num_keypoints, 1))
        nn.init.constant_(self.head[-1].bias, HEAD_BIAS)

    def forward(self, *xs):
        """xs: one tensor per input, in the order of self.inputs."""
        x = sum(self.stems[name](t) for name, t in zip(self.inputs, xs))
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        x = feats[-1]
        for fuse, skip in zip(self.fuse, reversed(feats[:-1])):
            x = fuse(x, skip)
        return self.head(x)

    def encoder_parameters(self):
        """Pretrained parameters (trained with a lower learning rate)."""
        return []


class MobileNetHeatmap(HeatmapModelBase):

    def __init__(self, inputs, num_keypoints, pretrained=True):
        super().__init__(inputs, num_keypoints)
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        from torchvision.ops.misc import Conv2dNormActivation
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        features = mobilenet_v3_small(weights=weights).features
        rgb_stem = features[0]   # 3 -> 16, stride 2, hardswish

        self.stems = nn.ModuleDict()
        if 'rgb' in self.inputs:
            self.stems['rgb'] = rgb_stem
        if 'depth' in self.inputs:
            stem = Conv2dNormActivation(pp.INPUT_CHANNELS['depth'], 16, kernel_size=3, stride=2,
                                        norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish)
            with torch.no_grad():
                # start with the pretrained edge filters applied to the (gray) depth channel
                stem[0].weight.zero_()
                stem[0].weight[:, 0] = rgb_stem[0].weight.sum(dim=1)
                stem[1].load_state_dict(rgb_stem[1].state_dict())
            self.stems['depth'] = stem
        self._pretrained = pretrained
        self.stages = nn.ModuleList([features[1], features[2:4], features[4:9]])   # 16@4, 24@8, 48@16
        self.fuse = nn.ModuleList([FuseBlock(48, 24, 32), FuseBlock(32, 16, 24)])
        self._build_head(24)

    def encoder_parameters(self):
        if not self._pretrained:
            return []
        params = list(self.stages.parameters())
        if 'rgb' in self.stems:
            params += list(self.stems['rgb'].parameters())
        return params


class TinyUNetHeatmap(HeatmapModelBase):

    def __init__(self, inputs, num_keypoints, pretrained=False):
        super().__init__(inputs, num_keypoints)
        self.stems = nn.ModuleDict({
            name: nn.Sequential(conv_bn_act(pp.INPUT_CHANNELS[name], 16, 3, 2), SepConv(16, 16, 3))
            for name in self.inputs})
        self.stages = nn.ModuleList([
            nn.Sequential(InvertedResidual(16, 24, 2, k=3), InvertedResidual(24, 24, k=3)),       # @4
            nn.Sequential(InvertedResidual(24, 32, 2, k=5), InvertedResidual(32, 32, k=5)),       # @8
            nn.Sequential(InvertedResidual(32, 48, 2, k=5), InvertedResidual(48, 48, k=5),
                          InvertedResidual(48, 48, k=5)),                                          # @16
            nn.Sequential(InvertedResidual(48, 64, 2, k=5), InvertedResidual(64, 64, k=5)),       # @32
        ])
        self.fuse = nn.ModuleList([FuseBlock(64, 48, 48), FuseBlock(48, 32, 32), FuseBlock(32, 24, 24)])
        self._build_head(24)


MODELS = {'mobilenet': MobileNetHeatmap, 'tiny_unet': TinyUNetHeatmap}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(name: str, inputs, num_keypoints: int, pretrained: bool = True) -> HeatmapModelBase:
    if name not in MODELS:
        raise ValueError(f'unknown model {name!r}, available: {", ".join(MODELS)}')
    model = MODELS[name](inputs, num_keypoints, pretrained=pretrained)
    n = count_parameters(model)
    if n >= MAX_PARAMS:
        raise ValueError(f'model {name} has {n} parameters (limit {MAX_PARAMS})')
    return model


class ProbabilityModel(nn.Module):
    """Wraps a model to output sigmoid probabilities (used for ONNX export and inference)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *xs):
        return torch.sigmoid(self.model(*xs))
