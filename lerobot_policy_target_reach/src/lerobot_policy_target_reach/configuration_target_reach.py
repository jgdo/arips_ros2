from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

from . import sampler_target_reach


@PreTrainedConfig.register_subclass("target_reach")
@dataclass
class TargetReachConfig(PreTrainedConfig):
    """Frozen ResNet18 image features + joint state -> joint pose at an annotated target frame.

    Training needs ``annotation_path`` (the visualizer's targets.json) and ``target_tag``: every
    frame of an episode whose target with that tag sits at frame ``t`` is trained to predict
    the pose at ``t``; episodes without the tag are skipped. Inference needs neither.
    """

    annotation_path: str | None = None
    target_tag: str | None = None
    # Pose recorded at the target frame: the reached position (state) or the commanded one (action).
    target_key: str = "observation.state"

    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    # Images are normalized with ImageNet statistics inside the model.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    # (height, width) images are resized to before the backbone; None keeps the camera size.
    image_size: tuple[int, int] | None = (240, 320)
    # Training-only augmentation, done on the GPU after resizing to image_size (see augment.py):
    # color = brightness/contrast/saturation/hue/sharpness, position = small rotation + shift.
    color_augmentation: bool = False
    position_augmentation: bool = False
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1

    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.target_key not in ("observation.state", "action"):
            raise ValueError(f"target_key must be 'observation.state' or 'action', got {self.target_key!r}")
        if self.chunk_size != 1 or self.n_action_steps != 1 or self.n_obs_steps != 1:
            raise ValueError("target_reach predicts a single pose: chunk_size, n_action_steps, n_obs_steps = 1")
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"vision_backbone must be a ResNet, got {self.vision_backbone}")
        if self.annotation_path and self.target_tag:
            sampler_target_reach.ACTIVE_CONFIG = self

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("target_reach needs at least one image input")
        if not self.robot_state_feature:
            raise ValueError("target_reach needs observation.state as input")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
