from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@PreTrainedConfig.register_subclass("doorhandle")
@dataclass
class DoorhandleConfig(PreTrainedConfig):
    """Kinect RGB-D frame + current joints -> door handle pre-grasp joint pose (+ per-joint std).

    Inputs: one RGB image (3 channels), one depth image (1 channel, unit ``depth_input_unit``) and
    ``observation.state`` ordered as ``state_joint_names``. Output (``action`` slot): the absolute
    target joints ordered as ``action_joint_names``, in metres / radians.

    All normalization is defined here (and therefore saved in every checkpoint's config.json); the
    LeRobot normalizer runs in IDENTITY mode and dataset statistics are not used.
    """

    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # -- joints and normalization ----------------------------------------------------------------
    state_joint_names: tuple[str, ...] = (
        "joint_slider", "joint_2", "joint_3", "joint_4", "joint_door_gripper", "kinect_joint",
    )
    action_joint_names: tuple[str, ...] = (
        "joint_slider", "joint_2", "joint_3", "joint_4", "joint_door_gripper",
    )
    # Joints listed here are mapped linearly [min, max] -> [-1, 1]; all others stay as they are
    # (rotary joints: 1 rad = 1). The same map is used for the state and the target.
    joint_norm_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"joint_slider": (0.5, 0.8)}
    )
    # Depth arrives in this unit ("mm" is LeRobot's default depth_output_unit, "m" also works),
    # is mapped [min, max] metres -> [-1, 1] and clipped; invalid pixels (0, NaN) get the value below.
    depth_input_unit: str = "mm"
    depth_norm_range: tuple[float, float] = (0.2, 0.8)
    depth_invalid_value: float = 1.0
    # RGB in [0, 1] is normalized with these (ImageNet, matching the pretrained backbone).
    rgb_mean: tuple[float, float, float] = IMAGENET_MEAN
    rgb_std: tuple[float, float, float] = IMAGENET_STD

    # -- network ---------------------------------------------------------------------------------
    # (height, width) both images are resized to before the backbone.
    image_size: tuple[int, int] = (288, 384)
    vision_backbone: str = "convnext_tiny"  # or "resnet18" (fast tests)
    pretrained_backbone_weights: str | None = "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
    fpn_channels: int = 128
    num_keypoints: int = 16
    keypoint_feature_dim: int = 32
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1

    # -- loss ------------------------------------------------------------------------------------
    nll_weight: float = 1.0
    # Huber loss on the door gripper position (forward kinematics), in millimetres.
    fk_weight: float = 0.01
    fk_huber_delta_mm: float = 10.0
    log_var_range: tuple[float, float] = (-14.0, 4.0)

    # -- training augmentation (GPU, after the resize; see augment.py) ---------------------------
    color_augmentation: bool = False  # brightness / contrast / saturation / hue / sharpness (RGB)
    color_max_num_transforms: int = 3
    geometric_augmentation: bool = False  # same zoom / rotation / shift / perspective on RGB and depth
    geometric_prob: float = 0.8
    geometric_zoom: tuple[float, float] = (0.9, 1.1)
    geometric_rotation_deg: float = 5.0
    geometric_shift: float = 0.05  # fraction of width / height
    geometric_perspective: float = 0.05  # projective tilt, normalized image coordinates
    depth_augmentation: bool = False  # depth noise, holes, pixel dropout
    depth_prob: float = 0.8
    depth_noise_std: float = 0.01  # relative (1 % of the depth)
    depth_max_holes: int = 3
    depth_hole_size: float = 0.15  # max hole side, fraction of the image side
    depth_dropout: float = 0.01  # fraction of pixels set invalid
    state_noise: bool = False  # gaussian noise on the current joints (raw units)
    # Per state joint (state_joint_names order): 5 mm slider, 1 deg for every rotary joint.
    state_noise_std: tuple[float, ...] = (0.005, 0.01745, 0.01745, 0.01745, 0.01745, 0.01745)

    # -- optimizer -------------------------------------------------------------------------------
    optimizer_lr: float = 1e-3
    backbone_lr_scale: float = 0.1
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.chunk_size != 1 or self.n_action_steps != 1 or self.n_obs_steps != 1:
            raise ValueError("doorhandle predicts a single pose: chunk_size, n_action_steps, n_obs_steps = 1")
        if self.vision_backbone not in ("convnext_tiny", "resnet18"):
            raise ValueError(f"vision_backbone must be convnext_tiny or resnet18, got {self.vision_backbone}")
        if self.depth_input_unit not in ("mm", "m"):
            raise ValueError(f"depth_input_unit must be 'mm' or 'm', got {self.depth_input_unit!r}")
        missing = [j for j in self.action_joint_names if j not in self.state_joint_names]
        if missing:
            raise ValueError(f"action joints {missing} are not in state_joint_names")
        for name, (lo, hi) in self.joint_norm_ranges.items():
            if hi <= lo:
                raise ValueError(f"joint_norm_ranges[{name}]: max must be > min")
        if self.depth_norm_range[1] <= self.depth_norm_range[0]:
            raise ValueError("depth_norm_range: max must be > min")
        if len(self.state_noise_std) != len(self.state_joint_names):
            raise ValueError("state_noise_std needs one value per state joint")

    def joint_offset_scale(self, names) -> tuple[list[float], list[float]]:
        """normalized = (raw - offset) / scale for the given joints."""
        offsets, scales = [], []
        for name in names:
            if name in self.joint_norm_ranges:
                lo, hi = self.joint_norm_ranges[name]
                offsets.append((lo + hi) / 2)
                scales.append((hi - lo) / 2)
            else:
                offsets.append(0.0)
                scales.append(1.0)
        return offsets, scales

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    def rgb_and_depth_keys(self) -> tuple[str, str]:
        rgb = [k for k, ft in self.image_features.items() if ft.shape[0] == 3]
        depth = [k for k, ft in self.image_features.items() if ft.shape[0] == 1]
        if len(rgb) != 1 or len(depth) != 1:
            raise ValueError(
                f"doorhandle needs exactly one RGB (3 channel) and one depth (1 channel) image, got "
                f"{ {k: tuple(ft.shape) for k, ft in self.image_features.items()} }"
            )
        return rgb[0], depth[0]

    def validate_features(self) -> None:
        self.rgb_and_depth_keys()
        if not self.robot_state_feature:
            raise ValueError("doorhandle needs observation.state as input")
        if self.robot_state_feature.shape[0] != len(self.state_joint_names):
            raise ValueError(
                f"observation.state has {self.robot_state_feature.shape[0]} values, "
                f"state_joint_names has {len(self.state_joint_names)}"
            )
        if self.action_feature.shape[0] != len(self.action_joint_names):
            raise ValueError(
                f"action has {self.action_feature.shape[0]} values, "
                f"action_joint_names has {len(self.action_joint_names)}"
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None
