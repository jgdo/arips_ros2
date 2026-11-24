from __future__ import annotations

import logging
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
    # Freeze the pretrained backbone (no gradients, BatchNorm statistics fixed). The 4-channel stem
    # conv stays trainable unless freeze_backbone_stem is set too, since its depth filter is not
    # pretrained (only initialised as the mean of the RGB filters).
    freeze_backbone: bool = False
    freeze_backbone_stem: bool = False
    fpn_channels: int = 128
    num_keypoints: int = 16
    keypoint_feature_dim: int = 32
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.1
    # 3D keypoints: feed the MLP, per keypoint, the attention-weighted mean position of its pixels in
    # base_link (back-projected depth, camera pose at the current tilt joint angle). Unlike the image
    # coordinates it does not depend on the camera tilt, and it stays correct under the geometric
    # augmentation (the point map is computed before the warp and moves with the pixels).
    xyz_keypoints: bool = False
    keypoint_uv: bool = True  # also feed the keypoint image coordinates (u, v)
    keypoint_depth: bool = True  # also feed the keypoint expected depth
    keypoint_descriptors: bool = True  # also feed the pooled keypoint descriptors (appearance)
    global_feature: bool = True  # also feed the average pooled stride 16 / 32 features (appearance)
    # Camera model for xyz_keypoints (defaults: the Gazebo Kinect of arips_gazebo/urdf/arips_sim.urdf.xacro).
    camera_hfov: float = 1.581  # horizontal field of view [rad], square pixels, centred principal point
    camera_tilt_joint: str = "kinect_joint"  # state joint of the camera tilt, axis -y of base_link
    camera_tilt_origin: tuple[float, float, float] = (0.14, -0.005, 1.4114)  # tilt joint in base_link
    camera_in_tilt_link: tuple[float, float, float] = (0.035, 0.0, 0.05)  # optical centre, looks along +x
    # Read the keypoint xyz at image resolution (attention logits upsampled bilinearly) instead of on
    # the stride 8 grid, where each cell averages the points of 8x8 pixels (handle + door behind it).
    xyz_fullres: bool = False
    # Auxiliary loss: keypoint 0's z must equal the target door gripper z (forward kinematics of the
    # target joints) plus a learned constant. Makes keypoint 0 track something at the handle height
    # (the handle) with millimetre precision. Huber in mm, delta xyz_aux_huber_delta_mm.
    xyz_aux_weight: float = 0.0
    xyz_aux_huber_delta_mm: float = 5.0
    # "z": only the height (offset = learned scalar). "xyz": the full position, keypoint 0 = target door
    # gripper position + R_z(target link_3 yaw) * learned offset (the pre-grasp pose is fixed relative
    # to the handle, so the offset is constant in the gripper frame).
    xyz_aux_dims: str = "z"
    # Also feed per keypoint the attention-weighted surface normal (from the point map, base_link):
    # gives the door's orientation relative to the robot (approach yaw, joint_4) directly.
    xyz_normals: bool = False
    # Auxiliary loss: keypoint 1's surface normal must point along the target link_3 yaw +
    # normal_aux_yaw_offset_deg (horizontal; the pre-grasp gripper is perpendicular to the door, so its
    # yaw is the door normal's yaw + 90 deg). Huber on the unit vectors in degree-like units (x 57.3).
    normal_aux_weight: float = 0.0
    normal_aux_yaw_offset_deg: float = -90.0
    # Full-resolution refinement of the keypoint attention used for the xyz readout (needs xyz_fullres):
    # a small conv net on the normalized RGB-D image + the upsampled stride 8 features adds per-pixel
    # logits to the upsampled ones, so the attention can follow the handle's edges.
    xyz_refine: bool = False
    xyz_refine_channels: int = 32
    # Slider target = keypoint 0 z - aux offset (z) - fixed kinematic height of the door gripper above
    # the slider position; replaces the MLP's slider mean (its log variance stays). Needs xyz_aux_weight.
    slider_from_keypoint: bool = False
    # Geometric base for the residual instead of the current joints: door gripper position = keypoint 0
    # xyz - the learned aux offset rotated by the link_3 yaw, link_3 yaw = yaw of keypoint 1's surface
    # normal - normal_aux_yaw_offset_deg, then analytic IK (kinematics.arm_ik) for joint_2..4 and the
    # slider from z. The MLP predicts a residual on that. Needs xyz_aux_dims="xyz" and xyz_normals.
    ik_from_keypoints: bool = False
    # Stop the joint losses' gradients through the IK base, so the keypoints are trained only by the
    # xyz / normal aux losses. Inference is the same either way. Tried against a training instability,
    # but clearly less accurate (the joint losses refine the keypoints through the IK), so off.
    ik_detach_base: bool = False
    # Linear path from (keypoint xyz, state) to the output next to the MLP (no dropout / nonlinearity
    # between the precise geometric features and the prediction).
    xyz_skip: bool = False
    # keypoint xyz is fed as (xyz - xyz_center) / xyz_scale, clipped to [-4, 4]
    xyz_center: tuple[float, float, float] = (0.35, -0.1, 1.05)
    xyz_scale: float = 0.25

    # Training-only supervision of keypoint image positions. It is never used at inference: the model
    # inputs stay the images + joints. Keypoint 0's soft-argmax (u, v) is pulled to the target door
    # gripper point and keypoint 1's to the current door gripper point. Both are projected into the
    # (augmented) image with the camera model above and the measured kinect_joint. Points outside the
    # image are not supervised. Huber in pixels. Labels only steer the attention to the handle and
    # the gripper, so a rough camera model is enough.
    uv_aux_weight: float = 0.0
    uv_aux_huber_delta_px: float = 2.0

    # Training precision. "bfloat16" makes lerobot-train run the forward pass under bf16 autocast
    # (backbone / FPN convolutions); augmentation, soft-argmax, the MLP head and the loss stay float32.
    dtype: str = "float32"

    # -- loss ------------------------------------------------------------------------------------
    nll_weight: float = 1.0
    # Huber loss on the door gripper position (forward kinematics), in millimetres.
    fk_weight: float = 0.01
    fk_huber_delta_mm: float = 10.0
    log_var_range: tuple[float, float] = (-14.0, 4.0)

    # -- training augmentation (GPU, after the resize; see augment.py) ---------------------------
    color_augmentation: bool = False  # brightness / contrast / saturation / hue / sharpness (RGB)
    color_max_num_transforms: int = 3
    # Same rotation + shift on RGB and depth. No zoom / perspective, so the depth values stay consistent.
    geometric_augmentation: bool = False
    geometric_prob: float = 0.8
    geometric_rotation_deg: float = 5.0
    geometric_shift: float = 0.05  # fraction of width / height
    depth_augmentation: bool = False  # depth noise, holes, pixel dropout
    depth_prob: float = 0.8
    depth_noise_mm: float = 0.5  # additive uniform noise in [-depth_noise_mm, +depth_noise_mm] mm
    depth_max_holes: int = 3
    depth_hole_size: float = 0.15  # max hole side, fraction of the image side
    depth_dropout: float = 0.01  # fraction of pixels set invalid
    state_noise: bool = False  # gaussian noise on the current joints (raw units)
    # Per state joint (state_joint_names order): 5 mm slider, 1 deg for every rotary joint.
    state_noise_std: tuple[float, ...] = (0.005, 0.01745, 0.01745, 0.01745, 0.01745, 0.01745)
    # Removed augmentation settings (zoom, perspective tilt, relative depth noise). Kept only so that
    # checkpoints trained with them still load (unknown config keys are rejected); they are ignored.
    geometric_zoom: tuple[float, float] | None = None
    geometric_perspective: float | None = None
    depth_noise_std: float | None = None

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
        removed = [name for name in ("geometric_zoom", "geometric_perspective", "depth_noise_std")
                   if getattr(self, name) is not None]
        if removed:
            logging.getLogger(__name__).warning(
                "doorhandle config: %s no longer exist and are ignored (no zoom / perspective "
                "augmentation; depth noise is depth_noise_mm)", ", ".join(removed))
        if self.chunk_size != 1 or self.n_action_steps != 1 or self.n_obs_steps != 1:
            raise ValueError("doorhandle predicts a single pose: chunk_size, n_action_steps, n_obs_steps = 1")
        if self.vision_backbone not in ("convnext_tiny", "resnet18"):
            raise ValueError(f"vision_backbone must be convnext_tiny or resnet18, got {self.vision_backbone}")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(f"dtype must be 'float32' or 'bfloat16', got {self.dtype!r}")
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
        if (self.xyz_fullres or self.xyz_aux_weight > 0 or self.xyz_skip) and not self.xyz_keypoints:
            raise ValueError("xyz_fullres, xyz_aux_weight and xyz_skip need xyz_keypoints")
        if self.normal_aux_weight > 0 and not self.xyz_normals:
            raise ValueError("normal_aux_weight needs xyz_normals")
        if self.xyz_normals and not self.xyz_keypoints:
            raise ValueError("xyz_normals needs xyz_keypoints")
        if self.xyz_refine and not self.xyz_fullres:
            raise ValueError("xyz_refine needs xyz_fullres")
        if self.ik_from_keypoints and (self.xyz_aux_weight <= 0 or self.xyz_aux_dims != "xyz"
                                       or not self.xyz_normals):
            raise ValueError("ik_from_keypoints needs xyz_aux_weight > 0, xyz_aux_dims='xyz' and xyz_normals")
        if self.slider_from_keypoint and (self.xyz_aux_weight <= 0 or "joint_slider" not in self.action_joint_names):
            raise ValueError("slider_from_keypoint needs xyz_aux_weight > 0 and joint_slider in action_joint_names")
        if self.xyz_aux_dims not in ("z", "xyz"):
            raise ValueError(f"xyz_aux_dims must be 'z' or 'xyz', got {self.xyz_aux_dims!r}")
        if self.xyz_keypoints and self.camera_tilt_joint not in self.state_joint_names:
            raise ValueError(f"camera_tilt_joint {self.camera_tilt_joint} is not in state_joint_names")
        if self.uv_aux_weight > 0 and self.camera_tilt_joint not in self.state_joint_names:
            raise ValueError(f"uv_aux_weight needs camera_tilt_joint {self.camera_tilt_joint} in state_joint_names")
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
