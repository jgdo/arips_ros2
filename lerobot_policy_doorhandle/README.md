# lerobot_policy_doorhandle

A LeRobot policy plugin (policy type `doorhandle`) that estimates the arm's **door handle pre-grasp joint pose** from:
- a single Kinect RGB-D frame;
- the robot's current joint state.

It replaces the pre-grasp pose that is currently hardcoded to the Gazebo world.

This is **not a ROS package**: it has an `AMENT_IGNORE`, so colcon skips it. It is a pip package that LeRobot discovers through its third-party plugin mechanism, the same way as `lerobot_policy_target_reach`. Nothing in it imports ROS.

```
rgb (3) + depth (1) ─> resize 288x384 ─> normalize ─> ConvNeXt-Tiny, 4-channel stem
                                                      (depth filter initialised as mean of RGB filters)
   ─> FPN, top-down fusion to stride 8 ─> 16 keypoint heatmaps ─> soft-argmax (u, v),
      expected depth, pooled descriptor per keypoint
   + global pooled feature ─┐
q_current (normalized) ─────┴─> MLP ─> Δq, log σ² per target joint

q_target = q_current[target joints] + Δq      (in normalized joint space, output in m / rad)
```

**Loss:**
- Gaussian NLL per joint, in normalized joint space;
- plus `fk_weight` × Huber loss on the door-gripper position, computed with the SCARA forward kinematics in `kinematics.py`, in mm.

## Install

Install into the same Python environment as `lerobot` (>= 0.6.1):

```bash
pip install -e path/to/arips_ros2/lerobot_policy_doorhandle
# in this workspace:
cd /home/rosuser/colcon_ws && uv pip install -e src/arips_ros2/lerobot_policy_doorhandle --python .venv/bin/python
```

After this, `lerobot-train --policy.type=doorhandle` works. In your own Python code, `import lerobot_policy_doorhandle`, or call `lerobot.utils.import_utils.register_third_party_plugins()`, before building the config from its type name.

The first training run downloads the ImageNet ConvNeXt-Tiny weights from torchvision (109 MB). Loading a trained checkpoint never downloads anything.

## Dataset contract

The policy expects a LeRobot dataset with these features:

| key | shape | content |
|---|---|---|
| `observation.images.kinect` | (3, H, W) | RGB, float in [0, 1]. Any resolution; it is resized to `image_size`. |
| `observation.images.kinect_depth` | (1, H, W) | Depth in **mm**: LeRobot depth video, `info.is_depth_map: true`, read back with the default `depth_output_unit="mm"`. 0 or NaN means invalid. |
| `observation.state` | (6,) | Current joints: `joint_slider` [m], `joint_2`, `joint_3`, `joint_4`, `joint_door_gripper`, `kinect_joint` [rad]. |
| `action` | (5,) | **Target** joints, constant over an episode: `joint_slider` [m], `joint_2`, `joint_3`, `joint_4`, `joint_door_gripper` [rad]. |

- **Image keys:** the names are free, but there must be exactly one 3-channel and one 1-channel image feature. The policy tells them apart by channel count.
- **Joint order:** given by `state_joint_names` and `action_joint_names` in the config. Every action joint must also be a state joint, because the prediction is a residual on top of it.
- **Frames per episode:** each episode is one approach movement. Every frame (the arm anywhere along its way) is a training sample with the same target: the pose at the end of the episode.

**Creating a dataset from ROS bags.** The bags are recorded with `ros2 run arips_grasp_dataset record_grasp_dataset`. Convert them in the ROS workspace:

```bash
ros2 run arips_lerobot pregrasp_bags_to_lerobot \
    --bags src/arips_lerobot_data/bags/pregrasp_gazebo_pull_right_30 \
    --output src/arips_lerobot_data/datasets/arips/pregrasp_gazebo_pull_right_30_288x384 \
    --tag pull_right \
    --image-size 288 384
```

`--tag` is the door type. It goes into the task string and into meta/pregrasp_source.json. `--image-size H W` is optional and resizes the frames before encoding: RGB with area interpolation, depth with nearest. It should match `image_size` in the policy config. Without it, frames are stored at the camera resolution (480×640), and decoding them becomes the bottleneck of the training.

`meta/pregrasp_source.json` maps each episode index to its source bag and that bag's recording metadata (door pose, noise, IK target).

## Normalization

All normalization is part of `DoorhandleConfig`. It is therefore saved in every checkpoint's `pretrained_model/config.json`. Dataset statistics are **not** used: `normalization_mapping` is IDENTITY, so the LeRobot pre- and post-processors pass the raw values through, and the policy normalizes internally.

| what | config field | default | mapping |
|---|---|---|---|
| prismatic joints | `joint_norm_ranges` | `{"joint_slider": [0.5, 0.8]}` m | linear, [min, max] → [-1, 1] |
| other (rotary) joints | (not listed) | | identity: 1 rad = 1 |
| depth | `depth_input_unit`, `depth_norm_range`, `depth_invalid_value` | `"mm"`, [0.2, 0.8] m, 1.0 | metres, then linear [min, max] → [-1, 1], clipped; invalid pixels set to 1.0 |
| RGB | `rgb_mean`, `rgb_std` | ImageNet | (x − mean) / std on [0, 1] images |
| image size | `image_size` | [288, 384] | RGB resized bilinear, depth resized nearest |

State and target use the same joint map, so the residual Δq lives in that space. The policy output is always in raw units (m / rad).

To read the normalization of a trained model:

```python
import json
cfg = json.load(open("CHECKPOINT/pretrained_model/config.json"))
print(cfg["joint_norm_ranges"], cfg["depth_norm_range"], cfg["rgb_mean"], cfg["rgb_std"], cfg["image_size"])

# or as an object (needs the plugin installed)
from lerobot.configs import PreTrainedConfig
import lerobot_policy_doorhandle  # registers the "doorhandle" type
cfg = PreTrainedConfig.from_pretrained("CHECKPOINT/pretrained_model")
offsets, scales = cfg.joint_offset_scale(cfg.action_joint_names)  # normalized = (raw - offset) / scale
```

## Training

```bash
lerobot-train \
    --policy.type=doorhandle \
    --dataset.repo_id=arips/pregrasp_gazebo_pull_right_30_288x384 \
    --dataset.root=src/arips_lerobot_data/datasets/arips/pregrasp_gazebo_pull_right_30_288x384 \
    --dataset.eval_split=0.1 \
    --policy.color_augmentation=true \
    --policy.geometric_augmentation=true \
    --policy.depth_augmentation=true \
    --policy.state_noise=true \
    --policy.push_to_hub=false \
    --steps=30000 --batch_size=32 --save_freq=5000 --num_workers=8 --policy.dtype=bfloat16 \
    --output_dir=src/arips_lerobot_data/train/doorhandle_pull_right_<date>
```

- **Speed:** see [Training speed](#training-speed) below. In short:
  - `--policy.dtype=bfloat16`;
  - `--num_workers=8`;
  - a dataset converted with `--image-size 288 384`;
  - optionally a fully frozen backbone.
- **Fast experiments:** `--policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1`.
- **Freezing the backbone:** `--policy.freeze_backbone=true` stops gradients to the pretrained backbone and keeps it in eval mode (fixed BatchNorm statistics, no ConvNeXt stochastic depth). The FPN, keypoint heads and MLP still train.
  - The 4-channel stem conv stays trainable, because its depth filter is not pretrained (it starts as the mean of the RGB filters).
  - Add `--policy.freeze_backbone_stem=true` to freeze the stem as well.
  - With ConvNeXt-Tiny, a frozen backbone (stem trainable) leaves about 1.3 M trainable parameters out of 29 M.
- **Learning rate:** the trainable part of the backbone gets `optimizer_lr * backbone_lr_scale` (1e-4); everything else gets `optimizer_lr` (1e-3). AdamW with cosine decay and warmup. LeRobot shrinks the schedule automatically for runs shorter than `scheduler_decay_steps`.
- **Logged values:**
  - `loss_nll`, `loss_fk`;
  - `fk_mm`, the mean door-gripper position error;
  - `mae_<joint>`, the absolute error in m / rad;
  - `sigma_mean`.

**Augmentation.** It runs on the GPU, during training only, after the resize. Every part is switched on by a flag and has configurable ranges:

| flag | does | parameters (defaults) |
|---|---|---|
| `color_augmentation` | brightness, contrast, saturation, hue, sharpness (RGB only); `color_max_num_transforms` of them per sample | ranges of LeRobot's `ImageTransformsConfig` |
| `geometric_augmentation` | one random rigid transform per sample (rotation about the image centre + shift; no zoom, no perspective, so depth values stay consistent with the image), applied identically to RGB (bilinear) and depth (nearest); pixels from outside the image become black / invalid depth | `geometric_prob` 0.8, `geometric_rotation_deg` 5, `geometric_shift` 0.05 |
| `depth_augmentation` | additive uniform noise of fixed size (independent of the distance), random rectangular holes, pixel dropout | `depth_prob` 0.8, `depth_noise_mm` 0.5 (±0.5 mm), `depth_max_holes` 3, `depth_hole_size` 0.15, `depth_dropout` 0.01 |
| `state_noise` | Gaussian noise on `observation.state` (raw units) | `state_noise_std` (0.005 m, then 1° for each rotary joint) |

State noise is added before the residual base is taken, so the network cannot simply copy q_current. It has to use the image.

### Geometric keypoints (xyz)

**Not for the real robot.** These options build the camera model, the `kinect_joint` tilt geometry and the arm's analytic IK into the model at inference. That needs a rigid, calibrated camera and arm mount. The real robot's mount is wobbly and cannot be fixed physically, so the deployed model must depend only on the images and the joint states. The options stay as a simulation experiment. The model used by `nav2_grab_policy.launch.py` is `doorhandle_pull_right_30_all_new_aug`, which uses none of them.

With only 30 training episodes, the appearance features (keypoint descriptors, global feature, u/v) let the MLP memorise the episodes. The prediction then regresses toward the mean door height, and validation error grows with training. In simulation the handle height is fully observable from geometry: the 3D z of the handle pixels, from depth plus the camera pose given by `kinect_joint`, predicts the slider label to 0.3–0.6 mm. These options make the model use that geometry. All default to off, so old checkpoints load unchanged.

| option | does |
|---|---|
| `xyz_keypoints` | Back-projects depth to a point map in base_link. Uses `camera_hfov`, `camera_tilt_joint` (state joint), `camera_tilt_origin` and `camera_in_tilt_link`, taken from the sim URDF. Feeds each keypoint's attention-weighted xyz and valid fraction to the MLP. Geometric augmentation warps the point map together with the image, so the points stay exact. |
| `xyz_fullres` | Reads the xyz out with full-resolution attention (upsampled logits) instead of stride-8 pooling. |
| `xyz_refine` | Adds a small full-resolution conv net that refines the attention logits (needs `xyz_fullres`). |
| `xyz_aux_weight`, `xyz_aux_dims` | Auxiliary Huber loss (`xyz_aux_huber_delta_mm`) that pulls keypoint 0 to a point fixed relative to the target door gripper, with a learned offset. `"z"` supervises only the height; `"xyz"` supervises the full point, with the offset rotated by the link_3 yaw. |
| `xyz_normals`, `normal_aux_weight` | Surface normals from the point map, read out per keypoint. The aux loss pulls keypoint 1's normal to the door normal, i.e. the target link_3 yaw + `normal_aux_yaw_offset_deg` (−90°). |
| `xyz_skip` | Adds a linear path from (keypoint xyz, state) to the output, next to the MLP. |
| `slider_from_keypoint` | Slider = keypoint 0 z − offset − `GRIPPER_Z_ABOVE_SLIDER`. Replaces the MLP's slider mean. |
| `ik_from_keypoints` | Residual base = analytic IK (`kinematics.arm_ik`) of the door gripper pose given by keypoint 0's xyz and keypoint 1's normal, instead of q_current. Needs `xyz_aux_dims="xyz"` and `xyz_normals`. |
| `keypoint_uv`, `keypoint_depth`, `keypoint_descriptors`, `global_feature` | Turn the appearance and image-position features off ("geometry only"). This was what stopped the memorisation. |

The validation numbers are in [Validation results](#validation-results) below.

### Training-only keypoint supervision (`uv_aux_weight`)

`uv_aux_weight` > 0 adds a training-only loss: keypoint 0's image position (u, v) is pulled to the target door gripper point and keypoint 1's to the current door gripper point. Both are projected into the augmented image with the camera model above and the measured `kinect_joint` (`uv_aux_huber_delta_px`, points outside the image are skipped). The labels only steer the attention to the handle and the gripper. Inference does not use the camera model, so the model's inputs stay the images and the joint states (`test_uv_aux_is_training_only` checks this). Not evaluated yet.

#### Validation results

All runs used the 30 pull_right training episodes. Evaluation is on the 5 held-out episodes of `pregrasp_gazebo_pull_right_val_5_288x384`, with door heights the model hasn't seen. The rows come from the training script's bf16 evaluation. `python -m lerobot_policy_doorhandle.evaluate` runs in fp32 and gives slightly lower numbers for the ik_from_keypoints model: 1.13 mm, 0.11°, 1.2 mm at 10k steps, and 0.51 mm, 0.10°, 0.6 mm at 20k.

| configuration | steps | slider MAE | j4 MAE | gripper position mean (first frame) |
|---|---|---|---|---|
| appearance features (all_new_aug, 005000) | 5k | 3.6 mm | 4.0° | 10.5 (18.3) mm |
| appearance features (all_new_aug, 025000) | 25k | 7.1 mm | 4.9° | 10.8 (14.1) mm |
| + xyz keypoints, z aux, geometry only | 10k | 1.3 mm | 4.3° | 8.4 (14.1) mm |
| + xyz aux (`xyz_aux_dims=xyz`) | 10k | 1.1 mm | 4.3° | 7.5 (12.3) mm |
| + normals + `ik_from_keypoints` | 10k | 1.2 mm | 0.2° | 1.4 (1.4) mm |
| same | 20k | 0.6 mm | 0.3° | 1.0 (1.0) mm |

The first frame is the hardest case, because the residual base is still far from the target.

- **Best geometric combination (simulation only, see the note above):**
  ```
  xyz_keypoints xyz_fullres xyz_skip xyz_aux_weight=0.1 xyz_aux_dims=xyz xyz_normals normal_aux_weight=0.1 ik_from_keypoints
  ```
  with the four appearance features off.
- **Steps:** 20k steps gave the best model. However, the IK path occasionally produces a non-finite gradient followed by a loss spike, and one of four 20k runs collapsed after about 7.5k steps; the other three ended at 0.5–0.9 mm. `lerobot-train` has no guard against non-finite gradients, so add one before using this config there (see the model README).
- **`ik_detach_base`:** detaching the IK base was clearly less accurate (j4 6° at 5k steps).

The geometric model (simulation only) is `src/arips_lerobot_data/train/doorhandle_pull_right_30_geo_ik`.

### Training speed

Measured on an RTX 3090 with ConvNeXt-Tiny, batch 16 and all augmentations on, while another job was using the same GPU. Treat the absolute numbers as roughly ±25 %.

GPU time of one step, measured with CUDA events on a fixed batch:

| configuration | fp32 | bf16 | backbone forward / backward (bf16) |
|---|---|---|---|
| unfrozen | 341 ms | 147 ms | 34 / 81 ms |
| `freeze_backbone` (stem trainable) | 164 ms | 105 ms | 35 / 44 ms |
| `freeze_backbone` + `freeze_backbone_stem` | 105 ms | 65 ms | 32 / 5 ms |

Every row also includes about 7 ms host-to-device copy at full resolution, 9 ms augmentation, 7 ms head and loss, and up to 6 ms optimizer.

- **bf16** (`--policy.dtype=bfloat16`): makes lerobot-train run the forward pass under bf16 autocast. The weights stay fp32, and augmentation, soft-argmax, the MLP head and the loss run in fp32. In the GPU benchmark it is 1.6–2.3× faster.
- **Trainable stem is expensive:** with a frozen backbone and a trainable stem, the gradient still has to pass back through the whole backbone to reach the stem. Only freezing the stem as well removes the backbone backward.
- **Data loader:** each sample decodes one AV1 RGB frame and one lossless HEVC depth frame. At 480×640 that takes about 40 ms per sample, of which depth is about 29 ms.
  - `--num_workers=8` was best. 12 workers oversubscribe 16 CPU cores and get slower again.
  - Converting the dataset at the model input size (`pregrasp_bags_to_lerobot --image-size 288 384`) brings this down to about 25 ms per sample. The loader then delivers about 260 samples/s with 8 workers instead of about 165, and the resize in the model is skipped.
  - Depth is stored with nearest interpolation, so it matches the model's own nearest resize up to the 12-bit quantization.

End to end in `lerobot-train`, measured as steady-state samples/s:

| setup | samples/s |
|---|---|
| unfrozen, fp32, 480×640 dataset, 4 workers (old default) | 49 |
| unfrozen, bf16, 288×384, 8 workers | 83 |
| frozen + trainable stem, bf16, 288×384, 8 workers | 114 |
| fully frozen, fp32, 480×640, 4 workers | 137 |
| fully frozen, bf16, 288×384, 8 workers | 177–222 |

With a fully frozen backbone, the main training process is the limit: it runs at 100 % of one CPU core launching kernels.

## Inference

```python
import torch
import lerobot_policy_doorhandle  # noqa: F401  registers the "doorhandle" policy type
from lerobot_policy_doorhandle.modeling_doorhandle import DoorhandlePolicy
from lerobot.policies.factory import make_pre_post_processors

path = "src/arips_lerobot_data/train/RUN/checkpoints/last/pretrained_model"
policy = DoorhandlePolicy.from_pretrained(path).eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=path)

def predict(rgb_uint8_hwc, depth_m_hw, joints):
    """rgb (H, W, 3) uint8 RGB, depth (H, W) float metres (0 = invalid), joints: 6 values in
    state_joint_names order. Returns (target joints, std), 5 values each, m / rad."""
    obs = {
        "observation.images.kinect": torch.as_tensor(rgb_uint8_hwc).permute(2, 0, 1).float() / 255.0,
        "observation.images.kinect_depth": torch.as_tensor(depth_m_hw).float()[None] * 1000.0,  # mm
        "observation.state": torch.as_tensor(joints, dtype=torch.float32),
    }
    obs = preprocessor(obs)  # adds the batch dimension, moves to policy.config.device
    target, sigma = policy.predict_with_uncertainty(obs)
    return postprocessor(target)[0].tolist(), sigma[0].cpu().tolist()
```

- **Image keys:** these must match `policy.config.input_features`.
- **Standard LeRobot path:** `policy.select_action(obs)` returns only the target (B, 5), in raw units.
- **Output semantics:**
  - The output is the **absolute** target joints in `action_joint_names` order: `joint_slider` [m], `joint_2`, `joint_3`, `joint_4`, `joint_door_gripper` [rad].
  - Send them to the arm as a joint goal, with `joint_door_gripper` as the gripper opening for the approach.
  - `sigma` is the predicted standard deviation per joint, in the same units. A large σ means the model is unsure, e.g. the handle is not visible, so don't move.
- **Moving camera:** the arm and Kinect move while approaching. The model is trained on frames from the whole approach, so you can re-query it at any time on the way.

## Evaluation

```bash
python -m lerobot_policy_doorhandle.evaluate \
    --policy src/arips_lerobot_data/train/RUN/checkpoints/last/pretrained_model \
    --dataset-root src/arips_lerobot_data/datasets/arips/pregrasp_gazebo_pull_right_30_288x384 \
    [--episodes 27 28 29] [--json result.json]
```

It prints:
- per-joint MAE (slider in mm, rotary joints in degrees);
- the mean predicted σ;
- the door-gripper position error from forward kinematics (mean, median, max, in mm);
- the error at the first frame (arm far away) and the last frame of every episode.

To evaluate on held-out data, pass the episodes that `--dataset.eval_split` kept out of training.

## Files

| file | content |
|---|---|
| `configuration_doorhandle.py` | `DoorhandleConfig`: all hyper-parameters, normalization and augmentation settings |
| `modeling_doorhandle.py` | `DoorhandlePolicy`: network, loss, `select_action`, `predict_with_uncertainty` |
| `augment.py` | batched GPU augmentation |
| `kinematics.py` | differentiable SCARA forward kinematics (from `arips_description`'s URDF), verified against the recorder's IK targets |
| `processor_doorhandle.py` | LeRobot pre- and post-processors (identity normalization) |
| `evaluate.py` | offline evaluation |
| `tests/test_doorhandle.py` | `python -m pytest tests` (synthetic data, resnet18, about 10 s) |
