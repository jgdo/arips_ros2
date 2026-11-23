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
    --output src/arips_lerobot_data/datasets/arips/pregrasp_gazebo_pull_right_30 \
    --tag pull_right          # door type, in the task string and meta/pregrasp_source.json
```

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
    --dataset.repo_id=arips/pregrasp_gazebo_pull_right_30 \
    --dataset.root=src/arips_lerobot_data/datasets/arips/pregrasp_gazebo_pull_right_30 \
    --dataset.eval_split=0.1 \
    --policy.color_augmentation=true \
    --policy.geometric_augmentation=true \
    --policy.depth_augmentation=true \
    --policy.state_noise=true \
    --policy.push_to_hub=false \
    --steps=30000 --batch_size=32 --save_freq=5000 --num_workers=8 \
    --output_dir=src/arips_lerobot_data/train/doorhandle_pull_right_<date>
```

- **Speed:** on an RTX 3090 this runs at about 6 steps/s at batch 8, using about 2.3 GB of GPU memory. The data loader (decoding HEVC video) is the usual bottleneck, so raise `--num_workers` if `data_s` dominates in the log.
- **Fast experiments:** `--policy.vision_backbone=resnet18 --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1`.
- **Learning rate:** the backbone gets `optimizer_lr * backbone_lr_scale` (1e-4); everything else gets `optimizer_lr` (1e-3). AdamW with cosine decay and warmup. LeRobot shrinks the schedule automatically for runs shorter than `scheduler_decay_steps`.
- **Logged values:**
  - `loss_nll`, `loss_fk`;
  - `fk_mm`, the mean door-gripper position error;
  - `mae_<joint>`, the absolute error in m / rad;
  - `sigma_mean`.

**Augmentation.** It runs on the GPU, during training only, after the resize. Every part is switched on by a flag and has configurable ranges:

| flag | does | parameters (defaults) |
|---|---|---|
| `color_augmentation` | brightness, contrast, saturation, hue, sharpness (RGB only); `color_max_num_transforms` of them per sample | ranges of LeRobot's `ImageTransformsConfig` |
| `geometric_augmentation` | one random homography per sample, applied identically to RGB (bilinear) and depth (nearest); pixels from outside the image become black / invalid depth | `geometric_prob` 0.8, `geometric_zoom` [0.9, 1.1], `geometric_rotation_deg` 5, `geometric_shift` 0.05, `geometric_perspective` 0.05 |
| `depth_augmentation` | relative Gaussian noise, random rectangular holes, pixel dropout | `depth_prob` 0.8, `depth_noise_std` 0.01, `depth_max_holes` 3, `depth_hole_size` 0.15, `depth_dropout` 0.01 |
| `state_noise` | Gaussian noise on `observation.state` (raw units) | `state_noise_std` (0.005 m, then 1° for each rotary joint) |

State noise is added before the residual base is taken, so the network cannot simply copy q_current. It has to use the image.

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
    --dataset-root src/arips_lerobot_data/datasets/arips/pregrasp_gazebo_pull_right_30 \
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
