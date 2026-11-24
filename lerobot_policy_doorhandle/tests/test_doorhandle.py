"""Unit tests (synthetic data, resnet18 without pretrained weights; runs on CPU or GPU)."""

import math

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig

from lerobot_policy_doorhandle.augment import depth_augment, random_homographies, warp
from lerobot_policy_doorhandle.configuration_doorhandle import DoorhandleConfig
from lerobot_policy_doorhandle.kinematics import (GRIPPER_Z_ABOVE_SLIDER, camera_points, gripper_position, link_3_pose,
                                                  project_points)
from lerobot_policy_doorhandle.modeling_doorhandle import DoorhandlePolicy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 120, 160


def make_config(**overrides) -> DoorhandleConfig:
    cfg = DoorhandleConfig(
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        image_size=(96, 128),
        device=DEVICE,
        push_to_hub=False,
        **overrides,
    )
    cfg.input_features = {
        "observation.images.kinect": PolicyFeature(FeatureType.VISUAL, (3, H, W)),
        "observation.images.kinect_depth": PolicyFeature(FeatureType.VISUAL, (1, H, W)),
        "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
    }
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (5,))}
    return cfg


def make_batch(b: int = 4) -> dict:
    g = torch.Generator().manual_seed(0)
    state = torch.tensor([0.65, 0.3, -0.5, 0.2, 0.0, -0.4]).repeat(b, 1)
    state += 0.1 * torch.randn(b, 6, generator=g)
    depth = 200 + 1500 * torch.rand(b, 1, H, W, generator=g)  # mm
    depth[:, :, :10] = 0  # invalid stripe
    return {
        "observation.images.kinect": torch.rand(b, 3, H, W, generator=g).to(DEVICE),
        "observation.images.kinect_depth": depth.to(DEVICE),
        "observation.state": state.to(DEVICE),
        "action": (state[:, :5] + 0.2 * torch.randn(b, 5, generator=g)).to(DEVICE),
    }


def all_augment_config(**overrides):
    return make_config(color_augmentation=True, geometric_augmentation=True, depth_augmentation=True,
                       state_noise=True, **overrides)


def test_train_step_and_gradients():
    cfg = all_augment_config()
    policy = DoorhandlePolicy(cfg).to(DEVICE).train()
    loss, info = policy(make_batch())
    assert torch.isfinite(loss)
    for key in ("loss_nll", "loss_fk", "fk_mm", "mae_joint_slider", "sigma_mean"):
        assert key in info
    loss.backward()
    for part in (policy.backbone.stages[0][0], policy.fpn, policy.heatmap_head, policy.descriptor_head,
                 policy.mlp, policy.out):
        grads = [p.grad for p in part.parameters()]
        assert all(g is not None for g in grads), part
        assert sum(g.abs().sum().item() for g in grads) > 0, part
    # the depth channel of the stem gets gradients too
    assert policy.backbone.stages[0][0].weight.grad[:, 3].abs().sum() > 0

    groups = policy.get_optim_params()
    assert groups[0]["lr"] == pytest.approx(cfg.optimizer_lr * cfg.backbone_lr_scale)
    n_params = sum(1 for p in policy.parameters() if p.requires_grad)
    assert len(groups[0]["params"]) + len(groups[1]["params"]) == n_params


@pytest.mark.parametrize("backbone", ["resnet18", "convnext_tiny"])
@pytest.mark.parametrize("freeze_stem", [False, True])
def test_freeze_backbone(backbone, freeze_stem):
    cfg = make_config(freeze_backbone=True, freeze_backbone_stem=freeze_stem)
    cfg.vision_backbone = backbone
    policy = DoorhandlePolicy(cfg).to(DEVICE).train()
    stem = policy.backbone.stem
    assert stem.in_channels == 4
    stem_ids = {id(p) for p in stem.parameters()}
    for p in policy.backbone.parameters():
        assert p.requires_grad == (id(p) in stem_ids and not freeze_stem)
    bns = [m for m in policy.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert all(not m.training for m in bns)
    assert not any(m.training for m in policy.backbone.modules())  # also no stochastic depth
    assert policy.fpn.training and policy.mlp.training

    before = [p.detach().clone() for p in policy.backbone.parameters()]
    stats = [m.running_mean.clone() for m in bns]
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=1e-2)
    loss, _ = policy(make_batch())
    loss.backward()
    opt.step()
    for p, b in zip(policy.backbone.parameters(), before):
        assert torch.equal(p, b) == (not p.requires_grad)
    assert all(torch.equal(m.running_mean, s) for m, s in zip(bns, stats))

    groups = policy.get_optim_params()
    assert len(groups) == (1 if freeze_stem else 2)
    n_trainable = sum(1 for p in policy.parameters() if p.requires_grad)
    assert sum(len(g["params"]) for g in groups) == n_trainable


def test_inference_shapes_and_initial_residual():
    policy = DoorhandlePolicy(make_config()).to(DEVICE).eval()
    batch = make_batch(3)
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (3, 1, 5)
    action = policy.select_action(batch)
    assert action.shape == (3, 5)
    # freshly initialised output layer: the prediction starts close to the current joints
    assert torch.allclose(action, batch["observation.state"][:, :5], atol=0.05)
    mean, sigma = policy.predict_with_uncertainty(batch)
    assert torch.allclose(mean, action)
    assert sigma.shape == (3, 5) and (sigma > 0).all()


def test_stem_depth_filter_is_rgb_mean():
    import torchvision

    from lerobot_policy_doorhandle.modeling_doorhandle import _widen_stem

    conv = torchvision.models.convnext_tiny().features[0][0]
    wide = _widen_stem(conv)
    assert wide.in_channels == 4
    assert torch.allclose(wide.weight[:, :3], conv.weight)
    assert torch.allclose(wide.weight[:, 3], conv.weight.mean(1))


def test_normalization():
    cfg = make_config()
    policy = DoorhandlePolicy(cfg).to(DEVICE)
    q = torch.tensor([[0.5, 1.0, -1.0, 0.3, 0.2], [0.8, 0.0, 0.0, 0.0, 0.0]], device=DEVICE)
    q_n = policy.normalize_action(q)
    assert torch.allclose(q_n[:, 0], torch.tensor([-1.0, 1.0], device=DEVICE))
    assert torch.allclose(q_n[:, 1:], q[:, 1:])  # rotary joints: identity
    assert torch.allclose(policy.unnormalize_action(q_n), q)
    d = torch.tensor([0.2, 0.5, 0.8, 2.0, 0.0, float("nan")], device=DEVICE)
    assert torch.allclose(policy.normalize_depth(d), torch.tensor([-1.0, 0.0, 1.0, 1.0, 1.0, 1.0], device=DEVICE))


def test_save_load_round_trip(tmp_path):
    cfg = make_config(joint_norm_ranges={"joint_slider": (0.45, 0.85)}, depth_norm_range=(0.25, 0.9),
                      rgb_mean=(0.5, 0.5, 0.5), rgb_std=(0.25, 0.25, 0.25))
    policy = DoorhandlePolicy(cfg).to(DEVICE).eval()
    batch = make_batch(2)
    before = policy.select_action(batch)
    policy.save_pretrained(tmp_path)

    loaded_cfg = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded_cfg, DoorhandleConfig)
    assert tuple(loaded_cfg.joint_norm_ranges["joint_slider"]) == (0.45, 0.85)
    assert tuple(loaded_cfg.depth_norm_range) == (0.25, 0.9)
    assert tuple(loaded_cfg.rgb_mean) == (0.5, 0.5, 0.5)
    assert tuple(loaded_cfg.rgb_std) == (0.25, 0.25, 0.25)
    assert tuple(loaded_cfg.state_joint_names) == cfg.state_joint_names

    loaded = DoorhandlePolicy.from_pretrained(tmp_path).to(DEVICE).eval()
    assert torch.allclose(loaded.select_action(batch), before, atol=1e-5)
    assert DoorhandlePolicy._loading_checkpoint is False


def test_augmentation():
    cfg = all_augment_config(depth_prob=1.0, geometric_prob=1.0)
    policy = DoorhandlePolicy(cfg).to(DEVICE)
    rgb = torch.rand(8, 3, 96, 128, device=DEVICE)
    depth = 0.3 + torch.rand(8, 1, 96, 128, device=DEVICE)
    rgb_a, depth_a, _ = policy.augment(rgb, depth)
    assert rgb_a.shape == rgb.shape and depth_a.shape == depth.shape
    assert rgb_a.min() >= 0 and rgb_a.max() <= 1
    assert (depth_a == 0).any()  # holes / dropout / outside the warped image
    valid = depth_a > 0
    assert depth_a[valid].min() > 0.25 and depth_a[valid].max() < 1.4

    # identity homography reproduces the image
    eye = torch.eye(3, device=DEVICE).expand(8, 3, 3)
    assert torch.allclose(warp(rgb, eye, "bilinear"), rgb, atol=1e-5)
    assert torch.equal(warp(depth, eye, "nearest"), depth)
    # no rotation / shift: identity; otherwise rigid (no zoom, no perspective)
    hom = random_homographies(2, 96, 128, 0.0, 0.0, DEVICE)
    assert torch.allclose(hom, torch.eye(3, device=DEVICE).expand(2, 3, 3), atol=1e-6)
    hom = random_homographies(64, 96, 128, 30.0, 0.2, DEVICE)
    assert torch.all(hom[:, 2, :2] == 0) and torch.all(hom[:, 2, 2] == 1)
    aspect = 128 / 96
    iso = hom[:, :2, :2] * torch.tensor([[1.0, aspect], [1 / aspect, 1.0]], device=DEVICE)
    assert torch.allclose(torch.linalg.det(iso), torch.ones(64, device=DEVICE), atol=1e-5)  # no scaling
    # a pure shift moves the depth image without changing its values
    shift = torch.eye(3, device=DEVICE).expand(8, 3, 3).clone()
    shift[:, 1, 2] = 2 * 4 / 96  # 4 pixels down in the input = content moves up
    moved = warp(depth, shift, "nearest")
    assert torch.equal(moved[..., :-4, :], depth[..., 4:, :])

    holes = depth_augment(depth, 0.0, 3, 0.5, 0.0)
    kept = holes > 0
    assert torch.allclose(holes[kept], depth[kept])

    # noise: uniform +-0.5 mm, the same at every distance
    noisy = depth_augment(depth, 0.5, 0, 0.0, 0.0)
    diff = noisy - depth
    assert diff.abs().max() <= 0.0005 + 1e-6 and diff.abs().max() > 0.0004
    assert diff.std().item() == pytest.approx(0.0005 / math.sqrt(3), rel=0.05)


def test_forward_kinematics():
    zero = torch.zeros(())
    # all joints at zero: link_0 at (0.14, 0.135), arm links pointing along -x of link_0 (= -y of base)
    x, y, z, yaw = link_3_pose(torch.tensor(0.1), zero, zero, zero)
    assert x.item() == pytest.approx(0.14)
    assert y.item() == pytest.approx(0.135 - 0.0962 - 0.17 - 0.15)
    assert z.item() == pytest.approx(0.1 + 0.292 + 0.06555 - 0.0516)
    assert yaw.item() == pytest.approx(0.0)
    # rotating joint_2 by 90 deg swings the 0.32 m arm about the joint_2 axis
    x, y, _, _ = link_3_pose(zero, torch.tensor(math.pi / 2), zero, zero)
    assert x.item() == pytest.approx(0.14 + 0.32)
    assert y.item() == pytest.approx(0.135 - 0.0962)
    p = gripper_position(zero, zero, zero, zero)
    assert p[0].item() == pytest.approx(0.14)
    assert p[1].item() == pytest.approx(0.135 - 0.0962 - 0.17 - 0.15 - 0.0533)
    assert p[2].item() == pytest.approx(0.292 + 0.06555 - 0.0516 - 0.0248)


def test_config_validation():
    with pytest.raises(ValueError):
        make_config(chunk_size=4)
    with pytest.raises(ValueError):
        make_config(joint_norm_ranges={"joint_slider": (0.8, 0.5)})
    cfg = make_config()
    cfg.input_features.pop("observation.images.kinect_depth")
    with pytest.raises(ValueError):
        cfg.validate_features()


def test_camera_points():
    origin, cam = (0.14, -0.005, 1.4114), (0.035, 0.0, 0.05)
    depth = torch.zeros(1, 1, 4, 6)
    depth[0, 0, 1:3, 2:4] = 0.5  # the 4 pixels around the image centre
    # tilt 0: the camera looks along base_link +x
    p = camera_points(depth, torch.zeros(1), 1.5, origin, cam)[0]
    centre = p[:, 1:3, 2:4].mean((-2, -1))
    assert torch.allclose(centre, torch.tensor([0.14 + 0.035 + 0.5, -0.005, 1.4114 + 0.05]), atol=1e-6)
    assert p[1, 1, 2] > p[1, 1, 3]  # image left = base_link +y
    assert p[2, 1, 2] > p[2, 2, 2]  # image up = base_link +z
    # tilt -pi/2 (kinect_joint axis is -y): looking straight down, camera "up" is base_link +x
    p = camera_points(depth, torch.tensor([-math.pi / 2]), 1.5, origin, cam)[0]
    centre = p[:, 1:3, 2:4].mean((-2, -1))
    assert torch.allclose(centre, torch.tensor([0.14 + 0.05, -0.005, 1.4114 - 0.035 - 0.5]), atol=1e-6)
    assert p[0, 1, 2] > p[0, 2, 2]


def test_xyz_keypoints_aux_and_structural_slider(tmp_path):
    cfg = all_augment_config(xyz_keypoints=True, xyz_fullres=True, xyz_refine=True, xyz_skip=True,
                             xyz_aux_weight=0.1, slider_from_keypoint=True, keypoint_uv=False,
                             keypoint_depth=False, keypoint_descriptors=False, global_feature=False)
    policy = DoorhandlePolicy(cfg).to(DEVICE).train()
    batch = make_batch()
    loss, info = policy(batch)
    assert torch.isfinite(loss) and "loss_xyz_aux" in info
    loss.backward()
    assert policy.xyz_aux_offset.grad is not None
    assert policy.refine[-1].weight.grad.abs().sum() > 0  # zero-initialised last layer
    policy.eval()
    mean, _ = policy.predict_with_uncertainty(batch)
    # slider = keypoint 0 z - offset - gripper height above the slider
    expected = policy._kp_xyz[:, 0, 2] - policy.xyz_aux_offset[-1] - GRIPPER_Z_ABOVE_SLIDER
    assert torch.allclose(mean[:, 0], expected, atol=1e-5)
    policy.save_pretrained(tmp_path)
    loaded = DoorhandlePolicy.from_pretrained(tmp_path).to(DEVICE).eval()
    assert torch.allclose(loaded.predict_with_uncertainty(batch)[0], mean, atol=1e-5)
    for dims in ("z", "xyz"):
        p = DoorhandlePolicy(make_config(xyz_keypoints=True, xyz_aux_weight=0.1, xyz_aux_dims=dims))
        assert p.aux_target(batch["action"].cpu()).shape == (4, 1 if dims == "z" else 3)


def test_arm_ik_inverts_fk():
    from lerobot_policy_doorhandle.kinematics import arm_ik
    q = torch.tensor([[0.77, -0.9, 2.1, 0.3], [0.6, -0.4, 1.9, -0.2]], dtype=torch.float64)
    g = gripper_position(*q.unbind(-1))
    yaw = link_3_pose(*q.unbind(-1))[3]
    j2, j3, j4 = arm_ik(g[:, 0], g[:, 1], yaw)
    assert torch.allclose(torch.stack([j2, j3, j4], 1), q[:, 1:], atol=1e-9)


def test_ik_from_keypoints():
    cfg = all_augment_config(xyz_keypoints=True, xyz_fullres=True, xyz_skip=True, xyz_aux_weight=0.1,
                             xyz_aux_dims="xyz", xyz_normals=True, normal_aux_weight=0.1,
                             ik_from_keypoints=True, keypoint_uv=False, keypoint_depth=False,
                             keypoint_descriptors=False, global_feature=False)
    policy = DoorhandlePolicy(cfg).to(DEVICE).train()
    batch = make_batch()
    loss, info = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.xyz_aux_offset.grad is not None
    policy.eval()
    mean, sigma = policy.predict_with_uncertainty(batch)
    assert torch.isfinite(mean).all() and torch.isfinite(sigma).all()


def test_project_points_inverts_camera_points():
    cfg = make_config()
    depth = 0.3 + 0.5 * torch.rand(2, 1, 96, 128)
    tilt = torch.tensor([0.2, 0.7])
    pts = camera_points(depth, tilt, cfg.camera_hfov, cfg.camera_tilt_origin, cfg.camera_in_tilt_link)
    for i, j in [(0, 0), (50, 70), (95, 127)]:
        u, v, z = project_points(pts[:, :, i, j], tilt, cfg.camera_hfov, (96, 128), cfg.camera_tilt_origin,
                                 cfg.camera_in_tilt_link)
        assert torch.allclose((u + 1) / 2 * 128 - 0.5, torch.full_like(u, float(j)), atol=1e-3)
        assert torch.allclose((v + 1) / 2 * 96 - 0.5, torch.full_like(v, float(i)), atol=1e-3)
        assert torch.allclose(z, depth[:, 0, i, j], atol=1e-5)


def test_uv_aux_is_training_only():
    cfg = all_augment_config(uv_aux_weight=0.01, keypoint_descriptors=False, global_feature=False)
    policy = DoorhandlePolicy(cfg).to(DEVICE).train()
    batch = make_batch()
    loss, info = policy(batch)
    assert torch.isfinite(loss) and "loss_uv_aux" in info
    loss.backward()
    assert policy.heatmap_head[-1].weight.grad.abs().sum() > 0
    # inference does not use the camera model: changing it leaves the prediction unchanged
    policy.eval()
    mean_a, _ = policy.predict_with_uncertainty(batch)
    policy.config.camera_hfov = 1.0
    policy.config.camera_tilt_origin = (0.0, 0.0, 0.5)
    mean_b, _ = policy.predict_with_uncertainty(batch)
    assert torch.equal(mean_a, mean_b)


def test_uv_aux_labels_follow_the_warp():
    cfg = make_config(uv_aux_weight=0.01)
    policy = DoorhandlePolicy(cfg).to(DEVICE)
    batch = make_batch(2)
    state, target = batch["observation.state"], batch["action"]
    policy._homographies = None
    uv0, _ = policy.uv_aux_target(state, target)
    shift = torch.eye(3, device=DEVICE).repeat(2, 1, 1)
    shift[:, 0, 2], shift[:, 1, 2] = 0.1, -0.05  # output -> input: input = output + t
    policy._homographies = shift
    uv1, _ = policy.uv_aux_target(state, target)
    assert torch.allclose(uv1, uv0 - torch.tensor([0.1, -0.05], device=DEVICE), atol=1e-5)
