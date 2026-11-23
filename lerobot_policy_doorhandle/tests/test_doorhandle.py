"""Unit tests (synthetic data, resnet18 without pretrained weights; runs on CPU or GPU)."""

import math

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig

from lerobot_policy_doorhandle.augment import depth_augment, random_homographies, warp
from lerobot_policy_doorhandle.configuration_doorhandle import DoorhandleConfig
from lerobot_policy_doorhandle.kinematics import gripper_position, link_3_pose
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
    rgb_a, depth_a = policy.augment(rgb, depth)
    assert rgb_a.shape == rgb.shape and depth_a.shape == depth.shape
    assert rgb_a.min() >= 0 and rgb_a.max() <= 1
    assert (depth_a == 0).any()  # holes / dropout / outside the warped image
    valid = depth_a > 0
    assert depth_a[valid].min() > 0.25 and depth_a[valid].max() < 1.4

    # identity homography reproduces the image
    eye = torch.eye(3, device=DEVICE).expand(8, 3, 3)
    assert torch.allclose(warp(rgb, eye, "bilinear"), rgb, atol=1e-5)
    assert torch.equal(warp(depth, eye, "nearest"), depth)
    # zoom only (no rotation / shift / tilt): magnifies around the centre
    hom = random_homographies(2, 96, 128, (2.0, 2.0), 0.0, 0.0, 0.0, DEVICE)
    assert torch.allclose(hom[0], torch.diag(torch.tensor([0.5, 0.5, 1.0], device=DEVICE)), atol=1e-6)

    holes = depth_augment(depth, 0.0, 3, 0.5, 0.0)
    kept = holes > 0
    assert torch.allclose(holes[kept], depth[kept])


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
