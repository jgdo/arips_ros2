import json

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets import EpisodeAwareSampler

import lerobot_policy_target_reach  # noqa: F401  (registers + installs the factory)
from lerobot_policy_target_reach import sampler_target_reach as st
from lerobot_policy_target_reach.configuration_target_reach import TargetReachConfig
from lerobot_policy_target_reach.modeling_target_reach import TargetReachPolicy
from lerobot_policy_target_reach.targets import load_tag_frames

# Three episodes of 10, 20 and 15 frames.
FROM = [0, 10, 30]
TO = [10, 30, 45]


@pytest.fixture
def annotations(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "version": 1,
        "types": ["lin", "circ", "coll"],
        "tags": ["a", "b"],
        "episodes": {
            "0": {"length": 10, "targets": [{"frame": 3, "type": "lin", "tag": "a"},
                                            {"frame": 7, "type": "circ", "tag": "b"}]},
            "2": {"length": 15, "targets": [{"frame": 14, "type": "coll", "tag": "a"}]},
        },
    }))
    return str(path)


@pytest.fixture(autouse=True)
def no_active_config():
    st.ACTIVE_CONFIG = None
    yield
    st.ACTIVE_CONFIG = None


def test_tag_frames(annotations):
    assert load_tag_frames(annotations, "a") == {0: 3, 2: 14}
    assert load_tag_frames(annotations, "b") == {0: 7}
    with pytest.raises(ValueError):
        load_tag_frames(annotations, "missing")


def test_sampler_yields_exactly_usable_frames(annotations):
    sampler = st.TargetReachSampler(annotations, "a", FROM, TO, shuffle=True, seed=1)
    expected = set(range(0, 10)) | set(range(30, 45))  # ep 0 all, ep 1 skipped, ep 2 all
    assert len(sampler) == len(expected)
    assert set(sampler) == expected
    assert set(sampler.indices) == expected


def test_sampler_respects_episode_subset(annotations):
    sampler = st.TargetReachSampler(annotations, "a", FROM, TO, episode_indices_to_use=[1, 2])
    assert set(sampler) == set(range(30, 45))
    with pytest.raises(ValueError):
        st.TargetReachSampler(annotations, "b", FROM, TO, episode_indices_to_use=[1, 2])


def test_sampler_resume_is_deterministic(annotations):
    a = st.TargetReachSampler(annotations, "a", FROM, TO, shuffle=True, seed=3)
    first = list(a), list(a)
    b = st.TargetReachSampler(annotations, "a", FROM, TO, shuffle=True, seed=3)
    b.load_state_dict({"epoch": 1, "start_index": 5})
    assert list(b) == first[1][5:]


def test_factory_returns_stock_sampler_without_target_reach():
    sampler = st.sampler_factory(FROM, TO, shuffle=True)
    assert type(sampler) is EpisodeAwareSampler
    assert len(sampler) == 45


def test_factory_uses_filtered_sampler_for_target_reach(annotations):
    TargetReachConfig(annotation_path=annotations, target_tag="b", input_features={}, output_features={})
    sampler = st.sampler_factory(FROM, TO, shuffle=True)
    assert isinstance(sampler, st.TargetReachSampler)
    assert set(sampler) == set(range(0, 8))


def _config(**kwargs):
    return TargetReachConfig(
        input_features={
            "observation.images.kinect": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
        pretrained_backbone_weights=None,
        image_size=None,
        device="cpu",
        **kwargs,
    )


def test_parameter_count_and_frozen_backbone():
    policy = TargetReachPolicy(_config())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    assert trainable == 265_216 + 1_024 + 131_328 + 1_285
    assert not any(p.requires_grad for p in policy.backbone.parameters())
    policy.train()
    assert not policy.backbone.training


class _Meta:
    def __init__(self, root):
        self.root = root
        self.total_episodes = 3


def test_forward_masks_and_targets(annotations, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Pose of each frame = [global index] * 5, so the expected target is easy to read off.
    index = np.arange(45)
    episode = np.repeat([0, 1, 2], [10, 20, 15])
    frame = np.concatenate([np.arange(10), np.arange(20), np.arange(15)])
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({
        "index": index, "episode_index": episode, "frame_index": frame,
        "observation.state": [[float(i)] * 5 for i in index],
    }), tmp_path / "data" / "chunk-000" / "file-000.parquet")
    stats = {"action": {"mean": np.zeros(5), "std": np.ones(5)}}

    policy = TargetReachPolicy(_config(annotation_path=annotations, target_tag="a"),
                               dataset_stats=stats, dataset_meta=_Meta(tmp_path))
    assert policy.target_last.tolist() == [3, -1, 44]
    assert policy.target_pose[0].tolist() == [3.0] * 5
    assert policy.target_pose[2].tolist() == [44.0] * 5

    batch = {
        "observation.images.kinect": torch.rand(4, 3, 48, 64),
        "observation.state": torch.zeros(4, 5),
        "episode_index": torch.tensor([0, 0, 1, 2]),
        "index": torch.tensor([2, 5, 12, 40]),  # valid, valid (after target), skipped episode, valid
    }
    loss, info = policy.forward(batch)
    assert info["valid_samples"] == 3
    loss.backward()
    assert policy.predict_action_chunk(batch).shape == (4, 1, 5)
    assert policy.select_action(batch).shape == (4, 5)


# -- GPU augmentation ------------------------------------------------------------


def test_augment_ops_match_torchvision():
    from torchvision.transforms.v2 import functional as tvf

    from lerobot_policy_target_reach import augment as aug

    torch.manual_seed(0)
    x = torch.rand(4, 3, 24, 32)
    f = torch.tensor([0.6, 0.9, 1.1, 1.4]).view(-1, 1, 1, 1)
    pairs = [
        (aug._brightness, tvf.adjust_brightness),
        (aug._contrast, tvf.adjust_contrast),
        (aug._saturation, tvf.adjust_saturation),
        (aug._sharpness, tvf.adjust_sharpness),
    ]
    for ours, ref in pairs:
        expected = torch.stack([ref(x[i], float(f[i])) for i in range(4)])
        torch.testing.assert_close(ours(x, f), expected, atol=1e-5, rtol=0)
    shift = torch.tensor([-0.05, -0.01, 0.02, 0.05])
    expected = torch.stack([tvf.adjust_hue(x[i], float(shift[i])) for i in range(4)])
    torch.testing.assert_close(aug._hue(x, shift), expected, atol=1e-4, rtol=0)


def test_augment_picks_three_per_sample_and_keeps_range():
    from lerobot_policy_target_reach.augment import GpuAugment

    torch.manual_seed(0)
    x = torch.rand(64, 3, 24, 32)
    out = GpuAugment(color=True, position=True)(x)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert not torch.equal(out, x)
    assert torch.equal(GpuAugment(color=False, position=False)(x), x)


def test_augment_only_in_training(annotations):
    cfg = TargetReachConfig(color_augmentation=True, position_augmentation=True, pretrained_backbone_weights=None)
    cfg.input_features = {
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(5,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(5,))}
    cfg.image_size = (24, 32)
    policy = TargetReachPolicy(cfg)
    calls = []
    policy.augment.forward = lambda img: calls.append(img.shape) or img
    batch = {"observation.images.cam": torch.rand(2, 3, 48, 64), "observation.state": torch.zeros(2, 5)}
    policy.eval()
    policy._predict(batch)
    assert calls == []
    policy.train()
    policy._predict(batch)
    assert calls == [torch.Size([2, 3, 24, 32])]  # augmented after resizing
