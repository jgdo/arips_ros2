import random

import cv2
import numpy as np
import pytest

from arips_keypoint_detection.learning import data as kd
from arips_keypoint_detection.learning import preprocess as pp

torch = pytest.importorskip('torch')


def test_resize_matrix_corner_convention():
    m = pp.resize_matrix((640, 480), (320, 240))
    # corner coordinate 100 in the reference = 50 at the input; affine works in center coords
    assert np.allclose(pp.apply_affine(m, [[100 - 0.5, 200 - 0.5]]) + 0.5, [[50, 100]])


def test_gaussian_decode_round_trip():
    kps = np.array([[10.3, 20.7], [0.0, 0.0]])
    hm = pp.gaussian_heatmaps(kps, np.array([True, False]), (80, 60), sigma=2.0)
    assert hm[1].max() == 0
    pts, conf = pp.decode_heatmaps(hm, stride=1)
    assert np.allclose(pts[0] - 0.5, kps[0], atol=1e-3)
    assert conf[0] > 0.9


def _dot_sample(tmp_path, xy):
    """Dataset sample: black image with a white dot at corner coords xy, annotated there."""
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(img, (int(xy[0]), int(xy[1])), 4, (255, 255, 255), -1)
    path = tmp_path / 'b_0000000001_000000000_rgb.png'
    cv2.imwrite(str(path), img)
    return kd.Sample(path, None, 'b', np.array([[xy[0] + 0.5, xy[1] + 0.5]]), np.array([True]))


def test_augmentation_keeps_image_and_keypoints_aligned(tmp_path):
    sample = _dot_sample(tmp_path, (300, 200))
    aug = kd.AugmentConfig(brightness=0, contrast=(1, 1), gamma=(1, 1), saturation=(1, 1), hue_deg=0, noise_std=0,
                           blur_prob=0, jpeg_prob=0, pan=0.05, zoom=(0.9, 1.1))
    ds = kd.KeypointDataset([sample], ['rgb'], (320, 240), (640, 480), 1, augment=aug)
    for _ in range(10):
        item = ds[0]
        # dot position in the augmented image vs. keypoint mapped with the same affine
        preview = item['preview'].numpy().max(axis=2).astype(np.float64)
        ys, xs = np.nonzero(preview > 128)
        dot = np.array([xs.mean(), ys.mean()])
        m = pp.invert_affine(item['m_inv'].numpy().astype(np.float64))
        kp = pp.apply_affine(m, sample.keypoints - 0.5)[0]
        assert np.linalg.norm(dot - kp) < 1.0
        # target heatmap peak decodes back to the annotated reference position
        pts, _ = pp.decode_heatmaps(item['heatmap'].numpy(), 4, item['m_inv'].numpy().astype(np.float64))
        assert np.linalg.norm(pts[0] - sample.keypoints[0]) < 0.1


def test_random_affine_identity_without_augmentation():
    cfg = kd.AugmentConfig(pan=0, rotate_deg=0, zoom=(1, 1), shear_deg=0)
    m = kd.random_affine((640, 480), (320, 240), cfg, random.Random(0))
    assert np.allclose(m, pp.resize_matrix((640, 480), (320, 240)))


def test_split_by_bag():
    samples = [kd.Sample(None, None, f'ds/ep{b}', np.zeros((1, 2)), np.ones(1, bool))
               for b in range(4) for _ in range(3)]
    train, val, val_bags = kd.split_by_bag(samples, val_bags=1)
    assert val_bags == ['ds/ep3'] and len(val) == 3 and len(train) == 9


@pytest.mark.parametrize('name', ['mobilenet', 'tiny_unet'])
@pytest.mark.parametrize('inputs', [['rgb'], ['depth'], ['rgb', 'depth']])
def test_models_forward_and_size(name, inputs):
    from arips_keypoint_detection.learning.models import MAX_PARAMS, build_model, count_parameters
    model = build_model(name, inputs, 3, pretrained=False).eval()
    xs = [torch.zeros(2, pp.INPUT_CHANNELS[i], 240, 320) for i in model.inputs]
    out = model(*xs)
    assert out.shape == (2, 3, 60, 80)
    assert count_parameters(model) < MAX_PARAMS
