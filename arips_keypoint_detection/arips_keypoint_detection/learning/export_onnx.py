"""Export a trained keypoint checkpoint to ONNX (+ metadata json next to it).

usage:
  ros2 run arips_keypoint_detection export_keypoints_onnx CHECKPOINT.pt [--out MODEL.onnx] [--threshold T]

The ONNX model takes one input per image type ('rgb': 1x3xHxW, 'depth': 1x2xHxW, see preprocess.py)
and outputs 'heatmaps' (1xKxH/4xW/4, probabilities). It is checked against PyTorch with onnxruntime
and cv2.dnn.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import preprocess as pp
from .models import ProbabilityModel, build_model


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt['meta']
    model = build_model(meta['model'], meta['inputs'], len(meta['keypoint_names']), pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval(), meta


def export(checkpoint, out: Path, threshold=None, opset=17) -> Path:
    model, meta = load_checkpoint(checkpoint)
    if threshold is not None:
        meta['threshold'] = threshold
    w, h = meta['input_size']
    dummy = tuple(torch.zeros(1, pp.INPUT_CHANNELS[name], h, w) for name in meta['inputs'])
    wrapped = ProbabilityModel(model).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapped, dummy, str(out), input_names=list(meta['inputs']), output_names=['heatmaps'],
                      opset_version=opset, dynamo=False)
    pp.save_meta(out, meta)

    # parity check
    rng = np.random.default_rng(0)
    feeds = {name: rng.standard_normal(t.shape).astype(np.float32) for name, t in zip(meta['inputs'], dummy)}
    with torch.no_grad():
        ref = wrapped(*[torch.from_numpy(feeds[n]) for n in meta['inputs']]).numpy()
    from .inference import KeypointDetector
    for backend in ('onnxruntime', 'cv2.dnn'):
        try:
            det = KeypointDetector(out, backend=backend)
        except Exception as e:  # noqa: BLE001
            print(f'{backend}: not available ({e})')
            continue
        hm = det.heatmaps(feeds)
        t0 = time.perf_counter()
        for _ in range(20):
            det.heatmaps(feeds)
        dt = (time.perf_counter() - t0) / 20
        print(f'{det.backend}: max diff to torch {np.abs(hm - ref[0]).max():.2e}, {dt * 1000:.1f} ms / frame')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--out', type=Path, help='onnx path (default: <checkpoint folder>/model.onnx)')
    parser.add_argument('--threshold', type=float, help='visibility threshold stored in the metadata')
    args = parser.parse_args(argv)
    out = export(args.checkpoint, args.out or args.checkpoint.parent / 'model.onnx', args.threshold)
    print(f'exported {out} (+ {out.with_suffix(".json").name})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
