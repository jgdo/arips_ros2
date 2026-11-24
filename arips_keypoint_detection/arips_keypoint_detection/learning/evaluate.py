"""Evaluate one or more keypoint models (checkpoint .pt or exported .onnx) on annotated datasets.

usage:
  ros2 run arips_keypoint_detection evaluate_keypoints MODEL [MODEL ...] --split-file DS/split_<tag>_<name>_test.json
  ros2 run arips_keypoint_detection evaluate_keypoints MODEL [MODEL ...] --data DIR [DIR ...]
      [--split val|train|all] [--val-bags N]
  common options: [--threshold T] [--preview DIR]

Prints one metrics row per model (use it to compare input combinations / architectures on the same
images) and a visibility table over thresholds. With --split-file the images of the given split
file(s) are used; otherwise --split selects from the same bag-wise split as training.
The splits a model was trained / tested on (stored in its metadata) are printed as well.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from . import data as kd
from . import preprocess as pp
from . import viz
from .metrics import keypoint_metrics


def predict_onnx(model_path, samples, ref_size):
    from .inference import KeypointDetector
    det = KeypointDetector(model_path)
    ds = kd.KeypointDataset(samples, inputs=det.inputs + (['rgb'] if 'rgb' not in det.inputs else []),
                            input_size=det.meta['input_size'], ref_size=ref_size,
                            num_keypoints=len(det.names))
    res = {'pred': [], 'conf': [], 'heatmaps': []}
    for s in samples:
        rgb, depth = ds.load_raw(s)
        pts, conf, hm = det.detect(rgb if 'rgb' in det.inputs else None, depth)
        res['pred'].append(pts)
        res['conf'].append(conf)
        res['heatmaps'].append(hm)
    return {k: np.stack(v) for k, v in res.items()}, det.meta


def predict_checkpoint(model_path, samples, ref_size):
    import torch
    from torch.utils.data import DataLoader
    from .export_onnx import load_checkpoint
    from .train import predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, meta = load_checkpoint(model_path, device)
    ds = kd.KeypointDataset(samples, inputs=meta['inputs'], input_size=meta['input_size'], ref_size=ref_size,
                            num_keypoints=len(meta['keypoint_names']), sigma=meta.get('sigma', 2.0))
    res = predict(model, DataLoader(ds, batch_size=32, num_workers=4), device, meta['heatmap_stride'])
    return res, meta


def describe_split(split) -> str:
    if not split:
        return 'unknown (model trained before splits were recorded)'
    if split.get('mode') == 'split_files':
        test = ', '.join(Path(p).name for p in split['test']) or 'none'
        return f'train split {", ".join(Path(p).name for p in split["train"])}, test split {test}'
    return f'{split.get("mode")} split, val bags {", ".join(split.get("val_bags", []))}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', type=Path)
    parser.add_argument('--split-file', nargs='+', type=Path, help='split json file(s) to evaluate on')
    parser.add_argument('--data', nargs='+', type=Path)
    parser.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    parser.add_argument('--val-bags', type=int, default=1)
    parser.add_argument('--val-fraction', type=float)
    parser.add_argument('--threshold', type=float, help='default: from the model metadata')
    parser.add_argument('--preview', type=Path, help='folder for per-model preview images')
    args = parser.parse_args(argv)
    if not args.split_file and not args.data:
        parser.error('either --split-file or --data is required')

    rows = []
    for model_path in args.models:
        meta = pp.load_meta(model_path) if model_path.suffix == '.onnx' else None
        if meta is None:
            import torch
            meta = torch.load(model_path, map_location='cpu', weights_only=False)['meta']
        names = meta['keypoint_names']
        if args.split_file:
            samples = kd.load_split_samples(args.split_file, meta['tag'], len(names))
            eval_desc = ', '.join(p.name for p in args.split_file)
        else:
            samples = kd.load_samples(args.data, meta['tag'], len(names))
            eval_desc = f'{args.split} images'
        if not args.split_file and args.split != 'all':
            train_s, val_s, _ = kd.split_by_bag(samples, args.val_bags, args.val_fraction)
            samples = val_s if args.split == 'val' else train_s
        ref_size = tuple(meta['ref_size'])
        if model_path.suffix == '.onnx':
            res, meta = predict_onnx(model_path, samples, ref_size)
        else:
            res, meta = predict_checkpoint(model_path, samples, ref_size)
        gt = np.stack([s.keypoints for s in samples])
        gt_vis = np.stack([s.visible for s in samples])
        thr = args.threshold if args.threshold is not None else meta.get('threshold', 0.3)
        m = keypoint_metrics(res['pred'], res['conf'], gt, gt_vis, thr, names)
        rows.append((model_path, meta, m))
        print(f'\n{model_path}  ({meta["model"]}, inputs {"+".join(meta["inputs"])}, {meta.get("num_params", "?")} '
              f'params) on {len(samples)} images ({eval_desc})')
        print(f'  trained with: {describe_split(meta.get("split"))}')
        print('  threshold  vis_P  vis_R  score  err_px  PCK@10  PCK@20')
        for t in sorted({0.1, 0.2, 0.3, 0.5, 0.7, thr}):
            mt = keypoint_metrics(res['pred'], res['conf'], gt, gt_vis, t, names)
            print(f'  {t:9.2f}  {mt["vis_precision"]:5.2f}  {mt["vis_recall"]:5.2f}  {mt["score"]:5.2f}  '
                  f'{mt["mean_err"]:6.2f}  {mt["pck10"]:6.3f}  {mt["pck20"]:6.3f}{"  <" if t == thr else ""}')
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            ds = kd.KeypointDataset(samples, inputs=['rgb'], input_size=meta['input_size'], ref_size=ref_size,
                                    num_keypoints=len(names))
            images = []
            for i, s in enumerate(samples[:16]):
                rgb, _ = ds.load_raw(s)
                images.append(viz.draw_keypoints(rgb, res['pred'][i], res['conf'][i], thr, names,
                                                 gt=s.keypoints, gt_vis=s.visible, heatmaps=res['heatmaps'][i]))
            out = args.preview / f'{model_path.parent.name}_{model_path.stem}.jpg'
            viz.save_rgb(out, viz.grid(images))
            print(f'  preview: {out}')

    if len(rows) > 1:
        print('\nsummary (threshold from metadata or --threshold):')
        print(f'  {"model":50s} {"inputs":10s} {"score":>6s} {"err_px":>7s} {"PCK@10":>7s} {"PCK@20":>7s} '
              f'{"vis_P":>6s} {"vis_R":>6s}')
        for path, meta, m in rows:
            label = str(path.parent.name + '/' + path.name)[-50:]
            print(f'  {label:50s} {"+".join(meta["inputs"]):10s} {m["score"]:6.2f} {m["mean_err"]:7.2f} '
                  f'{m["pck10"]:7.3f} {m["pck20"]:7.3f} {m["vis_precision"]:6.2f} {m["vis_recall"]:6.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
