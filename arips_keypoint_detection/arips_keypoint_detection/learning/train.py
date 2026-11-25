"""Train a heatmap keypoint model on one or more annotated keypoint datasets.

usage:
  # with split files (see create_split), the test split is optional:
  ros2 run arips_keypoint_detection train_keypoints --tag door_handle \
      --train-split DS/split_door_handle_foo_train.json [--test-split DS/split_door_handle_foo_test.json]
      --inputs rgb depth --model mobilenet|tiny_unet [--out DIR] [options]
  # without split files: all annotated images of the datasets, split by bag
  ros2 run arips_keypoint_detection train_keypoints --data DIR [DIR ...] --tag door_handle ...

Several split files (e.g. of different datasets) can be given and are combined. Without split
files the train/val split is by bag (the last --val-bags bags of all datasets, sorted by name),
never by frame, since neighbouring frames of a bag are near-duplicates.
Without a test split there is no validation and best.pt is the final model.

Output folder: best.pt / best.json (EMA weights of the best validation epoch + model metadata,
including the used splits under "split"), last.pt, log.csv, preview_best.jpg (validation
predictions), train_args.json, model.onnx / model.json (best.pt exported to ONNX after training,
skip with --no-export).
"""

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .. import dataset_io as dio
from . import data as kd
from . import preprocess as pp
from . import viz
from .losses import LOSSES
from .metrics import format_metrics, keypoint_metrics
from .models import build_model, count_parameters


class EMA:
    """Exponential moving average of all parameters and buffers."""

    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for ema_v, v in zip(self.model.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.lerp_(v, 1.0 - d)
            else:
                ema_v.copy_(v)


def model_inputs(batch, inputs, device):
    return [batch[name].to(device, non_blocking=True) for name in inputs]


@torch.no_grad()
def predict(model, loader, device, stride):
    """Runs the model on a (non-augmented) loader; returns arrays in reference coordinates."""
    model.eval()
    out = {'pred': [], 'conf': [], 'gt': [], 'gt_vis': [], 'heatmaps': [], 'preview': [], 'm_inv': []}
    for batch in loader:
        logits = model(*model_inputs(batch, model.inputs, device))   # fp32, like the exported model
        hms = torch.sigmoid(logits.float()).cpu().numpy()
        for i, hm in enumerate(hms):
            m_inv = batch['m_inv'][i].numpy().astype(np.float64)
            pts, conf = pp.decode_heatmaps(hm, stride, m_inv)
            out['pred'].append(pts)
            out['conf'].append(conf)
            out['heatmaps'].append(hm)
            out['m_inv'].append(m_inv)
        out['gt'].append(batch['keypoints'].numpy())
        out['gt_vis'].append(batch['visible'].numpy())
        out['preview'].append(batch['preview'].numpy())
    for k in ('pred', 'conf', 'heatmaps', 'm_inv'):
        out[k] = np.stack(out[k]) if out[k] else np.zeros((0,))
    for k in ('gt', 'gt_vis', 'preview'):
        out[k] = np.concatenate(out[k]) if out[k] else np.zeros((0,))
    return out


def preview_image(res, names, threshold, max_images=16):
    """Grid of network-input-sized previews with predictions (filled), GT (white rings) and heatmaps."""
    images = []
    for i in range(min(max_images, len(res['pred']))):
        m = pp.invert_affine(res['m_inv'][i])
        to_in = lambda p: pp.apply_affine(m, np.asarray(p) - 0.5) + 0.5   # noqa: E731
        images.append(viz.draw_keypoints(res['preview'][i], to_in(res['pred'][i]), res['conf'][i], threshold,
                                         names, gt=to_in(res['gt'][i]), gt_vis=res['gt_vis'][i],
                                         heatmaps=res['heatmaps'][i]))
    return viz.grid(images)


def build_meta(args, names, ref_size, inputs, num_params, split_info):
    return {
        'tag': args.tag,
        'keypoint_names': names,
        'inputs': inputs,
        'model': args.model,
        'input_size': list(args.input_size),
        'ref_size': list(ref_size),
        'heatmap_stride': 4,
        'sigma': args.sigma,
        'depth_range': list(pp.DEPTH_RANGE),
        'threshold': args.threshold,
        'num_params': num_params,
        'split': split_info,
    }


def load_keypoint_names(dataset_dirs, tag):
    for d in dataset_dirs:
        path = dio.config_path(d, tag)
        if path.is_file():
            return dio.load_config(path)[1]
    raise FileNotFoundError(f'no config_{tag}.json in {", ".join(map(str, dataset_dirs))}')


def split_run_name(split_file: Path) -> str:
    """split_door_handle_foo_train.json -> door_handle_foo, split_door_handle_foo_all.json -> door_handle_foo_all"""
    stem = Path(split_file).stem
    stem = stem[len(dio.SPLIT_PREFIX):] if stem.startswith(dio.SPLIT_PREFIX) else stem
    return stem[:-len('_train')] if stem.endswith('_train') else stem


def detect_ref_size(samples):
    import cv2
    img = cv2.imread(str(samples[0].rgb_path))
    return img.shape[1], img.shape[0]


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', nargs='+', type=Path, help='dataset folder(s), used when no split files are given')
    p.add_argument('--train-split', nargs='+', type=Path, help='split json file(s) with the training images')
    p.add_argument('--test-split', nargs='+', type=Path, help='split json file(s) with the test images (optional)')
    p.add_argument('--tag', required=True, help='annotation tag')
    p.add_argument('--inputs', nargs='+', default=['rgb'], choices=pp.VALID_INPUTS)
    p.add_argument('--model', default='mobilenet', choices=['mobilenet', 'tiny_unet'])
    p.add_argument('--out', type=Path,
                   help='output folder (default: keypoint_train/<train split name or tag>_<model>_<inputs>)')
    p.add_argument('--input-size', nargs=2, type=int, default=[320, 240], metavar=('W', 'H'))
    p.add_argument('--sigma', type=float, default=2.0, help='target Gaussian sigma in heatmap pixels')
    p.add_argument('--loss', default='qfl', choices=list(LOSSES))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--samples-per-epoch', type=int, default=1600,
                   help='augmented samples per epoch (at least the number of training images)')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight-decay', type=float, help='default 1e-4 (mobilenet) / 5e-4 (tiny_unet)')
    p.add_argument('--encoder-lr-scale', type=float, default=0.3, help='lr factor for pretrained weights')
    p.add_argument('--warmup-epochs', type=float, default=3)
    p.add_argument('--ema', type=float, default=0.998)
    p.add_argument('--val-bags', type=int, default=1, help='number of last bags used for validation')
    p.add_argument('--val-fraction', type=float, help='fraction of bags for validation (overrides --val-bags)')
    p.add_argument('--threshold', type=float, default=0.3, help='visibility threshold on heatmap peak')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--hflip', action='store_true', help='enable horizontal flip augmentation')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--overfit', type=int, help='sanity check: train and validate on the first N images, no aug')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-export', action='store_true', help='do not export best.pt to model.onnx after training')
    return p


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if not args.train_split and not args.data:
        parser.error('either --train-split or --data is required')
    if args.test_split and not args.train_split:
        parser.error('--test-split requires --train-split')
    torch.manual_seed(args.seed)
    inputs = pp.check_inputs(args.inputs)
    if args.weight_decay is None:
        args.weight_decay = 1e-4 if args.model == 'mobilenet' else 5e-4
    run_name = split_run_name(args.train_split[0]) if args.train_split else args.tag
    args.out = args.out or Path('keypoint_train') / f'{run_name}_{args.model}_{pp.inputs_name(inputs)}'
    args.out.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted({p.parent for p in args.train_split + (args.test_split or [])}) if args.train_split \
        else args.data
    names = load_keypoint_names(dataset_dirs, args.tag)
    if args.train_split:
        train_s = kd.load_split_samples(args.train_split, args.tag, len(names))
        val_s = kd.load_split_samples(args.test_split, args.tag, len(names)) if args.test_split else []
        split_info = {'mode': 'split_files',
                      'train': [str(p.resolve()) for p in args.train_split],
                      'test': [str(p.resolve()) for p in args.test_split or []]}
        split_desc = 'train split ' + ', '.join(p.name for p in args.train_split) + ', test split ' + \
            (', '.join(p.name for p in args.test_split) if args.test_split else 'none (no validation)')
    else:
        samples = kd.load_samples(args.data, args.tag, len(names))
        if args.overfit:
            train_s = val_s = samples[:args.overfit]
            val_bags = ['(overfit)']
            args.no_augment = True
        else:
            train_s, val_s, val_bags = kd.split_by_bag(samples, args.val_bags, args.val_fraction)
        split_info = {'mode': 'overfit' if args.overfit else 'bags',
                      'data': [str(Path(d).resolve()) for d in args.data], 'val_bags': val_bags}
        split_desc = f'val bags: {", ".join(val_bags)}'
    split_info |= {'num_train': len(train_s), 'num_test': len(val_s)}
    if not train_s:
        print(f'no training images annotated with tag {args.tag}', file=sys.stderr)
        return 1
    all_s = train_s + val_s
    if 'depth' in inputs and any(s.depth_path is None for s in all_s):
        n = sum(s.depth_path is None for s in all_s)
        print(f'warning: {n} images without depth image (depth input will be empty)', file=sys.stderr)
    ref_size = detect_ref_size(train_s)
    if 'depth' in inputs:
        no_depth = sum(s.depth_path is None for s in all_s)
        if no_depth == len(all_s):
            sys.exit('error: depth input requested, but no image has a depth image (rgb-only dataset?)')
        if no_depth:
            print(f'warning: {no_depth}/{len(all_s)} images have no depth image, they get an all-invalid depth input')
    print(f'train {len(train_s)}, val {len(val_s)} images ({split_desc}), keypoints {names}, inputs {inputs}, '
          f'ref size {ref_size}')

    augment = None if args.no_augment else kd.AugmentConfig(hflip=args.hflip)
    common = dict(inputs=inputs, input_size=args.input_size, ref_size=ref_size, num_keypoints=len(names),
                  sigma=args.sigma)
    train_ds = kd.KeypointDataset(train_s, augment=augment, **common)
    val_ds = kd.KeypointDataset(val_s, augment=None, **common)
    steps_per_epoch = math.ceil(max(args.samples_per_epoch, len(train_s)) / args.batch)
    train_loader = DataLoader(train_ds, batch_size=args.batch, drop_last=True, num_workers=args.workers,
                              sampler=RandomSampler(train_ds, replacement=True,
                                                    num_samples=steps_per_epoch * args.batch),
                              pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=min(args.workers, 4))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model, inputs, len(names), pretrained=not args.no_pretrained).to(device)
    n_params = count_parameters(model)
    print(f'model {args.model}: {n_params} parameters, device {device}')

    enc = {id(p) for p in model.encoder_parameters()}
    groups = [{'params': [p for p in model.parameters() if id(p) not in enc], 'lr_scale': 1.0}]
    if enc:
        groups.append({'params': [p for p in model.parameters() if id(p) in enc], 'lr_scale': args.encoder_lr_scale})
    for g in groups:
        g['lr'] = args.lr * g['lr_scale']
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, int(args.warmup_epochs * steps_per_epoch))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    ema = EMA(model, args.ema)
    loss_fn = LOSSES[args.loss]
    meta = build_meta(args, names, ref_size, inputs, n_params, split_info)

    def jsonable(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, list):
            return [jsonable(x) for x in v]
        return v

    with open(args.out / 'train_args.json', 'w') as f:
        json.dump({k: jsonable(v) for k, v in vars(args).items()} | {'split': split_info}, f, indent=2)

    log_path = args.out / 'log.csv'
    log_fields = ['epoch', 'train_loss', 'val_loss', 'score', 'mean_err', 'median_err', 'pck10', 'pck20',
                  'vis_precision', 'vis_recall', 'lr', 'time_s']
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(log_fields)

    best_score = float('inf')
    step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(*model_inputs(batch, inputs, device))
            loss = loss_fn(logits, batch['heatmap'].to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            ema.update(model, step)
            step += 1
            losses.append(loss.item())

        if not val_s:
            row = [epoch, np.mean(losses)] + [''] * 8 + [scheduler.get_last_lr()[0], time.time() - t0]
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([f'{v:.5g}' if isinstance(v, float) else v for v in row])
            print(f'epoch {epoch:3d}/{args.epochs}  loss {np.mean(losses):.4f}  (no test split)', flush=True)
            continue

        # validation with EMA weights
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits = ema.model(*model_inputs(batch, inputs, device))
                val_losses.append(loss_fn(logits, batch['heatmap'].to(device)).item())
        res = predict(ema.model, val_loader, device, meta['heatmap_stride'])
        m = keypoint_metrics(res['pred'], res['conf'], res['gt'], res['gt_vis'], args.threshold, names)
        improved = m['score'] < best_score
        if improved:
            best_score = m['score']
            torch.save({'model_state': ema.model.state_dict(), 'meta': meta, 'epoch': epoch, 'metrics': m},
                       args.out / 'best.pt')
            pp.save_meta(args.out / 'best.pt', meta | {'epoch': epoch, 'val_metrics': m})
            viz.save_rgb(args.out / 'preview_best.jpg', preview_image(res, names, args.threshold))
        row = [epoch, np.mean(losses), np.mean(val_losses), m['score'], m['mean_err'], m['median_err'], m['pck10'],
               m['pck20'], m['vis_precision'], m['vis_recall'], scheduler.get_last_lr()[0], time.time() - t0]
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([f'{v:.5g}' if isinstance(v, float) else v for v in row])
        print(f'epoch {epoch:3d}/{args.epochs}  loss {np.mean(losses):.4f}  val {np.mean(val_losses):.4f}  '
              f'{format_metrics(m)}{"  *" if improved else ""}', flush=True)

    torch.save({'model_state': ema.model.state_dict(), 'meta': meta, 'epoch': args.epochs}, args.out / 'last.pt')
    if not val_s:
        # no validation: the final model is the best one
        torch.save({'model_state': ema.model.state_dict(), 'meta': meta, 'epoch': args.epochs}, args.out / 'best.pt')
        pp.save_meta(args.out / 'best.pt', meta | {'epoch': args.epochs, 'val_metrics': None})
        print('no test split: best.pt = final model')
    else:
        best = json.loads((args.out / 'best.json').read_text())
        print(f'best epoch {best["epoch"]}: {format_metrics(best["val_metrics"])}')
    if not args.no_export:
        try:
            from .export_onnx import export
            onnx_path = export(args.out / 'best.pt', args.out / 'model.onnx')
            print(f'exported {onnx_path} (+ {onnx_path.with_suffix(".json").name})')
        except Exception as e:  # noqa: BLE001
            print(f'warning: ONNX export failed ({e}); run export_keypoints_onnx manually', file=sys.stderr)
    print(f'saved to {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
