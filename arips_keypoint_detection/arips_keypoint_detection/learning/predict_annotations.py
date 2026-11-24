"""Annotate dataset images with a trained keypoint model, so its predictions can be inspected in the
annotation tool.

usage:
  ros2 run arips_keypoint_detection predict_annotations MODEL --split-file DS/split_<tag>_..._test.json
  ros2 run arips_keypoint_detection predict_annotations MODEL --data DS [DS ...]
      [--tag NAME] [--threshold T] [--all-keypoints] [--overwrite]

MODEL is a checkpoint (.pt), an exported model (.onnx) or a training folder (best.pt is used, or
model.onnx if there is no checkpoint). For every image it writes <rgb file>_<NAME>.json in the normal
annotation format, plus config_<NAME>.json with the model's keypoint names, next to the images. NAME
defaults to the model folder name. Keypoints below the threshold are left out, as the annotation tool
treats them as not visible; --all-keypoints writes every keypoint anyway.
Open the result with: keypoint_annotator --dataset DS, then choose the tag NAME.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .. import dataset_io as dio
from . import data as kd
from . import preprocess as pp
from .evaluate import predict_checkpoint, predict_onnx


def resolve_model(path: Path) -> Path:
    if path.is_dir():
        for name in ('best.pt', 'model.onnx'):
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f'no best.pt or model.onnx in {path}')
    return path


def load_model_meta(model_path: Path) -> dict:
    if model_path.suffix == '.onnx':
        meta = pp.load_meta(model_path)
        if meta is None:
            raise FileNotFoundError(f'metadata json for {model_path} not found')
        return meta
    import torch
    return torch.load(model_path, map_location='cpu', weights_only=False)['meta']


def collect_images(split_files, data_dirs) -> dict[Path, list[Path]]:
    """dataset folder -> rgb images (all images, annotated for some tag or not)."""
    out = {}
    for split in split_files or []:
        split = Path(split)
        images = [p for p in dio.load_split(split) if p.is_file()]
        missing = len(dio.load_split(split)) - len(images)
        if missing:
            print(f'warning: {split.name}: {missing} image(s) not found, skipped')
        out.setdefault(split.parent, []).extend(images)
    for d in data_dirs or []:
        out.setdefault(Path(d), []).extend(dio.list_rgb_images(d))
    return {d: sorted(set(imgs)) for d, imgs in out.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', type=Path, help='.pt checkpoint, .onnx model or training folder')
    parser.add_argument('--split-file', nargs='+', type=Path, help='annotate the images of these split file(s)')
    parser.add_argument('--data', nargs='+', type=Path, help='annotate all images of these dataset folder(s)')
    parser.add_argument('--tag', help='tag for the written annotations (default: model folder name)')
    parser.add_argument('--threshold', type=float, help='visibility threshold (default: from the model metadata)')
    parser.add_argument('--all-keypoints', action='store_true', help='also write keypoints below the threshold')
    parser.add_argument('--overwrite', action='store_true', help='replace existing annotations with this tag')
    args = parser.parse_args(argv)
    if not args.split_file and not args.data:
        parser.error('either --split-file or --data is required')

    model_path = resolve_model(args.model)
    tag = args.tag or dio.sanitize_name(model_path.parent.name)
    if not dio.is_valid_tag(tag):
        parser.error(f'invalid tag {tag!r}, allowed: letters, digits, _ and -; use --tag')
    meta = load_model_meta(model_path)
    if tag == meta['tag']:
        parser.error(f'tag {tag!r} is the training tag of the model, refusing to overwrite the ground truth')
    names = list(meta['keypoint_names'])
    thr = args.threshold if args.threshold is not None else meta.get('threshold', 0.3)

    datasets = collect_images(args.split_file, args.data)
    if not args.overwrite:
        for d, images in datasets.items():
            existing = [p for p in images if dio.has_annotation(p, tag)]
            if dio.config_path(d, tag).is_file() or existing:
                parser.error(f'{d}: annotations with tag {tag!r} already exist ({len(existing)} images), '
                             'use --overwrite or another --tag')

    print(f'model {model_path} ({meta["model"]}, inputs {"+".join(meta["inputs"])}), tag {tag!r}, threshold {thr}')
    for d, images in datasets.items():
        if not images:
            print(f'{d}: no images')
            continue
        samples = kd._make_samples(d, images, meta['tag'], len(names), annotated_only=False)[0]
        ref_size = tuple(meta['ref_size'])
        if model_path.suffix == '.onnx':
            res, _ = predict_onnx(model_path, samples, ref_size)
        else:
            res, _ = predict_checkpoint(model_path, samples, ref_size)
        n_kp = np.zeros(len(names), int)
        for s, pts, conf in zip(samples, res['pred'], res['conf']):
            kps = {k: (float(x), float(y)) for k, ((x, y), c) in enumerate(zip(pts, conf))
                   if args.all_keypoints or c >= thr}
            for k in kps:
                n_kp[k] += 1
            dio.save_annotation(s.rgb_path, tag, kps)
        cfg = dio.save_config(d, tag, names)
        counts = ', '.join(f'{n}: {c}' for n, c in zip(names, n_kp))
        print(f'{d}: annotated {len(samples)} images ({counts} visible), config {cfg.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
