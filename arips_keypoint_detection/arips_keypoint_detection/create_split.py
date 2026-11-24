"""Create train/test (or all) split files from the annotated images of a dataset.

usage:
  ros2 run arips_keypoint_detection create_split DATASET/config_<tag>.json [NAME] (--test_ratio R | --all)
      [--by-bag] [--seed S] [--overwrite]

Writes next to the config:
  split_<tag>[_<NAME>]_train.json and split_<tag>[_<NAME>]_test.json   (random, test_ratio of the images)
  split_<tag>[_<NAME>]_all.json                                         (--all)
with content {"images": ["<rgb file name>", ...]}. Only images annotated for <tag> are used.

--by-bag assigns whole bags (recordings, the <bag> prefix of the image names) to either train or
test, so near-identical neighbouring frames never end up on both sides. The test split then gets
the random set of bags whose image count is closest to test_ratio.
"""

import argparse
import random
import sys
from pathlib import Path

from . import dataset_io as dio


def bag_of(image_name: str) -> str:
    parsed = dio.parse_frame_name(image_name)
    return parsed[0] if parsed else image_name


def split_by_bag(images: list[str], test_ratio: float, rng: random.Random) -> dict:
    """Whole bags go to train or test; the test image count is kept as close as possible to the ratio."""
    bags: dict[str, list[str]] = {}
    for name in images:
        bags.setdefault(bag_of(name), []).append(name)
    order = sorted(bags)
    target = len(images) * test_ratio

    def greedy(perm):
        chosen, n = [], 0
        for bag in perm:
            # take the bag if that brings the test size closer to the target
            if abs(n + len(bags[bag]) - target) < abs(n - target):
                chosen.append(bag)
                n += len(bags[bag])
        return chosen, abs(n - target)

    # greedy over many random bag orders, then a random choice among the candidates closest to the target
    candidates = []
    for _ in range(2000):
        perm = order[:]
        rng.shuffle(perm)
        candidates.append(greedy(perm))
    best = min(dev for _, dev in candidates)
    test_bags = rng.choice([bags_ for bags_, dev in candidates if dev <= best + 1e-9])
    if test_ratio > 0 and not test_bags and len(order) > 1:
        test_bags = [min(order, key=lambda b: len(bags[b]))]   # at least one test bag
    if len(test_bags) == len(order) and test_ratio < 1:
        test_bags.pop()                                         # keep at least one train bag
    test_set = set(test_bags)
    print(f'{len(order)} bags, test bags: {", ".join(sorted(test_set)) or "none"}')
    return {'train': [n for b in order if b not in test_set for n in bags[b]],
            'test': [n for b in test_bags for n in bags[b]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help='config_<tag>.json of the dataset')
    parser.add_argument('name', nargs='?', help='split name (optional)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--test_ratio', '--test-ratio', type=float, help='fraction of images in the test split')
    mode.add_argument('--all', action='store_true', help='put all annotated images into one _all split')
    parser.add_argument('--by-bag', action='store_true',
                        help='split whole bags (recordings) instead of single images')
    parser.add_argument('--seed', type=int, help='random seed (default: random)')
    parser.add_argument('--overwrite', action='store_true', help='replace existing split files')
    args = parser.parse_args(argv)

    if args.test_ratio is not None and not 0.0 <= args.test_ratio <= 1.0:
        parser.error('--test_ratio must be between 0 and 1')
    if args.by_bag and args.all:
        parser.error('--by-bag only works with --test_ratio')
    if args.name is not None and not dio.is_valid_tag(args.name):
        parser.error('the split name may only contain letters, digits, _ and -')
    tag, _ = dio.load_config(args.config)
    dataset_dir = args.config.parent
    images = [p.name for p in dio.list_rgb_images(dataset_dir) if dio.has_annotation(p, tag)]
    if not images:
        print(f'no images annotated with tag {tag!r} in {dataset_dir}', file=sys.stderr)
        return 1

    if args.all:
        splits = {'all': images}
    elif args.by_bag:
        splits = split_by_bag(images, args.test_ratio, random.Random(args.seed))
    else:
        shuffled = images[:]
        random.Random(args.seed).shuffle(shuffled)
        n_test = round(len(images) * args.test_ratio)
        splits = {'train': shuffled[n_test:], 'test': shuffled[:n_test]}

    paths = {kind: dio.split_path(dataset_dir, tag, args.name, kind) for kind in splits}
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        print('split file(s) already exist, use --overwrite: ' + ', '.join(p.name for p in existing), file=sys.stderr)
        return 1
    for kind, names in splits.items():
        dio.save_split(paths[kind], names)
        print(f'{paths[kind]}: {len(names)} images')
    return 0


if __name__ == '__main__':
    sys.exit(main())
