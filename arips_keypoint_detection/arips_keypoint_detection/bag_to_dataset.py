"""Extract RGB(-D) frames from a folder of rosbags into a flat keypoint dataset folder.

For every kept RGB frame the nearest depth image (same bag, within --max-dt) is written too:
  <bag>_<sec>_<nsec>_rgb.jpg     (original jpeg/png bytes, no re-encoding)
  <bag>_<sec>_<nsec>_depth.png   (uint16 millimetres, 0 = invalid)
each named after its own header stamp. Frames without a matching depth image are dropped.
RGB-only: with --no-depth, or for bags that don't contain the depth topic, only the RGB images are
written (no frames are dropped).
Existing files are kept (annotations are never touched) unless --overwrite is given.

usage:
  ros2 run arips_keypoint_detection extract_rgbd_dataset --bags DIR [--output DIR] [--skip N]
      [--rgb-topic T] [--depth-topic T | --no-depth] [--max-dt S] [--overwrite]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from . import dataset_io as dio

DEFAULT_RGB_TOPIC = '/kinect/rgb/image_raw/compressed'
DEFAULT_DEPTH_TOPIC = '/kinect/depth/image_raw/compressedDepth'


def stamp_ns(msg) -> int:
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def find_bags(root: Path, exclude: Path) -> list[Path]:
    exclude = exclude.resolve()
    bags = []
    for meta in sorted(root.rglob('metadata.yaml')):
        bag = meta.parent.resolve()
        if bag == exclude or exclude in bag.parents:
            continue
        bags.append(meta.parent)
    return bags


def read_streams(bag: Path, rgb_topic: str, depth_topic: str | None):
    """Sorted lists of (stamp_ns, msg) for the rgb and depth topic, and their types.

    depth_topic None, or not in the bag: depth is an empty list and its type None (rgb only).
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if rgb_topic not in types:
        raise RuntimeError(f'topic {rgb_topic} not in bag (has: {", ".join(sorted(types))})')
    topics = [rgb_topic] + ([depth_topic] if depth_topic and depth_topic in types else [])
    classes = {topic: get_message(types[topic]) for topic in topics}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(classes)))

    rgb, depth = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, classes[topic])
        (rgb if topic == rgb_topic else depth).append((stamp_ns(msg), msg))
    rgb.sort(key=lambda x: x[0])
    depth.sort(key=lambda x: x[0])
    return rgb, depth, types[rgb_topic], types[depth_topic] if len(topics) > 1 else None


def rgb_bytes_and_ext(msg, msg_type: str) -> tuple[bytes, str]:
    if msg_type.endswith('CompressedImage'):
        fmt = msg.format.lower()
        ext = 'png' if 'png' in fmt else 'jpg'
        return bytes(msg.data), ext
    ok, buf = cv2.imencode('.jpg', dio.decode_raw_rgb_bgr(msg), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError('jpeg encoding failed')
    return buf.tobytes(), 'jpg'


def decode_depth(msg, msg_type: str) -> np.ndarray:
    if msg_type.endswith('CompressedImage'):
        return dio.decode_compressed_depth(msg)
    return dio.decode_raw_depth(msg)


def convert_bag(bag: Path, out: Path, args) -> tuple[int, int, int, bool]:
    """Returns (written, already existing, dropped without depth, with depth)."""
    rgb, depth, rgb_type, depth_type = read_streams(bag, args.rgb_topic,
                                                    None if args.no_depth else args.depth_topic)
    bag_name = dio.sanitize_name(bag.name)
    with_depth = depth_type is not None
    depth_stamps = [t for t, _ in depth]
    written = existing = dropped = 0
    for t, rgb_msg in rgb[::args.skip]:
        if not with_depth:
            data, ext = rgb_bytes_and_ext(rgb_msg, rgb_type)
            rgb_path = out / dio.frame_name(bag_name, t, 'rgb', ext)
            if rgb_path.exists() and not args.overwrite:
                existing += 1
            else:
                rgb_path.write_bytes(data)
                written += 1
            continue
        j = dio.nearest_index(depth_stamps, t)
        if j is None or abs(depth_stamps[j] - t) > args.max_dt * 1e9:
            dropped += 1
            continue
        data, ext = rgb_bytes_and_ext(rgb_msg, rgb_type)
        rgb_path = out / dio.frame_name(bag_name, t, 'rgb', ext)
        depth_path = out / dio.frame_name(bag_name, depth_stamps[j], 'depth', 'png')
        if rgb_path.exists() and depth_path.exists() and not args.overwrite:
            existing += 1
            continue
        rgb_path.write_bytes(data)
        dio.save_depth_m(depth_path, decode_depth(depth[j][1], depth_type))
        written += 1
    return written, existing, dropped, with_depth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bags', required=True, type=Path, help='folder containing rosbag folders')
    parser.add_argument('--output', type=Path, help='dataset folder (default: <bags>/dataset)')
    parser.add_argument('--skip', type=int, default=1, help='keep every N-th RGB frame (default 1 = all)')
    parser.add_argument('--rgb-topic', default=DEFAULT_RGB_TOPIC)
    depth = parser.add_mutually_exclusive_group()
    depth.add_argument('--depth-topic', default=DEFAULT_DEPTH_TOPIC,
                       help='depth topic; bags without it are extracted rgb-only')
    depth.add_argument('--no-depth', action='store_true', help='extract only rgb images')
    parser.add_argument('--max-dt', type=float, default=0.05,
                        help='max rgb/depth stamp difference in seconds (default 0.05)')
    parser.add_argument('--overwrite', action='store_true', help='rewrite existing image files')
    args = parser.parse_args(argv)

    if args.skip < 1:
        parser.error('--skip must be >= 1')
    if not args.bags.is_dir():
        parser.error(f'{args.bags} is not a folder')
    out = args.output or args.bags / 'dataset'
    bags = find_bags(args.bags, out)
    if not bags:
        print(f'no rosbags (metadata.yaml) found below {args.bags}', file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    failed = []
    for bag in bags:
        try:
            written, existing, dropped, with_depth = convert_bag(bag, out, args)
        except Exception as e:  # noqa: BLE001 - keep converting the other bags
            print(f'{bag.name}: FAILED: {e}', file=sys.stderr)
            failed.append(bag.name)
            continue
        total += written
        if with_depth:
            print(f'{bag.name}: {written} frames written, {existing} already existed, '
                  f'{dropped} dropped (no depth within {args.max_dt} s)')
        else:
            why = '--no-depth' if args.no_depth else f'no {args.depth_topic} in bag'
            print(f'{bag.name}: {written} rgb frames written, {existing} already existed (rgb only: {why})')
    print(f'{total} frames written to {out}')
    if failed:
        print(f'{len(failed)} bag(s) failed: {", ".join(failed)}', file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
