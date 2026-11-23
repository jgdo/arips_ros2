"""Convert a folder of pre-grasp approach bags (record_grasp_dataset) into a LeRobot dataset.

One bag = one episode. Every RGB frame becomes one dataset frame with
- observation.images.kinect        RGB (480, 640, 3) uint8
- observation.images.kinect_depth  depth (480, 640, 1) float32 metres, stored as a LeRobot depth video
  (both at --image-size H W instead if given; e.g. 288 384 = the doorhandle policy input size, which
  makes video decoding in training ~1.6x faster and skips the resize in the model)
- observation.state                joints at the image time (interpolated), see STATE_JOINTS
- action                           target joints = last joint state of the bag, see ACTION_JOINTS

usage:
  ros2 run arips_lerobot pregrasp_bags_to_lerobot --bags DIR --output ROOT --tag pull_right
      [--repo-id arips/NAME] [--image-size H W] [--overwrite]
"""

import argparse
import bisect
import json
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

RGB_TOPIC = '/kinect/rgb/image_raw/compressed'
DEPTH_TOPIC = '/kinect/depth/image_raw/compressedDepth'
JOINT_TOPIC = '/feetech_joint_state_broadcaster/joint_states'

STATE_JOINTS = ['joint_slider', 'joint_2', 'joint_3', 'joint_4', 'joint_door_gripper', 'kinect_joint']
ACTION_JOINTS = ['joint_slider', 'joint_2', 'joint_3', 'joint_4', 'joint_door_gripper']

ROBOT_TYPE = 'arips_arm_door'
TASK = 'move the arm to the door handle pre-grasp pose'


class EpisodeError(Exception):
    pass


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def decode_rgb(msg) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise EpisodeError('RGB image could not be decoded')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_compressed_depth(msg) -> np.ndarray:
    """compressed_depth_image_transport -> float32 metres, 0 = invalid."""
    raw = bytes(msg.data)
    encoding = msg.format.split(';')[0].strip()
    _, quant_a, quant_b = struct.unpack('<iff', raw[:12])
    img = cv2.imdecode(np.frombuffer(raw[12:], np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise EpisodeError(f'depth image could not be decoded ({msg.format})')
    if encoding == '32FC1':
        # Inverse depth quantization: v = quant_a / depth - quant_b, v = 0 marks invalid pixels.
        v = img.astype(np.float32)
        with np.errstate(divide='ignore'):
            depth = np.where(v > 0, quant_a / (v - quant_b), 0.0)
    elif encoding == '16UC1':
        depth = img.astype(np.float32) / 1000.0
    else:
        raise EpisodeError(f'unsupported depth encoding {msg.format!r}')
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return depth[..., None]


def find_bags(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob('metadata.yaml'))


def read_bag(bag: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for topic in (RGB_TOPIC, DEPTH_TOPIC, JOINT_TOPIC):
        if topic not in types:
            raise EpisodeError(f'topic {topic} missing')
    classes = {topic: get_message(types[topic]) for topic in (RGB_TOPIC, DEPTH_TOPIC, JOINT_TOPIC)}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(classes)))

    rgb, depth, joint_t, joint_q = [], [], [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, classes[topic])
        if topic == JOINT_TOPIC:
            pos = dict(zip(msg.name, msg.position))
            if not all(j in pos for j in STATE_JOINTS):
                continue
            joint_t.append(stamp_s(msg.header))
            joint_q.append([pos[j] for j in STATE_JOINTS])
        elif topic == RGB_TOPIC:
            rgb.append((stamp_s(msg.header), msg))
        else:
            depth.append((stamp_s(msg.header), msg))
    if not rgb or not depth or len(joint_t) < 2:
        raise EpisodeError(f'empty stream (rgb {len(rgb)}, depth {len(depth)}, joints {len(joint_t)})')
    rgb.sort(key=lambda x: x[0])
    depth.sort(key=lambda x: x[0])
    order = np.argsort(joint_t)
    return rgb, depth, np.asarray(joint_t)[order], np.asarray(joint_q, dtype=np.float64)[order]


def build_frames(bag: Path, fps: float):
    rgb, depth, joint_t, joint_q = read_bag(bag)
    period = 1.0 / fps

    rgb_t = np.array([t for t, _ in rgb])
    gaps = np.diff(rgb_t)
    if len(gaps) and gaps.max() > 1.5 * period:
        raise EpisodeError(f'RGB gap of {gaps.max() * 1000:.0f} ms')

    depth_t = [t for t, _ in depth]
    pairs = []
    for t, rgb_msg in rgb:
        i = bisect.bisect_left(depth_t, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(depth_t)), key=lambda j: abs(depth_t[j] - t))
        ok = abs(depth_t[best] - t) <= 0.5 * period and joint_t[0] - period <= t <= joint_t[-1] + period
        pairs.append((t, rgb_msg, best if ok else None))
    # The recorder starts/stops all streams at once, so the first or last RGB frame may lack its
    # depth image: trim those, but a missing pair in the middle rejects the episode.
    while pairs and pairs[0][2] is None:
        pairs.pop(0)
    while pairs and pairs[-1][2] is None:
        pairs.pop()
    if len(pairs) < 2:
        raise EpisodeError('no RGB frame with a matching depth image')
    missing = [t for t, _, best in pairs if best is None]
    if missing:
        raise EpisodeError(f'{len(missing)} RGB frames without depth image / joint state, first at t={missing[0]:.3f}')

    target = joint_q[-1, :len(ACTION_JOINTS)].astype(np.float32)
    frames = []
    for t, rgb_msg, best in pairs:
        state = np.array([np.interp(t, joint_t, joint_q[:, k]) for k in range(len(STATE_JOINTS))],
                         dtype=np.float32)
        frames.append({
            'observation.images.kinect': decode_rgb(rgb_msg),
            'observation.images.kinect_depth': decode_compressed_depth(depth[best][1]),
            'observation.state': state,
            'action': target,
        })
    return frames


def resize_frame(frame: dict, height: int, width: int) -> dict:
    """RGB with area interpolation (antialiased), depth with nearest (no blending of invalid / edge depths)."""
    rgb = cv2.resize(frame['observation.images.kinect'], (width, height), interpolation=cv2.INTER_AREA)
    depth = cv2.resize(frame['observation.images.kinect_depth'], (width, height), interpolation=cv2.INTER_NEAREST)
    return {**frame, 'observation.images.kinect': rgb,
            'observation.images.kinect_depth': depth.reshape(height, width, 1)}


def features(height: int, width: int) -> dict:
    return {
        'observation.images.kinect': {
            'dtype': 'video', 'shape': (height, width, 3), 'names': ['height', 'width', 'channels'],
        },
        'observation.images.kinect_depth': {
            'dtype': 'video', 'shape': (height, width, 1), 'names': ['height', 'width', 'channels'],
            'info': {'is_depth_map': True},
        },
        'observation.state': {'dtype': 'float32', 'shape': (len(STATE_JOINTS),), 'names': STATE_JOINTS},
        'action': {'dtype': 'float32', 'shape': (len(ACTION_JOINTS),), 'names': ACTION_JOINTS},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bags', required=True, type=Path, help='folder with one bag folder per episode')
    parser.add_argument('--output', required=True, type=Path, help='dataset root folder to create')
    parser.add_argument('--tag', required=True,
                        help='door type tag, e.g. pull_right; part of the task string and the metadata')
    parser.add_argument('--repo-id', help='default: arips/<output folder name>')
    parser.add_argument('--fps', type=int, default=30, help='camera frame rate (default 30)')
    parser.add_argument('--image-size', type=int, nargs=2, metavar=('H', 'W'),
                        help='store RGB (area interpolation) and depth (nearest) at this size; default: camera size')
    parser.add_argument('--overwrite', action='store_true', help='delete an existing output folder first')
    args = parser.parse_args(argv)

    from lerobot.datasets import LeRobotDataset  # heavy import, after argument parsing

    bags = find_bags(args.bags)
    if not bags:
        sys.exit(f'no bags found under {args.bags}')
    if args.output.exists():
        if not args.overwrite:
            sys.exit(f'{args.output} exists, use --overwrite')
        shutil.rmtree(args.output)
    repo_id = args.repo_id or f'arips/{args.output.name}'
    task = f'{TASK} ({args.tag})'

    dataset = None
    sources, skipped = [], []
    for bag in bags:
        try:
            frames = build_frames(bag, args.fps)
        except EpisodeError as error:
            print(f'skip {bag.name}: {error}', flush=True)
            skipped.append({'bag': bag.name, 'reason': str(error)})
            continue
        if args.image_size:
            frames = [resize_frame(f, *args.image_size) for f in frames]
        if dataset is None:
            h, w = frames[0]['observation.images.kinect'].shape[:2]
            dataset = LeRobotDataset.create(repo_id=repo_id, fps=args.fps, features=features(h, w),
                                            root=args.output, robot_type=ROBOT_TYPE, use_videos=True)
        for frame in frames:
            dataset.add_frame({**frame, 'task': task})
        dataset.save_episode()

        meta_file = bag / 'metadata.json'
        sources.append({
            'episode_index': len(sources),
            'bag': str(bag.relative_to(args.bags)),
            'frames': len(frames),
            'target': dict(zip(ACTION_JOINTS, frames[0]['action'].tolist())),
            'recording': json.loads(meta_file.read_text()) if meta_file.exists() else None,
        })
        print(f'episode {len(sources) - 1}: {bag.name}, {len(frames)} frames', flush=True)

    if dataset is None:
        sys.exit('no valid episode')
    dataset.finalize()
    (args.output / 'meta' / 'pregrasp_source.json').write_text(json.dumps({
        'tag': args.tag,
        'task': task,
        'source_bags': str(args.bags.resolve()),
        'state_joints': STATE_JOINTS,
        'action_joints': ACTION_JOINTS,
        'action_meaning': 'target joint state = last joint state of the bag (pre-grasp pose)',
        'image_size': list(args.image_size) if args.image_size else None,
        'episodes': sources,
        'skipped': skipped,
    }, indent=1))
    print(f'{len(sources)} episodes written to {args.output} ({len(skipped)} skipped)')


if __name__ == '__main__':
    main()
