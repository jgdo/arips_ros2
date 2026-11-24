"""Dataset layout and file formats for RGB-D keypoint datasets (no ROS dependency).

Dataset folder (flat):
  <bag>_<sec:010d>_<nsec:09d>_rgb.jpg      RGB image (or .png), named after its own header stamp
  <bag>_<sec:010d>_<nsec:09d>_depth.png    depth, uint16 millimetres, 0 = invalid
  <rgb file name>_<tag>.json               keypoints of one image for one annotation tag:
                                           {"0": [x, y], "1": [x, y], ...} in RGB pixel coords,
                                           invisible keypoints are left out ({} = none visible)
  config_<tag>.json                        {"tag": <tag>, "keypoints": {"0": "name0", ...}}

RGB and depth images belong together if they share the <bag> prefix and their stamps are
closest (see find_depth_for_rgb).
"""

import bisect
import json
import re
import struct
from pathlib import Path

import cv2
import numpy as np

RGB_EXTENSIONS = ('.jpg', '.jpeg', '.png')
_TAG_RE = re.compile(r'^[A-Za-z0-9_-]+$')
_FRAME_RE = re.compile(r'^(?P<bag>.+)_(?P<sec>\d{10})_(?P<nsec>\d{9})_(?P<kind>rgb|depth)\.(?P<ext>\w+)$')
CONFIG_PREFIX = 'config_'


# ---------------------------------------------------------------------------------------------
# file names
# ---------------------------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Make a bag name safe to use in file names."""
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', name).strip('_')
    return name or 'bag'


def is_valid_tag(tag: str) -> bool:
    return bool(_TAG_RE.match(tag))


def frame_name(bag: str, stamp_ns: int, kind: str, ext: str) -> str:
    """File name of one frame; kind is 'rgb' or 'depth', ext without dot."""
    sec, nsec = divmod(int(stamp_ns), 1_000_000_000)
    return f'{sanitize_name(bag)}_{sec:010d}_{nsec:09d}_{kind}.{ext.lstrip(".")}'


def parse_frame_name(path) -> tuple[str, int, str] | None:
    """(bag, stamp_ns, kind) of a frame file, or None if the name doesn't match."""
    m = _FRAME_RE.match(Path(path).name)
    if m is None:
        return None
    return m['bag'], int(m['sec']) * 1_000_000_000 + int(m['nsec']), m['kind']


def list_rgb_images(dataset_dir) -> list[Path]:
    """All RGB frames in the dataset, sorted by name."""
    out = []
    for p in Path(dataset_dir).iterdir():
        if p.suffix.lower() in RGB_EXTENSIONS and p.is_file():
            parsed = parse_frame_name(p)
            if parsed is not None and parsed[2] == 'rgb':
                out.append(p)
    return sorted(out, key=lambda p: p.name)


def list_depth_images(dataset_dir) -> list[Path]:
    return sorted((p for p in Path(dataset_dir).glob('*_depth.png')
                   if parse_frame_name(p) is not None), key=lambda p: p.name)


class DepthIndex:
    """Nearest-stamp lookup of depth images per bag."""

    def __init__(self, dataset_dir):
        self._by_bag: dict[str, tuple[list[int], list[Path]]] = {}
        for p in list_depth_images(dataset_dir):
            bag, stamp, _ = parse_frame_name(p)
            stamps, paths = self._by_bag.setdefault(bag, ([], []))
            stamps.append(stamp)
            paths.append(p)
        for bag, (stamps, paths) in self._by_bag.items():
            order = np.argsort(stamps)
            self._by_bag[bag] = ([stamps[i] for i in order], [paths[i] for i in order])

    def find(self, rgb_path, max_dt: float = 0.05) -> Path | None:
        parsed = parse_frame_name(rgb_path)
        if parsed is None or parsed[0] not in self._by_bag:
            return None
        bag, stamp, _ = parsed
        stamps, paths = self._by_bag[bag]
        i = nearest_index(stamps, stamp)
        if i is None or abs(stamps[i] - stamp) > max_dt * 1e9:
            return None
        return paths[i]


def find_depth_for_rgb(rgb_path, max_dt: float = 0.05) -> Path | None:
    """Depth image belonging to an RGB image (same bag, nearest stamp within max_dt seconds)."""
    return DepthIndex(Path(rgb_path).parent).find(rgb_path, max_dt)


def nearest_index(sorted_values, value) -> int | None:
    """Index of the element of sorted_values closest to value, None if empty."""
    if len(sorted_values) == 0:
        return None
    i = bisect.bisect_left(sorted_values, value)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(sorted_values)]
    return min(candidates, key=lambda j: abs(sorted_values[j] - value))


# ---------------------------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------------------------

def depth_m_to_mm(depth_m: np.ndarray) -> np.ndarray:
    """float metres (0/nan/inf = invalid) -> uint16 millimetres (0 = invalid)."""
    d = np.nan_to_num(np.asarray(depth_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mm = np.round(d * 1000.0)
    mm = np.where((mm > 0) & (mm <= 65535), mm, 0)
    return mm.astype(np.uint16)


def save_depth_m(path, depth_m: np.ndarray) -> None:
    if not cv2.imwrite(str(path), depth_m_to_mm(np.squeeze(depth_m))):
        raise OSError(f'could not write {path}')


def load_depth_m(path) -> np.ndarray:
    """Depth png -> float32 metres (H, W), 0 = invalid."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f'could not read {path}')
    return img.astype(np.float32) / 1000.0


def decode_compressed_depth(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage from compressed_depth_image_transport -> float32 metres."""
    raw = bytes(msg.data)
    encoding = msg.format.split(';')[0].strip()
    _, quant_a, quant_b = struct.unpack('<iff', raw[:12])
    img = cv2.imdecode(np.frombuffer(raw[12:], np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'depth image could not be decoded ({msg.format})')
    if encoding == '32FC1':
        # inverse depth quantization: v = quant_a / depth - quant_b, v = 0 marks invalid pixels
        v = img.astype(np.float32)
        with np.errstate(divide='ignore'):
            depth = np.where(v > 0, quant_a / (v - quant_b), 0.0)
    elif encoding == '16UC1':
        depth = img.astype(np.float32) / 1000.0
    else:
        raise ValueError(f'unsupported depth encoding {msg.format!r}')
    return np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _raw_image_array(msg) -> np.ndarray:
    dtype = {'16UC1': np.uint16, 'mono16': np.uint16, '32FC1': np.float32}.get(msg.encoding, np.uint8)
    dtype = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding, 1)
    row = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
    width_bytes = msg.width * channels * dtype.itemsize
    arr = np.ascontiguousarray(row[:, :width_bytes]).view(dtype)
    return arr.reshape(msg.height, msg.width, channels).squeeze(-1) if channels == 1 else \
        arr.reshape(msg.height, msg.width, channels)


def decode_raw_depth(msg) -> np.ndarray:
    """sensor_msgs/Image (16UC1 mm or 32FC1 m) -> float32 metres."""
    arr = _raw_image_array(msg)
    if msg.encoding in ('16UC1', 'mono16'):
        depth = arr.astype(np.float32) / 1000.0
    elif msg.encoding == '32FC1':
        depth = arr.astype(np.float32)
    else:
        raise ValueError(f'unsupported depth encoding {msg.encoding!r}')
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def decode_raw_rgb_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image -> BGR uint8 (for cv2.imwrite)."""
    arr = _raw_image_array(msg)
    conv = {'rgb8': cv2.COLOR_RGB2BGR, 'rgba8': cv2.COLOR_RGBA2BGR, 'bgra8': cv2.COLOR_BGRA2BGR,
            'mono8': cv2.COLOR_GRAY2BGR}
    if msg.encoding == 'bgr8':
        return arr
    if msg.encoding in conv:
        return cv2.cvtColor(arr, conv[msg.encoding])
    raise ValueError(f'unsupported rgb encoding {msg.encoding!r}')


# ---------------------------------------------------------------------------------------------
# annotations / config
# ---------------------------------------------------------------------------------------------

def annotation_path(rgb_path, tag: str) -> Path:
    """frame_123.jpg + door_handle -> frame_123.jpg_door_handle.json (next to the image)."""
    rgb_path = Path(rgb_path)
    return rgb_path.with_name(f'{rgb_path.name}_{tag}.json')


def has_annotation(rgb_path, tag: str) -> bool:
    return annotation_path(rgb_path, tag).is_file()


def load_annotation(rgb_path, tag: str) -> dict[int, tuple[float, float]] | None:
    """Keypoints {index: (x, y)} of an image, None if not annotated yet."""
    path = annotation_path(rgb_path, tag)
    if not path.is_file():
        return None
    with open(path) as f:
        data = json.load(f)
    return {int(k): (float(v[0]), float(v[1])) for k, v in data.items()}


def save_annotation(rgb_path, tag: str, keypoints: dict[int, tuple[float, float]]) -> Path:
    path = annotation_path(rgb_path, tag)
    # one line per keypoint: {"0": [x, y], ...}
    lines = [f'  "{k}": [{round(float(x), 1)}, {round(float(y), 1)}]' for k, (x, y) in sorted(keypoints.items())]
    text = '{\n' + ',\n'.join(lines) + '\n}\n' if lines else '{}\n'
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)
    return path


def config_path(dataset_dir, tag: str) -> Path:
    return Path(dataset_dir) / f'{CONFIG_PREFIX}{tag}.json'


def save_config(dataset_dir, tag: str, keypoint_names: list[str]) -> Path:
    if not is_valid_tag(tag):
        raise ValueError(f'invalid tag {tag!r}, allowed: letters, digits, _ and -')
    path = config_path(dataset_dir, tag)
    _write_json_atomic(path, {'tag': tag, 'keypoints': {str(i): n for i, n in enumerate(keypoint_names)}})
    return path


def load_config(path) -> tuple[str, list[str]]:
    """(tag, keypoint names ordered by index)."""
    with open(path) as f:
        data = json.load(f)
    kps = data['keypoints']
    return data['tag'], [kps[k] for k in sorted(kps, key=int)]


def list_configs(dataset_dir) -> list[tuple[str, list[str]]]:
    """All (tag, keypoint names) configs of a dataset, sorted by tag."""
    out = []
    for p in sorted(Path(dataset_dir).glob(f'{CONFIG_PREFIX}*.json')):
        try:
            out.append(load_config(p))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def _write_json_atomic(path: Path, data) -> None:
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------------------------

SPLIT_PREFIX = 'split_'


def split_path(dataset_dir, tag: str, name: str | None, kind: str) -> Path:
    """split_<tag>[_<name>]_<kind>.json, kind e.g. 'train', 'test' or 'all'."""
    middle = f'{tag}_{name}' if name else tag
    return Path(dataset_dir) / f'{SPLIT_PREFIX}{middle}_{kind}.json'


def save_split(path, image_names: list[str]) -> Path:
    path = Path(path)
    _write_json_atomic(path, {'images': sorted(image_names)})
    return path


def load_split(path) -> list[Path]:
    """Image paths of a split file (names are relative to the folder of the split file)."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return [path.parent / name for name in data['images']]


def list_splits(dataset_dir) -> list[Path]:
    return sorted(Path(dataset_dir).glob(f'{SPLIT_PREFIX}*.json'))
