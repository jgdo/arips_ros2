"""Annotation config presets: yaml files in the package's config/presets folder.

Preset format:
  tag: door_handle            # default tag for the new annotation
  description: ...            # optional, shown in the annotation tool
  keypoints: [name0, name1]   # keypoint names in index order
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import dataset_io as dio

PACKAGE = 'arips_keypoint_detection'


@dataclass
class Preset:
    name: str            # file name without .yaml
    tag: str
    keypoints: list[str]
    description: str = ''
    path: Path | None = None


def presets_dir() -> Path:
    """Installed share/<pkg>/config/presets, or the source folder when running from the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        d = Path(get_package_share_directory(PACKAGE)) / 'config' / 'presets'
        if d.is_dir():
            return d
    except Exception:  # noqa: BLE001 - package not installed / ament index not available
        pass
    return Path(__file__).resolve().parents[1] / 'config' / 'presets'


def load_preset(path) -> Preset:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    tag = str(data.get('tag', path.stem))
    keypoints = [str(k) for k in data.get('keypoints') or []]
    if not dio.is_valid_tag(tag):
        raise ValueError(f'{path}: invalid tag {tag!r}')
    if not keypoints:
        raise ValueError(f'{path}: no keypoints')
    if len(set(keypoints)) != len(keypoints):
        raise ValueError(f'{path}: duplicate keypoint names')
    return Preset(path.stem, tag, keypoints, str(data.get('description', '')), path)


def list_presets(directory=None) -> list[Preset]:
    """All valid presets, sorted by name; invalid files are skipped with a warning."""
    directory = Path(directory) if directory else presets_dir()
    presets = []
    for p in sorted(directory.glob('*.yaml')) if directory.is_dir() else []:
        try:
            presets.append(load_preset(p))
        except (ValueError, yaml.YAMLError, OSError) as e:
            print(f'warning: skipping preset {p}: {e}')
    return presets
