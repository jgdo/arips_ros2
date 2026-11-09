"""Filtered sampler: lerobot_train only ever draws frames that have a target.

``lerobot_train`` always builds ``EpisodeAwareSampler`` from its module globals. Importing this
plugin replaces that name with a factory that returns ``TargetReachSampler`` while a
``TargetReachConfig`` is active in the process, and the stock sampler otherwise (ACT, SmolVLA, …
are unaffected).
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from lerobot.datasets import EpisodeAwareSampler

from .targets import load_tag_frames

logger = logging.getLogger(__name__)

# Set by TargetReachConfig.__post_init__; read by the factory below.
ACTIVE_CONFIG = None
PATCHED_MODULES: list[str] = []


class TargetReachSampler(EpisodeAwareSampler):
    """``EpisodeAwareSampler`` over every frame of each episode that has the tag.

    Shuffling, ``(seed, epoch)`` determinism and ``state_dict`` resume come from the base class;
    only the per-episode frame ranges differ.
    """

    def __init__(self, annotation_path: str, target_tag: str, dataset_from_indices, dataset_to_indices,
                 episode_indices_to_use=None, drop_n_first_frames: int = 0, drop_n_last_frames: int = 0,
                 **kwargs):
        super().__init__(dataset_from_indices, dataset_to_indices, episode_indices_to_use, 0, 0, **kwargs)
        frames = load_tag_frames(annotation_path, target_tag)
        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        candidates = (
            range(len(from_indices)) if episode_indices_to_use is None else sorted(set(episode_indices_to_use))
        )

        starts, lengths, skipped = [], [], 0
        for ep in candidates:
            t = frames.get(int(ep))
            if t is None:
                skipped += 1
                continue
            length = to_indices[ep] - from_indices[ep]
            if t >= length:
                raise ValueError(f"episode {ep}: target frame {t} outside episode (length {length})")
            starts.append(from_indices[ep])
            lengths.append(length)
        if not starts:
            raise ValueError(f"no episode of this dataset has tag {target_tag!r}")

        self._starts = np.asarray(starts, dtype=np.int64)
        self._cum_lengths = np.cumsum(np.asarray(lengths, dtype=np.int64))
        self._num_frames = int(self._cum_lengths[-1])
        total = int(sum(to_indices[ep] - from_indices[ep] for ep in candidates))
        logger.info(
            "target_reach: tag %r, %d episodes used, %d skipped (no tag), usable frames %d/%d (%.1f%%)",
            target_tag, len(starts), skipped, self._num_frames, total, 100 * self._num_frames / max(total, 1),
        )


def sampler_factory(*args, **kwargs):
    cfg = ACTIVE_CONFIG
    if cfg is None:
        return EpisodeAwareSampler(*args, **kwargs)
    return TargetReachSampler(cfg.annotation_path, cfg.target_tag, *args, **kwargs)


def install() -> None:
    """Point ``lerobot_train``'s ``EpisodeAwareSampler`` at the factory (also when run as ``-m``)."""
    for name in ("lerobot.scripts.lerobot_train", "__main__"):
        module = sys.modules.get(name)
        if module is None:
            continue
        if name == "__main__" and getattr(module.__spec__, "name", None) != "lerobot.scripts.lerobot_train":
            continue
        if getattr(module, "EpisodeAwareSampler", None) in (EpisodeAwareSampler, sampler_factory):
            module.EpisodeAwareSampler = sampler_factory
            PATCHED_MODULES.append(name)
