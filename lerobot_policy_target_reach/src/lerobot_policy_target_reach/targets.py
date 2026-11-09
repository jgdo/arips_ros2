"""Reads the visualizer's targets.json (see lerobot-dataset-visualizer/backend/target_annotations.py).

Training rule: an episode whose targets contain ``tag`` at frame ``t`` contributes all of its frames,
each labelled with the pose at ``t``. Episodes without the tag are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_tag_frames(annotation_path: str | Path, tag: str) -> dict[int, int]:
    """``{episode_index: target frame}`` for every episode that has ``tag``."""
    data = json.loads(Path(annotation_path).read_text())
    if tag not in data.get("tags", []):
        raise ValueError(f"tag {tag!r} not in {annotation_path} (tags: {data.get('tags', [])})")
    frames: dict[int, int] = {}
    for episode, entry in data.get("episodes", {}).items():
        hits = [t["frame"] for t in entry.get("targets", []) if t["tag"] == tag]
        if len(hits) > 1:
            raise ValueError(f"episode {episode} uses tag {tag!r} {len(hits)} times")
        if hits:
            frames[int(episode)] = int(hits[0])
    if not frames:
        raise ValueError(f"no episode in {annotation_path} has tag {tag!r}")
    return frames
