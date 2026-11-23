from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_doorhandle import DoorhandleConfig


def make_doorhandle_pre_post_processors(
    config: DoorhandleConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # The normalization mapping is IDENTITY everywhere: the policy normalizes its inputs itself with
    # the ranges in the config, so these pipelines only add the batch dim and move to the device.
    return make_default_pre_post_processors(config, dataset_stats, normalizer_device=config.device)
