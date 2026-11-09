"""LeRobot policy plugin ``target_reach``; imported by lerobot's register_third_party_plugins()."""

from . import sampler_target_reach
from .configuration_target_reach import TargetReachConfig

sampler_target_reach.install()

__all__ = ["TargetReachConfig"]
