"""Environments + systems: dataset/simulation frame sources and the AV pipelines under test.

- ``MockEnv`` / ``MockSystem`` — simulator-free closed loop (no CARLA).
- ``CarlaEnv`` / ``CarlaSystem`` — closed-loop CARLA via avcarla (all carla/avcarla imports are
  lazy, so importing this package never requires CARLA).
"""

from .carla import CarlaEnv, CarlaSystem
from .mock import MockEnv, MockSystem

__all__ = ["CarlaEnv", "CarlaSystem", "MockEnv", "MockSystem"]
