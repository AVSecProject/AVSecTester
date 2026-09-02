"""Environments + systems: dataset/simulation frame sources and the AV pipelines under test.

- ``MockEnv`` / ``MockSystem`` — simulator-free closed loop (no CARLA).
- ``CarlaEnv`` / ``CarlaSystem`` — closed-loop CARLA via avcarla (lazy import).
"""

from .mock import MockEnv, MockSystem

__all__ = ["MockEnv", "MockSystem"]
