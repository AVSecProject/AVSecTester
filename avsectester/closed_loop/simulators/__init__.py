"""World-model simulators — :class:`~avsectester.closed_loop.simulator.Simulator` implementations.

``CarlaSimulator`` (real avcarla closed loop, in ``carla.py``) and ``NuRecSimulator`` (in-process NuRec
reconstruction, in ``nurec.py``); both lazy-import their heavy deps so this package imports without a
CARLA stack. Per-simulator view adapters for :mod:`avsectester.closed_loop.simulators.viz` live alongside each
simulator (e.g. ``carla.camera_view`` / ``carla.lidar_bev``).
"""

from .carla import CarlaSimulator
from .nurec import (
    EgoPose,
    KinematicBicycle,
    NuRecSimulator,
    NuRecRenderer,
    Renderer,
    StubRenderer,
    TrajectoryFollower,
)

__all__ = [
    "CarlaSimulator",
    "EgoPose",
    "KinematicBicycle",
    "NuRecSimulator",
    "NuRecRenderer",
    "Renderer",
    "StubRenderer",
    "TrajectoryFollower",
]
