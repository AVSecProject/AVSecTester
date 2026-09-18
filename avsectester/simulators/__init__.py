"""World-model simulators — ``WorldBackend`` implementations.

``CarlaBackend`` (real avcarla closed loop, in ``carla.py``) and ``NuRecBackend`` (in-process NuRec
reconstruction, in ``nurec.py``); both lazy-import their heavy deps so this package imports without a
CARLA stack. Per-simulator view adapters for :mod:`avsectester.simulators.viz` live alongside each
backend (e.g. ``carla.camera_view`` / ``carla.lidar_bev``).
"""

from .carla import CarlaBackend
from .nurec import (
    EgoPose,
    KinematicBicycle,
    NuRecBackend,
    NuRecRenderer,
    Renderer,
    StubRenderer,
    TrajectoryFollower,
)

__all__ = [
    "CarlaBackend",
    "EgoPose",
    "KinematicBicycle",
    "NuRecBackend",
    "NuRecRenderer",
    "Renderer",
    "StubRenderer",
    "TrajectoryFollower",
]
