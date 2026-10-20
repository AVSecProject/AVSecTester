"""Supplied simulators — :class:`~avsectester.closed_loop.simulator.Simulator` configurations and
their live :class:`~avsectester.closed_loop.simulator.SimulatorRuntime` worlds.

``CarlaSimulator`` (real avcarla closed loop, in ``carla.py``) and ``NuRecSimulator`` (in-process NuRec
reconstruction, in ``nurec.py``); both lazy-import their heavy deps so this package imports without
CARLA dependencies. RecordedReplaySimulator replays recorded nuScenes frames. Each simulator runtime exposes
its calibrated cameras through sensor_geometry and sensor_view.
"""

from .recorded import RecordedReplaySimulator, RecordedReplaySimulatorRuntime
from .carla import CarlaSimulator, CarlaSimulatorRuntime
from .nurec import (
    EgoPose,
    KinematicBicycle,
    NuRecSimulator,
    NuRecSimulatorRuntime,
    NuRecRenderer,
    Renderer,
    StubRenderer,
    TrajectoryFollower,
)

__all__ = [
    "RecordedReplaySimulator",
    "RecordedReplaySimulatorRuntime",
    "CarlaSimulator",
    "CarlaSimulatorRuntime",
    "EgoPose",
    "KinematicBicycle",
    "NuRecSimulator",
    "NuRecSimulatorRuntime",
    "NuRecRenderer",
    "Renderer",
    "StubRenderer",
    "TrajectoryFollower",
]
