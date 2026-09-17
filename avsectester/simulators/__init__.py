"""World-model simulators — WorldBackend implementations (CARLA lives in avsectester.scenario)."""

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
    "EgoPose",
    "KinematicBicycle",
    "NuRecBackend",
    "NuRecRenderer",
    "Renderer",
    "StubRenderer",
    "TrajectoryFollower",
]
