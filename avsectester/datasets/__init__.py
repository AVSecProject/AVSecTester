"""Stateless recorded-data and recipe providers; no simulator ownership.

"""

from .base import (
    ActorState, SensorDataReference, Dataset, FrameReference, FrameData, ScenarioDescriptor,
    ScenarioStart, Unavailable,
)

__all__ = [
    "ActorState", "SensorDataReference", "Dataset", "FrameReference", "FrameData",
    "ScenarioDescriptor", "ScenarioStart", "Unavailable",
]
