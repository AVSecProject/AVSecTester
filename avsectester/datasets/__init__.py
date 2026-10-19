"""Stateless recorded-data and recipe providers; no simulator ownership.

The old ``scenarios.datasets`` imports remain compatibility adapters during migration.
"""

from .base import (
    ActorState, SensorDataReference, Dataset, FrameReference, FrameData, ScenarioDescriptor,
    ScenarioStart, Unavailable,
)

__all__ = [
    "ActorState", "SensorDataReference", "Dataset", "FrameReference", "FrameData",
    "ScenarioDescriptor", "ScenarioStart", "Unavailable",
]
