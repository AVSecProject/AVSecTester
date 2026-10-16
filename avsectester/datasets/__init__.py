"""Stateless recorded-data and recipe providers; no simulator ownership.

The old ``scenarios.datasets`` imports remain compatibility adapters during migration.
"""

from .base import (
    ActorState, DataReference, Dataset, FrameRef, ReferenceFrame, ScenarioDescriptor,
    SourceData, Unavailable,
)

__all__ = [
    "ActorState", "DataReference", "Dataset", "FrameRef", "ReferenceFrame",
    "ScenarioDescriptor", "SourceData", "Unavailable",
]
