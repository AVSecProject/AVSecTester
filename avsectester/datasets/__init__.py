"""Stateless recorded-data and recipe providers, without simulator ownership.
"""

from .base import (
    SensorDataReference, Dataset, FrameReference, FrameData, ScenarioDescriptor,
    ScenarioStart,
)


__all__ = [
    'SensorDataReference',
    'Dataset',
    'FrameReference',
    'FrameData',
    'ScenarioDescriptor',
    'ScenarioStart',
]
