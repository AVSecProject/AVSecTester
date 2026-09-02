"""Frame — one time-step of AV data, common to dataset replay and simulation.

A ``Frame`` is the single unit that flows through the framework. A dataset yields recorded
frames; a simulator yields frames from its current state. Both carry the same fields, so
attacks, systems, and metrics never need to know which source produced them. Sensor payloads
follow avstack conventions (``LidarData`` / ``ImageData`` / ``ObjectState`` ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Frame:
    index: int = 0
    timestamp: float = 0.0
    #: sensor payloads by channel, e.g. {"lidar": LidarData, "camera": ImageData, "gps": ...}
    sensors: dict[str, Any] = field(default_factory=dict)
    #: ego state (pose + velocity), avstack ObjectState
    ego: Any = None
    #: per-sensor calibration, by channel
    calibration: dict[str, Any] = field(default_factory=dict)
    #: ground-truth objects (for metrics); may be None on real data
    ground_truth: Any = None
    #: anything else: weather, scenario tags, route, ...
    meta: dict[str, Any] = field(default_factory=dict)
