"""Source contracts. References describe source positions, never simulator handles."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from avsectester.scenarios.scene import SceneGT


def nonnegative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class FrameRef:
    scenario_id: str
    position: int
    frame_id: str | int
    timestamp: float

    def __post_init__(self) -> None:
        nonnegative(self.position, "position")
        if not self.scenario_id or not math.isfinite(self.timestamp):
            raise ValueError("A frame needs a scenario identity and a finite source timestamp")


@dataclass(frozen=True)
class ScenarioDescriptor:
    id: str
    kind: str
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Unavailable:
    reason: str


@dataclass(frozen=True)
class DataReference:
    """Read-only payload location; adapters decide how the referenced data is decoded."""

    uri: str
    sensor: str
    timestamp: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActorState:
    """World-from-actor SE(3), world velocity, stable identity in canonical right-handed axes."""

    pose: Any
    velocity: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ReferenceFrame:
    frame: FrameRef
    scene: SceneGT | None
    actors: Mapping[str, ActorState] = field(default_factory=dict)
    sensors: Mapping[str, DataReference] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceData:
    """Detached source payload for reset, containing no attack or selection metadata."""

    kind: str
    scenario_id: str
    start: FrameRef
    payload: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported SourceData schema: {self.schema_version}")
        if self.scenario_id != self.start.scenario_id:
            raise ValueError("Source and start must identify the same scenario")
        object.__setattr__(self, "payload", deepcopy(dict(self.payload)))


class Dataset(ABC):
    """Read source data with identical behavior before and after simulator execution.

    Providers may cache immutable source-file reads. Generated reference sequences belong to
    experiment preparation, not this object. All returned mutable records are detached copies.
    """

    @abstractmethod
    def scenarios(self) -> Iterator[ScenarioDescriptor]: ...

    @abstractmethod
    def frames(self, scenario_id: str) -> Sequence[FrameRef]: ...

    @abstractmethod
    def reference(self, scenario_id: str, frame: FrameRef) -> ReferenceFrame | Unavailable: ...

    def sensor_data(
        self, scenario_id: str, frame: FrameRef, sensor: str,
    ) -> DataReference | None:
        reference = self.reference(scenario_id, frame)
        if isinstance(reference, Unavailable):
            return None
        return deepcopy(reference.sensors.get(sensor))

    @abstractmethod
    def native(self, scenario_id: str) -> Any: ...

    @abstractmethod
    def source_data(self, scenario_id: str, start: FrameRef) -> SourceData: ...

    def checked_frame(self, scenario_id: str, frame: FrameRef) -> FrameRef:
        frames = self.frames(scenario_id)
        if (frame.scenario_id != scenario_id or frame.position >= len(frames)
                or frame != frames[frame.position]):
            raise ValueError("Frame is outside or does not match the source sequence")
        return frame

    def decision_window(
        self, scenario_id: str, start: FrameRef, decision_steps: int,
    ) -> tuple[FrameRef, ...]:
        """Require N+1 actual references; never pad or truncate."""
        nonnegative(decision_steps, "decision_steps")
        self.checked_frame(scenario_id, start)
        frames = self.frames(scenario_id)
        stop = start.position + decision_steps
        if stop >= len(frames):
            raise ValueError("Decision range exceeds terminal reference support")
        return tuple(frames[start.position:stop + 1])
