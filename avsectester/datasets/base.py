"""Dataset records. A frame reference identifies a dataset position, never a simulator handle."""

from __future__ import annotations

import math
from numbers import Real
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from avsectester.geometry.poses import ActorPose


def nonnegative(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class FrameReference:
    """One frame of a scenario's timeline: its position, native frame ID and scenario time (s)."""

    scenario_id: str
    position: int
    frame_id: str | int
    timestamp: float

    def __post_init__(self) -> None:
        nonnegative(self.position, "position")
        if not self.scenario_id or not math.isfinite(self.timestamp):
            raise ValueError("A frame needs a scenario identity and a finite scenario timestamp")


@dataclass(frozen=True)
class ScenarioDescriptor:
    id: str
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Unavailable:
    """The dataset has no frame data for this frame, for example a CARLA recipe."""

    reason: str


@dataclass(frozen=True)
class SensorDataReference:
    """Location of one recorded sensor payload; adapters decide how it is decoded."""

    uri: str
    sensor: str
    timestamp: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActorState:
    """World-from-actor SE(3), world velocity, stable identity in canonical right-handed axes.

    ``category`` is the actor's class, such as ``vehicle`` or a dataset-native category,
    when known. ``extent`` is the full (length, width, height) in metres, not half extents.
    ``box_center`` is the bounding-box centre in actor coordinates; ``pose`` remains the
    actor/rig origin, so world box centres are ``pose @ [*box_center, 1]``.
    ``box_rotation`` is the local right-handed (roll, pitch, yaw) in degrees.
    """

    pose: Any
    velocity: tuple[float, float, float] | None = None
    category: str | None = None
    extent: tuple[float, float, float] | None = None
    box_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.box_center) != 3 or any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in self.box_center
        ):
            raise ValueError("Actor box_center must be three finite coordinates")
        object.__setattr__(self, "box_center", tuple(float(value) for value in self.box_center))
        from avsectester.geometry.poses import validated_vector

        rotation = validated_vector(self.box_rotation, "box_rotation")
        object.__setattr__(self, "box_rotation", tuple(float(value) for value in rotation))
        if self.extent is not None:
            if (len(self.extent) != 3 or any(
                isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0 for value in self.extent
            )):
                raise ValueError("Actor extent must be three finite nonnegative dimensions")
            object.__setattr__(self, "extent", tuple(float(value) for value in self.extent))


    def box_transform(self) -> np.ndarray:
        """World-from-box-centre transform, including local roll/pitch/yaw degree rotation."""
        import numpy as np
        from avsectester.geometry.poses import rotation_matrix

        local = np.eye(4)
        local[:3, :3] = rotation_matrix(self.box_rotation)
        local[:3, 3] = self.box_center
        return np.asarray(self.pose) @ local

    def box_pose(self) -> ActorPose:
        """Return the world box-centre pose, preserving unknown dimensions as None."""
        from avsectester.geometry.poses import ActorPose

        return ActorPose(self.box_transform(), self.extent)


@dataclass(frozen=True)
class FrameData:
    """World-frame actor states and recorded sensor references at one dataset position.

    ``ego`` holds the recorded ego state; ``actors`` excludes ego and uses stable actor IDs.
    Sensor-reference metadata carries calibration needed by recorded replay.
    """

    frame: FrameReference
    actors: Mapping[str, ActorState] = field(default_factory=dict)
    sensor_data: Mapping[str, SensorDataReference] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    ego: ActorState | None = None


@dataclass(frozen=True)
class ScenarioStart:
    """What a simulator needs to start a scenario at a frame; no attack or selection metadata.

    ``kind`` is the dataset class name (``NuRecDataset``, ``NuScenesDataset`` or ``CarlaDataset``);
    ``payload`` is that dataset's start data and is copied on construction.
    """

    kind: str
    scenario_id: str
    frame: FrameReference
    payload: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported ScenarioStart schema: {self.schema_version}")
        if self.scenario_id != self.frame.scenario_id:
            raise ValueError("The start frame must belong to the scenario")
        object.__setattr__(self, "payload", deepcopy(dict(self.payload)))


class Dataset(ABC):
    """Read scenarios with identical behavior before and after simulator execution.

    Datasets may cache immutable dataset-file reads. Frame data generated by a simulator belongs
    to experiment preparation, not this object. All returned mutable records are detached copies.
    """

    @abstractmethod
    def scenarios(self) -> Iterator[ScenarioDescriptor]: ...

    @abstractmethod
    def frames(self, scenario_id: str) -> Sequence[FrameReference]: ...

    @abstractmethod
    def frame_data(self, scenario_id: str, frame: FrameReference) -> FrameData | Unavailable: ...

    @abstractmethod
    def native(self, scenario_id: str) -> Any: ...

    @abstractmethod
    def scenario_start(self, scenario_id: str, frame: FrameReference) -> ScenarioStart: ...

    def checked_frame(self, scenario_id: str, frame: FrameReference) -> FrameReference:
        frames = self.frames(scenario_id)
        if (frame.scenario_id != scenario_id or frame.position >= len(frames)
                or frame != frames[frame.position]):
            raise ValueError("Frame is outside or does not match the scenario's frames")
        return frame

    def decision_window(
        self, scenario_id: str, start: FrameReference, decision_steps: int,
    ) -> tuple[FrameReference, ...]:
        """Require N+1 actual frames (the last is the terminal state); never pad or truncate."""
        nonnegative(decision_steps, "decision_steps")
        self.checked_frame(scenario_id, start)
        frames = self.frames(scenario_id)
        stop = start.position + decision_steps
        if stop >= len(frames):
            raise ValueError("Decision range exceeds the scenario's terminal frame")
        return tuple(frames[start.position:stop + 1])
