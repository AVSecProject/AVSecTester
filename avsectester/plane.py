"""The data plane: the messages that cross between the simulator and the AV stack.

The sim<->stack contract is exactly two messages — **sensor data down, control up**:

* the backend renders an :class:`Observation` (sensor data + the ego's self-knowledge) and sends it;
* the AV box replies with a :class:`Control` command.

The backend keeps a full *world state* internally (ground truth, used for rendering + scoring) but
never sends it to the box — the box sees only its observations. Runtime stage handlers can modify
model-visible messages or explicitly access native world operations through their context.

This module is **pure data** — no avstack / avcarla / carla imports — so the interface does not
depend on any backend or assume a modular vs end-to-end stack. Sensor and state payloads retain
adapter-specific types, so transport across a process boundary requires appropriate codecs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Control:
    """The AV box's output — a vehicle control *command* (the decision).

    The *physical* response to the command (vehicle dynamics) is the backend's shared job, so the
    same command drives the same car regardless of which AV stack produced it. Actuator-level by
    default; a stack that plans a path can carry it in ``trajectory`` for a backend that follows one.
    """

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    trajectory: Any = None


@dataclass
class Observation:
    """What the backend sends the AV box each frame — and the only thing it sends.

    ``sensor_data`` is derived by the backend from its (hidden) world state, the ego ``vehicle_state``
    and ``calibration``. ``calibration`` maps each sensor to the vehicle frame (intrinsics + extrinsic)
    — the bridge from world/vehicle coordinates into each sensor. No world/ground-truth state is
    included: the box sees only what a real vehicle senses about the scene and itself.
    """

    t: float
    frame: int
    sensor_data: dict[str, Any] = field(default_factory=dict)  # sensor_id -> sensor payload
    calibration: dict[str, Any] = field(default_factory=dict)  # sensor_id -> intrinsics + extrinsic
    vehicle_state: Any = None  # ego pose/velocity (what localization reports)
    ego_speed: float = 0.0  # reported speed (m/s), available to the driving model


@dataclass(frozen=True)
class WorldSnapshot:
    """An independent snapshot of the backend's actual ego state.

    Nested state objects belong to this snapshot. Editing them does not actuate the world.
    Native backend operations are the explicit interface for changing physical state.
    """

    t: float
    frame: int
    vehicle_state: Any = None
    ego_speed: float = 0.0


@dataclass
class StateEstimate:
    """Model-visible ego state and speed produced by a localization adapter."""

    vehicle_state: Any = None
    ego_speed: float = 0.0


@dataclass
class FrameRecord:
    frame: int = 0
    t: float = 0.0
    n_detections: int = 0  # optional stack telemetry (0 for stacks with no detection stage)
    speed: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


@dataclass
class Trace:
    """The driving record of one run — the observable a security test scores."""

    records: list[FrameRecord] = field(default_factory=list)
    replay_scenario: dict | None = None

    @property
    def final_speed(self) -> float:
        return self.records[-1].speed if self.records else 0.0

    @property
    def peak_speed(self) -> float:
        """Fastest the ego went during the run — used to tell whether it ever really drove."""
        return max((r.speed for r in self.records), default=0.0)

    @property
    def braking_frames(self) -> int:
        return sum(r.throttle == 0.0 and r.brake > 0.0 for r in self.records)

    @property
    def mean_detections(self) -> float:
        return sum(r.n_detections for r in self.records) / max(len(self.records), 1)
