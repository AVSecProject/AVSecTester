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
    backend must support the command representation produced by the stack. Actuator commands use
    ``throttle``, ``steer`` and ``brake``. Trajectory commands use ``trajectory`` in the selected
    adapter's coordinates and time convention. Built-in backends reject fields they cannot consume.
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


    def set_calibration(self, sensor_id: str, calibration: Any) -> None:
        """Replace model-visible calibration and its native payload reference together.

        Operates on this observation only. A backend or runtime copy isolates it from
        physical sensor configuration. Native calibration types and units are preserved.
        """
        if sensor_id not in self.sensor_data:
            raise KeyError(f"Unknown observation sensor: {sensor_id!r}")
        payload = self.sensor_data[sensor_id]
        if hasattr(payload, "calibration"):
            payload.calibration = calibration
        self.calibration[sensor_id] = calibration


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


@dataclass(frozen=True)
class InterventionOutput:
    """A named output captured by an attack, defense or localization callback.

    ``step`` is the driving-decision index. ``frame`` and ``t`` describe the physical world
    at the callback, so a ``step.end`` output belongs to the resulting frame. ``value`` is
    copied when recorded and does not share mutable state with the originating plugin.
    """

    name: str
    value: Any
    step: int
    stage: str
    frame: int
    t: float
    sensor: str | None = None


@dataclass
class Trace:
    """The driving record of one run — the observable a security test scores."""

    records: list[FrameRecord] = field(default_factory=list)
    replay_scenario: dict | None = None
    outputs: list[InterventionOutput] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

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
