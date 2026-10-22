"""Closed-loop data types: the messages between the simulator and the AVStack, and world state.

The simulator<->AVStack contract is exactly two messages — **sensor data down, control up**:

* the simulator renders an :class:`Observation` (sensor data + the ego's self-knowledge) and sends it;
* the AV box replies with a :class:`Control` command.

The simulator keeps a full *world state* internally (ground truth, used for rendering + scoring) but
never sends it to the box — the box sees only its observations. Stage hook handlers can modify
model-visible messages or explicitly access native world operations through their context.

This module is **pure data** — no avstack / avcarla / carla imports — so the interface does not
depend on any simulator or assume a modular vs end-to-end AVStack. Sensor and state payloads retain
adapter-specific types, so transport across a process boundary requires appropriate codecs.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from avsectester.datasets.base import ActorState

import numpy as np


@dataclass
class Control:
    """The AV box's output — a vehicle control *command* (the decision).

    The *physical* response to the command (vehicle dynamics) is the simulator's shared job, so the
    simulator must support the command representation produced by the AVStack. Actuator commands use
    ``throttle``, ``steer`` and ``brake``. Trajectory commands use ``trajectory`` in the selected
    adapter's coordinates and time convention. Built-in simulators reject fields they cannot consume.
    """

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    trajectory: Any = None


@dataclass
class Observation:
    """What the simulator sends the AV box each frame — and the only thing it sends.

    ``sensor_data`` is derived by the simulator from its (hidden) world state, the ego ``vehicle_state``
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

        Operates on this observation only. A simulator or hook-dispatcher copy isolates it from
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
    """An independent snapshot of the simulator's actual ego state and other actors' states.

    ``actors`` maps stable actor IDs, other than the ego, to ``ActorState`` records (canonical
    world pose, velocity and category). Nested state objects belong to this snapshot. Editing
    them does not actuate the world; simulator-runtime actor methods do. ``ego`` is canonical
    physical truth. ``vehicle_state`` is its native SDK representation for adapter callbacks;
    neither is the potentially perturbed model-visible Observation.vehicle_state.
    ``traffic_lights`` maps semantic-map control IDs to red, yellow, green, off or unknown.
    Simulators without live signal-state evidence leave the mapping empty.
    ``objects`` holds framework-inserted world objects separately from native actors.
    """

    t: float
    frame: int
    vehicle_state: Any = None
    ego_speed: float = 0.0
    actors: Mapping[str, Any] = field(default_factory=dict)
    ego: ActorState | None = None
    traffic_lights: Mapping[str, str] = field(default_factory=dict)
    objects: Mapping[str, ActorState] = field(default_factory=dict)


@dataclass
class StateEstimate:
    """Model-visible ego state and speed produced by a localization adapter."""

    vehicle_state: Any = None
    ego_speed: float = 0.0


@dataclass(frozen=True)
class RecordedOutput:
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


@dataclass(frozen=True)
class Detection:
    """One 3D detection in the capture-time true ego frame (x forward, y left, z up).

    ``position`` is the cuboid centre in metres, ``extent`` its full length, width and
    height in metres, and ``yaw`` its heading in radians about ego z. Sensor mounting
    extrinsics define this frame, independently of the driving system's localization.
    Its origin is the true ego actor origin, not the ego cuboid centre. Evaluation uses
    ``StepRecord.physical_pre.ego`` to transform ground-truth actors into this frame.
    """

    position: tuple[float, float, float]
    extent: tuple[float, float, float]
    yaw: float
    category: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        import math

        for name in ("position", "extent"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(float(v)) for v in value):
                raise ValueError(f"Detection {name} must contain three finite values")
            if name == "extent" and any(v < 0 for v in value):
                raise ValueError("Detection extent must be nonnegative")
            object.__setattr__(self, name, tuple(float(v) for v in value))
        object.__setattr__(self, "yaw", float(self.yaw))
        if self.score is not None:
            object.__setattr__(self, "score", float(self.score))
        if self.category is not None and not isinstance(self.category, str):
            raise TypeError("Detection category must be a string or None")
        if not math.isfinite(self.yaw):
            raise ValueError("Detection yaw must be finite")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("Detection score must be finite")


def validate_command_modes(command: Control, modes: frozenset[str] | None) -> None:
    """Validate the fields consumed by a simulator or independently called dynamics model."""
    if not isinstance(command, Control):
        raise TypeError(f"Expected Control, got {type(command).__name__}")
    if modes is None:
        return
    if command.trajectory is not None and "trajectory" not in modes:
        raise ValueError("Simulator does not support trajectory commands")
    active = [name for name in ("throttle", "brake", "steer") if getattr(command, name) != 0.0]
    if active and "actuator" not in modes:
        raise ValueError(f"Simulator does not support actuator command fields: {active}")


def read_camera_rgb(observation: Observation, camera: str | None = None) -> tuple[str, np.ndarray] | None:
    """``(sensor_id, (H,W,3) uint8 RGB)`` for a camera payload, or None. Duck-typed over the canonical
    ndarray payload (NuRec/dataset) and an avstack ``ImageData`` (``.rgb_image``) — the read side of the
    sensor-plane seam, so transforms need no simulator-specific image wrapper."""
    data = observation.sensor_data
    if not data:
        return None
    if camera is not None and camera not in data:
        return None
    key = camera if camera is not None else next(iter(data))
    payload = data[key]
    rgb = getattr(payload, "rgb_image", None)
    if rgb is None and isinstance(payload, np.ndarray) and payload.ndim == 3 and payload.shape[2] >= 3:
        rgb = payload
    if rgb is None:
        return None
    return key, np.asarray(rgb)[:, :, :3].astype(np.uint8)


def write_camera_rgb(observation: Observation, key: str, rgb: np.ndarray) -> Observation:
    """Return a copy of ``observation`` with sensor ``key``'s image replaced by ``rgb``. An ndarray
    payload is replaced; an ``ImageData``-like payload is shallow-copied with only its pixel buffer
    swapped (the write side of the seam, shared by attacks and augmentations)."""
    payload = observation.sensor_data[key]
    new_rgb = np.asarray(rgb).astype(np.uint8)
    if isinstance(payload, np.ndarray):
        new_payload: Any = new_rgb
    else:
        new_payload = copy.copy(payload)
        new_payload.data = new_rgb
    new_data = dict(observation.sensor_data)
    new_data[key] = new_payload
    return replace(observation, sensor_data=new_data)


def write_sensor_data(observation: Observation, sensor: str, data: Any) -> Observation:
    """Return an observation copy with one sensor's data replaced.

    Arrays retain their representation and dtype regardless of shape. Native sensor wrappers
    are shallow-copied with only ``data`` replaced, preserving their calibration, timestamps
    and native options. Camera-specific writes use ``write_camera_rgb`` instead.
    """
    payload = observation.sensor_data[sensor]
    if isinstance(payload, np.ndarray) or not hasattr(payload, "data"):
        replacement = data
    else:
        replacement = copy.copy(payload)
        replacement.data = data
    sensor_data = dict(observation.sensor_data)
    sensor_data[sensor] = replacement
    return replace(observation, sensor_data=sensor_data)
