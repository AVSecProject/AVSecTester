"""Data types exchanged between framework components.

These include driving-system observations, commands, state estimates and detections,
world and actor states, reasons evidence is unavailable, attack image target regions,
and outputs recorded by callbacks.
"""

from __future__ import annotations

import copy
import math
from numbers import Real
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from avsectester.validation import timestamp_to_microseconds

from avsectester.geometry.poses import (
    rotation_matrix, validated_vector,
)


@dataclass(frozen=True)
class PointCloud:
    """One LiDAR scan in the sensor frame at the end of the scan.

    ``points`` is an (N, 3) float32 array in meters, with x forward, y left and z up.
    ``intensity`` is an (N,) float32 array in [0, 1], or None when the source has no
    intensity. Array fields are not validated at construction so recording can restore
    payload references without loading arrays. Producers validate shapes, dtypes, finite
    values and matching lengths before constructing a cloud.
    """

    points: np.ndarray
    intensity: np.ndarray | None = None


@dataclass(frozen=True)
class ActorState:
    """An actor's world-from-origin pose, world velocity and optional bounding-box geometry.

    ``category`` is the actor's class, such as ``vehicle`` or a dataset-native category,
    when known. ``extent`` is the full (length, width, height) in meters, not half extents.
    ``box_center`` is the bounding-box center in actor coordinates. ``pose`` remains the
    actor/rig origin, so world box centers are ``pose @ [*box_center, 1]``.
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
        """World-from-box-center transform, including local roll/pitch/yaw degree rotation."""
        local = np.eye(4)
        local[:3, :3] = rotation_matrix(self.box_rotation)
        local[:3, 3] = self.box_center
        return np.asarray(self.pose) @ local


@dataclass(frozen=True)
class Unavailable:
    """A dataset cannot supply the requested evidence, with an explicit reason."""

    reason: str


@dataclass
class TargetSpec:
    """The object to attack, with an image-space xyxy ``box`` and an optional class label."""

    box: tuple[float, float, float, float]
    label: str | None = None


@dataclass
class Control:
    """The driving system's output — a vehicle control *command* (the decision).

    The *physical* response to the command (vehicle dynamics) is the simulator's shared job, so the
    simulator must support the command representation produced by the AVStack. Actuator commands use
    ``throttle``, ``steer`` and ``brake``. Trajectory commands use ``trajectory`` in the selected
    adapter's coordinates and time convention. Built-in simulators reject unsupported fields.
    """

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    trajectory: Any = None


@dataclass(frozen=True)
class TimedStateEstimate:
    """One localization result at its scheduler time and native observation time."""

    time_us: int
    t: float
    vehicle_state: Any
    ego_speed: float


@dataclass(frozen=True)
class DeliveredAcquisition:
    """One newly delivered frame and its estimated pose at frame end.

    Data and calibration are borrowed references. A missing capture ID denotes a direct
    adapter input rather than a scheduler acquisition.
    """

    time_us: int
    t: float
    frame: int
    data: Any
    calibration: Any
    state: TimedStateEstimate
    capture_id: int | None = None
    frame_start_us: int | None = None


@dataclass
class Observation:
    """One driving decision's model-visible inputs and newly delivered frames.

    ``sensor_data`` is derived by the simulator from its hidden world state, ego state
    and ``calibration``. Calibration contains each sensor's intrinsics and extrinsics
    relative to the vehicle frame. No separate world/ground-truth state is included.
    The driving system sees only what a real vehicle senses about the scene and itself.
    ``route`` is model-visible navigation intent in ego coordinates (X forward, Y left),
    an (N, 3) array in meters with zero Z, or None. Hooks and attacks may modify it.
    ``deliveries`` maps a sensor to an empty tuple when every frame due since the previous
    decision was dropped.
    """

    t: float
    frame: int
    sensor_data: dict[str, Any] = field(default_factory=dict)  # sensor_id -> sensor payload
    calibration: dict[str, Any] = field(default_factory=dict)  # sensor_id -> intrinsics + extrinsic
    vehicle_state: Any = None  # ego pose/velocity (what localization reports)
    ego_speed: float = 0.0  # reported speed (m/s), available to the driving model
    route: np.ndarray | None = None
    deliveries: dict[str, tuple[DeliveredAcquisition, ...]] = field(default_factory=dict)
    state_estimates: tuple[TimedStateEstimate, ...] = ()

    def _decision_state(self) -> TimedStateEstimate:
        latest = self.state_estimates[-1] if self.state_estimates else None
        if (latest is not None and latest.vehicle_state is self.vehicle_state
                and latest.ego_speed == self.ego_speed):
            return latest
        time_us = latest.time_us if latest is not None else timestamp_to_microseconds(self.t)
        return TimedStateEstimate(time_us, time_us / 1e6, self.vehicle_state, self.ego_speed)

    def estimated_states(self) -> tuple[TimedStateEstimate, ...]:
        """New localization results, with current observation edits on the decision state."""
        current = self._decision_state()
        return (*self.state_estimates[:-1], current) if self.state_estimates else (current,)

    def delivered(self, sensor_id: str) -> tuple[DeliveredAcquisition, ...]:
        """Read this decision's incremental acquisitions with current owner-field edits.

        Only the latest payload and calibration come from the observation mappings.
        Frames ending at the decision timestamp see its current state edits.
        Stripped payloads read as None. When there are no delivery records, a sensor in
        ``sensor_data`` produces one acquisition from the current observation. An unknown
        sensor with neither delivery records nor sensor data raises KeyError.
        """
        acquisitions = self.deliveries.get(sensor_id, ())
        current = self._decision_state()
        if not acquisitions:
            if sensor_id not in self.sensor_data:
                raise KeyError(f"Unknown observation sensor: {sensor_id!r}")
            return (DeliveredAcquisition(current.time_us, self.t, self.frame,
                    self.sensor_data[sensor_id], self.calibration.get(sensor_id), current),)
        result = []
        for index, acquisition in enumerate(acquisitions):
            changes: dict[str, Any] = {}
            if index == len(acquisitions) - 1:
                changes.update(data=self.sensor_data.get(sensor_id), calibration=self.calibration.get(sensor_id))
            if acquisition.time_us == current.time_us:
                changes['state'] = current
            if all(getattr(acquisition, key) is value for key, value in changes.items()):
                result.append(acquisition)
            else:
                result.append(replace(acquisition, **changes))
        return tuple(result)

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


def without_sensor_payloads(observation: Observation, *, keep_sensor_ids: Iterable[str] = (),
                            strip_deliveries: bool = True) -> Observation:
    """Select observation payload owners and optionally retain only delivery metadata.

    Selected sensor payloads keep their existing references. Calibration, state and frame
    metadata remain available. No array is copied by this operation.
    """
    retained = set(keep_sensor_ids)
    deliveries = ({sensor: tuple(replace(item, data=None) for item in values)
                   for sensor, values in observation.deliveries.items()}
                  if strip_deliveries else observation.deliveries)
    return replace(observation,
        sensor_data={sensor: data for sensor, data in observation.sensor_data.items() if sensor in retained},
        deliveries=deliveries)


@dataclass(frozen=True)
class WorldSnapshot:
    """An independent snapshot of the simulator's actual ego state and other actors' states.

    ``actors`` maps stable actor IDs, other than the ego, to ``ActorState`` records (canonical
    world pose, velocity and category). Nested state objects belong to this snapshot. Editing
    them does not actuate the world. Simulator-runtime actor methods do. ``ego`` is canonical
    physical truth. ``vehicle_state`` is its native SDK representation for adapter callbacks.
    Neither is the potentially perturbed model-visible Observation.vehicle_state.
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
    """Model-visible ego state and speed produced by the driving system's localizer."""

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

    ``position`` is the cuboid center in meters, ``extent`` its full length, width and
    height in meters, and ``yaw`` its heading in radians about ego z. Sensor mounting
    extrinsics define this frame, independently of the driving system's localization.
    Its origin is the true ego actor origin, not the ego cuboid center. Evaluation uses
    the associated CaptureRecord physical state to transform ground-truth actors into this frame.
    Records without an associated capture use ``StepRecord.physical_pre.ego``.
    """

    position: tuple[float, float, float]
    extent: tuple[float, float, float]
    yaw: float
    category: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:

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
    """Return the selected sensor ID and H×W×3 uint8 RGB pixels, or ``None`` if unavailable.

    Accept arrays and image wrappers exposing ``rgb_image``. With no camera name, inspect
    only the first sensor entry.
    """
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
    """Return an observation copy with sensor ``key`` replaced by pixels cast to uint8.

    Replace array payloads directly. Shallow-copy image wrappers and replace their pixel buffer.
    Other fields and sensor entries retain their values.
    """
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
    ``PointCloud`` has no ``data`` attribute and is replaced as a whole, for example
    ``write_sensor_data(observation, "lidar", PointCloud(points, intensity))``.
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


__all__ = [
    'PointCloud',
    'ActorState',
    'Unavailable',
    'TargetSpec',
    'Control',
    'DeliveredAcquisition',
    'TimedStateEstimate',
    'Observation',
    'WorldSnapshot',
    'StateEstimate',
    'RecordedOutput',
    'Detection',
    'validate_command_modes',
    'read_camera_rgb',
    'write_camera_rgb',
    'write_sensor_data',
]
