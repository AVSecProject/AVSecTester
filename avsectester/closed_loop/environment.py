"""World settings and deterministic sensor corruptions.

Sensor corruptions run after attack sensor actions and before ObservationPipeline hooks. World settings
are validated against the simulator and passed to its reset without automatic conversion.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from avsectester.data_types import Observation, read_camera_rgb, write_camera_rgb
from avsectester.closed_loop.sensor_corruptions import CORRUPTIONS
from avsectester.type_paths import import_type_path


def normalize_environment(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Copy and validate the public ``{world, sensors}`` structure.

    Each sensor entry has ``type``, optional ``options`` and an optional ``camera``. A ``type``
    without a colon names a built-in ``Corruption`` class. ``package.module:name`` names a
    custom corruption, constructed with ``options``. Custom corruptions implement
    ``apply(rgb, rng) -> rgb`` or are callables with that signature.
    """
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("environment must be a mapping with world and sensors")
    unknown = set(value) - {"world", "sensors"}
    if unknown:
        raise ValueError(f"Unknown environment settings: {sorted(unknown)}. Use world or sensors")
    world, sensors = value.get("world", {}), value.get("sensors", [])
    if not isinstance(world, Mapping):
        raise TypeError("environment.world must be a mapping")
    if not isinstance(sensors, Sequence) or isinstance(sensors, (str, bytes)):
        raise TypeError("environment.sensors must be a list of corruption specifications")
    entries = []
    for entry in sensors:
        if not isinstance(entry, Mapping):
            raise TypeError("Sensor environment entries must be mappings")
        entry = deepcopy(dict(entry))
        unknown = set(entry) - {"type", "options", "camera"}
        if unknown:
            raise ValueError(f"Unknown sensor environment keys: {sorted(unknown)}. "
                             "Use type, options and camera")
        kind = entry.get("type")
        if not isinstance(kind, str) or not kind:
            raise ValueError("Sensor environment entries require type")
        camera = entry.get("camera")
        if camera is not None and (not isinstance(camera, str) or not camera):
            raise ValueError("Sensor environment camera must be a nonempty sensor name")
        options = entry.setdefault("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("Sensor environment options must be a mapping")
        if ":" in kind:
            if kind.count(":") != 1 or not all(kind.split(":")):
                raise ValueError(f"Custom sensor corruption type {kind!r} must be 'package.module:name'")
        else:
            if kind not in CORRUPTIONS:
                raise ValueError(f"Unknown sensor corruption class {kind!r}. Use a built-in class "
                                 "name or 'package.module:name' for a custom corruption")
            allowed = ({"severity", "warm"} if kind == "ColorTemp" else
                       {"severity", "transform"} if kind == "AlbumentationsCorruption" else {"severity"})
            if set(options) - allowed:
                raise ValueError(f"Unknown {kind} options: {sorted(set(options) - allowed)}")
            if kind == "AlbumentationsCorruption":
                from avsectester.closed_loop.sensor_corruptions import ALBUMENTATIONS_TRANSFORMS
                transform = options.get("transform")
                if not isinstance(transform, str) or transform not in ALBUMENTATIONS_TRANSFORMS:
                    raise ValueError("AlbumentationsCorruption requires a supported transform: "
                                     + ", ".join(sorted(ALBUMENTATIONS_TRANSFORMS)))
            severity = options.get("severity", 0.5)
            if isinstance(severity, bool) or not isinstance(severity, (int, float)) or not 0 <= severity <= 1:
                raise ValueError("Sensor corruption severity must be between 0 and 1")
        entries.append(entry)
    return {"world": deepcopy(dict(world)), "sensors": entries}


def merge_environments(*values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge world mappings recursively. An explicit later sensors list replaces the earlier list.
    """
    def overlay(before: dict[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
        for key, value in after.items():
            before[key] = (overlay(dict(before[key]), value)
                           if isinstance(before.get(key), Mapping) and isinstance(value, Mapping)
                           else deepcopy(value))
        return before

    result: dict[str, Any] = {"world": {}, "sensors": []}
    for value in values:
        normalized = normalize_environment(value)
        result["world"] = overlay(result["world"], normalized["world"])
        if "sensors" in value:
            result["sensors"] = normalized["sensors"]
    return result


def validate_environment(value: Mapping[str, Any], simulator: Any) -> dict[str, Any]:
    """Reject unsupported world settings before opening any simulator resources."""
    environment = normalize_environment(value)
    simulator.validate_world_settings(environment["world"])
    return environment


class SensorEnvironment:
    """One run's corruptions, seeded by sensor stream and integer acquisition time."""

    def __init__(self, environment: Mapping[str, Any], seed: int = 0) -> None:
        normalized = normalize_environment(environment)
        self.seed = seed
        self.corruptions = []
        self._default_camera: str | None = None
        for spec in normalized["sensors"]:
            if ":" in spec["type"]:
                constructor = import_type_path(spec["type"])
                corruption = constructor(**spec["options"])
            else:
                corruption = CORRUPTIONS[spec["type"]](**spec["options"])
            apply = getattr(corruption, "apply", corruption)
            if not callable(apply):
                raise TypeError("A sensor environment corruption must implement apply(rgb, rng)")
            self.corruptions.append((spec.get("camera"), apply))

    def bind(self, observation: Observation) -> None:
        """Resolve omitted camera names from the initial full capture without corrupting it."""
        capture = read_camera_rgb(observation)
        if capture is not None:
            self._default_camera = capture[0]

    def apply(self, observation: Observation, position: int, *,
              interval_capture_index: Mapping[str, int] | None = None,
              available_sensors: frozenset[str] | None = None) -> Observation:
        """Apply corruptions with stable sensor and acquisition-time randomness.

        ``position`` is the containing dataset grid position. ``interval_capture_index``
        gives each due camera's own capture index within that decision interval, starting
        at zero. Index zero uses the three-part random seed.
        ``available_sensors`` names the configured streams. A configured camera absent
        from this frame batch is skipped. Other missing cameras raise ``ValueError``.
        """
        result = observation
        for index, (camera, apply) in enumerate(self.corruptions):
            if camera is None:
                if self._default_camera is None:
                    initial = read_camera_rgb(result)
                    if initial is not None:
                        self._default_camera = initial[0]
                camera = self._default_camera
            if camera is not None and camera not in result.sensor_data:
                if available_sensors is not None and camera in available_sensors:
                    continue
                raise ValueError(f"Sensor environment requested unavailable camera: {camera!r}")
            capture = read_camera_rgb(result, camera)
            if capture is None:
                raise ValueError(f"Sensor environment requires a readable RGB camera: {camera!r}")
            sensor_id, rgb = capture
            capture_index = (interval_capture_index or {}).get(sensor_id, 0)
            key = (self.seed, int(position), index)
            rng = np.random.default_rng(key if capture_index == 0 else (*key, capture_index))
            corrupted = np.asarray(apply(rgb.copy(), rng))
            if corrupted.shape != rgb.shape or corrupted.dtype != np.uint8:
                raise ValueError("Sensor environment corruptions must return uint8 RGB with unchanged shape")
            result = write_camera_rgb(result, sensor_id, corrupted)
        return result


__all__ = [
    'normalize_environment',
    'merge_environments',
    'validate_environment',
    'SensorEnvironment',
]
