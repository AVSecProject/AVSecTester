"""World settings and deterministic sensor-environment transforms.

Sensor transforms run after attack sensor effects and before Vehicle I/O hooks. World settings
are validated against the simulator and passed to its reset without automatic conversion.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import import_module
from typing import Any

import numpy as np

from avsectester.closed_loop.datatypes import Observation, read_camera_rgb, write_camera_rgb
from avsectester.closed_loop.sensor_transforms import CORRUPTIONS


def normalize_environment(value: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Copy and validate the public ``{world, sensors}`` structure.

    Each sensor entry has ``type``, optional ``options`` and an optional ``camera``. A ``type``
    without a colon names a built-in ``Corruption`` class; ``package.module:name`` names a
    custom transform, constructed with ``options``. Custom transforms implement
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
        raise TypeError("environment.sensors must be a list of transform specifications")
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
                raise ValueError(f"Custom sensor transform type {kind!r} must be 'package.module:name'")
        else:
            if kind not in CORRUPTIONS:
                raise ValueError(f"Unknown sensor corruption class {kind!r}. Use a built-in class "
                                 "name or 'package.module:name' for a custom transform")
            allowed = {"severity", "warm"} if kind == "ColorTemp" else {"severity"}
            if set(options) - allowed:
                raise ValueError(f"Unknown {kind} options: {sorted(set(options) - allowed)}")
            severity = options.get("severity", 0.5)
            if isinstance(severity, bool) or not isinstance(severity, (int, float)) or not 0 <= severity <= 1:
                raise ValueError("Sensor corruption severity must be between 0 and 1")
        entries.append(entry)
    return {"world": deepcopy(dict(world)), "sensors": entries}


def merge_environments(*values: Mapping[str, Any]) -> dict[str, Any]:
    """Merge world mappings recursively. An explicit later sensors list replaces the earlier list."""
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
    """One run's transform instances, with randomness indexed by seed and absolute dataset frame position."""

    def __init__(self, environment: Mapping[str, Any], seed: int = 0) -> None:
        normalized = normalize_environment(environment)
        self.seed = seed
        self.transforms = []
        for spec in normalized["sensors"]:
            if ":" in spec["type"]:
                module, symbol = spec["type"].split(":", 1)
                constructor: Any = import_module(module)
                for part in symbol.split("."):
                    constructor = getattr(constructor, part)
                transform = constructor(**spec["options"])
            else:
                transform = CORRUPTIONS[spec["type"]](**spec["options"])
            apply = getattr(transform, "apply", transform)
            if not callable(apply):
                raise TypeError("A sensor environment transform must implement apply(rgb, rng)")
            self.transforms.append((spec.get("camera"), apply))

    def apply(self, observation: Observation, position: int) -> Observation:
        result = observation
        for index, (camera, apply) in enumerate(self.transforms):
            if camera is not None and camera not in result.sensor_data:
                raise ValueError(f"Sensor environment requested unavailable camera: {camera!r}")
            capture = read_camera_rgb(result, camera)
            if capture is None:
                raise ValueError(f"Sensor environment requires a readable RGB camera: {camera!r}")
            sensor_id, rgb = capture
            rng = np.random.default_rng((self.seed, int(position), index))
            transformed = np.asarray(apply(rgb.copy(), rng))
            if transformed.shape != rgb.shape or transformed.dtype != np.uint8:
                raise ValueError("Sensor environment transforms must return uint8 RGB with unchanged shape")
            result = write_camera_rgb(result, sensor_id, transformed)
        return result
