"""Shared JSON-compatible serialization and independent in-memory snapshots.

``encode`` and ``decode`` convert registered dataclasses and native values to and from
JSON-compatible values. ``snapshot`` preserves independent in-memory data, including native
wrappers. This module belongs at the bottom layer because driving adapters also snapshot
native model results. Lazy registrations import no model or simulator during construction.
Users register their dataclasses with ``DEFAULT_REGISTRY.register_dataclass(MyMeasurement)``;
native encoders deliberately decode to ordinary data.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from avsectester.type_paths import import_type_path, split_type_path

_TAG = "_avsectester_type"
_KEYS = "_avsectester_keys"


class SerializationRegistry:
    """Explicit dataclass reconstruction and data-only native encoding.

    ``snapshot`` preserves independent in-memory native values. Native ``snapshot`` callbacks
    receive ``(value, snapshot_child)`` and must return a distinct native-compatible value.
    Optional encode/decode callbacks let storage transports handle arrays or dataclass fields
    while sharing the same dataclass registrations and native conversion rules.
    """

    def __init__(self) -> None:
        self._dataclasses: dict[str, type | None] = {}
        self._native_types: dict[tuple[str, str], tuple[
            Callable[[Any], Any], Callable[[Any, Callable[[Any], Any]], Any] | None,
        ]] = {}

    def register_dataclass(self, cls_or_path: type | str) -> None:
        """Register a dataclass, or a path imported only when that dataclass is decoded."""
        dataclass_type: type | None
        if isinstance(cls_or_path, str):
            path = cls_or_path
            split_type_path(path)
            dataclass_type = None
        elif isinstance(cls_or_path, type) and is_dataclass(cls_or_path):
            dataclass_type = cls_or_path
            path = f"{dataclass_type.__module__}:{dataclass_type.__qualname__}"
        else:
            raise TypeError("register_dataclass requires a dataclass type or its module path")
        if path in self._dataclasses:
            raise ValueError(f"Duplicate dataclass registration: {path}")
        self._dataclasses[path] = dataclass_type

    def register_native(
        self, module: str, name: str, encode: Callable[[Any], Any], *,
        snapshot: Callable[[Any, Callable[[Any], Any]], Any] | None = None,
    ) -> None:
        """Register native conversion without importing the native module.

        Resolution follows the value's MRO, so a subclass registration takes precedence
        over its base. Imported module aliases, including extension SDK exports, work too.
        """
        if not module or not name or not isinstance(module, str) or not isinstance(name, str):
            raise ValueError("Native registrations require nonempty module and class names")
        if not callable(encode) or (snapshot is not None and not callable(snapshot)):
            raise TypeError("Native encode and snapshot callbacks must be callable")
        key = split_type_path(f"{module}:{name}")
        if key in self._native_types:
            raise ValueError(f"Duplicate native registration: {module}:{name}")
        self._native_types[key] = encode, snapshot

    def snapshot(self, value: Any) -> Any:
        """Take an independent snapshot, preserving native wrappers and container subclasses."""
        return _snapshot(value, {}, self)

    def _native(self, value: Any) -> tuple[
        Callable[[Any], Any], Callable[[Any, Callable[[Any], Any]], Any] | None,
    ] | None:
        for base in type(value).__mro__:
            callbacks = self._native_registration(base)
            if callbacks is not None:
                return callbacks
        return None

    def _native_registration(self, native_type: type) -> tuple[
        Callable[[Any], Any], Callable[[Any, Callable[[Any], Any]], Any] | None,
    ] | None:
        direct = self._native_types.get((native_type.__module__, native_type.__qualname__))
        if direct is not None:
            return direct
        for (module_name, name), callbacks in self._native_types.items():
            target = sys.modules.get(module_name)
            for part in name.split("."):
                target = getattr(target, part, None)
            if target is native_type:
                return callbacks
        return None

    def _snapshotter(self, value: Any) -> Callable[[Any, Callable[[Any], Any]], Any] | None:
        for base in type(value).__mro__:
            native = self._native_registration(base)
            if native is not None and native[1] is not None:
                return native[1]
        return None

    def encode(
        self, value: Any, *, encode_child: Callable[[Any], Any] | None = None,
        encode_field: Callable[[str, Any], Any] | None = None,
        encode_leaf: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Convert a registered value into tagged JSON-compatible data."""
        child = encode_child or self.encode
        field_encoder = encode_field or (lambda name, item: child(item))
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        native = self._native(value)
        if native is not None:
            converted = native[0](value)
            if converted is value:
                raise TypeError("A native encoder must not return its input")
            return child(converted)
        if _is_tensor(value):
            return child(value.detach().cpu().numpy())
        if isinstance(value, np.generic):
            return child(value.item())
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields:
                raise TypeError("Object/structured arrays require an explicit native registration")
            if encode_leaf is not None:
                return encode_leaf(value)
            return {_TAG: "array", "dtype": value.dtype.str, "shape": list(value.shape),
                    "data": base64.b64encode(np.ascontiguousarray(value).tobytes()).decode("ascii")}
        if isinstance(value, bytes):
            if encode_leaf is not None:
                return encode_leaf(value)
            return {_TAG: "bytes", "data": base64.b64encode(value).decode("ascii")}
        if isinstance(value, Path):
            return {_TAG: "Path", "value": str(value)}
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Serialized mappings require string keys")
            if _TAG in value or _KEYS in value:
                return {_TAG: "mapping", "entries": [
                    {"key": key, "value": child(item)} for key, item in value.items()
                ]}
            return {_KEYS: list(value), **{key: child(item) for key, item in value.items()}}
        if isinstance(value, list):
            return [child(item) for item in value]
        if isinstance(value, tuple):
            return {_TAG: "tuple", "length": len(value),
                    **{f"item_{index}": child(item) for index, item in enumerate(value)}}
        if is_dataclass(value) and not isinstance(value, type):
            path = f"{type(value).__module__}:{type(value).__qualname__}"
            if path not in self._dataclasses:
                raise TypeError(f"Unregistered dataclass type {path}; use register_dataclass")
            return {_TAG: "dataclass", "type": path, "fields": {
                item.name: field_encoder(item.name, getattr(value, item.name))
                for item in fields(value) if item.init
            }}
        raise TypeError(
            f"Unregistered type {type(value).__module__}:{type(value).__qualname__}; "
            "use register_dataclass for dataclasses or register_native for native values"
        )

    def decode(self, data: Any, *, decode_child: Callable[[Any], Any] | None = None) -> Any:
        """Reconstruct registered dataclasses; encoded native values remain ordinary data."""
        child = decode_child or self.decode
        if isinstance(data, list):
            return [child(item) for item in data]
        if not isinstance(data, dict):
            return data
        if _KEYS in data:
            return {key: child(data[key]) for key in data[_KEYS]}
        tag = data.get(_TAG)
        if tag == "dataclass":
            path = data["type"]
            if not isinstance(path, str) or path not in self._dataclasses:
                raise TypeError(f"Unregistered dataclass type {path!r}; use register_dataclass before decoding")
            dataclass_type = self._dataclasses[path]
            if dataclass_type is None:
                resolved = import_type_path(path)
                if not isinstance(resolved, type) or not is_dataclass(resolved):
                    raise TypeError(f"Registered type {path} is not a dataclass type")
                dataclass_type = resolved
                self._dataclasses[path] = dataclass_type
            values = data["fields"]
            return dataclass_type(**{
                item.name: child(values[item.name]) for item in fields(dataclass_type)
                if item.init and item.name in values
            })
        if tag == "mapping":
            return {entry["key"]: child(entry["value"]) for entry in data["entries"]}
        if tag == "tuple":
            return tuple(child(data[f"item_{index}"]) for index in range(data["length"]))
        if tag == "Path":
            return Path(data["value"])
        if tag in {"array", "bytes"}:
            raw = base64.b64decode(data["data"], validate=True)
            if tag == "bytes":
                return raw
            dtype = np.dtype(data["dtype"])
            if dtype.hasobject or dtype.fields:
                raise ValueError("Unsafe array dtype in serialized data")
            return np.frombuffer(raw, dtype=dtype).reshape(data["shape"]).copy()
        if tag is not None:
            raise ValueError(f"Unknown serialized encoding: {tag!r}")
        return {key: child(item) for key, item in data.items()}


def _snapshot(value: Any, memo: dict[int, Any], registry: SerializationRegistry) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    key = id(value)
    if key in memo:
        return memo[key]
    snapshot = registry._snapshotter(value)
    if snapshot is not None:
        result = snapshot(value, lambda child: _snapshot(child, memo, registry))
        if result is value:
            raise TypeError("A Native snapshot callback must return an independent snapshot")
        memo[key] = result
        return result
    if _is_tensor(value):
        result = value.detach().clone()
        memo[key] = result
        return result
    if type(value) is dict:
        result = {}
        memo[key] = result
        for k, v in value.items():
            result[deepcopy(k, memo)] = _snapshot(v, memo, registry)
        return result
    if type(value) is list:
        result = []
        memo[key] = result
        result.extend(_snapshot(v, memo, registry) for v in value)
        return result
    if type(value) is tuple:
        result = tuple(_snapshot(v, memo, registry) for v in value)
        memo[key] = result
        return result
    if isinstance(value, dict):
        result = copy(value)
        memo[key] = result
        result.clear()
        for k, v in value.items():
            result[deepcopy(k, memo)] = _snapshot(v, memo, registry)
        _snapshot_attributes(value, result, memo, registry)
        return result
    if isinstance(value, list):
        result = copy(value)
        memo[key] = result
        result[:] = [_snapshot(v, memo, registry) for v in value]
        _snapshot_attributes(value, result, memo, registry)
        return result
    if isinstance(value, tuple):
        items = [_snapshot(v, memo, registry) for v in value]
        if hasattr(value, "_fields"):
            result = getattr(type(value), "_make")(items)
        else:
            try:
                result = type(value)(items)
            except Exception:
                result = deepcopy(value, memo)
        memo[key] = result
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = object.__new__(type(value))
        memo[key] = result
        for f in fields(value):
            object.__setattr__(result, f.name, _snapshot(getattr(value, f.name), memo, registry))
        return result
    return deepcopy(value, memo)


def _snapshot_attributes(
    source: Any, result: Any, memo: dict[int, Any], registry: SerializationRegistry,
) -> None:
    """Replace the attributes ``copy`` shared with ``source`` by independent snapshots."""
    state = getattr(source, "__dict__", None)
    if state:
        target = vars(result)
        for name, item in state.items():
            target[name] = _snapshot(item, memo, registry)


def _is_tensor(value: Any) -> bool:
    return any(base.__module__.split(".")[0] == "torch" and base.__name__ == "Tensor"
               for base in type(value).__mro__)


DEFAULT_REGISTRY = SerializationRegistry()
for _module, _names in (
    ("avsectester.data_types", (
        "ActorState", "Unavailable", "TargetSpec", "Control", "Observation", "WorldSnapshot",
        "StateEstimate", "RecordedOutput", "Detection",
    )),
    ("avsectester.datasets.base", ("FrameReference",)),
    ("avsectester.closed_loop.simulators.base", ("ClockSnapshot", "SensorGeometry")),
    ("avsectester.closed_loop.simulators.ego_dynamics", ("EgoPose",)),
    ("avsectester.rendering.compositing", ("ObjectVisibility",)),
    ("avsectester.geometry.cameras", ("PinholeCamera", "FThetaCamera")),
    ("avsectester.attack.attack", ("AttackSegment",)),
    ("avsectester.maps.scenario_map", ("ScenarioMap", "Lane", "TrafficControl", "StopLine")),
):
    for _name in _names:
        DEFAULT_REGISTRY.register_dataclass(f"{_module}:{_name}")
