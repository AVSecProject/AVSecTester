"""Translate ROS message fields into NumPy buffers in the relay process."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from math import prod
from operator import attrgetter
from typing import Any

import numpy as np

from avsectester.structured_arrays import item_bytes


POINT_CLOUD = "sensor_msgs/msg/PointCloud2"
"""The ROS type name for a structured point cloud."""
OCCUPANCY_GRID = "nav_msgs/msg/OccupancyGrid"
"""The ROS type name for an occupancy grid."""

_POINT_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
_POINT_TYPE_CODES = {dtype: code for code, dtype in _POINT_TYPES.items()}


_NUMERIC_TYPES = {
    "boolean": "?", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1",
    "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4",
    "int64": "i8", "uint64": "u8",
}


def _native_message_type(slot: Any) -> type:
    return getattr(import_module(".".join(slot.namespaces)), slot.name)


@dataclass(frozen=True)
class _Layout:
    dtype: np.dtype
    read: Callable[[Any], Any]
    make: Callable[[Any], Any]


def _identity(value: Any) -> Any:
    return value


@cache
def _record_layout(message_type: type) -> _Layout | None:
    names = tuple(message_type.get_fields_and_field_types())
    if not names:
        return None
    members = [_field_layout(slot) for slot in message_type.SLOT_TYPES]
    if any(member is None for member in members):
        return None
    layouts = tuple(member for member in members if member is not None)
    dtype = np.dtype([(name, layout.dtype)
                      for name, layout in zip(names, layouts, strict=True)])
    readers = tuple((attrgetter(name), layout.read)
                    for name, layout in zip(names, layouts, strict=True))
    writers = tuple(layout.make for layout in layouts)

    def read(message: Any) -> tuple:
        return tuple(read(getter(message)) for getter, read in readers)

    def make(item: tuple) -> Any:
        return message_type(**{name: write(member)
                               for name, write, member in zip(names, writers, item, strict=True)})

    return _Layout(dtype, read, make)


def _field_layout(slot: Any) -> _Layout | None:
    if hasattr(slot, "namespaces"):
        return _record_layout(_native_message_type(slot))
    if hasattr(slot, "size") and hasattr(slot, "value_type"):
        element = _field_layout(slot.value_type)
        if element is None:
            return None
        return _Layout(np.dtype((element.dtype, (slot.size,))),
                       lambda values: [element.read(value) for value in values],
                       lambda values: [element.make(value) for value in values.tolist()])
    numeric = _NUMERIC_TYPES.get(getattr(slot, "typename", None))
    return None if numeric is None else _Layout(np.dtype(numeric), _identity, _identity)


def _sequence_layout(slot: Any) -> _Layout | None:
    element = getattr(slot, "value_type", None)
    return _record_layout(_native_message_type(element)) if hasattr(element, "namespaces") else None


@cache
def _field_slots(message_type: type) -> dict[str, tuple[Any, _Layout | None]]:
    return {name: (slot, _sequence_layout(slot))
            for name, slot in zip(message_type.get_fields_and_field_types(),
                                  message_type.SLOT_TYPES, strict=True)}


def _message_fields(value: Any, *, omit: str | None = None) -> dict[str, Any]:
    return {name: _field_value(getattr(value, name), slot, layout)
            for name, (slot, layout) in _field_slots(type(value)).items() if name != omit}


def _field_value(value: Any, slot: Any, layout: _Layout | None) -> Any:
    if hasattr(slot, "namespaces"):
        return _message_fields(value)
    element = getattr(slot, "value_type", None)
    if element is None:
        return value
    if layout is not None:
        return np.fromiter(map(layout.read, value), dtype=layout.dtype, count=len(value))
    if hasattr(element, "namespaces"):
        return [_message_fields(item) for item in value]
    dtype = _NUMERIC_TYPES.get(getattr(element, "typename", None))
    return list(value) if dtype is None else np.asarray(value, dtype=dtype)


def header_stamp_ns(message: Any) -> int | None:
    """Read a header or direct ROS stamp where the type provides one."""
    header = getattr(message, "header", None)
    stamp = getattr(header if header is not None else message, "stamp", None)
    if stamp is None:
        return None
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _point_cloud_array(message: Any) -> np.ndarray:
    names, formats, offsets = [], [], []
    endian = ">" if message.is_bigendian else "<"
    for field in message.fields:
        names.append(field.name)
        offsets.append(field.offset)
        dtype = np.dtype(endian + _POINT_TYPES[field.datatype])
        formats.append((dtype, (field.count,)) if field.count != 1 else dtype)
    width, height, step = message.width, message.height, message.point_step
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": step})
    if height <= 1:
        return np.ndarray((width if height else 0,), dtype=dtype, buffer=message.data)
    row_step = max(message.row_step, width * step)
    return np.ndarray((height, width), dtype=dtype, buffer=message.data, strides=(row_step, step))


def to_fields(message: Any, type_name: str) -> dict[str, Any]:
    """Read ROS fields, packing fixed-numeric message sequences into structured arrays."""
    fields = _message_fields(message, omit="data" if type_name == POINT_CLOUD else None)
    if type_name == POINT_CLOUD:
        fields["data"] = _point_cloud_array(message)
    return fields


def _point_cloud_fields(fields: dict[str, Any]) -> dict[str, Any]:
    points = fields["data"]
    if not isinstance(points, np.ndarray) or points.dtype.names is None or points.ndim not in (1, 2):
        raise ValueError("PointCloud2 data must be a one- or two-dimensional structured array")
    layout = []
    for name in points.dtype.names:
        dtype, offset = points.dtype.fields[name][:2]
        base, shape = dtype.subdtype or (dtype, ())
        code = _POINT_TYPE_CODES.get(base.str[1:])
        if code is None:
            raise ValueError(f"PointCloud2 field {name!r} has unsupported NumPy dtype {dtype}")
        layout.append({"name": name, "offset": offset, "datatype": code, "count": prod(shape)})
    height, width = (1, points.shape[0]) if points.ndim == 1 else points.shape
    step = points.dtype.itemsize
    endian = ">" if fields["is_bigendian"] else "<"
    points = points.astype(points.dtype.newbyteorder(endian), copy=False)
    return {**fields, "data": item_bytes(points), "fields": layout, "height": height, "width": width,
            "point_step": step, "row_step": step * width}


def _make_message(fields: Mapping[str, Any], message_type: type, path: str,
                  set_message_fields: Callable[[Any, Mapping[str, Any]], None]) -> Any:
    message = message_type()
    primitive = {}
    slots = _field_slots(message_type)
    for name, value in fields.items():
        field = f"{path}.{name}"
        if name not in slots:
            raise ValueError(f"Unknown ROS field {field}")
        slot, layout = slots[name]
        element = getattr(slot, "value_type", None)
        if layout is not None:
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != layout.dtype:
                raise ValueError(f"{field} requires a one-dimensional array with dtype {layout.dtype}")
            setattr(message, name, [layout.make(item) for item in value.tolist()])
        elif hasattr(slot, "namespaces"):
            if not isinstance(value, Mapping):
                raise ValueError(f"{field} requires a mapping")
            setattr(message, name,
                    _make_message(value, _native_message_type(slot), field, set_message_fields))
        elif hasattr(element, "namespaces"):
            if not isinstance(value, (tuple, list)) or any(not isinstance(item, Mapping) for item in value):
                raise ValueError(f"{field} requires a sequence of mappings")
            nested_type = _native_message_type(element)
            setattr(message, name, [
                _make_message(item, nested_type, f"{field}[{index}]", set_message_fields)
                for index, item in enumerate(value)
            ])
        else:
            target = getattr(message, name)
            primitive[name] = value.tolist() if isinstance(target, list) and isinstance(value, np.ndarray) else value
    if primitive:
        set_message_fields(message, primitive)
    return message


def to_message(fields: Mapping[str, Any], message_type: type, type_name: str) -> Any:
    """Build a ROS message from fields using its declared layouts.

    Fixed layouts require one-dimensional structured arrays with their declared dtype.
    Nested singleton messages and variable-layout records require mappings.
    """
    from rosidl_runtime_py import set_message_fields

    if type_name == POINT_CLOUD:
        fields = _point_cloud_fields(dict(fields))
    elif type_name == OCCUPANCY_GRID:
        data = fields["data"]
        expected = fields["info"]["height"] * fields["info"]["width"]
        if not isinstance(data, np.ndarray) or data.dtype != np.int8 or data.size != expected:
            raise ValueError("OccupancyGrid data must be an int8 array matching its height and width")
        fields = {**fields, "data": data.reshape(-1)}
    return _make_message(fields, message_type, type_name, set_message_fields)


__all__ = ["OCCUPANCY_GRID", "POINT_CLOUD", "header_stamp_ns", "to_fields", "to_message"]
