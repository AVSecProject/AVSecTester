"""Translate ROS message fields into NumPy buffers in the relay process."""

from __future__ import annotations

from array import array
from math import prod
from typing import Any

import numpy as np

from avsectester.structured_arrays import item_bytes


POINT_CLOUD = "sensor_msgs/msg/PointCloud2"
"""The ROS type name for a structured point cloud."""
OCCUPANCY_GRID = "nav_msgs/msg/OccupancyGrid"
"""The ROS type name for an occupancy grid."""

_POINT_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
_POINT_TYPE_CODES = {dtype: code for code, dtype in _POINT_TYPES.items()}


def _fields(value: Any) -> Any:
    if hasattr(value, "get_fields_and_field_types"):
        return {name: _fields(getattr(value, name)) for name in value.get_fields_and_field_types()}
    if isinstance(value, (array, np.ndarray)):
        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        if value and all(type(item) in (int, float, bool) for item in value):
            return np.asarray(value)
        return [_fields(item) for item in value]
    return value


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
    """Read ROS fields, including fields and row strides of structured clouds."""
    fields = {name: _fields(getattr(message, name)) for name in message.get_fields_and_field_types()
              if name != "data" or type_name != POINT_CLOUD}
    if type_name == POINT_CLOUD:
        fields["data"] = _point_cloud_array(message)
    elif type_name == OCCUPANCY_GRID:
        fields["data"] = np.asarray(fields["data"], dtype=np.int8)
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


def publishable_fields(fields: dict[str, Any], type_name: str) -> dict[str, Any]:
    """Prepare fields for ROS assignment after a payload replacement."""
    if type_name == POINT_CLOUD:
        return _point_cloud_fields(fields)
    if type_name == OCCUPANCY_GRID:
        data = fields["data"]
        expected = fields["info"]["height"] * fields["info"]["width"]
        if not isinstance(data, np.ndarray) or data.dtype != np.int8 or data.size != expected:
            raise ValueError("OccupancyGrid data must be an int8 array matching its height and width")
        return {**fields, "data": data.reshape(-1)}
    return fields


__all__ = ["OCCUPANCY_GRID", "POINT_CLOUD", "header_stamp_ns", "publishable_fields", "to_fields"]
