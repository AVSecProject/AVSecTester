"""Immutable semantic map records in canonical world coordinates, measured in meters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
import math
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from avsectester.geometry.poses import validated_transform


# Control kinds that stop traffic at a line, and the stop-line kind each one gives.
STOP_LINE_KIND_BY_CONTROL_KIND: Mapping[str, str] = MappingProxyType({
    "stop_sign": "stop_sign", "yield_sign": "yield_sign", "traffic_light": "traffic_light",
})
'Mapping from traffic-control kind to the corresponding stop-line kind.'
_STOP_LINE_KINDS = frozenset({"stop_sign", "yield_sign", "traffic_light", "unknown"})
_CONTROL_KINDS = frozenset({"stop_sign", "yield_sign", "speed_limit", "traffic_light", "other"})


def _points(value: Any, name: str, minimum: int) -> np.ndarray:
    points = np.array(value, dtype=float, copy=True)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum
            or not np.isfinite(points).all()):
        raise ValueError(f"{name} must contain at least {minimum} finite three-dimensional points")
    points.setflags(write=False)
    return points


def _identity(value: str, name: str = "id") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _identities(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of IDs")
    return tuple(_identity(value, name) for value in values)


@dataclass(frozen=True)
class Lane:
    """One lane with direction-ordered centerline and left/right boundary polylines."""

    id: str
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    lane_type: Literal["driving", "shoulder", "bike", "parking", "other"] = "driving"
    predecessors: tuple[str, ...] = ()
    successors: tuple[str, ...] = ()
    left_neighbor: str | None = None
    right_neighbor: str | None = None
    speed_limit_mps: float | None = None

    def __post_init__(self) -> None:
        _identity(self.id)
        for name in ("centerline", "left_boundary", "right_boundary"):
            object.__setattr__(self, name, _points(getattr(self, name), name, 2))
        if self.lane_type not in {"driving", "shoulder", "bike", "parking", "other"}:
            raise ValueError("Unsupported lane_type")
        for name in ("predecessors", "successors"):
            object.__setattr__(self, name, _identities(getattr(self, name), name))
        for name in ("left_neighbor", "right_neighbor"):
            value = getattr(self, name)
            if value is not None:
                _identity(value, name)
        if self.speed_limit_mps is not None:
            if (isinstance(self.speed_limit_mps, bool) or not math.isfinite(self.speed_limit_mps)
                    or self.speed_limit_mps < 0):
                raise ValueError("speed_limit_mps must be finite and nonnegative")
            object.__setattr__(self, "speed_limit_mps", float(self.speed_limit_mps))


@dataclass(frozen=True)
class TrafficControl:
    """A static control facing approaching traffic, with (thickness, width, height) extent.

    The pose origin is either the face center or the source's signal position, as declared
    by ``pose_reference``. A signal position need not locate the visible sign or lamp head.
    """

    id: str
    kind: Literal["stop_sign", "yield_sign", "speed_limit", "traffic_light", "other"]
    native_category: str
    pose: np.ndarray
    extent: tuple[float, float, float]
    pose_reference: Literal["face_center", "signal_position"]
    lane_ids: tuple[str, ...] = ()
    stop_line_id: str | None = None

    def __post_init__(self) -> None:
        _identity(self.id)
        if self.kind not in _CONTROL_KINDS:
            raise ValueError("Unsupported traffic-control kind")
        if not isinstance(self.native_category, str):
            raise TypeError("native_category must be a string")
        if self.pose_reference not in {"face_center", "signal_position"}:
            raise ValueError("pose_reference must be face_center or signal_position")
        object.__setattr__(self, "pose", validated_transform(self.pose))
        extent = np.asarray(self.extent, dtype=float)
        if extent.shape != (3,) or not np.isfinite(extent).all() or (extent < 0).any():
            raise ValueError("Traffic-control extent must be finite nonnegative thickness, width, height")
        object.__setattr__(self, "extent", tuple(float(value) for value in extent))
        object.__setattr__(self, "lane_ids", _identities(self.lane_ids, "lane_ids"))
        if self.stop_line_id is not None:
            _identity(self.stop_line_id, "stop_line_id")


@dataclass(frozen=True)
class StopLine:
    """A world-coordinate transverse line associated with lanes and optionally a control."""

    id: str
    points: np.ndarray
    native_category: str
    kind: Literal["stop_sign", "traffic_light", "yield_sign", "unknown"] = "unknown"
    lane_ids: tuple[str, ...] = ()
    control_id: str | None = None

    def __post_init__(self) -> None:
        _identity(self.id)
        object.__setattr__(self, "points", _points(self.points, "points", 2))
        if not isinstance(self.native_category, str):
            raise TypeError("native_category must be a string")
        if self.kind not in _STOP_LINE_KINDS:
            raise ValueError("Unsupported stop-line kind")
        object.__setattr__(self, "lane_ids", _identities(self.lane_ids, "lane_ids"))
        if self.control_id is not None:
            _identity(self.control_id, "control_id")


@dataclass(frozen=True)
class ScenarioMap:
    """Semantic layers in canonical right-handed world axes, with explicit missing-layer reasons.

    drivable_area polygons represent their set union; they need not be dissolved or disjoint.
    Each crosswalks or intersections polygon is one crosswalk or intersection. Mapping inputs
    are copied, and every geometric array is an independent read-only copy.
    """

    lanes: Mapping[str, Lane] = field(default_factory=dict)
    drivable_area: tuple[np.ndarray, ...] = ()
    road_boundaries: tuple[np.ndarray, ...] = ()
    crosswalks: tuple[np.ndarray, ...] = ()
    intersections: tuple[np.ndarray, ...] = ()
    traffic_controls: Mapping[str, TrafficControl] = field(default_factory=dict)
    stop_lines: Mapping[str, StopLine] = field(default_factory=dict)
    unavailable_layers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, record_type in (("lanes", Lane), ("traffic_controls", TrafficControl),
                                  ("stop_lines", StopLine)):
            records = dict(getattr(self, name))
            if any(not isinstance(value, record_type) or key != value.id
                   for key, value in records.items()):
                raise ValueError(f"{name} must map record IDs to matching map records")
            object.__setattr__(self, name, records)
        for name in ("drivable_area", "road_boundaries", "crosswalks", "intersections"):
            minimum = 2 if name == "road_boundaries" else 3
            object.__setattr__(self, name,
                               tuple(_points(points, name, minimum) for points in getattr(self, name)))
        unavailable = dict(self.unavailable_layers)
        for name, reason in unavailable.items():
            _identity(name, "layer name")
            if name not in _UNAVAILABLE_KEYS:
                raise ValueError(f"Unsupported unavailable-layer key {name!r}; use a ScenarioMap "
                                 "layer, or layer.partial, layer.<record field> or "
                                 "traffic_controls.<kind>")
            _identity(reason, "unavailable-layer reason")
        object.__setattr__(self, "unavailable_layers", unavailable)


_LAYERS = ("lanes", "drivable_area", "road_boundaries", "crosswalks", "intersections",
           "traffic_controls", "stop_lines")
# A whole layer, part of a layer (some records unreadable), one record field, or one control kind.
_UNAVAILABLE_KEYS = frozenset({
    *_LAYERS,
    *(f"{layer}.partial" for layer in _LAYERS),
    *(f"lanes.{item.name}" for item in fields(Lane) if item.name != "id"),
    *(f"traffic_controls.{item.name}" for item in fields(TrafficControl) if item.name != "id"),
    *(f"stop_lines.{item.name}" for item in fields(StopLine) if item.name != "id"),
    *(f"traffic_controls.{kind}" for kind in _CONTROL_KINDS),
})


__all__ = [
    'STOP_LINE_KIND_BY_CONTROL_KIND',
    'Lane',
    'TrafficControl',
    'StopLine',
    'ScenarioMap',
]
