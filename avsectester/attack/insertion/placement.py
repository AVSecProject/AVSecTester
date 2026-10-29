"""Insertion positions and orientations shared by attack metadata."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from avsectester.geometry import poses

_ANCHORS = {
    "center": (0, 0, 0),
    "front_center": (0.5, 0, 0),
    "rear_center": (-0.5, 0, 0),
    "left_center": (0, 0.5, 0),
    "right_center": (0, -0.5, 0),
    "top_center": (0, 0, 0.5),
    "bottom_center": (0, 0, -0.5),
}


def validate_attached_mount(
    anchor: str, offset_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Validate host-independent mounting options and return a finite metric offset.

    Selection may use these options before choosing an actual host. Construct an
    ``AttachedPlacement`` only after that host's stable actor ID is known.
    """
    if not isinstance(anchor, str) or anchor not in _ANCHORS:
        raise ValueError(f"Unknown anchor {anchor!r}. Choose from {tuple(_ANCHORS)}")
    x, y, z = poses.validated_vector(offset_m, "offset_m")
    return float(x), float(y), float(z)


@dataclass(frozen=True)
class AttachedPlacement:
    """A user-specified host-local point. Named anchors refer to bounding-box faces."""

    host: str
    anchor: str = "center"
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a nonempty logical actor ID")
        object.__setattr__(self, "offset_m", validate_attached_mount(self.anchor, self.offset_m))

    def local_position(self, extent: tuple[float, float, float]) -> np.ndarray:
        """Return this mounting point in host-local axes for full bounding-box dimensions."""
        dimensions = poses.validated_vector(extent, "extent")
        if (dimensions < 0).any():
            raise ValueError("extent must be nonnegative")
        return np.asarray(_ANCHORS[self.anchor]) * dimensions + self.offset_m


@dataclass(frozen=True)
class WorldPlacement:
    """A fixed position in the canonical world frame, in meters."""

    position_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", tuple(poses.validated_vector(self.position_m, "position_m")))


@dataclass(frozen=True)
class Orientation:
    """Rotation relative to a host, absolute in world, or directed at the ego.

    ``face_ego`` points the asset's +X axis at the ego-local ``target_offset_m``.
    It keeps +Z as close to world up as possible. Directly vertical directions use world +Y
    as the secondary up reference. Explicit Euler angles apply to the other two modes only.
    """

    mode: str = "fixed_world"
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.mode not in {"follow_host", "fixed_world", "face_ego"}:
            raise ValueError(f"Unknown orientation mode {self.mode!r}")
        rotation = poses.validated_vector(self.rotation_deg, "rotation_deg")
        offset = poses.validated_vector(self.target_offset_m, "target_offset_m")
        if self.mode == "face_ego" and np.any(rotation != 0):
            raise ValueError("face_ego determines orientation, so rotation_deg must be zero")
        if self.mode != "face_ego" and np.any(offset != 0):
            raise ValueError("target_offset_m is only used by face_ego")
        object.__setattr__(self, "rotation_deg", tuple(rotation))
        object.__setattr__(self, "target_offset_m", tuple(offset))


def placement_from_spec(value: Any) -> AttachedPlacement | WorldPlacement:
    """Read an explicit placement object or a ``{type, options}`` specification."""
    if isinstance(value, Mapping):
        if set(value) - {"type", "options"} or "type" not in value:
            raise ValueError("Placement requires only type and options")
        classes = {f"{kind.__module__}:{kind.__name__}": kind
                   for kind in (AttachedPlacement, WorldPlacement)}
        kind = value["type"]
        if not isinstance(kind, str) or kind not in classes:
            raise ValueError(f"Unknown placement type {kind!r}; choose from {tuple(classes)}")
        options = value.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("Placement options must be a mapping")
        return classes[kind](**options)
    if not isinstance(value, (AttachedPlacement, WorldPlacement)):
        raise TypeError("placement must be AttachedPlacement or WorldPlacement")
    return value


def orientation_from_spec(value: Any) -> Orientation:
    """Read an explicit orientation. Default image insertions face the ego."""
    if value is None:
        return Orientation("face_ego")
    if isinstance(value, Mapping):
        if (set(value) - {"type", "options"}
                or value.get("type") != f"{Orientation.__module__}:{Orientation.__name__}"):
            raise ValueError("Orientation requires its full module:Class type and options")
        options = value.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError("Orientation options must be a mapping")
        return Orientation(**options)
    if not isinstance(value, Orientation):
        raise TypeError("orientation must be Orientation")
    return value


__all__ = [
    'validate_attached_mount',
    'AttachedPlacement',
    'WorldPlacement',
    'Orientation',
    'placement_from_spec',
    'orientation_from_spec',
]
