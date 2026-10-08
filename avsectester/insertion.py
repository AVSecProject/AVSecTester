"""Explicit insertion geometry, independent of a simulator or attack implementation.

World and actor frames use metres and right-handed x-forward, y-left, z-up axes. An actor
pose is centred on its bounding box. Assets use the same axes, with their front along +X.
Placement determines position, while orientation independently determines rotation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return array


def _transform(value) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = array[:3, :3]
    if not (
        np.allclose(array[3], [0, 0, 0, 1], atol=1e-8)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("transform must be a right-handed rigid pose")
    array.setflags(write=False)
    return array


def rotation_matrix(rotation_deg) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll) for right-handed degree angles."""
    roll, pitch, yaw = np.deg2rad(_vector(rotation_deg, "rotation_deg"))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass(frozen=True)
class ActorPose:
    """World pose of a bounding-box centre and its full (length, width, height)."""

    transform: np.ndarray
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        extent = _vector(self.extent, "extent")
        if (extent < 0).any():
            raise ValueError("extent must be nonnegative")
        object.__setattr__(self, "transform", _transform(self.transform))
        object.__setattr__(self, "extent", tuple(float(v) for v in extent))


@dataclass(frozen=True)
class PlaneSurface:
    """A textured rectangle, with corners ordered TL, TR, BR, BL from its front.

    Asset surfaces use local coordinates. Resolved surfaces use world coordinates. A surface
    unpacks as ``(corners, texture)`` for the existing planar image compositor.
    """

    corners: np.ndarray
    texture: np.ndarray

    def __post_init__(self):
        corners = np.array(self.corners, dtype=float, copy=True)
        texture = np.array(self.texture, copy=True)
        if corners.shape != (4, 3) or not np.isfinite(corners).all():
            raise ValueError("corners must be four finite three-dimensional points")
        if texture.ndim != 3 or texture.shape[2] != 4 or 0 in texture.shape:
            raise ValueError("texture must be a nonempty RGBA image")
        if texture.dtype != np.uint8:
            raise ValueError("texture must use uint8 RGBA values")
        u, v = corners[1] - corners[0], corners[3] - corners[0]
        scale = np.linalg.norm(u) * np.linalg.norm(v)
        if scale <= 0 or not np.isclose(u @ v, 0.0, atol=1e-8 * scale):
            raise ValueError("surface edges must form a nondegenerate rectangle")
        if not np.allclose(corners[2], corners[1] + corners[3] - corners[0]):
            raise ValueError("surface corners must form a rectangle in TL, TR, BR, BL order")
        corners.setflags(write=False)
        texture.setflags(write=False)
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "texture", texture)

    def __iter__(self):
        yield self.corners
        yield self.texture


class PlanarAsset(Protocol):
    """Extension point for assets composed of local textured rectangular surfaces."""

    def planes(self) -> Sequence[PlaneSurface]: ...


@dataclass(frozen=True)
class PlaneAsset:
    """RGBA patch or sign, centred at the origin in the local YZ plane.

    The front normal is +X and the top is +Z. Transparent pixels are not part of the silhouette.
    A file path is loaded once at construction. Image loading requires Pillow.
    """

    texture: np.ndarray | str | Path
    width_m: float
    height_m: float
    texture_path: str | None = field(default=None, init=False)

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.width_m, self.height_m)):
            raise ValueError("asset width and height must be finite and positive")
        if isinstance(self.texture, (str, Path)):
            from PIL import Image

            object.__setattr__(self, "texture_path", str(self.texture))
            with Image.open(self.texture) as image:
                texture = np.array(image.convert("RGBA"))
        else:
            texture = np.array(self.texture, copy=True)
        # Reuse the same validation as custom asset surfaces.
        surface = PlaneSurface(self._corners(), texture)
        object.__setattr__(self, "texture", surface.texture)

    def _corners(self):
        w, h = self.width_m / 2, self.height_m / 2
        # Viewed from +X with +Z up, screen-right is +Y.
        return np.array([[0, -w, h], [0, w, h], [0, w, -h], [0, -w, -h]])

    def planes(self) -> tuple[PlaneSurface, ...]:
        return (PlaneSurface(self._corners(), self.texture),)


_ANCHORS = {
    "center": (0, 0, 0),
    "front_center": (0.5, 0, 0),
    "rear_center": (-0.5, 0, 0),
    "left_center": (0, 0.5, 0),
    "right_center": (0, -0.5, 0),
    "top_center": (0, 0, 0.5),
    "bottom_center": (0, 0, -0.5),
}


@dataclass(frozen=True)
class AttachedPlacement:
    """A user-specified host-local point. Named anchors refer to bounding-box faces."""

    host: str
    anchor: str = "center"
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a nonempty logical actor ID")
        if self.anchor not in _ANCHORS:
            raise ValueError(f"Unknown anchor {self.anchor!r}. Choose from {tuple(_ANCHORS)}")
        object.__setattr__(self, "offset_m", tuple(_vector(self.offset_m, "offset_m")))


@dataclass(frozen=True)
class WorldPlacement:
    """A fixed position in the canonical world frame, in metres."""

    position_m: tuple[float, float, float]

    def __post_init__(self):
        object.__setattr__(self, "position_m", tuple(_vector(self.position_m, "position_m")))


@dataclass(frozen=True)
class Orientation:
    """Rotation relative to a host, absolute in world, or directed at the victim.

    ``face_victim`` points the asset's +X axis at the victim-local ``target_offset_m``.
    It keeps +Z as close to world up as possible. Directly vertical directions use world +Y
    as the secondary up reference. Explicit Euler angles apply to the other two modes only.
    """

    mode: str = "fixed_world"
    rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.mode not in {"follow_host", "fixed_world", "face_victim"}:
            raise ValueError(f"Unknown orientation mode {self.mode!r}")
        rotation = _vector(self.rotation_deg, "rotation_deg")
        offset = _vector(self.target_offset_m, "target_offset_m")
        if self.mode == "face_victim" and np.any(rotation != 0):
            raise ValueError("face_victim determines orientation, so rotation_deg must be zero")
        if self.mode != "face_victim" and np.any(offset != 0):
            raise ValueError("target_offset_m is only used by face_victim")
        object.__setattr__(self, "rotation_deg", tuple(rotation))
        object.__setattr__(self, "target_offset_m", tuple(offset))


@dataclass(frozen=True)
class Insertion:
    """An independently configured object. No placement coordinates are searched or adjusted."""

    id: str
    asset: PlanarAsset
    placement: AttachedPlacement | WorldPlacement
    orientation: Orientation = field(default_factory=Orientation)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("insertion ID must be a nonempty string")
        if not callable(getattr(self.asset, "planes", None)):
            raise TypeError("asset must expose local planes()")
        if not isinstance(self.placement, (AttachedPlacement, WorldPlacement)):
            raise TypeError("placement must be AttachedPlacement or WorldPlacement")
        if self.orientation.mode == "follow_host" and not isinstance(
            self.placement, AttachedPlacement
        ):
            raise ValueError("follow_host orientation requires an attached placement")


@dataclass(frozen=True)
class ResolvedInsertion:
    """An insertion's world pose at one instant, shared by rendering and selection."""

    id: str
    pose: np.ndarray
    asset: PlanarAsset
    host_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "pose", _transform(self.pose))

    def planes(self) -> tuple[PlaneSurface, ...]:
        rotation, translation = self.pose[:3, :3], self.pose[:3, 3]
        return tuple(
            PlaneSurface(surface.corners @ rotation.T + translation, surface.texture)
            for surface in self.asset.planes()
        )


def resolve_insertion(
    insertion: Insertion,
    actors: Mapping[str, ActorPose],
    victim: ActorPose,
) -> ResolvedInsertion:
    """Resolve a user-specified installation against current actor poses without mutation.

    Missing hosts raise ``KeyError``. They are never replaced by another nearby actor.
    The same operation is used during initial screening and during insertion rendering.
    """
    placement, orientation = insertion.placement, insertion.orientation
    host = None
    if isinstance(placement, AttachedPlacement):
        if placement.host not in actors:
            raise KeyError(f"Insertion {insertion.id!r} requires missing host {placement.host!r}")
        host = actors[placement.host]
        local = np.asarray(_ANCHORS[placement.anchor]) * host.extent + placement.offset_m
        position = host.transform[:3, :3] @ local + host.transform[:3, 3]
    else:
        position = np.asarray(placement.position_m)

    if orientation.mode == "face_victim":
        target = victim.transform[:3, :3] @ orientation.target_offset_m + victim.transform[:3, 3]
        forward = target - position
        distance = np.linalg.norm(forward)
        if distance < 1e-9:
            raise ValueError("Cannot face a victim reference point coincident with the insertion")
        forward = forward / distance
        up = np.array([0.0, 0.0, 1.0])
        if abs(forward @ up) > 1 - 1e-8:
            up = np.array([0.0, 1.0, 0.0])
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        rotation = np.column_stack((forward, left, np.cross(forward, left)))
    else:
        rotation = rotation_matrix(orientation.rotation_deg)
        if orientation.mode == "follow_host":
            rotation = host.transform[:3, :3] @ rotation

    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return ResolvedInsertion(
        insertion.id,
        pose,
        insertion.asset,
        placement.host if isinstance(placement, AttachedPlacement) else None,
    )
