"""Explicit insertion geometry, independent of a simulator or attack implementation.

World and actor frames use meters and right-handed x-forward, y-left, z-up axes. An actor
pose identifies its origin; box_transform() locates its bounding-box center. Assets use the same axes, with their front along +X.
Placement determines position, while orientation independently determines rotation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np


from avsectester.geometry import poses
from avsectester.data_types import ActorState
from avsectester.rendering.objects import PlaneSurface
from avsectester.rendering import textures
from avsectester.attack.insertion.placement import AttachedPlacement, WorldPlacement, Orientation


class PlanarAsset(Protocol):
    """Extension point for assets composed of local textured rectangular surfaces."""

    def planes(self) -> Sequence[PlaneSurface]:
        """Resolve the asset into textured planes in its local coordinate frame."""
        ...


@dataclass(frozen=True)
class PlaneAsset:
    """RGBA patch or sign, centered at the origin in the local YZ plane.

    The front normal is +X and the top is +Z. Transparent pixels are not part of the silhouette.
    A file path is loaded once at construction. Image loading requires Pillow.
    """

    texture: np.ndarray | str | Path
    width_m: float
    height_m: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.width_m, self.height_m)):
            raise ValueError("asset width and height must be finite and positive")
        texture = textures.load_rgba_image(self.texture)
        # Reuse the same validation as custom asset surfaces.
        surface = PlaneSurface(self._corners(), texture)
        object.__setattr__(self, "texture", surface.texture)

    def _corners(self):
        w, h = self.width_m / 2, self.height_m / 2
        # Viewed from +X with +Z up, screen-right is +Y.
        return np.array([[0, -w, h], [0, w, h], [0, w, -h], [0, -w, -h]])

    def planes(self) -> tuple[PlaneSurface, ...]:
        """Resolve the asset into textured planes in its local coordinate frame."""
        return (PlaneSurface(self._corners(), self.texture),)


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
        object.__setattr__(self, "pose", poses.validated_transform(self.pose))

    def planes(self) -> tuple[PlaneSurface, ...]:
        """Return textured world-coordinate planes transformed by the insertion pose."""
        rotation, translation = self.pose[:3, :3], self.pose[:3, 3]
        return tuple(
            replace(surface, corners=surface.corners @ rotation.T + translation)
            for surface in self.asset.planes()
        )


def resolve_insertion(
    insertion: Insertion,
    actors: Mapping[str, ActorState],
    ego: ActorState,
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
        if host.extent is None:
            raise ValueError(f"Insertion host {placement.host!r} bounding-box extent is unavailable")
        local = placement.local_position(host.extent)
        position = host.box_transform()[:3, :3] @ local + host.box_transform()[:3, 3]
    else:
        position = np.asarray(placement.position_m)

    if orientation.mode == "face_ego":
        target = ego.box_transform()[:3, :3] @ orientation.target_offset_m + ego.box_transform()[:3, 3]
        forward = target - position
        distance = np.linalg.norm(forward)
        if distance < 1e-9:
            raise ValueError("Cannot face an ego reference point coincident with the insertion")
        forward = forward / distance
        up = np.array([0.0, 0.0, 1.0])
        if abs(forward @ up) > 1 - 1e-8:
            up = np.array([0.0, 1.0, 0.0])
        left = np.cross(up, forward)
        left /= np.linalg.norm(left)
        rotation = np.column_stack((forward, left, np.cross(forward, left)))
    else:
        rotation = poses.rotation_matrix(orientation.rotation_deg)
        if orientation.mode == "follow_host":
            rotation = host.box_transform()[:3, :3] @ rotation

    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return ResolvedInsertion(
        insertion.id,
        pose,
        insertion.asset,
        placement.host if isinstance(placement, AttachedPlacement) else None,
    )


__all__ = [
    'PlanarAsset',
    'PlaneAsset',
    'Insertion',
    'ResolvedInsertion',
    'resolve_insertion',
]
