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


from avsectester.geometry import poses
from avsectester.rendering.objects import PlaneSurface
from avsectester.rendering import images
from avsectester.attack.placement import AttachedPlacement, WorldPlacement, Orientation


@dataclass(frozen=True)
class ImageTexture:
    """An image loaded from an explicit image path."""

    image: str | Path

    def __post_init__(self) -> None:
        if not isinstance(self.image, (str, Path)):
            raise TypeError("image must be a file path")
        object.__setattr__(self, "_image", images.load_rgba_image(self.image))

    def rgba(self) -> np.ndarray:
        return self._image.copy()


@dataclass(frozen=True)
class CheckerboardTexture:
    """An opaque square checkerboard with an explicit pixel size and cell count."""

    size: int = 256
    cells: int = 8
    colors: tuple = ((255, 0, 0), (255, 255, 255))

    def __post_init__(self) -> None:
        for name in ("size", "cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.cells > self.size:
            raise ValueError("cells cannot exceed the texture size")
        colors = np.asarray(self.colors)
        if (colors.shape != (2, 3) or not np.issubdtype(colors.dtype, np.integer)
                or ((colors < 0) | (colors > 255)).any()):
            raise ValueError("colors must contain two RGB integer colors in [0, 255]")

    def rgba(self) -> np.ndarray:
        positions = np.arange(self.size) * self.cells // self.size
        pattern = (positions[:, None] + positions[None, :]) % 2
        rgb = np.asarray(self.colors, dtype=np.uint8)[pattern]
        return np.concatenate((rgb, np.full((self.size, self.size, 1), 255, np.uint8)), axis=2)


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

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.width_m, self.height_m)):
            raise ValueError("asset width and height must be finite and positive")
        texture = images.load_rgba_image(self.texture)
        # Reuse the same validation as custom asset surfaces.
        surface = PlaneSurface(self._corners(), texture)
        object.__setattr__(self, "texture", surface.texture)

    def _corners(self):
        w, h = self.width_m / 2, self.height_m / 2
        # Viewed from +X with +Z up, screen-right is +Y.
        return np.array([[0, -w, h], [0, w, h], [0, w, -h], [0, -w, -h]])

    def planes(self) -> tuple[PlaneSurface, ...]:
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
        rotation, translation = self.pose[:3, :3], self.pose[:3, 3]
        return tuple(
            PlaneSurface(surface.corners @ rotation.T + translation, surface.texture)
            for surface in self.asset.planes()
        )


def resolve_insertion(
    insertion: Insertion,
    actors: Mapping[str, poses.ActorPose],
    ego: poses.ActorPose,
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
        position = host.transform[:3, :3] @ local + host.transform[:3, 3]
    else:
        position = np.asarray(placement.position_m)

    if orientation.mode == "face_ego":
        target = ego.transform[:3, :3] @ orientation.target_offset_m + ego.transform[:3, 3]
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
            rotation = host.transform[:3, :3] @ rotation

    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return ResolvedInsertion(
        insertion.id,
        pose,
        insertion.asset,
        placement.host if isinstance(placement, AttachedPlacement) else None,
    )
