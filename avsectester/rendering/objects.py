"""Textured world-object descriptions, independent of simulators and attacks."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avsectester.validation import require_nonnegative_integer
from avsectester.geometry.poses import validated_transform
from .harmonizers import Harmonizer
from .textures import Texture, load_rgba_image


def _post_texture(height_pixels: int = 256, width_pixels: int = 16,
              color: tuple = (150, 152, 155)) -> np.ndarray:
    """A galvanized sign post texture: flat grey, shaded across its width like a round/U-channel post."""
    shade = 0.65 + 0.35 * np.sin(np.linspace(0.15, math.pi - 0.15, width_pixels))  # darker at the edges
    img = np.zeros((height_pixels, width_pixels, 4), np.uint8)
    img[..., :3] = (np.asarray(color, np.float32)[None, None, :] * shade[None, :, None]).astype(np.uint8)
    img[..., 3] = 255
    return img


@dataclass(frozen=True)
class PlaneSurface:
    """A textured rectangle, with corners ordered TL, TR, BR, BL from its front.

    Asset surfaces use local coordinates; resolved surfaces use world coordinates.
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


@dataclass(frozen=True)
class ObjectAsset:
    """Local rectangular faces in x-forward, y-left, z-up axes, with bottom-centre origin.

    ``extent`` gives full (length, width, height). The horizontal footprint is centred on
    the origin and the bottom is z=0. A planar object may have zero extent on one axis.
    """

    surfaces: tuple[PlaneSurface, ...]
    extent: tuple[float, float, float]
    category: str

    def __post_init__(self) -> None:
        surfaces = tuple(self.surfaces)
        if not surfaces or any(not isinstance(surface, PlaneSurface) for surface in surfaces):
            raise ValueError("ObjectAsset requires at least one PlaneSurface")
        if not isinstance(self.category, str) or not self.category.strip():
            raise ValueError("ObjectAsset.category must be a nonempty string")
        extent = np.asarray(self.extent, dtype=float)
        if extent.shape != (3,) or not np.isfinite(extent).all() or (extent < 0).any():
            raise ValueError("ObjectAsset.extent must contain three finite nonnegative dimensions")
        corners = np.concatenate([surface.corners for surface in surfaces])
        lower = np.array([-extent[0] / 2, -extent[1] / 2, 0.])
        upper = np.array([extent[0] / 2, extent[1] / 2, extent[2]])
        if np.any(corners < lower - 1e-6) or np.any(corners > upper + 1e-6):
            raise ValueError("ObjectAsset surfaces must lie inside its bottom-centred bounding box")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(self, "extent", tuple(float(value) for value in extent))

    @classmethod
    def from_surfaces(cls, surfaces: Sequence[PlaneSurface], category: str) -> ObjectAsset:
        """Infer full extent from local corners without moving the bottom-centre origin."""
        surfaces = tuple(surfaces)
        if not surfaces or any(not isinstance(surface, PlaneSurface) for surface in surfaces):
            raise ValueError("ObjectAsset requires at least one PlaneSurface")
        corners = np.concatenate([surface.corners for surface in surfaces])
        extent = (2 * np.max(np.abs(corners[:, 0])), 2 * np.max(np.abs(corners[:, 1])),
                  np.max(corners[:, 2]))
        return cls(surfaces, extent, category)


    @classmethod
    def from_panel(
        cls, texture: np.ndarray | str | Path | Texture, *, category: str, width_m: float,
        mount_height_m: float = 0.0, post_count: int = 0, post_width_m: float = 0.07,
    ) -> ObjectAsset:
        """Build a +X-facing panel and optional support posts above a ground-level origin.

        The image's aspect ratio determines the face height. Posts stand 3 cm behind
        the face and reach its centre; several posts are spaced across +/-35% of its
        width. Surfaces list the posts first and the panel last.
        """
        face = load_rgba_image(texture)
        if not np.isfinite([width_m, mount_height_m, post_width_m]).all():
            raise ValueError("Panel dimensions must be finite")
        if width_m <= 0 or mount_height_m < 0 or post_width_m <= 0:
            raise ValueError("Panel width_m and post_width_m must be positive, mount_height_m nonnegative")
        require_nonnegative_integer(post_count, "post_count")
        height = width_m * face.shape[0] / face.shape[1]

        def rectangle(x, y, width, bottom, top, image):
            return PlaneSurface(np.array([
                [x, y - width / 2, top], [x, y + width / 2, top],
                [x, y + width / 2, bottom], [x, y - width / 2, bottom],
            ]), image)

        surfaces = []
        if post_count:
            offsets = [0.0] if post_count == 1 else np.linspace(-0.35, 0.35, post_count) * width_m
            for offset in offsets:
                surfaces.append(rectangle(-0.03, offset, post_width_m, 0,
                                          mount_height_m + height / 2, _post_texture()))
        surfaces.append(rectangle(0, 0, width_m, mount_height_m,
                                  mount_height_m + height, face))
        return cls.from_surfaces(surfaces, category)


@dataclass(frozen=True)
class PlacedObject:
    """One asset at a rigid world-from-object transform with its own optional harmonizer."""

    object_id: str
    asset: ObjectAsset
    world_from_object: np.ndarray
    harmonizer: Harmonizer | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("PlacedObject.object_id must be a nonempty string")
        if not isinstance(self.asset, ObjectAsset):
            raise TypeError("PlacedObject.asset must be ObjectAsset")
        if self.harmonizer is not None and not callable(self.harmonizer):
            raise TypeError("PlacedObject.harmonizer must be callable or None")
        object.__setattr__(self, "world_from_object", validated_transform(self.world_from_object))

    def world_surfaces(self) -> tuple[PlaneSurface, ...]:
        """The asset's surfaces with corners in world coordinates."""
        rotation, translation = self.world_from_object[:3, :3], self.world_from_object[:3, 3]
        return tuple(PlaneSurface(surface.corners @ rotation.T + translation, surface.texture)
                     for surface in self.asset.surfaces)
