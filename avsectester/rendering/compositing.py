"""Depth-aware composition of world objects and shared visible-region appearance blending."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from avsectester.geometry.cameras import Camera, camera_ray_mask
from avsectester.data_types import ActorState
from avsectester.geometry.boxes import box_ray_distances
from .harmonizers import Harmonizer
from .objects import PlacedObject
from .surfaces import render_surfaces


@dataclass(frozen=True)
class ObjectVisibility:
    """One world object's camera visibility and the evidence used to measure it."""

    occlusion_source: str
    ignored_occluders: tuple[str, ...]
    visible_pixels: int


@dataclass(frozen=True)
class CompositeResult:
    """Composite pixels, optical z-depth, visibility, occlusion source and ignored_occluders.

    Cuboid depth is evaluated only inside candidate projection regions, padded by two pixels.
    Outside those regions it is infinity. Missing actor extents are reported in input order.
    """

    image: np.ndarray
    depth: np.ndarray
    visible_mask: np.ndarray
    occlusion_source: str
    ignored_occluders: tuple[str, ...] = ()
    object_visibility: Mapping[str, ObjectVisibility] = field(default_factory=dict)


def _harmonize_visible(output, active, background, harmonizer, source=None):
    if harmonizer is not None and active.any():
        composite = output if source is None else source
        mask = active.astype(np.uint8) * 255
        harmonized = harmonizer(composite.copy(), mask, background.copy())
        output[active] = harmonized[active]
    return output


def blend_rgba(
    image: np.ndarray, rgba: np.ndarray, visible_mask: np.ndarray, *,
    harmonizer: Harmonizer | None = None, soften: float = 0.0,
) -> np.ndarray:
    """Blend sampled RGBA only at visible pixels, optionally softening and harmonizing there.

    The caller owns the geometric mask. Alpha values, including values below 128, retain their
    original blending meaning. Appearance filtering never expands the supplied mask.
    """
    if rgba.shape != (*image.shape[:2], 4) or visible_mask.shape != image.shape[:2]:
        raise ValueError("RGBA samples and visibility must match the image dimensions")
    if not np.isfinite(soften) or soften < 0:
        raise ValueError("soften must be finite and nonnegative")
    if harmonizer is not None and not callable(harmonizer):
        raise TypeError("harmonizer must be callable or None")
    active = np.asarray(visible_mask, dtype=bool)
    if soften > 0:
        import cv2

        rgba = cv2.GaussianBlur(rgba.astype(np.float32), (0, 0), soften)
    output = image.copy()
    alpha = rgba[active, 3:4].astype(np.float32) / 255
    output[active] = (image[active] * (1 - alpha) + rgba[active, :3] * alpha).astype(np.uint8)
    return _harmonize_visible(output, active, image, harmonizer)


def composite_objects(
    image: np.ndarray, camera: Camera, cam_from_world: np.ndarray,
    objects: Sequence[PlacedObject], *, scene_depth: np.ndarray | None = None,
    occluders: Mapping[str, ActorState] | None = None,
    occlusion_tolerance_m: float = 0.1,
) -> CompositeResult:
    """Render the nearest object surfaces and composite their unoccluded pixels.

    Aligned scene z-depth, when supplied, is the sole occlusion evidence. Otherwise actor
    cuboids with known extents approximate the scene inside the candidate projection regions.
    Each object's harmonizer sees the same unharmonized composite and changes only that object's
    visible pixels. Actor IDs without known extents are returned as ``ignored_occluders``.
    """
    image = np.asarray(image)
    shape = (camera.height, camera.width)
    if image.shape != (*shape, 3) or image.dtype != np.uint8:
        raise ValueError("image must be calibrated uint8 RGB with shape (H, W, 3)")
    if not np.isfinite(occlusion_tolerance_m) or occlusion_tolerance_m < 0:
        raise ValueError("occlusion_tolerance_m must be finite and nonnegative")
    objects = tuple(objects)
    if any(not isinstance(item, PlacedObject) for item in objects):
        raise TypeError("objects must contain PlacedObject values")
    if len({item.object_id for item in objects}) != len(objects):
        raise ValueError("Placed object IDs must be unique")
    nearest = np.full(shape, np.inf)
    rgba = np.zeros((*shape, 4), np.uint8)
    owners = np.full(shape, -1, dtype=np.int64)
    candidate_pixels = np.zeros(shape, dtype=bool)
    for index, placed in enumerate(objects):
        colors, depth = render_surfaces(placed.world_surfaces(), camera, cam_from_world)
        rows, columns = np.nonzero(np.isfinite(depth))
        if columns.size:
            x0, x1 = max(0, columns.min() - 2), min(shape[1], columns.max() + 3)
            y0, y1 = max(0, rows.min() - 2), min(shape[0], rows.max() + 3)
            candidate_pixels[y0:y1, x0:x1] = True
        closer = depth < nearest
        nearest[closer], rgba[closer], owners[closer] = depth[closer], colors[closer], index
    ignored_occluders = []
    if scene_depth is not None:
        background_depth = np.asarray(scene_depth, dtype=float)
        if background_depth.shape != shape:
            raise ValueError("scene_depth must match the calibrated image dimensions")
        occlusion_source = "depth"
    else:
        image_y, image_x = np.nonzero(candidate_pixels)
        pixels = np.column_stack((image_x + .5, image_y + .5))
        rays = np.asarray(camera.unproject(pixels))
        valid = camera_ray_mask(camera, rays)
        world_from_cam = np.linalg.inv(cam_from_world)
        ray_depth = np.full(len(rays), np.inf)
        for actor_id, actor in (occluders or {}).items():
            if actor.extent is None:
                ignored_occluders.append(actor_id)
                continue
            distance = box_ray_distances(actor.box_transform(), actor.extent, rays, world_from_cam)
            hit = valid & np.isfinite(distance)
            ray_depth[hit] = np.minimum(ray_depth[hit], distance[hit] * rays[hit, 2])
        background_depth = np.full(shape, np.inf)
        background_depth[image_y, image_x] = ray_depth
        occlusion_source = "actor_cuboids"
    visible = (np.isfinite(nearest) & (background_depth > 0)
               & (nearest <= background_depth + occlusion_tolerance_m))
    output = blend_rgba(image, rgba, visible)
    unharmonized = output.copy()
    ignored = tuple(ignored_occluders)
    object_visibility = {}
    for index, placed in enumerate(objects):
        active = visible & (owners == index)
        _harmonize_visible(output, active, image, placed.harmonizer, unharmonized)
        object_visibility[placed.object_id] = ObjectVisibility(
            occlusion_source, ignored, int(np.count_nonzero(active)))
    return CompositeResult(output, np.minimum(background_depth, nearest), visible, occlusion_source,
                           ignored, object_visibility)
