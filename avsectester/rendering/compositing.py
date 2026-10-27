"""Depth-aware composition of world objects, differentiable in surface colors, and visible-region blending."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from avsectester.geometry.cameras import Camera, camera_ray_mask
from avsectester.data_types import ActorState
from avsectester.geometry.boxes import box_ray_distances
from .harmonizers import Harmonizer
from .objects import PlacedObject
from .surfaces import PlaneSampling, _render_region, prepare_surface, sample_texture

if TYPE_CHECKING:
    import torch


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


@dataclass(frozen=True)
class SurfaceFootprint:
    """One surface's sampling over its clipped image region and the visible pixels it owns."""

    object_index: int
    surface_index: int
    sampling: PlaneSampling
    alpha: np.ndarray
    premultiplied: np.ndarray
    owned: np.ndarray


def _object_mask(
    footprints: Sequence[SurfaceFootprint], index: int, shape: tuple[int, ...],
) -> np.ndarray:
    """The visible pixels owned by one planned object's surfaces."""
    mask = np.zeros(shape, dtype=bool)
    for footprint in footprints:
        if footprint.object_index == index:
            x0, y0, x1, y1 = footprint.sampling.bounds
            mask[y0:y1, x0:x1] |= footprint.owned
    return mask


@dataclass(frozen=True)
class CompositePlan:
    """Texture-independent geometry and composition for one camera image.

    Sampling, coverage, nearest-surface ownership and occlusion are fixed when planning; only
    surface colors remain free. Apply and composite share the floating-point blend and all
    differentiable harmonizers, differing only in final 8-bit rounding. Non-differentiable
    harmonizers support deployment through apply only. Each object's harmonizer sees the same
    unadjusted composite and original background, and changes only its visible pixels.
    Differentiable harmonizers also see the nominal composite, in which surfaces with a
    nominal texture show it instead of their texture or replacement colors.
    Depth combines scene and surface evidence; scene_visibility keeps only visible surfaces.
    """

    objects: tuple[PlacedObject, ...]
    depth: np.ndarray
    visible_mask: np.ndarray
    occlusion_source: str
    ignored_occluders: tuple[str, ...]
    object_visibility: Mapping[str, ObjectVisibility]
    footprints: tuple[SurfaceFootprint, ...] = field(repr=False)

    def composite(self, image: torch.Tensor,
                  textures: Mapping[tuple[str, int], torch.Tensor] | None = None) -> torch.Tensor:
        """Blend surfaces and differentiable harmonizers over a (3, H, W) image in [0, 1].

        Textures maps (object_id, surface index) to straight RGB (3, h, w) colors at the
        surface texture's size. Asset alpha preserves its shape; omitted surfaces keep their
        asset colors. Gradients reach the image and replacement colors, not planned geometry.
        Each harmonizer receives the same unadjusted blend, its visible mask, the original
        background and the nominal blend. Non-differentiable harmonizers make this operation
        unavailable.
        """
        for placed in self.objects:
            harmonizer = placed.harmonizer
            if harmonizer is not None and not getattr(harmonizer, "differentiable", False):
                raise ValueError(
                    f"Harmonizers are not differentiable: object {placed.object_id!r} uses "
                    f"{type(harmonizer).__name__}")
        textures = dict(textures or {})
        return self._harmonize_tensor(self._blend(image, textures), image, textures)

    def _harmonize_tensor(self, blended: torch.Tensor, background: torch.Tensor,
                          textures: dict[tuple[str, int], torch.Tensor]) -> torch.Tensor:
        """Apply differentiable harmonizers to independent copies of one unadjusted blend."""
        import torch

        harmonized = [(index, placed, placed.harmonizer) for index, placed in enumerate(self.objects)
                      if placed.harmonizer is not None and getattr(placed.harmonizer, "differentiable", False)]
        nominal_textures = {
            (placed.object_id, surface_index): torch.as_tensor(
                surface.nominal_texture[..., :3].copy(), dtype=background.dtype, device=background.device,
            ).permute(2, 0, 1) / 255
            for _, placed, _ in harmonized
            for surface_index, surface in enumerate(placed.asset.surfaces)
            if surface.nominal_texture is not None
        }
        nominal = (self._blend(background, {**textures, **nominal_textures}) if nominal_textures
                   else blended)
        output = blended
        for index, _, harmonizer in harmonized:
            active = torch.as_tensor(
                _object_mask(self.footprints, index, self.depth.shape), device=blended.device)
            if active.any():
                adjusted = harmonizer.harmonize_tensor(
                    blended.clone(), active, background.clone(), nominal.clone())
                output = torch.where(active[None], adjusted, output)
        return output

    def apply(self, image: np.ndarray) -> CompositeResult:
        """Deploy uint8 RGB, rounding once after differentiable composition.

        Differentiable harmonizers use the same floating-point calculation as composite.
        Non-differentiable harmonizers run afterwards through their NumPy interface, each seeing
        the rounded unadjusted composite and original background.
        """
        import torch

        image = np.asarray(image)
        if image.shape != (*self.depth.shape, 3) or image.dtype != np.uint8:
            raise ValueError("image must be calibrated uint8 RGB with shape (H, W, 3)")
        with torch.no_grad():
            values = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255
            blended = self._blend(values, {})
            harmonized = self._harmonize_tensor(blended, values, {}).permute(1, 2, 0).numpy()
            unadjusted = blended.permute(1, 2, 0).numpy()
        output = image.copy()
        visible = self.visible_mask
        output[visible] = np.rint(np.clip(harmonized[visible], 0, 1) * 255).astype(np.uint8)
        numpy_harmonized = [
            (index, placed) for index, placed in enumerate(self.objects)
            if placed.harmonizer is not None and not getattr(placed.harmonizer, "differentiable", False)]
        if numpy_harmonized:
            unharmonized = image.copy()
            unharmonized[visible] = np.rint(np.clip(unadjusted[visible], 0, 1) * 255).astype(np.uint8)
            for index, placed in numpy_harmonized:
                active = _object_mask(self.footprints, index, self.depth.shape)
                _harmonize_visible(output, active, image, placed.harmonizer, unharmonized)
        return CompositeResult(output, self.depth, visible, self.occlusion_source,
                               self.ignored_occluders, self.object_visibility)

    def _blend(self, image: torch.Tensor, textures: dict[tuple[str, int], torch.Tensor]) -> torch.Tensor:
        import torch

        if image.ndim != 3 or tuple(image.shape) != (3, *self.depth.shape):
            raise ValueError("image must be a (3, H, W) tensor at the planned camera size")
        assets = {placed.object_id: placed.asset for placed in self.objects}
        for key, colors in textures.items():
            if (not isinstance(key, tuple) or len(key) != 2 or key[0] not in assets
                    or not isinstance(key[1], int) or not 0 <= key[1] < len(assets[key[0]].surfaces)):
                raise KeyError(f"Unknown planned surface {key!r}; keys are (object_id, surface index)")
            texture = assets[key[0]].surfaces[key[1]].texture
            if not isinstance(colors, torch.Tensor) or tuple(colors.shape) != (3, *texture.shape[:2]):
                raise ValueError(f"Colours for {key!r} must be a (3, {texture.shape[0]}, {texture.shape[1]}) tensor")
        layer = image.new_zeros((4, *image.shape[1:]))
        for footprint in self.footprints:
            colors = textures.get((self.objects[footprint.object_index].object_id, footprint.surface_index))
            if colors is None:
                sampled = torch.as_tensor(footprint.premultiplied, dtype=image.dtype, device=image.device)
            else:
                alpha = torch.as_tensor(footprint.alpha, dtype=colors.dtype, device=colors.device)[None]
                sampled = sample_texture(torch.cat((colors * alpha, alpha)), footprint.sampling)
                sampled = sampled.to(dtype=image.dtype, device=image.device)
            x0, y0, x1, y1 = footprint.sampling.bounds
            owned = torch.as_tensor(footprint.owned, device=image.device)
            layer[:, y0:y1, x0:x1] += sampled * owned
        return image * (1 - layer[3:]) + layer[:3]


@dataclass(frozen=True)
class ObjectComposition:
    """How placed objects were composed into one capture: the image beneath them and the plan.

    ``plan.apply(background).image`` is the composed capture, and ``plan.composite`` over
    ``background`` reproduces it differentiably in surface colors.
    """

    background: np.ndarray
    plan: CompositePlan


def plan_composite(
    camera: Camera, cam_from_world: np.ndarray, objects: Sequence[PlacedObject], *,
    scene_depth: np.ndarray | None = None, occluders: Mapping[str, ActorState] | None = None,
    occlusion_tolerance_m: float = 0.1, scene_visibility: np.ndarray | None = None,
) -> CompositePlan:
    """Plan the nearest object surfaces and their unoccluded pixels in one camera.

    ``scene_visibility``, when supplied, is the sole occlusion evidence: a boolean image that is
    true where the scene does not hide these objects, measured for example by rendering them
    inside the scene. It excludes ``scene_depth`` and ``occluders``, and the plan's depth then
    holds only the visible object surfaces. Otherwise aligned scene z-depth, when supplied, is
    the sole occlusion evidence; without it, actor cuboids with known extents approximate the
    scene inside the candidate projection regions. Actor IDs without known extents are returned
    as ``ignored_occluders``. Surface order breaks exact depth ties, so the result does not
    depend on object order otherwise.
    """
    shape = (camera.height, camera.width)
    if not np.isfinite(occlusion_tolerance_m) or occlusion_tolerance_m < 0:
        raise ValueError("occlusion_tolerance_m must be finite and nonnegative")
    if scene_visibility is not None:
        if scene_depth is not None or occluders:
            raise ValueError("scene_visibility excludes scene_depth and occluders")
        scene_visibility = np.asarray(scene_visibility)
        if scene_visibility.shape != shape or scene_visibility.dtype != bool:
            raise ValueError("scene_visibility must be a boolean image at the calibrated dimensions")
    objects = tuple(objects)
    if any(not isinstance(item, PlacedObject) for item in objects):
        raise TypeError("objects must contain PlacedObject values")
    if len({item.object_id for item in objects}) != len(objects):
        raise ValueError("Placed object IDs must be unique")
    nearest = np.full(shape, np.inf)
    owners = np.full(shape, -1, dtype=np.int64)
    candidate_pixels = np.zeros(shape, dtype=bool)
    regions = []
    for object_index, placed in enumerate(objects):
        rows, columns = [], []
        for surface_index, surface in enumerate(placed.world_surfaces()):
            sampler = prepare_surface(surface, camera, cam_from_world)
            bounds = sampler.image_bounds(camera)
            if bounds is None:
                continue
            x0, y0, x1, y1 = bounds
            sampling, premultiplied, depth = _render_region(sampler, camera, bounds)
            covered_rows, covered_columns = np.nonzero(np.isfinite(depth))
            rows.append(covered_rows + y0)
            columns.append(covered_columns + x0)
            region, region_owners = nearest[y0:y1, x0:x1], owners[y0:y1, x0:x1]
            closer = depth < region
            region[closer], region_owners[closer] = depth[closer], len(regions)
            regions.append((object_index, surface_index, sampling, surface.texture[..., 3] / 255,
                            premultiplied.transpose(2, 0, 1).astype(np.float32)))
        if rows and np.concatenate(rows).size:
            all_rows, all_columns = np.concatenate(rows), np.concatenate(columns)
            candidate_pixels[max(0, all_rows.min() - 2):min(shape[0], all_rows.max() + 3),
                             max(0, all_columns.min() - 2):min(shape[1], all_columns.max() + 3)] = True
    ignored_occluders = []
    if scene_visibility is not None:
        # Hidden pixels get a scene depth of zero, which no surface can be in front of.
        background_depth = np.where(scene_visibility, np.inf, 0.0)
        occlusion_source = "scene_visibility"
    elif scene_depth is not None:
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
    footprints = []
    for index, (object_index, surface_index, sampling, alpha, premultiplied) in enumerate(regions):
        x0, y0, x1, y1 = sampling.bounds
        owned = visible[y0:y1, x0:x1] & (owners[y0:y1, x0:x1] == index)
        if owned.any():
            footprints.append(SurfaceFootprint(
                object_index, surface_index, sampling, alpha, premultiplied, owned))
    ignored = tuple(ignored_occluders)
    visibility = {placed.object_id: ObjectVisibility(
        occlusion_source, ignored, int(np.count_nonzero(_object_mask(footprints, index, shape))))
        for index, placed in enumerate(objects)}
    depth = (np.where(visible, nearest, np.inf) if scene_visibility is not None
             else np.minimum(background_depth, nearest))
    return CompositePlan(objects, depth, visible, occlusion_source, ignored, visibility, tuple(footprints))


def composite_objects(
    image: np.ndarray, camera: Camera, cam_from_world: np.ndarray,
    objects: Sequence[PlacedObject], *, scene_depth: np.ndarray | None = None,
    occluders: Mapping[str, ActorState] | None = None,
    occlusion_tolerance_m: float = 0.1,
) -> CompositeResult:
    """Plan and apply a composite; see ``plan_composite`` and ``CompositePlan.apply``.

    Each object's harmonizer sees the same unharmonized composite and changes only that object's
    visible pixels.
    """
    image = np.asarray(image)
    if image.shape != (camera.height, camera.width, 3) or image.dtype != np.uint8:
        raise ValueError("image must be calibrated uint8 RGB with shape (H, W, 3)")
    return plan_composite(camera, cam_from_world, objects, scene_depth=scene_depth, occluders=occluders,
                          occlusion_tolerance_m=occlusion_tolerance_m).apply(image)


__all__ = [
    'SurfaceFootprint',
    'ObjectVisibility',
    'CompositeResult',
    'blend_rgba',
    'CompositePlan',
    'ObjectComposition',
    'plan_composite',
    'composite_objects',
]
