"""Depth-based visibility and explicitly approximate cuboid visibility for insertions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import chain
import numpy as np

from avsectester.data_types import ActorState
from avsectester.validation import require_positive_integer
from .assets import ResolvedInsertion
from avsectester.geometry.cameras import Camera, camera_ray_mask
from avsectester.geometry.boxes import box_ray_distances
from avsectester.rendering.surfaces import prepare_surface, rasterize
from avsectester.rendering import PlacedObject
from .types import Visibility, VisibilityEvidence


class _VisibilityEstimator:
    def __init__(self, *, tolerance_m: float = 0.002, max_reference_pixels: int = 4_000_000):
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("tolerance_m must be finite and nonnegative")
        require_positive_integer(max_reference_pixels, "max_reference_pixels")
        self.tolerance_m = tolerance_m
        self.max_reference_pixels = max_reference_pixels

    def _prepare(self, subject, camera, cam_from_world, source, camera_name):
        projection, reason = rasterize(subject.planes(), camera, cam_from_world, self.max_reference_pixels)
        shape = (camera.height, camera.width)
        reference = np.zeros(shape, dtype=bool)
        rgba = np.zeros((*shape, 4), np.uint8)
        depth = np.full(shape, np.inf)
        if projection is None:
            visibility = (
                Visibility(0.0, source, camera_name)
                if reason == "Target is behind the camera"
                else None
            )
            return None, VisibilityEvidence(
                visibility, reference, reference.copy(), depth, 0, reason
            )
        count = int(projection.reference.sum())
        if count == 0:
            return None, VisibilityEvidence(
                None, reference, reference.copy(), depth, 0, "No target pixels"
            )
        height, width = projection.reference.shape
        ys, xs = np.nonzero(projection.reference)
        image_x, image_y = xs + projection.x0, ys + projection.y0
        valid = (
            (image_x >= 0) & (image_x < camera.width) & (image_y >= 0) & (image_y < camera.height)
        )
        ray_indices = ys * width + xs
        rays = projection.rays[ray_indices]
        valid &= camera_ray_mask(camera, rays)
        selected = ray_indices[valid]
        image_x, image_y = image_x[valid], image_y[valid]
        reference[image_y, image_x] = True
        rgba[image_y, image_x] = projection.rgba[selected]
        depth[image_y, image_x] = projection.depth[selected]
        state = (projection, selected, image_x, image_y)
        return state, VisibilityEvidence(
            None, reference, reference.copy(), depth, count, sampled_rgba=rgba
        )

    @staticmethod
    def _finish(evidence, visible, source, camera_name, reason=""):
        visibility = (
            None
            if reason
            else Visibility(float(visible.sum() / evidence.reference_pixels), source, camera_name)
        )
        return VisibilityEvidence(
            visibility,
            evidence.reference_mask,
            visible,
            evidence.target_depth,
            evidence.reference_pixels,
            reason,
            evidence.sampled_rgba,
            evidence.ignored_occluders,
        )


def _surface_occlusion_distances(
    subject_id, other_insertions, world_objects, camera, cam_from_world, rays,
):
    proposed = (surface for other in other_insertions if other.id != subject_id
                for surface in other.planes())
    inserted = (surface for placed in world_objects for surface in placed.world_surfaces())
    distance = np.full(len(rays), np.inf)
    for surface in chain(proposed, inserted):
        distance = np.minimum(distance, prepare_surface(surface, camera, cam_from_world).sample(rays)[0])
    return distance


class CuboidVisibilityEstimator(_VisibilityEstimator):
    """Estimate opacity occlusion by known 3D boxes and other proposed planar insertions.

    Keep the host in ``occluders`` so that body panels can obscure their mounted objects. Missing
    buildings, vegetation and other unlabeled geometry are not inferred. Result source is always
    ``actor_cuboids``. Proposed insertions and world-object surfaces use their alpha masks.
    Actor IDs without known extents are skipped and recorded in the evidence.
    """

    def estimate(
        self,
        subject: ResolvedInsertion,
        camera: Camera,
        cam_from_world: np.ndarray,
        *,
        occluders: Mapping[str, ActorState],
        camera_name: str,
        other_insertions: Sequence[ResolvedInsertion] = (),
        world_objects: Sequence[PlacedObject] = (),
    ) -> VisibilityEvidence:
        """Measure target visibility using the configured camera and occlusion evidence."""
        source = "actor_cuboids"
        state, evidence = self._prepare(subject, camera, cam_from_world, source, camera_name)
        evidence = replace(evidence, ignored_occluders=tuple(
            identity for identity, actor in occluders.items() if actor.extent is None))
        if state is None:
            return evidence
        projection, indices, xs, ys = state
        rays = projection.rays[indices]
        blocking_depth = np.full(len(rays), np.inf)
        world_from_cam = np.linalg.inv(cam_from_world)
        # Native actor IDs and insertion IDs belong to different namespaces. Even an
        # identically named host must remain an occluder of its mounted surface.
        for actor in occluders.values():
            if actor.extent is None:
                continue
            blocking_depth = np.minimum(
                blocking_depth, box_ray_distances(actor.box_transform(), actor.extent, rays, world_from_cam)
            )
        blocking_depth = np.minimum(blocking_depth, _surface_occlusion_distances(
            subject.id, other_insertions, world_objects, camera, cam_from_world, rays))
        visible = np.zeros_like(evidence.reference_mask)
        target = projection.depth[indices] / rays[:, 2]
        visible[ys, xs] = target <= blocking_depth + self.tolerance_m
        return self._finish(evidence, visible, source, camera_name)


class DepthVisibilityEstimator(_VisibilityEstimator):
    """Compare known insertion surfaces with calibrated, aligned scene depth.

    ``depth_convention`` must be ``'z'`` for optical-axis depth or ``'range'`` for ray distance.
    The depth image must represent the same camera pose, lens and instant as the target geometry.
    Positive infinity means no scene intersection. NaN or nonpositive depth at a target pixel
    makes the result unknown rather than silently visible.
    """

    def __init__(self, *, tolerance_m: float = 0.0002, max_reference_pixels: int = 4_000_000):
        super().__init__(tolerance_m=tolerance_m, max_reference_pixels=max_reference_pixels)

    def estimate(
        self,
        subject: ResolvedInsertion,
        camera: Camera,
        cam_from_world: np.ndarray,
        *,
        scene_depth: np.ndarray,
        depth_convention: str,
        camera_name: str,
        other_insertions: Sequence[ResolvedInsertion] = (),
    ) -> VisibilityEvidence:
        """Measure target visibility using the configured camera and occlusion evidence."""
        if depth_convention not in {"z", "range"}:
            raise ValueError("depth_convention must be 'z' or 'range'")
        scene_depth = np.asarray(scene_depth, dtype=float)
        if scene_depth.shape != (camera.height, camera.width):
            raise ValueError("scene_depth must have the calibrated camera's image shape")
        source = "depth"
        state, evidence = self._prepare(subject, camera, cam_from_world, source, camera_name)
        if state is None:
            return evidence
        projection, indices, xs, ys = state
        measured = scene_depth[ys, xs]
        visible = np.zeros_like(evidence.reference_mask)
        if np.any(np.isnan(measured) | (measured <= 0)):
            return self._finish(
                evidence, visible, source, camera_name, "Invalid depth at target pixels"
            )
        rays = projection.rays[indices]
        target_z = projection.depth[indices]
        # Both conventions compare the nearest covering surface along each pixel's own rays.
        target = target_z if depth_convention == "z" else target_z / rays[:, 2]
        visible[ys, xs] = target <= measured + self.tolerance_m
        distance = _surface_occlusion_distances(
            subject.id, other_insertions, (), camera, cam_from_world, rays)
        visible[ys, xs] &= target_z / rays[:, 2] <= distance + self.tolerance_m
        return self._finish(evidence, visible, source, camera_name)


def _image_mask(shape, image_bounds):
    if image_bounds is None:
        return np.ones(shape, dtype=bool)
    if len(image_bounds) != 4 or any(not isinstance(v, (int, np.integer)) for v in image_bounds):
        raise ValueError("Image bounds must be integer (left, top, right, bottom)")
    left, top, right, bottom = image_bounds
    height, width = shape
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Image bounds must lie inside the reference canvas")
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def visibility_from_masks(
    reference, visible, *, camera: str, image_bounds=None
) -> Visibility | None:
    """Divide visible in-image pixels by the FULL unoccluded reference silhouette.

    Both masks use the reference canvas. ``image_bounds=None`` means that the camera covers the
    whole canvas and the reference is known to fit. An empty reference is unknown, not zero.
    A nonempty reference entirely outside image_bounds has zero visibility.
    """
    reference, visible = np.asarray(reference, dtype=bool), np.asarray(visible, dtype=bool)
    if reference.ndim != 2 or visible.shape != reference.shape:
        raise ValueError("Visibility masks must be aligned two-dimensional arrays")
    in_image = _image_mask(reference.shape, image_bounds)
    denominator = int(reference.sum())
    if denominator == 0:
        return None
    return Visibility(
        float(np.count_nonzero(reference & visible & in_image) / denominator),
        source="rendered_masks",
        camera=camera,
    )


def visibility_from_depth(
    footprint,
    target_depth,
    scene_depth,
    *,
    camera: str,
    tolerance_m: float = 0.0002,
    image_bounds=None,
) -> Visibility | None:
    """Compare target and scene depth in the same convention and pixel projection.

    footprint is the FULL projected silhouette, including portions outside the camera image.
    target_depth may come from an isolated reference render or known surface geometry. The
    target and camera poses must match the scene measurement. Depth outside image_bounds is
    not needed and may be NaN. Target depth inside the in-image footprint must be finite and
    positive; positive infinite scene depth means no scene intersection.
    """
    footprint = np.asarray(footprint, dtype=bool)
    target_depth, scene_depth = np.asarray(target_depth), np.asarray(scene_depth)
    if (
        footprint.ndim != 2
        or target_depth.shape != footprint.shape
        or scene_depth.shape != footprint.shape
    ):
        raise ValueError("Footprint and depth maps must be aligned two-dimensional arrays")
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Depth tolerance must be finite and nonnegative")
    required = footprint & _image_mask(footprint.shape, image_bounds)
    if not np.all(
        np.isfinite(target_depth[required])
        & (target_depth[required] > 0)
        & ~np.isnan(scene_depth[required])
        & (scene_depth[required] > 0)
    ):
        raise ValueError("Depth evidence is missing or invalid inside the target footprint")
    result = visibility_from_masks(
        footprint,
        target_depth <= scene_depth + tolerance_m,
        camera=camera,
        image_bounds=image_bounds,
    )
    return None if result is None else Visibility(result.fraction, "depth", camera)


__all__ = [
    'CuboidVisibilityEstimator',
    'DepthVisibilityEstimator',
    'visibility_from_masks',
    'visibility_from_depth',
]
