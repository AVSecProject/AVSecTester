"""Depth-based visibility and explicitly approximate cuboid visibility for insertions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from avsectester.insertion import ActorPose, ResolvedInsertion
from .cameras import Camera
from .geometry import box_intersections, prepare_surface, rasterize
from .types import Visibility, VisibilityEvidence


class _VisibilityEstimator:
    def __init__(self, *, tolerance_m: float = 0.002, max_reference_pixels: int = 4_000_000):
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("tolerance_m must be finite and nonnegative")
        if not isinstance(max_reference_pixels, int) or max_reference_pixels <= 0:
            raise ValueError("max_reference_pixels must be a positive integer")
        self.tolerance_m = tolerance_m
        self.max_reference_pixels = max_reference_pixels

    def _prepare(self, subject, camera, cam_from_world, source, camera_name):
        projection, reason = rasterize(subject, camera, cam_from_world, self.max_reference_pixels)
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
        if hasattr(camera, "max_angle"):
            valid &= np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2]) <= camera.max_angle
        selected = ray_indices[valid]
        image_x, image_y = image_x[valid], image_y[valid]
        reference[image_y, image_x] = True
        rgba[image_y, image_x] = projection.rgba[selected]
        depth[image_y, image_x] = projection.distance[selected] * projection.rays[selected, 2]
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
        )


class CuboidVisibilityEstimator(_VisibilityEstimator):
    """Estimate opacity occlusion by known 3D boxes and other proposed planar insertions.

    Keep the host in ``occluders`` so that body panels can obscure their mounted objects. Missing
    buildings, vegetation and other unlabeled geometry are not inferred. Result source is always
    ``cuboid_estimate``. Explicit insertion surfaces use their alpha masks.
    """

    def estimate(
        self,
        subject: ResolvedInsertion,
        camera: Camera,
        cam_from_world: np.ndarray,
        *,
        occluders: Mapping[str, ActorPose],
        camera_name: str,
        other_insertions: Sequence[ResolvedInsertion] = (),
    ) -> VisibilityEvidence:
        source = "cuboid_estimate"
        state, evidence = self._prepare(subject, camera, cam_from_world, source, camera_name)
        if state is None:
            return evidence
        projection, indices, xs, ys = state
        rays = projection.rays[indices]
        blocking_depth = np.full(len(rays), np.inf)
        world_from_cam = np.linalg.inv(cam_from_world)
        # Native actor IDs and insertion IDs belong to different namespaces. Even an
        # identically named host must remain an occluder of its mounted surface.
        for actor in occluders.values():
            blocking_depth = np.minimum(
                blocking_depth, box_intersections(actor, rays, world_from_cam)
            )
        for other in other_insertions:
            if other.id != subject.id:
                for surface in other.planes():
                    blocking_depth = np.minimum(
                        blocking_depth,
                        prepare_surface(surface, camera, cam_from_world).sample(rays)[0],
                    )
        visible = np.zeros_like(evidence.reference_mask)
        visible[ys, xs] = projection.distance[indices] <= blocking_depth + self.tolerance_m
        return self._finish(evidence, visible, source, camera_name)


class DepthVisibilityEstimator(_VisibilityEstimator):
    """Compare known insertion surfaces with calibrated, aligned scene depth.

    ``depth_convention`` must be ``'z'`` for optical-axis depth or ``'range'`` for ray distance.
    The depth image must represent the same camera pose, lens and instant as the target geometry.
    Invalid depth at a target pixel makes the result unknown rather than silently visible.
    """

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
        if depth_convention not in {"z", "range"}:
            raise ValueError("depth_convention must be 'z' or 'range'")
        scene_depth = np.asarray(scene_depth, dtype=float)
        if scene_depth.shape != (camera.height, camera.width):
            raise ValueError("scene_depth must have the calibrated camera's image shape")
        source = "depth_comparison"
        state, evidence = self._prepare(subject, camera, cam_from_world, source, camera_name)
        if state is None:
            return evidence
        projection, indices, xs, ys = state
        measured = scene_depth[ys, xs]
        visible = np.zeros_like(evidence.reference_mask)
        if np.any(~np.isfinite(measured) | (measured <= 0)):
            return self._finish(
                evidence, visible, source, camera_name, "Invalid depth at target pixels"
            )
        rays = projection.rays[indices]
        target = projection.distance[indices]
        if depth_convention == "z":
            target = target * rays[:, 2]
        visible[ys, xs] = target <= measured + self.tolerance_m
        for other in other_insertions:
            if other.id != subject.id:
                for surface in other.planes():
                    distance = prepare_surface(surface, camera, cam_from_world).sample(rays)[0]
                    visible[ys, xs] &= projection.distance[indices] <= distance + self.tolerance_m
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
    not needed and may be NaN. Depth inside the target's in-image footprint must be valid.
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
        & np.isfinite(scene_depth[required])
        & (scene_depth[required] > 0)
    ):
        raise ValueError("Depth evidence is missing or invalid inside the target footprint")
    result = visibility_from_masks(
        footprint,
        target_depth <= scene_depth + tolerance_m,
        camera=camera,
        image_bounds=image_bounds,
    )
    return None if result is None else Visibility(result.fraction, "depth_comparison", camera)
