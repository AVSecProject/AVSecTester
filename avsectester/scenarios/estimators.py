"""Visibility providers for explicit insertion geometry.

The target silhouette is rasterized using the actual camera lens, including the portion outside
the image. CARLA depth measurements and NuRec cuboid approximations share that same footprint.
The cuboid method only accounts for supplied geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from avsectester.insertion import ActorPose, PlaneSurface, ResolvedInsertion
from avsectester.scenarios.scene import CameraCalib, PlacementCandidate, Visibility
from avsectester.simulators.camera_models import FThetaCamera, PinholeCamera, transform


class _ForwardPolynomialCamera(FThetaCamera):
    """Invert a monotonic f-theta polynomial when a dataset omits the inverse calibration."""

    def unproject(self, uv):
        uv = np.asarray(uv, dtype=float)
        c, d, e = self.linear_cde
        affine = np.array([[c, d], [e, 1.0]])
        offsets = (uv - [self.cx, self.cy]) @ np.linalg.inv(affine).T
        radius = np.linalg.norm(offsets, axis=1)
        low, high = np.zeros(len(uv)), np.full(len(uv), np.pi / 2)
        angles = np.linspace(0, np.pi / 2, 512)
        if (np.diff(self._poly(self.angle_to_pixeldist, angles)) <= 0).any():
            raise ValueError("F-theta projection must be monotonic over the front hemisphere")
        for _ in range(48):
            middle = (low + high) / 2
            smaller = self._poly(self.angle_to_pixeldist, middle) < radius
            low = np.where(smaller, middle, low)
            high = np.where(smaller, high, middle)
        theta = (low + high) / 2
        scale = np.divide(np.sin(theta), radius, out=np.zeros_like(radius), where=radius > 1e-12)
        rays = np.column_stack((offsets * scale[:, None], np.cos(theta)))
        rays[radius > self._poly(self.angle_to_pixeldist, np.pi / 2)] = np.nan
        return rays


def camera_from_calibration(calib: CameraCalib):
    """Return an optical-frame project/unproject lens from the supported calibration forms."""
    model = calib.model
    if callable(getattr(model, "unproject", None)) and callable(getattr(model, "project", None)):
        return model
    if hasattr(model, "t_sensor_rig") and hasattr(model, "angle_to_pixeldist"):
        # The dataset wrapper projects rig coordinates. The estimator already applies extrinsics.
        return _ForwardPolynomialCamera(
            cx=float(model.principal_point[0]),
            cy=float(model.principal_point[1]),
            angle_to_pixeldist=tuple(model.angle_to_pixeldist),
            width=calib.width,
            height=calib.height,
            max_angle=float(model.max_angle),
            linear_cde=tuple(model.linear_cde),
        )
    if model is None:
        raise ValueError(f"Camera {calib.name!r} has no projection calibration")
    intrinsic = np.asarray(model, dtype=float)
    if intrinsic.shape == (3,):
        f, cx, cy = intrinsic
        intrinsic = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("Camera model must be a calibrated lens, K matrix, or (f, cx, cy)")
    return PinholeCamera(intrinsic.copy(), calib.width, calib.height)


def _edge_points(corners: np.ndarray) -> np.ndarray:
    # Fisheye projection bends straight 3D edges. Sample along them before choosing a raster ROI.
    t = np.linspace(0, 1, 65)[:, None]
    return np.concatenate([corners[i] + t * (corners[(i + 1) % 4] - corners[i]) for i in range(4)])


def resolved_to_target(
    subject: ResolvedInsertion,
    ego: ActorPose,
    cameras: Mapping[str, CameraCalib],
) -> PlacementCandidate:
    """Expose a resolved insertion in the scene's ego frame for ordinary visual filters.

    Projection boxes are only a field-of-view aid. They are not used as visibility silhouettes.
    Missing or unsupported camera calibration leaves that camera's projection unavailable.
    """
    ego_from_world = np.linalg.inv(ego.transform)
    pose = ego_from_world @ subject.pose
    planes = subject.planes()
    local_corners = np.concatenate([surface.corners for surface in subject.asset.planes()])
    extent = np.ptp(local_corners, axis=0)
    box2d = {}
    for name, calib in cameras.items():
        if calib.cam_to_ego is None:
            continue
        try:
            camera = camera_from_calibration(calib)
        except (ValueError, TypeError, NotImplementedError):
            continue
        cam_from_world = np.linalg.inv(np.asarray(calib.cam_to_ego)) @ ego_from_world
        pixels = []
        for surface in planes:
            corners = transform(cam_from_world, surface.corners)
            if np.all(corners[:, 2] > 1e-5):
                projected = camera.project(_edge_points(corners))
                if np.isfinite(projected).all():
                    pixels.append(projected)
        if pixels:
            pixels = np.concatenate(pixels)
            box2d[name] = tuple(np.concatenate((pixels.min(axis=0), pixels.max(axis=0))))
    return PlacementCandidate(
        track_id=subject.id,
        category="insertion",
        center=tuple(pose[:3, 3]),
        extent=tuple(extent),
        yaw=float(np.arctan2(pose[1, 0], pose[0, 0])),
        box2d=box2d,
        pose=pose,
    )


@dataclass(frozen=True)
class VisibilityEvidence:
    """Visibility and image-sized diagnostic geometry from one camera and instant.

    ``reference_pixels`` counts the full projected alpha silhouette, including out-of-image
    pixels. ``reference_mask`` and ``visible_mask`` are cropped to the real image. ``target_depth``
    uses camera-axis Z and contains infinity away from target surfaces. A missing visibility means
    the complete footprint or required depth could not be established.
    """

    visibility: Visibility | None
    reference_mask: np.ndarray
    visible_mask: np.ndarray
    target_depth: np.ndarray
    reference_pixels: int
    reason: str = ""


@dataclass
class _Projection:
    reference: np.ndarray
    distance: np.ndarray
    rays: np.ndarray
    x0: int
    y0: int


def _plane_intersections(surface: PlaneSurface, rays: np.ndarray, cam_from_world) -> np.ndarray:
    """Distance along camera unit rays to opaque texels of a world-space rectangle."""
    corners = transform(cam_from_world, surface.corners)
    origin, u, v = corners[0], corners[1] - corners[0], corners[3] - corners[0]
    normal = np.cross(u, v)
    denominator = rays @ normal
    distance = np.divide(
        origin @ normal,
        denominator,
        out=np.full(len(rays), np.inf),
        where=np.abs(denominator) > 1e-12,
    )
    valid = np.isfinite(distance) & (distance > 1e-5) & np.isfinite(rays).all(axis=1)
    offsets = rays * np.where(valid, distance, 0)[:, None] - origin
    s, t = offsets @ u / (u @ u), offsets @ v / (v @ v)
    valid &= (s >= 0) & (s < 1) & (t >= 0) & (t < 1)
    indices = np.flatnonzero(valid)
    h, w = surface.texture.shape[:2]
    if len(indices):
        # Match the compositor's pixel-centre coordinates and transparent image border.
        px, py = s[indices] * w - 0.5, t[indices] * h - 0.5
        ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - ix, py - iy
        alpha = np.zeros(len(indices))
        for dx, dy, weight in (
            (0, 0, (1 - fx) * (1 - fy)),
            (1, 0, fx * (1 - fy)),
            (0, 1, (1 - fx) * fy),
            (1, 1, fx * fy),
        ):
            x, y = ix + dx, iy + dy
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            alpha[inside] += surface.texture[y[inside], x[inside], 3] * weight[inside]
        valid[indices] &= alpha > 127
    return np.where(valid, distance, np.inf)


def _rasterize(subject, camera, cam_from_world, max_reference_pixels):
    surfaces = subject.planes()
    if not surfaces:
        return None, "Asset has no planar surfaces"
    projected = []
    any_front = False
    for surface in surfaces:
        corners = transform(cam_from_world, surface.corners)
        if np.all(corners[:, 2] <= 1e-5):
            continue
        any_front = True
        if np.any(corners[:, 2] <= 1e-5):
            return None, "Target intersects the camera plane, so its full projection is unavailable"
        pixels = camera.project(_edge_points(corners))
        if not np.isfinite(pixels).all():
            return None, "Camera projection is undefined for part of the target"
        projected.append(pixels)
    if not any_front:
        return None, "Target is behind the camera"
    pixels = np.concatenate(projected)
    # Use Python integers so a very large near-camera footprint cannot overflow the
    # allocation guard's width*height calculation before we reject it.
    x0, y0 = (int(value) - 2 for value in np.floor(pixels.min(axis=0)))
    x1, y1 = (int(value) + 2 for value in np.ceil(pixels.max(axis=0)))
    width, height = x1 - x0, y1 - y0
    if width * height > max_reference_pixels:
        return None, "Full target footprint exceeds max_reference_pixels"
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    rays = np.asarray(camera.unproject(np.column_stack((xs.ravel() + 0.5, ys.ravel() + 0.5))))
    distance = np.full(len(rays), np.inf)
    for surface in surfaces:
        distance = np.minimum(distance, _plane_intersections(surface, rays, cam_from_world))
    reference = np.isfinite(distance)
    return _Projection(reference.reshape(height, width), distance, rays, x0, y0), ""


def projected_silhouette(subject, camera, cam_from_world) -> np.ndarray:
    """Opaque insertion pixels inside the camera image, before scene occlusion.

    Use the visibility estimator's ray/surface and alpha rules. Image-only projection does
    not require a finite offscreen footprint, so a surface crossing the camera plane can
    still be tested. Row batches bound temporary memory for large camera images.
    """
    mask = np.zeros((camera.height, camera.width), dtype=bool)
    surfaces = subject.planes()
    for top in range(0, camera.height, 32):
        bottom = min(top + 32, camera.height)
        xs, ys = np.meshgrid(np.arange(camera.width), np.arange(top, bottom))
        rays = np.asarray(camera.unproject(np.column_stack((xs.ravel() + 0.5, ys.ravel() + 0.5))))
        valid = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 1e-5)
        if hasattr(camera, "max_angle"):
            valid &= np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2]) <= camera.max_angle
        hits = np.zeros(len(rays), dtype=bool)
        for surface in surfaces:
            hits |= np.isfinite(_plane_intersections(surface, rays, cam_from_world))
        mask[top:bottom] = (valid & hits).reshape(bottom - top, camera.width)
    return mask


def _box_intersections(actor: ActorPose, rays: np.ndarray, world_from_cam) -> np.ndarray:
    """Ray/slab intersection with an oriented box, in metres along camera unit rays."""
    if any(v <= 0 for v in actor.extent):
        return np.full(len(rays), np.inf)
    box_from_cam = np.linalg.inv(actor.transform) @ world_from_cam
    origin = box_from_cam[:3, 3]
    directions = rays @ box_from_cam[:3, :3].T
    half = np.asarray(actor.extent) / 2
    parallel = np.abs(directions) < 1e-12
    near = np.divide(
        -half - origin, directions, out=np.full_like(directions, -np.inf), where=~parallel
    )
    far = np.divide(
        half - origin, directions, out=np.full_like(directions, np.inf), where=~parallel
    )
    lower, upper = np.minimum(near, far), np.maximum(near, far)
    lower[parallel], upper[parallel] = -np.inf, np.inf
    miss = np.any(parallel & (np.abs(origin) > half), axis=1)
    entry, leave = lower.max(axis=1), upper.min(axis=1)
    # If the camera starts inside a box, the first forward surface is its exit.
    hit = np.where(entry > 1e-5, entry, leave)
    return np.where(~miss & (leave >= entry) & (hit > 1e-5), hit, np.inf)


class _VisibilityEstimator:
    def __init__(self, *, tolerance_m: float = 0.002, max_reference_pixels: int = 4_000_000):
        if not np.isfinite(tolerance_m) or tolerance_m < 0:
            raise ValueError("tolerance_m must be finite and nonnegative")
        if not isinstance(max_reference_pixels, int) or max_reference_pixels <= 0:
            raise ValueError("max_reference_pixels must be a positive integer")
        self.tolerance_m = tolerance_m
        self.max_reference_pixels = max_reference_pixels

    def _prepare(self, subject, camera, cam_from_world, source, camera_name):
        projection, reason = _rasterize(subject, camera, cam_from_world, self.max_reference_pixels)
        shape = (camera.height, camera.width)
        reference = np.zeros(shape, dtype=bool)
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
        depth[image_y, image_x] = projection.distance[selected] * projection.rays[selected, 2]
        state = (projection, selected, image_x, image_y)
        return state, VisibilityEvidence(None, reference, reference.copy(), depth, count)

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
        camera,
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
                blocking_depth, _box_intersections(actor, rays, world_from_cam)
            )
        for other in other_insertions:
            if other.id != subject.id:
                for surface in other.planes():
                    blocking_depth = np.minimum(
                        blocking_depth, _plane_intersections(surface, rays, cam_from_world)
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
        camera,
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
                    distance = _plane_intersections(surface, rays, cam_from_world)
                    visible[ys, xs] &= projection.distance[indices] <= distance + self.tolerance_m
        return self._finish(evidence, visible, source, camera_name)
