"""Shared planar sampling for image composition and visibility measurements.

Texture filtering depends on the full projected surface, never the cropped image ROI.
Both consumers use the same pixel centres, bilinear samples and alpha > 127 silhouette.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .objects import PlaneSurface
from avsectester.geometry.cameras import Camera, NEAR_Z, camera_ray_mask, transform


def edge_points(corners: np.ndarray) -> np.ndarray:
    """Sample curved image boundaries produced by projecting straight 3D edges."""
    t = np.linspace(0, 1, 65)[:, None]
    return np.concatenate([corners[i] + t * (corners[(i + 1) % 4] - corners[i]) for i in range(4)])


@dataclass
class SurfaceSampler:
    """A camera-space rectangle and its once-prefiltered RGBA texture."""

    corners: np.ndarray
    texture: np.ndarray
    bounds: tuple[int, int, int, int] | None
    reason: str = ""

    def sample(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ray distances to opaque texels and RGBA samples for all input rays."""
        import cv2

        origin = self.corners[0]
        u, v = self.corners[1] - origin, self.corners[3] - origin
        normal = np.cross(u, v)
        denominator = rays @ normal
        distance = np.divide(
            origin @ normal,
            denominator,
            out=np.full(len(rays), np.inf),
            where=np.abs(denominator) > 1e-12,
        )
        valid = np.isfinite(distance) & (distance > 0) & np.isfinite(rays).all(axis=1)
        valid &= distance * rays[:, 2] > NEAR_Z
        offsets = rays * np.where(valid, distance, 0)[:, None] - origin
        s, t = offsets @ u / (u @ u), offsets @ v / (v @ v)
        valid &= (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        indices = np.flatnonzero(valid)
        rgba = np.zeros((len(rays), 4), np.uint8)
        h, w = self.texture.shape[:2]
        # remap uses signed-short dimensions. Batching also bounds its temporary maps.
        for start in range(0, len(indices), 16384):
            batch = indices[start : start + 16384]
            x = (s[batch] * w - 0.5).astype(np.float32)[:, None]
            y = (t[batch] * h - 0.5).astype(np.float32)[:, None]
            rgba[batch] = cv2.remap(
                self.texture,
                x,
                y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )[:, 0]
        return np.where(valid & (rgba[:, 3] > 127), distance, np.inf), rgba


def prepare_surface(surface: PlaneSurface, camera: Camera, cam_from_world) -> SurfaceSampler:
    """Choose a texture resolution from the full projection, including off-image pixels."""
    import cv2

    corners = transform(cam_from_world, surface.corners)
    if np.all(corners[:, 2] <= NEAR_Z):
        return SurfaceSampler(corners, surface.texture, None, "Target is behind the camera")
    if np.any(corners[:, 2] <= NEAR_Z):
        return SurfaceSampler(
            corners,
            surface.texture,
            None,
            "Target intersects the camera plane, so its full projection is unavailable",
        )
    pixels = camera.project(edge_points(corners))
    if not np.isfinite(pixels).all():
        return SurfaceSampler(
            corners, surface.texture, None, "Camera projection is undefined for part of the target"
        )
    x0, y0 = (int(value) - 2 for value in np.floor(pixels.min(axis=0)))
    x1, y1 = (int(value) + 2 for value in np.ceil(pixels.max(axis=0)))
    span = max(x1 - x0, y1 - y0, 4)
    h, w = surface.texture.shape[:2]
    scale = min(1.0, 2.0 * span / max(h, w))
    texture = surface.texture
    if scale < 1:
        texture = cv2.resize(
            texture, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA
        )
    return SurfaceSampler(corners, texture, (x0, y0, x1, y1))


@dataclass
class Projection:
    reference: np.ndarray
    distance: np.ndarray
    rays: np.ndarray
    rgba: np.ndarray
    x0: int
    y0: int


def rasterize(surfaces: Sequence[PlaneSurface], camera: Camera, cam_from_world, max_reference_pixels):
    """Full unbounded alpha silhouette, nearest-surface distances and matching colors."""
    samplers = [prepare_surface(s, camera, cam_from_world) for s in surfaces]
    if not samplers:
        return None, "Asset has no planar surfaces"
    front = [s for s in samplers if s.reason != "Target is behind the camera"]
    if not front:
        return None, "Target is behind the camera"
    for sampler in front:
        if sampler.bounds is None:
            return None, sampler.reason
    x0 = min(s.bounds[0] for s in front)
    y0 = min(s.bounds[1] for s in front)
    x1 = max(s.bounds[2] for s in front)
    y1 = max(s.bounds[3] for s in front)
    width, height = x1 - x0, y1 - y0
    if width * height > max_reference_pixels:
        return None, "Full target footprint exceeds max_reference_pixels"
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    rays = np.asarray(camera.unproject(np.column_stack((xs.ravel() + 0.5, ys.ravel() + 0.5))))
    distance = np.full(len(rays), np.inf)
    rgba = np.zeros((len(rays), 4), np.uint8)
    for sampler in front:
        candidate, colors = sampler.sample(rays)
        closer = candidate < distance
        distance[closer], rgba[closer] = candidate[closer], colors[closer]
    return Projection(
        np.isfinite(distance).reshape(height, width), distance, rays, rgba, x0, y0
    ), ""


def sample_image(
    sampler: SurfaceSampler, camera: Camera, *, margin: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Image-sized RGBA and optical-axis depth, using the full projection's sampling scale."""
    rgba = np.zeros((camera.height, camera.width, 4), np.uint8)
    depth = np.full((camera.height, camera.width), np.inf)
    if sampler.reason == "Target is behind the camera":
        return rgba, depth
    bounds = sampler.bounds or (0, 0, camera.width, camera.height)
    if sampler.bounds is not None:
        padding = margin - 2
        bounds = (
            bounds[0] - padding,
            bounds[1] - padding,
            bounds[2] + padding,
            bounds[3] + padding,
        )
    x0, y0 = max(0, bounds[0]), max(0, bounds[1])
    x1, y1 = min(camera.width, bounds[2]), min(camera.height, bounds[3])
    if x1 <= x0 or y1 <= y0:
        return rgba, depth
    for top in range(y0, y1, 32):
        bottom = min(top + 32, y1)
        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(top, bottom))
        rays = np.asarray(camera.unproject(np.column_stack((xs.ravel() + 0.5, ys.ravel() + 0.5))))
        distances, colors = sampler.sample(rays)
        valid = camera_ray_mask(camera, rays)
        colors[~valid] = 0
        z = np.full(len(rays), np.inf)
        hits = valid & np.isfinite(distances)
        z[hits] = distances[hits] * rays[hits, 2]
        rgba[top:bottom, x0:x1] = colors.reshape(bottom - top, x1 - x0, 4)
        depth[top:bottom, x0:x1] = z.reshape(bottom - top, x1 - x0)
    return rgba, depth


def render_surfaces(
    surfaces: Sequence[PlaneSurface], camera: Camera, cam_from_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the nearest opaque surface per image pixel, independently of input ordering."""
    rgba = np.zeros((camera.height, camera.width, 4), np.uint8)
    nearest = np.full((camera.height, camera.width), np.inf)
    for surface in surfaces:
        colors, depth = sample_image(prepare_surface(surface, camera, cam_from_world), camera)
        closer = depth < nearest
        nearest[closer], rgba[closer] = depth[closer], colors[closer]
    return rgba, nearest

