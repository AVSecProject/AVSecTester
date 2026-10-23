"""Prefiltered planar texture sampling shared by composition, visibility and optimization.

Every image pixel averages ``SUBSAMPLES`` x ``SUBSAMPLES`` rays. Each ray reads a trilinear
mipmap of the premultiplied RGBA texture at the level of its own footprint, so distant
textures are filtered instead of aliased and every texel can receive a gradient. Rendering,
visibility measurement and differentiable optimization call the same ``sample_texture``;
only the texture tensor differs. A pixel belongs to a surface when its filtered coverage
(premultiplied alpha) is positive, and its depth is the plane depth along the pixel's rays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from .objects import PlaneSurface
from avsectester.geometry.cameras import Camera, NEAR_Z, camera_ray_mask
from avsectester.geometry.poses import transform

if TYPE_CHECKING:
    import torch

SUBSAMPLES = 2
"""Rays per pixel along each image axis."""

# Coverage that rounds to a nonzero 8-bit alpha.
_COVERED = 0.5 / 255


def edge_points(corners: np.ndarray) -> np.ndarray:
    """Sample curved image boundaries produced by projecting straight 3D edges."""
    t = np.linspace(0, 1, 65)[:, None]
    return np.concatenate([corners[i] + t * (corners[(i + 1) % 4] - corners[i]) for i in range(4)])


@dataclass
class SurfaceSampler:
    """A camera-space rectangle, its RGBA texture and its full projected pixel bounds.

    ``bounds`` is an exclusive-stop pixel rectangle around the whole projection, including any
    part outside the image. It is None when the projection is unavailable, with ``reason``.
    """

    corners: np.ndarray
    texture: np.ndarray
    bounds: tuple[int, int, int, int] | None
    reason: str = ""

    def intersect(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distances to the unbounded plane along each ray and its (s, t) plane coordinates.

        Rays that miss the plane or hit it behind the camera return infinite distance and
        coordinates (-1, -1), outside the textured rectangle.
        """
        origin = self.corners[0]
        u, v = self.corners[1] - origin, self.corners[3] - origin
        normal = np.cross(u, v)
        denominator = rays @ normal
        distance = np.divide(origin @ normal, denominator, out=np.full(len(rays), np.inf),
                             where=np.abs(denominator) > 1e-12)
        valid = np.isfinite(distance) & (distance > 0) & np.isfinite(rays).all(axis=1)
        reached = np.where(valid, distance, 0.)
        valid &= reached * np.where(valid, rays[:, 2], 0.) > NEAR_Z
        offsets = np.where(valid[:, None], rays, 0.) * np.where(valid, distance, 0.)[:, None] - origin
        coordinates = np.column_stack((offsets @ u / (u @ u), offsets @ v / (v @ v)))
        coordinates[~valid] = -1.
        return np.where(valid, distance, np.inf), coordinates

    def sample(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distances to opaque texels and the full-resolution RGBA they hit, for any rays.

        Occlusion tests use this point query: a ray is blocked where the bilinear texture alpha
        exceeds 127. Rendered colours come from ``sample_texture`` instead.
        """
        import cv2

        distance, coordinates = self.intersect(rays)
        s, t = coordinates[:, 0], coordinates[:, 1]
        valid = np.isfinite(distance) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        indices = np.flatnonzero(valid)
        rgba = np.zeros((len(rays), 4), np.uint8)
        h, w = self.texture.shape[:2]
        # remap uses signed-short dimensions. Batching also bounds its temporary maps.
        for start in range(0, len(indices), 16384):
            batch = indices[start:start + 16384]
            x = (s[batch] * w - 0.5).astype(np.float32)[:, None]
            y = (t[batch] * h - 0.5).astype(np.float32)[:, None]
            rgba[batch] = cv2.remap(self.texture, x, y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))[:, 0]
        return np.where(valid & (rgba[:, 3] > 127), distance, np.inf), rgba

    def image_bounds(self, camera: Camera) -> tuple[int, int, int, int] | None:
        """The projection bounds clipped to the image, the whole image when they are unavailable,
        or None when the surface is behind the camera or projects outside the image."""
        if self.reason == "Target is behind the camera":
            return None
        bounds = self.bounds or (0, 0, camera.width, camera.height)
        x0, y0 = max(0, bounds[0]), max(0, bounds[1])
        x1, y1 = min(camera.width, bounds[2]), min(camera.height, bounds[3])
        return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def prepare_surface(surface: PlaneSurface, camera: Camera, cam_from_world) -> SurfaceSampler:
    """Transform a surface into the camera and bound its full projection, including off-image pixels."""
    corners = transform(cam_from_world, surface.corners)
    if np.all(corners[:, 2] <= NEAR_Z):
        return SurfaceSampler(corners, surface.texture, None, "Target is behind the camera")
    if np.any(corners[:, 2] <= NEAR_Z):
        return SurfaceSampler(corners, surface.texture, None,
                              "Target intersects the camera plane, so its full projection is unavailable")
    pixels = camera.project(edge_points(corners))
    if not np.isfinite(pixels).all():
        return SurfaceSampler(corners, surface.texture, None,
                              "Camera projection is undefined for part of the target")
    x0, y0 = (int(value) - 2 for value in np.floor(pixels.min(axis=0)))
    x1, y1 = (int(value) + 2 for value in np.ceil(pixels.max(axis=0)))
    return SurfaceSampler(corners, surface.texture, (x0, y0, x1, y1))


@dataclass(frozen=True)
class PlaneSampling:
    """Footprint geometry of one plane over an integer pixel region, independent of texture size.

    ``coordinates`` has shape ``(SUBSAMPLES**2, H, W, 2)``: plane (s, t) for every subpixel ray,
    (-1, -1) where a ray misses the plane or the lens. ``derivatives`` has shape ``(H, W, 2, 2)``:
    the change of (s, t) per pixel along image x and y. ``depth`` is the mean optical-axis depth
    of the rays that hit the unbounded plane, and positive infinity where none do.
    """

    bounds: tuple[int, int, int, int]
    coordinates: np.ndarray
    derivatives: np.ndarray
    depth: np.ndarray


def plane_sampling(sampler: SurfaceSampler, camera: Camera,
                   bounds: tuple[int, int, int, int]) -> PlaneSampling:
    """Unproject the region's subpixel rays through the lens and intersect them with the plane."""
    x0, y0, x1, y1 = bounds
    height, width = y1 - y0, x1 - x0
    ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    offsets = (np.arange(SUBSAMPLES) + .5) / SUBSAMPLES
    coordinates = np.empty((SUBSAMPLES, SUBSAMPLES, height, width, 2))
    z_sum, hits = np.zeros(height * width), np.zeros(height * width)
    for row, oy in enumerate(offsets):
        for column, ox in enumerate(offsets):
            rays = np.asarray(camera.unproject(
                np.column_stack((xs.ravel() + ox, ys.ravel() + oy))), dtype=float)
            distance, st = sampler.intersect(rays)
            valid = np.isfinite(distance) & camera_ray_mask(camera, rays)
            st[~valid] = -1.
            coordinates[row, column] = st.reshape(height, width, 2)
            z_sum += np.where(valid, distance, 0.) * np.where(valid, rays[:, 2], 0.)
            hits += valid
    step = offsets[-1] - offsets[0] if SUBSAMPLES > 1 else 1.
    hit = (coordinates >= -.5).all(axis=-1)
    along_x = coordinates[:, -1] - coordinates[:, 0]
    along_y = coordinates[-1] - coordinates[0]
    known_x = (hit[:, -1] & hit[:, 0]).all(axis=0)
    known_y = (hit[-1] & hit[0]).all(axis=0)
    derivatives = np.zeros((height, width, 2, 2))
    derivatives[..., 0, :] = np.where(known_x[..., None], along_x.mean(axis=0) / step, 0.)
    derivatives[..., 1, :] = np.where(known_y[..., None], along_y.mean(axis=0) / step, 0.)
    depth = np.divide(z_sum, hits, out=np.full(height * width, np.inf), where=hits > 0)
    return PlaneSampling(bounds, coordinates.reshape(SUBSAMPLES ** 2, height, width, 2),
                         derivatives, depth.reshape(height, width))


def premultiplied_texture(rgba: np.ndarray) -> torch.Tensor:
    """A float32 ``(4, H, W)`` premultiplied tensor from a uint8 RGBA image."""
    import torch

    values = torch.as_tensor(np.array(rgba, copy=True), dtype=torch.float32).permute(2, 0, 1) / 255
    return torch.cat((values[:3] * values[3:], values[3:]))


def _pyramid(texture: torch.Tensor) -> list[torch.Tensor]:
    """Area-averaged levels down to one texel; differentiable with respect to the texture."""
    import torch.nn.functional as F

    levels = [texture]
    while max(levels[-1].shape[1:]) > 1:
        height, width = levels[-1].shape[1:]
        levels.append(F.interpolate(levels[-1][None], size=(max(1, height // 2), max(1, width // 2)),
                                    mode="area")[0])
    return levels


def sample_texture(texture: torch.Tensor, sampling: PlaneSampling) -> torch.Tensor:
    """Filtered premultiplied RGBA ``(4, H, W)`` of a ``(4, h, w)`` premultiplied texture.

    The mean of the subpixel rays; each is a trilinear mipmap lookup at ``log2(rho) -
    log2(SUBSAMPLES)``, where ``rho`` is the pixel footprint in texels. Coverage is geometric:
    a ray outside the rectangle contributes nothing, and lookups clamp to the edge texels, so
    filtering never spreads a surface beyond its edges. Gradients reach every texel inside
    the footprints.
    """
    import torch
    import torch.nn.functional as F

    if texture.ndim != 3 or texture.shape[0] != 4:
        raise ValueError("texture must be a premultiplied (4, H, W) tensor")
    texture_height, texture_width = texture.shape[1:]
    texels = sampling.derivatives * np.array([texture_width, texture_height], dtype=float)
    rho = np.maximum(np.linalg.norm(texels[..., 0, :], axis=-1), np.linalg.norm(texels[..., 1, :], axis=-1))
    pyramid = _pyramid(texture)
    top = len(pyramid) - 1
    level = np.clip(np.log2(np.maximum(rho, 1e-12)) - math.log2(SUBSAMPLES), 0, top).ravel()
    lower = np.floor(level).astype(int)
    weight = torch.as_tensor(level - lower, dtype=texture.dtype, device=texture.device)
    pixels = level.size
    coordinates = sampling.coordinates.reshape(-1, pixels, 2)
    grids = torch.as_tensor(coordinates * 2 - 1, dtype=texture.dtype, device=texture.device)
    inside = torch.as_tensor(((coordinates >= 0) & (coordinates <= 1)).all(axis=-1),
                             dtype=texture.dtype, device=texture.device)
    total = texture.new_zeros((4, pixels))
    for index in np.unique(lower).tolist():
        selected = np.flatnonzero(lower == index)
        chosen = torch.as_tensor(selected, device=texture.device)
        grid = grids[:, chosen][None]
        nearer = F.grid_sample(pyramid[index][None], grid, mode="bilinear", padding_mode="border",
                               align_corners=False)[0]
        farther = F.grid_sample(pyramid[min(index + 1, top)][None], grid, mode="bilinear",
                                padding_mode="border", align_corners=False)[0]
        blend = weight[chosen]
        total[:, chosen] = ((nearer * (1 - blend) + farther * blend) * inside[:, chosen]).mean(dim=1)
    return total.reshape(4, *sampling.depth.shape)


def _straight_rgba(premultiplied: np.ndarray) -> np.ndarray:
    """uint8 straight RGBA from float premultiplied ``(H, W, 4)`` values in [0, 1]."""
    alpha = premultiplied[..., 3:4]
    color = np.divide(premultiplied[..., :3], alpha, out=np.zeros_like(premultiplied[..., :3]),
                      where=alpha > 0)
    return np.rint(np.clip(np.concatenate((color, alpha), axis=-1), 0, 1) * 255).astype(np.uint8)


def _render_region(sampler: SurfaceSampler, camera: Camera,
                   bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Premultiplied float RGBA and covered depth of one surface over a pixel region."""
    import torch

    sampling = plane_sampling(sampler, camera, bounds)
    with torch.no_grad():
        premultiplied = sample_texture(premultiplied_texture(sampler.texture), sampling)
    premultiplied = premultiplied.permute(1, 2, 0).numpy().astype(float)
    covered = (premultiplied[..., 3] > _COVERED) & np.isfinite(sampling.depth)
    return premultiplied, np.where(covered, sampling.depth, np.inf)


@dataclass
class Projection:
    """Nearest-surface samples over the full, unclipped footprint of several surfaces.

    ``reference`` marks covered pixels of the footprint rectangle starting at ``(x0, y0)``.
    ``depth`` is optical-axis depth, ``rays`` the unit pixel-centre rays and ``rgba`` uint8
    straight colours, all flattened over that rectangle.
    """

    reference: np.ndarray
    depth: np.ndarray
    rays: np.ndarray
    rgba: np.ndarray
    x0: int
    y0: int


def rasterize(surfaces: Sequence[PlaneSurface], camera: Camera, cam_from_world, max_reference_pixels):
    """Full unbounded silhouette, nearest-surface depth and matching colours, or a reason."""
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
    nearest = np.full((height, width), np.inf)
    premultiplied = np.zeros((height, width, 4))
    for sampler in front:
        colors, depth = _render_region(sampler, camera, (x0, y0, x1, y1))
        closer = depth < nearest
        nearest[closer], premultiplied[closer] = depth[closer], colors[closer]
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    rays = np.asarray(camera.unproject(np.column_stack((xs.ravel() + 0.5, ys.ravel() + 0.5))))
    return Projection(np.isfinite(nearest), nearest.ravel(), rays,
                      _straight_rgba(premultiplied).reshape(-1, 4), x0, y0), ""


def render_surfaces(
    surfaces: Sequence[PlaneSurface], camera: Camera, cam_from_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Image-sized uint8 RGBA and optical-axis depth of the nearest covering surface per pixel.

    Covered pixels have finite depth; their alpha is the filtered coverage, so edges are
    antialiased. The result does not depend on surface order.
    """
    nearest = np.full((camera.height, camera.width), np.inf)
    premultiplied = np.zeros((camera.height, camera.width, 4))
    for surface in surfaces:
        sampler = prepare_surface(surface, camera, cam_from_world)
        bounds = sampler.image_bounds(camera)
        if bounds is None:
            continue
        x0, y0, x1, y1 = bounds
        colors, depth = _render_region(sampler, camera, bounds)
        region = nearest[y0:y1, x0:x1]
        closer = depth < region
        region[closer] = depth[closer]
        premultiplied[y0:y1, x0:x1][closer] = colors[closer]
    return _straight_rgba(premultiplied), nearest
