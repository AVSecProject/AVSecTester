"""Lens-aware projection and depth visibility using only geometric data."""

from __future__ import annotations

from itertools import product

import numpy as np

from .cameras import Camera, NEAR_Z, camera_ray_mask, transform
from .poses import ActorPose


def _plane_samples(
    camera: Camera, cam_from_world: np.ndarray, corners: np.ndarray,
    *, bounds: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 3):
        raise ValueError("Plane corners must be four finite three-dimensional points")
    corners = transform(cam_from_world, corners)
    origin = corners[0]
    u, v = corners[1] - origin, corners[3] - origin
    normal = np.cross(u, v)
    basis = np.column_stack((u, v))
    gram = basis.T @ basis
    if not np.isfinite(corners).all() or np.linalg.det(gram) <= 1e-18:
        raise ValueError("Projection needs a finite nondegenerate plane")
    height, width = camera.height, camera.width
    x0, y0, x1, y1 = (0, 0, width, height) if bounds is None else bounds
    ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    pixels = np.column_stack((xs.ravel() + .5, ys.ravel() + .5))
    rays = np.asarray(camera.unproject(pixels), dtype=float)
    denominator = rays @ normal
    distance = np.divide(origin @ normal, denominator,
                         out=np.full(len(rays), np.inf), where=np.abs(denominator) > 1e-12)
    valid = camera_ray_mask(camera, rays)
    valid &= np.isfinite(distance) & (distance > 0)
    valid &= distance * rays[:, 2] > NEAR_Z
    offsets = rays * np.where(valid, distance, 0.)[:, None] - origin
    coordinates = offsets @ basis @ np.linalg.inv(gram)
    valid &= np.isfinite(coordinates).all(axis=1)
    valid &= ((coordinates >= 0.) & (coordinates <= 1.)).all(axis=1)
    return rays, distance, coordinates, valid


def plane_sampling_grid(
    camera: Camera, cam_from_world: np.ndarray, corners: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Image-sized texture grid and geometric mask for a world-coordinate plane.

    Pixel centres are unprojected through the actual lens and intersected with the plane.
    The float32 grid has shape (H, W, 2), using ``grid_sample(align_corners=False)`` with
    texture edges at -1/+1. The float32 geometric mask has shape (1, H, W). Invalid samples
    use grid coordinate 2 so that NaNs never reach differentiable texture sampling.
    """
    _, _, coordinates, valid = _plane_samples(camera, cam_from_world, corners)
    grid = np.where(valid[:, None], coordinates * 2. - 1., 2.)
    return (grid.reshape(camera.height, camera.width, 2).astype(np.float32),
            valid.reshape(1, camera.height, camera.width).astype(np.float32))


def project_box(
    camera: Camera, cam_from_world: np.ndarray, world_from_box: np.ndarray,
    extent: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Project a box-centred rigid pose and full dimensions to an image-clipped rectangle.

    Box edges crossing the camera plane are clipped to a small positive optical depth.
    A box entirely behind that plane or outside the image returns None.
    """
    dimensions = np.asarray(extent, dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or (dimensions < 0).any():
        raise ValueError("Box extent must contain three finite nonnegative dimensions")
    corners = np.asarray(tuple(product((-0.5, 0.5), repeat=3))) * dimensions
    camera_points = transform(cam_from_world @ world_from_box, corners)
    if not np.isfinite(camera_points).all():
        raise ValueError("Box projection requires finite poses")
    front = camera_points[:, 2] > NEAR_Z
    if not front.any():
        return None
    clipped = list(camera_points[front])
    for first in range(8):
        for direction in (1, 2, 4):
            second = first ^ direction
            if first >= second or front[first] == front[second]:
                continue
            a, b = camera_points[first], camera_points[second]
            fraction = (NEAR_Z - a[2]) / (b[2] - a[2])
            clipped.append(a + fraction * (b - a))
    pixels = np.asarray(camera.project(np.asarray(clipped)))
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("Box has an undefined camera projection")
    lower = np.maximum(pixels.min(axis=0), (0, 0))
    upper = np.minimum(pixels.max(axis=0), (camera.width, camera.height))
    if (upper <= lower).any():
        return None
    return float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])


def visible_plane_mask(
    camera: Camera, cam_from_world: np.ndarray, depth: np.ndarray, corners: np.ndarray,
    tolerance_m: float,
    *, bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Return geometric plane pixels no farther than scene z-depth plus tolerance.

    This mask describes the rectangular plane, independently of texture alpha. Positive
    infinity means no scene intersection. NaN and nonpositive scene depths are not visible.
    Optional bounds are an exclusive-stop image rectangle; out-of-image portions are
    clipped, and pixels outside the resulting rectangle are false without ray queries.
    """
    scene_depth = np.asarray(depth, dtype=float)
    if scene_depth.shape != (camera.height, camera.width):
        raise ValueError("Depth must have the calibrated camera's image shape")
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("Depth tolerance must be finite and nonnegative")
    if bounds is None:
        bounds = (0, 0, camera.width, camera.height)
    if len(bounds) != 4 or any(isinstance(value, (bool, np.bool_))
                              or not isinstance(value, (int, np.integer)) for value in bounds):
        raise ValueError("Bounds must be integer (x0, y0, x1, y1) coordinates")
    x0, y0, x1, y1 = (int(value) for value in bounds)
    if x1 < x0 or y1 < y0:
        raise ValueError("Bounds stop coordinates must not precede their starts")
    x0, x1 = max(0, min(camera.width, x0)), max(0, min(camera.width, x1))
    y0, y1 = max(0, min(camera.height, y0)), max(0, min(camera.height, y1))
    rays, distance, _, valid = _plane_samples(
        camera, cam_from_world, corners, bounds=(x0, y0, x1, y1))
    measured = scene_depth[y0:y1, x0:x1].ravel()
    plane_depth = distance * rays[:, 2]
    visible = valid & (measured > 0) & (plane_depth <= measured + tolerance_m)
    result = np.zeros((camera.height, camera.width), dtype=bool)
    result[y0:y1, x0:x1] = visible.reshape(y1 - y0, x1 - x0)
    return result


def box_ray_distances(
    actor: ActorPose, rays: np.ndarray, world_from_cam: np.ndarray,
) -> np.ndarray:
    """Ray/slab intersection with an oriented box, in metres along camera unit rays."""
    if actor.extent is None:
        raise ValueError("Actor bounding-box extent is unavailable for ray intersections")
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
    hit = np.where(entry > NEAR_Z, entry, leave)
    return np.where(~miss & (leave >= entry) & (hit > NEAR_Z), hit, np.inf)
