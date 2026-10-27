"""AlpaSim-style ego ground contact using CPU mesh-ray intersections.

Only the vehicle box's height and orientation change. The plane-fit correction rotates about
its predicted center, matching AlpaSim PhysicsBackend.update_pose without requiring Warp.
"""
from __future__ import annotations

import math

import numpy as np

from avsectester.geometry.mesh import MeshRaycaster
from avsectester.geometry.poses import validated_transform, validated_vector


def _plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    center = points.mean(axis=0)
    centered = points - center
    vectors, singular_values, _ = np.linalg.svd(centered.T @ centered)
    if singular_values[1] <= 1e-12:
        return None
    normal = vectors[:, -1]
    if normal[2] < 0:
        normal = -normal
    return center, normal


def constrain_to_ground(
    raycaster: MeshRaycaster, world_from_box: np.ndarray, extent: tuple[float, float, float],
) -> tuple[np.ndarray, str]:
    """Fit a vehicle's bottom to nearby ground, preserving its predicted center's XY.

    Six successful intersections are required from a 4-by-4 bottom grid, queried in both
    vertical directions within 100 m. Corrections exceeding 1.5 m or 10 degrees are rejected.
    The returned status is ``successful_update``, ``insufficient_intersections``,
    ``high_translation`` or ``high_rotation``. Failed updates return the original pose.
    Like AlpaSim, rotating about the box center after centroid alignment approximates contact;
    it does not perform another translation to eliminate the small residual on a slope.
    """
    pose = validated_transform(world_from_box)
    dimensions = validated_vector(extent, "extent")
    if (dimensions <= 0).any():
        raise ValueError("extent must contain positive box dimensions")
    x, y = np.meshgrid(np.linspace(-dimensions[0] / 2, dimensions[0] / 2, 4),
                       np.linspace(-dimensions[1] / 2, dimensions[1] / 2, 4))
    bottom = np.column_stack((x.ravel(), y.ravel(), np.full(16, -dimensions[2] / 2)))
    bottom = bottom @ pose[:3, :3].T + pose[:3, 3]
    origins = np.concatenate((bottom, bottom))
    directions = np.repeat([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], 16, axis=0)
    distances = raycaster.distances(origins, directions)
    hit = np.isfinite(distances) & (distances >= 0) & (distances <= 100.0)
    if np.count_nonzero(hit) < 6:
        return pose.copy(), "insufficient_intersections"
    original_plane = _plane(origins[hit])
    ground_plane = _plane(origins[hit] + distances[hit, None] * directions[hit])
    if original_plane is None or ground_plane is None:
        return pose.copy(), "insufficient_intersections"
    original_center, original_normal = original_plane
    ground_center, ground_normal = ground_plane
    displacement = ground_center - original_center
    if np.linalg.norm(displacement) > 1.5:
        return pose.copy(), "high_translation"
    cosine = float(np.clip(original_normal @ ground_normal, -1.0, 1.0))
    if math.acos(cosine) > math.radians(10.0):
        return pose.copy(), "high_rotation"
    axis = np.cross(original_normal, ground_normal)
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    correction = np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)
    updated = pose.copy()
    updated[:3, :3] = correction @ pose[:3, :3]
    updated[:3, 3] += displacement
    updated[:2, 3] = pose[:2, 3]
    return updated, "successful_update"


__all__ = [
    'constrain_to_ground',
]
