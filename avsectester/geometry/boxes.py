"""Ray intersections and top-down overlap for box-centred rigid transforms."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .cameras import NEAR_Z
from .poses import validated_transform, validated_vector


def box_ray_distances(
    world_from_box: ArrayLike, extent: ArrayLike, rays: np.ndarray, world_from_cam: ArrayLike,
) -> np.ndarray:
    """Ray/slab intersection distances in metres along camera-frame unit rays.

    ``extent`` is the full box size, and ``world_from_box`` locates its centre.
    A ray starting inside the box reaches its exit face. Misses and boxes with
    a zero dimension return positive infinity.
    """
    dimensions = validated_vector(extent, "extent")
    if (dimensions < 0).any():
        raise ValueError("extent must be nonnegative")
    rays = np.asarray(rays, dtype=float)
    if rays.ndim != 2 or rays.shape[1:] != (3,) or not np.isfinite(rays).all():
        raise ValueError("rays must have finite coordinates with shape (N, 3)")
    if (dimensions == 0).any():
        return np.full(len(rays), np.inf)
    box_from_cam = np.linalg.inv(validated_transform(world_from_box)) @ validated_transform(world_from_cam)
    origin = box_from_cam[:3, 3]
    directions = rays @ box_from_cam[:3, :3].T
    half = dimensions / 2
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


def oriented_boxes_overlap(
    world_from_box_a: ArrayLike, extent_a: ArrayLike,
    world_from_box_b: ArrayLike, extent_b: ArrayLike,
) -> bool:
    """Whether full-extent boxes overlap in the horizontal plane, including touching.

    The separating-axis test uses each box's yaw and horizontal dimensions.
    Known zero dimensions remain valid points or line segments; height is not
    used for this top-down collision measure.
    """
    poses = (validated_transform(world_from_box_a), validated_transform(world_from_box_b))
    extents = [validated_vector(value, "extent") for value in (extent_a, extent_b)]
    if any((extent < 0).any() for extent in extents):
        raise ValueError("extent must be nonnegative")
    rotations = []
    for pose in poses:
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        rotations.append(np.array([[math.cos(yaw), -math.sin(yaw)],
                                   [math.sin(yaw), math.cos(yaw)]]))
    offset = poses[1][:2, 3] - poses[0][:2, 3]
    for rotation in rotations:
        for axis in rotation.T:
            radii = [float(np.sum(extent[:2] / 2 * np.abs(basis.T @ axis)))
                     for extent, basis in zip(extents, rotations)]
            if abs(float(offset @ axis)) > sum(radii):
                return False
    return True
