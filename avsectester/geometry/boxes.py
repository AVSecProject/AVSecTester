"""Ray intersections, top-down overlap and separation for box-centered rigid transforms."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .cameras import NEAR_Z
from .poses import validated_transform, validated_vector


def box_ray_distances(
    world_from_box: ArrayLike, extent: ArrayLike, rays: np.ndarray, world_from_cam: ArrayLike,
) -> np.ndarray:
    """Ray/slab intersection distances in meters along camera-frame unit rays.

    ``extent`` is the full box size, and ``world_from_box`` locates its center.
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


def _top_down_boxes(
    world_from_box_a: ArrayLike, extent_a: ArrayLike,
    world_from_box_b: ArrayLike, extent_b: ArrayLike,
) -> tuple[tuple[np.ndarray, np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Validate both boxes and retain only their yaw rotations for horizontal geometry."""
    poses = (validated_transform(world_from_box_a), validated_transform(world_from_box_b))
    extents = [validated_vector(value, "extent") for value in (extent_a, extent_b)]
    if any((extent < 0).any() for extent in extents):
        raise ValueError("extent must be nonnegative")
    rotations = []
    for pose in poses:
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        rotations.append(np.array([[math.cos(yaw), -math.sin(yaw)],
                                   [math.sin(yaw), math.cos(yaw)]]))
    return poses, extents, rotations


def _overlap(poses: tuple[np.ndarray, np.ndarray], extents: list[np.ndarray],
             rotations: list[np.ndarray]) -> bool:
    """Separating-axis test on validated top-down boxes, including touching."""
    offset = poses[1][:2, 3] - poses[0][:2, 3]
    for rotation in rotations:
        for axis in rotation.T:
            radii = [float(np.sum(extent[:2] / 2 * np.abs(basis.T @ axis)))
                     for extent, basis in zip(extents, rotations)]
            if abs(float(offset @ axis)) > sum(radii):
                return False
    return True


def oriented_boxes_overlap(
    world_from_box_a: ArrayLike, extent_a: ArrayLike,
    world_from_box_b: ArrayLike, extent_b: ArrayLike,
) -> bool:
    """Whether full-extent boxes overlap in the horizontal plane, including touching.

    The separating-axis test uses each box's yaw and horizontal dimensions.
    Known zero dimensions remain valid points or line segments. Height is not
    used for this top-down collision measure.
    """
    return _overlap(*_top_down_boxes(world_from_box_a, extent_a, world_from_box_b, extent_b))


def oriented_boxes_distance(
    world_from_box_a: ArrayLike, extent_a: ArrayLike,
    world_from_box_b: ArrayLike, extent_b: ArrayLike,
) -> float:
    """Minimum horizontal distance between full-extent boxes, in meters.

    Uses the same validated yaw-only rectangles as ``oriented_boxes_overlap`` and returns
    zero when that function reports overlap or touching. Otherwise the distance is the
    smallest vertex-to-opposite-edge distance. Zero horizontal dimensions form points or
    line segments. Height, roll and pitch do not change the horizontal dimensions.
    """
    poses, extents, rotations = _top_down_boxes(
        world_from_box_a, extent_a, world_from_box_b, extent_b)
    if _overlap(poses, extents, rotations):
        return 0.0
    # Work relative to the first center to avoid cancelling large world coordinates.
    offset = poses[1][:2, 3] - poses[0][:2, 3]
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    first, second = [corners * extent[:2] / 2 @ rotation.T
                     for extent, rotation in zip(extents, rotations)]
    second += offset
    distance = np.inf
    for vertices, other in ((first, second), (second, first)):
        edges = np.roll(other, -1, axis=0) - other
        relative = vertices[:, None, :] - other
        lengths_squared = np.einsum("ij,ij->i", edges, edges)
        fractions = np.divide(
            np.einsum("vij,ij->vi", relative, edges), lengths_squared,
            out=np.zeros((4, 4)), where=lengths_squared > 0)
        nearest = other + np.clip(fractions, 0., 1.)[..., None] * edges
        separation = vertices[:, None, :] - nearest
        distance = min(distance, float(np.hypot(separation[..., 0], separation[..., 1]).min()))
    # Corner/edge arithmetic can round to zero when the shared SAT reports separation.
    # Reserve exact zero for that overlap decision without adding another tolerance.
    return float(max(distance, np.nextafter(0., 1.)))


__all__ = [
    'box_ray_distances',
    'oriented_boxes_overlap',
    'oriented_boxes_distance',
]
