"""Lens-aware projected bounds of boxes, using only geometric data."""

from __future__ import annotations

from itertools import product

import numpy as np

from .cameras import Camera, FThetaCamera, NEAR_Z, camera_ray_mask
from .poses import transform


def _windshield_points(camera: FThetaCamera, points: np.ndarray) -> np.ndarray:
    """Optical points moved along the windshield's deflection, keeping their distances.

    The calibrated windshield model is extrapolated near the lens's maximum angle, where its
    deflected ray can lose its forward component. Points outside the lens's field, or deflected
    out of it, project outside the image and stay in place.
    """
    if camera.windshield is None:
        return points
    flat = points.reshape(-1, 3)
    moved = flat.copy()
    seen = np.flatnonzero(camera_ray_mask(camera, flat))
    if seen.size:
        deflected = camera.windshield.distort(flat[seen])
        kept = camera_ray_mask(camera, deflected)
        moved[seen[kept]] = deflected[kept] * np.linalg.norm(flat[seen[kept]], axis=1, keepdims=True)
    return moved.reshape(points.shape)


def _ftheta_box_bounds(camera: FThetaCamera, corners: np.ndarray) -> np.ndarray:
    """Conservative bounds of curved projected edges using interval arithmetic.

    In normalized perspective coordinates an edge stays inside its endpoint rectangle.
    Bound the radial polynomial and theta/tan(theta) separately on each short segment;
    unlike corner sampling, this includes edge extrema between the vertices.
    """
    coefficients = np.asarray(camera.angle_to_pixeldist, dtype=float)
    if not len(coefficients) or coefficients[0] != 0:
        return np.array([[0., 0.], [camera.width, camera.height]])
    edges = np.asarray([(first, first ^ axis) for first in range(8)
                        for axis in (1, 2, 4) if first < first ^ axis])
    fractions = np.linspace(0., 1., 9)
    points = (corners[edges[:, :1]] * (1 - fractions[None, :, None])
              + corners[edges[:, 1:]] * fractions[None, :, None])
    points = _windshield_points(camera, points)
    normalized = points[:, :, :2] / points[:, :, 2:]
    lower = np.minimum(normalized[:, :-1], normalized[:, 1:]).reshape(-1, 2)
    upper = np.maximum(normalized[:, :-1], normalized[:, 1:]).reshape(-1, 2)
    nearest = np.maximum(np.maximum(lower, -upper), 0.)
    radius_min = np.linalg.norm(nearest, axis=1)
    radius_max = np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper)), axis=1)
    angles = np.column_stack((np.arctan(radius_min), np.arctan(radius_max)))
    quotient = coefficients[1:]
    values = np.polynomial.polynomial.polyval(angles, quotient)
    polynomial_min, polynomial_max = values.min(axis=1), values.max(axis=1)
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(quotient))
    for root in roots:
        if abs(root.imag) > 1e-12:
            continue
        inside = (angles[:, 0] <= root.real) & (root.real <= angles[:, 1])
        value = np.polynomial.polynomial.polyval(root.real, quotient)
        polynomial_min = np.where(inside, np.minimum(polynomial_min, value), polynomial_min)
        polynomial_max = np.where(inside, np.maximum(polynomial_max, value), polynomial_max)
    factors = np.divide(angles, np.tan(angles), out=np.ones_like(angles), where=angles != 0)
    products = np.stack((polynomial_min * factors[:, 0], polynomial_min * factors[:, 1],
                         polynomial_max * factors[:, 0], polynomial_max * factors[:, 1]), axis=1)
    scale_min, scale_max = products.min(axis=1), products.max(axis=1)
    c, d, e = camera.linear_cde
    affine = np.array([[c, d], [e, 1.]])
    offset_min = lower @ np.maximum(affine, 0).T + upper @ np.minimum(affine, 0).T
    offset_max = upper @ np.maximum(affine, 0).T + lower @ np.minimum(affine, 0).T
    screen = np.stack((offset_min * scale_min[:, None], offset_min * scale_max[:, None],
                       offset_max * scale_min[:, None], offset_max * scale_max[:, None]))
    principal = np.array([camera.cx, camera.cy])
    return np.array([screen.min(axis=(0, 1)) + principal, screen.max(axis=(0, 1)) + principal])


def _ftheta_partial_box_bounds(camera: FThetaCamera, corners: np.ndarray) -> np.ndarray:
    """Bound forward projections of a box crossing the camera plane.

    A lateral rectangle bounds ray azimuths even as optical z approaches zero.
    Polynomial extrema bound the radial distance independently of that azimuth;
    no division by near-zero optical z or full-image ray query is needed.
    """
    corners = _windshield_points(camera, corners)
    lower, upper = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
    nearest = np.maximum(np.maximum(lower, -upper), 0.)
    radius_min = np.linalg.norm(nearest)
    if radius_min == 0:
        return np.array([[0., 0.], [camera.width, camera.height]])
    lateral = [(x, y) for x in (lower[0], upper[0]) for y in (lower[1], upper[1])]
    if lower[0] <= 0 <= upper[0]:
        lateral.extend([(0., lower[1]), (0., upper[1])])
    if lower[1] <= 0 <= upper[1]:
        lateral.extend([(lower[0], 0.), (upper[0], 0.)])
    directions = np.asarray(lateral)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    angle_min = np.arctan2(radius_min, corners[:, 2].max())
    angles = [angle_min, np.pi / 2]
    coefficients = np.asarray(camera.angle_to_pixeldist, dtype=float)
    roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
    angles.extend(root.real for root in roots if abs(root.imag) <= 1e-12
                  and angle_min <= root.real <= np.pi / 2)
    radial = np.polynomial.polynomial.polyval(angles, coefficients)
    radial_min, radial_max = radial.min(), radial.max()
    direction_min, direction_max = directions.min(axis=0), directions.max(axis=0)
    products = np.array([direction_min * radial_min, direction_min * radial_max,
                         direction_max * radial_min, direction_max * radial_max])
    offset_min, offset_max = products.min(axis=0), products.max(axis=0)
    c, d, e = camera.linear_cde
    affine = np.array([[c, d], [e, 1.]])
    screen_min = offset_min @ np.maximum(affine, 0).T + offset_max @ np.minimum(affine, 0).T
    screen_max = offset_max @ np.maximum(affine, 0).T + offset_min @ np.minimum(affine, 0).T
    return np.array([screen_min, screen_max]) + [camera.cx, camera.cy]


def project_box(
    camera: Camera, cam_from_world: np.ndarray, world_from_box: np.ndarray,
    extent: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Project a box-centred rigid pose and full dimensions to an image-clipped rectangle.

    Box edges crossing the camera plane are clipped to a small positive optical depth.
    F-theta bounds include the curved edge projections conservatively, after deflecting edge
    samples inside the lens's field through any windshield. A box entirely behind that plane or
    outside the image returns None.
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
    if isinstance(camera, FThetaCamera):
        pixels = (_ftheta_box_bounds(camera, camera_points) if front.all()
                  else _ftheta_partial_box_bounds(camera, np.asarray(clipped)))
    else:
        pixels = np.asarray(camera.project(np.asarray(clipped)))
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError("Box has an undefined camera projection")
    lower = np.maximum(pixels.min(axis=0), (0, 0))
    upper = np.minimum(pixels.max(axis=0), (camera.width, camera.height))
    if (upper <= lower).any():
        return None
    return float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])
