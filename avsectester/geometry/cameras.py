"""Camera lens models shared by datasets and simulators.

Both models use optical axes: X right, Y down, Z forward. ``PinholeCamera``
represents calibrated CARLA and nuScenes cameras. ``FThetaCamera`` represents
NuRec wide-angle lenses using angle-to-pixel polynomials, optionally seen through a
``WindshieldDistortion``. Per-pixel ray casting supports distorted lenses whose projection
cannot be described by a homography.
The module depends only on NumPy and standard-library types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


BODY_FROM_OPTICAL = np.array([
    [0., 0., 1., 0.],
    [-1., 0., 0., 0.],
    [0., -1., 0., 0.],
    [0., 0., 0., 1.],
])
"""Map optical points (x right, y down, z forward) to body axes (x forward, y left, z up)."""

NEAR_Z: float = 1e-5
"""Minimum positive optical z used for geometric projection and ray intersections, in metres."""


class Camera(Protocol):
    """Optical-frame lens contract. Coordinates address pixel centres at (x + .5, y + .5).

    ``project`` maps N camera points to N pixel coordinates. ``unproject`` returns N
    unit rays. A lens may additionally expose ``max_angle`` in radians.
    """

    width: int
    height: int

    def project(self, pts_cam: np.ndarray) -> np.ndarray: ...

    def unproject(self, uv: np.ndarray) -> np.ndarray: ...


def camera_ray_mask(camera: Camera, rays: np.ndarray) -> np.ndarray:
    """Finite forward camera rays inside the lens's optional maximum viewing angle."""
    valid = np.isfinite(rays).all(axis=1) & (rays[:, 2] > 0)
    if hasattr(camera, "max_angle"):
        valid &= np.arctan2(np.hypot(rays[:, 0], rays[:, 1]), rays[:, 2]) <= camera.max_angle
    return valid


# ---------------------------------------------------------------------------------------------------
# Lens models (camera frame: x right, y down, z forward)
# ---------------------------------------------------------------------------------------------------
@dataclass
class PinholeCamera:
    """Ideal pinhole camera with 3x3 intrinsics ``K``."""

    K: np.ndarray
    width: int
    height: int

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """(N,3) camera-frame points -> (N,2) pixels (valid only for z > 0)."""
        p = np.asarray(pts_cam, dtype=np.float64)
        uv = (np.asarray(self.K, dtype=np.float64) @ p.T).T
        return uv[:, :2] / uv[:, 2:3]

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """(N,2) pixels -> (N,3) unit rays in the camera frame."""
        uv = np.asarray(uv, dtype=np.float64)
        rays = (np.linalg.inv(self.K) @ np.c_[uv, np.ones(len(uv))].T).T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _bivariate_order(coefficients: tuple) -> int:
    terms = 0
    for order in range(len(coefficients)):
        terms += order + 1
        if terms == len(coefficients):
            return order
    raise ValueError("Windshield polynomial length must be a triangular number (order + 1)(order + 2) / 2")


def _bivariate(coefficients: tuple, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate sum_k p_k(x) * y**k, where p_k has degree order - k; blocks are stored for k = 0, 1, ..."""
    order, start = _bivariate_order(coefficients), 0
    result = np.zeros_like(x)
    for degree in range(order, -1, -1):
        block = np.asarray(coefficients[start:start + degree + 1], dtype=np.float64)
        result += np.polynomial.polynomial.polyval(x, block) * y ** (order - degree)
        start += degree + 1
    return result


@dataclass(frozen=True)
class WindshieldDistortion:
    """Refraction of camera rays by the vehicle windshield, as in NVIDIA ncore's bivariate model.

    A unit optical ray has angles ``phi = asin(x)`` and ``theta = asin(y)``. Bivariate polynomials
    map these angles, and the deflected ray is ``(sin phi', sin theta', sqrt(1 - x'^2 - y'^2))``
    with the original sign of z. ``horizontal_poly`` and ``vertical_poly`` map scene-side rays to
    lens-side rays; the ``_inverse`` polynomials map back. Coefficients use ncore's order: blocks
    of decreasing degree in phi, multiplied by increasing powers of theta.
    """

    horizontal_poly: tuple
    vertical_poly: tuple
    horizontal_poly_inverse: tuple
    vertical_poly_inverse: tuple

    def __post_init__(self) -> None:
        for name in ("horizontal_poly", "vertical_poly", "horizontal_poly_inverse", "vertical_poly_inverse"):
            values = tuple(float(value) for value in getattr(self, name))
            _bivariate_order(values)
            object.__setattr__(self, name, values)

    @staticmethod
    def _deflect(rays: np.ndarray, horizontal: tuple, vertical: tuple) -> np.ndarray:
        r = np.atleast_2d(np.asarray(rays, dtype=np.float64))
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
        phi, theta = np.arcsin(np.clip(r[:, 0], -1, 1)), np.arcsin(np.clip(r[:, 1], -1, 1))
        x, y = np.sin(_bivariate(horizontal, phi, theta)), np.sin(_bivariate(vertical, phi, theta))
        return np.stack([x, y, np.sqrt(np.clip(1 - x * x - y * y, 0, 1)) * np.sign(r[:, 2])], axis=1)

    def distort(self, rays: np.ndarray) -> np.ndarray:
        """Scene-side rays to the lens-side rays that the f-theta lens projects."""
        return self._deflect(rays, self.horizontal_poly, self.vertical_poly)

    def undistort(self, rays: np.ndarray) -> np.ndarray:
        """Lens-side rays to the scene-side rays they observe."""
        return self._deflect(rays, self.horizontal_poly_inverse, self.vertical_poly_inverse)


@dataclass
class FThetaCamera:
    """NVIDIA f-theta fisheye: ``r_px = angle_to_pixeldist(theta)`` with ``theta`` the ray's angle to
    the optical axis, ``r_px`` its pixel distance from the principal point (``cx``, ``cy``).

    ``pixeldist_to_angle`` is the inverse polynomial used by :meth:`unproject`. Polynomials are
    coefficient lists in increasing order (c0 + c1*x + c2*x^2 ...), exactly as NuRec's ``ftheta_param``
    and ``.usdz`` calibration store them. ``linear_cde`` is the small affine screen correction
    ``[[c, d], [e, 1]]`` applied to the radial offset (identity ``(1, 0, 0)`` in NuRec).
    :meth:`unproject` inverts that affine correction before the radial polynomial. A
    ``windshield`` deflects scene rays before the lens projects them, and unprojected lens rays
    are mapped back to the scene, matching NuRec's rendering cameras.
    """

    cx: float
    cy: float
    angle_to_pixeldist: tuple
    width: int
    height: int
    pixeldist_to_angle: tuple = ()
    max_angle: float = math.pi / 2
    linear_cde: tuple = (1.0, 0.0, 0.0)
    windshield: WindshieldDistortion | None = None


    @staticmethod
    def _poly(coeffs: tuple, x: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(x, np.asarray(coeffs, dtype=np.float64))

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_cam, dtype=np.float64))
        if self.windshield is not None:
            p = self.windshield.distort(p)
        rho = np.hypot(p[:, 0], p[:, 1])
        theta = np.arctan2(rho, p[:, 2])
        r = self._poly(self.angle_to_pixeldist, theta)
        scale = np.divide(r, rho, out=np.zeros_like(r), where=rho > 1e-12)
        du, dv = p[:, 0] * scale, p[:, 1] * scale
        c, d, e = self.linear_cde
        return np.stack([self.cx + c * du + d * dv, self.cy + e * du + dv], axis=1)

    def in_view(self, pts_cam: np.ndarray) -> np.ndarray:
        """Which camera-frame points project: in front of the camera and within ``max_angle``."""
        p = np.atleast_2d(np.asarray(pts_cam, dtype=np.float64))
        return camera_ray_mask(self, p)

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        if not len(self.pixeldist_to_angle):
            raise NotImplementedError("unproject needs the inverse polynomial pixeldist_to_angle")
        uv = np.asarray(uv, dtype=np.float64)
        # invert the affine screen correction [[c, d], [e, 1]] to recover the radial offset (dx, dy)
        c, d, e = self.linear_cde
        u0, v0 = uv[:, 0] - self.cx, uv[:, 1] - self.cy
        det = c - d * e
        dx = (u0 - d * v0) / det
        dy = (-e * u0 + c * v0) / det
        r = np.hypot(dx, dy)
        theta = self._poly(self.pixeldist_to_angle, r)
        s = np.sin(theta)
        inv = np.divide(s, r, out=np.zeros_like(r), where=r > 1e-12)
        rays = np.stack([dx * inv, dy * inv, np.cos(theta)], axis=1)
        if self.windshield is not None:
            rays = self.windshield.undistort(rays)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)
