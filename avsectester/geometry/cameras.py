"""Camera lens models and pose transforms shared by datasets and simulators.

Both models use optical axes: X right, Y down, Z forward. ``PinholeCamera``
represents calibrated CARLA and nuScenes cameras. ``FThetaCamera`` represents
NuRec wide-angle lenses using angle-to-pixel polynomials. Per-pixel ray casting
supports distorted lenses whose projection cannot be described by a homography.
The module depends only on NumPy and standard-library types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, overload, Protocol

import numpy as np
from numpy.typing import ArrayLike


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


@dataclass
class FThetaCamera:
    """NVIDIA f-theta fisheye: ``r_px = angle_to_pixeldist(theta)`` with ``theta`` the ray's angle to
    the optical axis, ``r_px`` its pixel distance from the principal point (``cx``, ``cy``).

    ``pixeldist_to_angle`` is the inverse polynomial used by :meth:`unproject`. Polynomials are
    coefficient lists in increasing order (c0 + c1*x + c2*x^2 ...), exactly as NuRec's ``ftheta_param``
    and ``.usdz`` calibration store them. ``linear_cde`` is the small affine screen correction
    ``[[c, d], [e, 1]]`` applied to the radial offset (identity ``(1, 0, 0)`` in NuRec).
    :meth:`unproject` inverts that affine correction before the radial polynomial.
    """

    cx: float
    cy: float
    angle_to_pixeldist: tuple
    width: int
    height: int
    pixeldist_to_angle: tuple = ()
    max_angle: float = math.pi / 2
    linear_cde: tuple = (1.0, 0.0, 0.0)

    @classmethod
    def from_nurec(cls, spec: Any) -> FThetaCamera:
        """Build from a NuRec ``CameraSpec`` (``RGBRenderRequest.camera_intrinsics``)."""
        f = spec.ftheta_param
        cde = f.linear_cde
        return cls(
            cx=f.principal_point_x,
            cy=f.principal_point_y,
            angle_to_pixeldist=tuple(f.angle_to_pixeldist_poly),
            pixeldist_to_angle=tuple(f.pixeldist_to_angle_poly),
            width=spec.resolution_w,
            height=spec.resolution_h,
            max_angle=f.max_angle or math.pi / 2,
            linear_cde=(cde.linear_c or 1.0, cde.linear_d, cde.linear_e),
        )

    @staticmethod
    @overload
    def _poly(coeffs: tuple, x: np.ndarray) -> np.ndarray: ...

    @staticmethod
    @overload
    def _poly(coeffs: tuple, x: float) -> float: ...

    @staticmethod
    def _poly(coeffs: tuple, x: np.ndarray | float) -> np.ndarray | float:
        return np.polynomial.polynomial.polyval(x, np.asarray(coeffs, dtype=np.float64))

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_cam, dtype=np.float64))
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
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------------------
# Poses (4x4 homogeneous, "a_from_b" maps b-frame points into frame a)
# ---------------------------------------------------------------------------------------------------
def quat_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_pose(rotation: np.ndarray, translation: ArrayLike) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def pose_from_proto(pose: Any) -> np.ndarray:
    """A NuRec/AlpaSim ``common.Pose`` (vec + quat) -> 4x4 matrix."""
    q, v = pose.quat, pose.vec
    return make_pose(quat_to_matrix(q.w, q.x, q.y, q.z), (v.x, v.y, v.z))


def planar_rig_pose(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    """world_from_rig for a 2-D ego pose (rig: x forward, y left, z up), as NuRecRenderer builds it."""
    c, s = math.cos(yaw), math.sin(yaw)
    return make_pose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), (x, y, z))


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]
