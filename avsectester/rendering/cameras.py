"""Camera lens models and pose transforms for geometry-based insertion.

Both models use optical axes: X right, Y down, Z forward. ``PinholeCamera``
represents calibrated CARLA and nuScenes cameras. ``FThetaCamera`` represents
NuRec wide-angle lenses using angle-to-pixel polynomials. Per-pixel ray casting
supports distorted lenses whose projection cannot be described by a homography.
The module depends only on NumPy and standard-library types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Camera(Protocol):
    """Optical-frame lens contract. Coordinates address pixel centres at (x + .5, y + .5).

    ``project`` maps N camera points to N pixel coordinates. ``unproject`` returns N
    unit rays. A lens may additionally expose ``max_angle`` in radians.
    """

    width: int
    height: int

    def project(self, pts_cam: np.ndarray) -> np.ndarray: ...

    def unproject(self, uv: np.ndarray) -> np.ndarray: ...


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
    ``[[c, d], [e, 1]]`` applied to the radial offset (identity ``(1, 0, 0)`` in NuRec); :meth:`unproject`
    supports only the identity. This is the one f-theta lens model in the package — the scenario layer's
    rig-frame camera (:class:`avsectester.scenarios.datasets.nurec.FThetaCamera`) projects through it.
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
    def _poly(coeffs: tuple, x: np.ndarray) -> np.ndarray:
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
        return (p[:, 2] > 0) & (np.arctan2(np.hypot(p[:, 0], p[:, 1]), p[:, 2]) <= self.max_angle)

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


def make_pose(rotation: np.ndarray, translation) -> np.ndarray:
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


def camera_from_calibration(calib) -> Camera:
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


def project_to_pixels(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Standard pinhole projection: camera-frame points (x right, y down, z forward) -> (N,2) px."""
    uv = (K @ np.asarray(pts_cam, dtype=np.float64).T).T
    return uv[:, :2] / uv[:, 2:3]


def cam_coords(pts_world: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    """Standard extrinsic: world points -> camera frame (x right, y down, z fwd). NuRec/pinhole."""
    pts = np.asarray(pts_world, dtype=np.float64)
    homog = np.c_[pts, np.ones(len(pts))]
    return (np.asarray(world_to_cam, dtype=np.float64) @ homog.T).T[:, :3]


def carla_cam_coords(pts_world: np.ndarray, cam_inverse_matrix: np.ndarray) -> np.ndarray:
    """CARLA UE convention: world -> camera then reorder to standard (x right, y down, z fwd).

    ``cam_inverse_matrix`` = ``camera.get_transform().get_inverse_matrix()``; UE camera axes are
    (x fwd, y right, z up), so the standard camera frame is ``[y, -z, x]``.
    """
    pc = cam_coords(pts_world, cam_inverse_matrix)  # UE camera coords (x fwd, y right, z up)
    return np.stack([pc[:, 1], -pc[:, 2], pc[:, 0]], axis=1)
