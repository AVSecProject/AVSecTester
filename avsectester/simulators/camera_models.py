"""Camera models + poses for **world-anchored** insertion — project a 3-D point to a pixel and back.

The planar-warp path (:func:`avsectester.simulators.patch_insertion.warp_patch`) needs only an image
quad, which is enough when the target is found in image space (a detector box). Anchoring an inserted
object at a fixed *world* position (a sign on the roadside, seen from a moving ego) instead needs the
real camera: its lens model (``project`` / ``unproject``) and its pose. Both models here use the
standard camera frame (x right, y down, z forward):

  * :class:`PinholeCamera` — ``K`` intrinsics (nuScenes, CARLA).
  * :class:`FThetaCamera` — NVIDIA's f-theta fisheye (NuRec / AlpaSim wide cameras): pixel distance
    from the principal point is a polynomial of the ray's angle to the optical axis. A 120-deg lens is
    far from pinhole, so a 4-corner homography is not exact; rendering by per-pixel ray casting
    (:func:`avsectester.simulators.patch_insertion.render_plane`) is exact for any model here.

Pure numpy; no simulator imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


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
    stores them. Only the identity ``linear_cde`` (c=1, d=e=0) is supported, which is what NuRec ships.
    """

    cx: float
    cy: float
    angle_to_pixeldist: tuple
    pixeldist_to_angle: tuple
    width: int
    height: int
    max_angle: float = math.pi / 2

    @classmethod
    def from_nurec(cls, spec: Any) -> FThetaCamera:
        """Build from a NuRec ``CameraSpec`` (``RGBRenderRequest.camera_intrinsics``)."""
        f = spec.ftheta_param
        cde = f.linear_cde
        if (cde.linear_c, cde.linear_d, cde.linear_e) not in ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)):
            raise NotImplementedError("non-identity f-theta linear_cde is not supported")
        return cls(
            cx=f.principal_point_x,
            cy=f.principal_point_y,
            angle_to_pixeldist=tuple(f.angle_to_pixeldist_poly),
            pixeldist_to_angle=tuple(f.pixeldist_to_angle_poly),
            width=spec.resolution_w,
            height=spec.resolution_h,
            max_angle=f.max_angle or math.pi / 2,
        )

    @staticmethod
    def _poly(coeffs: tuple, x: np.ndarray) -> np.ndarray:
        return np.polynomial.polynomial.polyval(x, np.asarray(coeffs, dtype=np.float64))

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_cam, dtype=np.float64)
        rho = np.hypot(p[:, 0], p[:, 1])
        theta = np.arctan2(rho, p[:, 2])
        r = self._poly(self.angle_to_pixeldist, theta)
        scale = np.divide(r, rho, out=np.zeros_like(r), where=rho > 1e-12)
        return np.stack([self.cx + p[:, 0] * scale, self.cy + p[:, 1] * scale], axis=1)

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        dx, dy = uv[:, 0] - self.cx, uv[:, 1] - self.cy
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
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


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
