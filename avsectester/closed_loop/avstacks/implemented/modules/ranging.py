"""Known-height stop-sign ranging and camera-guided LiDAR vehicle-face ranging."""

from __future__ import annotations

import numpy as np

from avsectester.geometry import CameraCalibration
from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_positive_number


_CAMERA_NEAR_Z_M = .5
_BOX_HORIZONTAL_INSET = .1
_BOX_VERTICAL_FRACTIONS = (.1, .6)
_MIN_CLUSTER_POINTS = 5
_NEAR_CLUSTER_PERCENTILE = 20.
_NEAR_CLUSTER_MARGIN_M = 1.
_FACE_X_PERCENTILE = 10.
_FACE_Y_PERCENTILES = (95., 5.)


def estimate_stop_sign_position(
    box_xyxy: tuple[float, float, float, float] | np.ndarray,
    calibration: CameraCalibration,
    sign_height_m: float = .762,
) -> np.ndarray:
    """Estimate the sign center in rig meters from its angular image height.

    The top/bottom box midpoints define angle theta and range
    sign_height_m / (2 * tan(theta / 2)). Range error is proportional to
    the difference between assumed and actual sign height. Pixel coordinates
    share the camera's continuous pixel-center convention. No offset is added.
    """
    require_positive_number(sign_height_m, "sign_height_m")
    if not isinstance(calibration, CameraCalibration):
        raise TypeError("calibration must be CameraCalibration")
    box = np.asarray(box_xyxy, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or not (
        box[2] > box[0] and box[3] > box[1]
    ):
        raise ValueError("box_xyxy must be a finite, nonempty image box")
    centre_x = (box[0] + box[2]) / 2
    pixels = np.array([[centre_x, box[1]], [centre_x, box[3]],
                       [centre_x, (box[1] + box[3]) / 2]])
    rays = np.asarray(calibration.camera.unproject(pixels), dtype=float)
    if rays.shape != (3, 3) or not np.isfinite(rays).all():
        raise ValueError("Camera unprojection must return three finite rays")
    norms = np.linalg.norm(rays, axis=1)
    if (norms <= 0).any():
        raise ValueError("Camera unprojection returned an empty ray")
    rays = rays / norms[:, None]
    theta = float(np.arccos(np.clip(np.dot(rays[0], rays[1]), -1., 1.)))
    if theta <= 1e-9 or theta >= np.pi:
        raise ValueError("Stop-sign box must span a nonzero angle below pi")
    distance = sign_height_m / (2 * np.tan(theta / 2))
    camera_position = rays[2] * distance
    transform = validated_transform(calibration.rig_from_camera)
    return transform[:3, :3] @ camera_position + transform[:3, 3]


def estimate_vehicle_face_points(
    box_xyxy: tuple[float, float, float, float] | np.ndarray,
    calibration: CameraCalibration,
    rig_points: np.ndarray,
) -> np.ndarray:
    """Estimate left, center and right vehicle-face points from box-selected LiDAR returns.

    rig_points contains finite (N, 3) rig coordinates in meters. Only points more than 0.5 m
    ahead of the camera are used. Selection removes 10 percent of the box width on either side
    and keeps the region from 10 to 60 percent of its height. The nearest cluster is within
    one meter of the 20th percentile horizontal distance from the camera.

    Return a (3, 3) rig-coordinate array with z=0. Each x is the cluster's 10th percentile x.
    Left and right y are its 95th and 5th percentiles, and center y is their midpoint.
    Invalid geometry or fewer than five selected or nearest-cluster points raise ValueError.
    A calibration that is not CameraCalibration raises TypeError.
    """
    if not isinstance(calibration, CameraCalibration):
        raise TypeError("calibration must be CameraCalibration")
    box = np.asarray(box_xyxy, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or not (
        box[2] > box[0] and box[3] > box[1]
    ):
        raise ValueError("box_xyxy must be a finite, nonempty image box")
    points = np.asarray(rig_points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("rig_points must contain finite (N, 3) rig coordinates")
    transform = validated_transform(calibration.rig_from_camera)
    camera_from_rig = np.linalg.inv(transform)
    camera_points = points @ camera_from_rig[:3, :3].T + camera_from_rig[:3, 3]
    front = camera_points[:, 2] > _CAMERA_NEAR_Z_M
    if np.count_nonzero(front) < _MIN_CLUSTER_POINTS:
        raise ValueError("Fewer than five LiDAR points in the vehicle box region")
    pixels = np.asarray(calibration.camera.project(camera_points[front]), dtype=float)
    width, height = box[2:] - box[:2]
    inside = (
        (pixels[:, 0] >= box[0] + _BOX_HORIZONTAL_INSET * width)
        & (pixels[:, 0] <= box[2] - _BOX_HORIZONTAL_INSET * width)
        & (pixels[:, 1] >= box[1] + _BOX_VERTICAL_FRACTIONS[0] * height)
        & (pixels[:, 1] <= box[1] + _BOX_VERTICAL_FRACTIONS[1] * height)
    )
    selected = points[front][inside]
    if len(selected) < _MIN_CLUSTER_POINTS:
        raise ValueError("Fewer than five LiDAR points in the vehicle box region")
    ranges = np.linalg.norm(selected[:, :2] - transform[:2, 3], axis=1)
    near = selected[ranges <= np.percentile(ranges, _NEAR_CLUSTER_PERCENTILE) + _NEAR_CLUSTER_MARGIN_M]
    if len(near) < _MIN_CLUSTER_POINTS:
        raise ValueError("Fewer than five LiDAR points in the nearest vehicle cluster")
    left_y, right_y = np.percentile(near[:, 1], _FACE_Y_PERCENTILES)
    face = np.zeros((3, 3))
    face[:, 0] = np.percentile(near[:, 0], _FACE_X_PERCENTILE)
    face[:, 1] = (left_y, (left_y + right_y) / 2, right_y)
    return face


__all__ = [
    'estimate_vehicle_face_points',
    'estimate_stop_sign_position',
]
