"""Monocular stop-sign position from calibrated angular height."""

from __future__ import annotations

import numpy as np

from avsectester.geometry import CameraCalibration
from avsectester.geometry.poses import validated_transform
from avsectester.validation import require_positive_number


def estimate_stop_sign_position(
    box_xyxy: tuple[float, float, float, float] | np.ndarray,
    calibration: CameraCalibration,
    sign_height_m: float = .762,
) -> np.ndarray:
    """Estimate the sign centre in rig metres from its angular image height.

    The top/bottom box midpoints define angle theta and range
    sign_height_m / (2 * tan(theta / 2)). Range error is proportional to
    the difference between assumed and actual sign height. Pixel coordinates
    share the camera's continuous pixel-centre convention; no offset is added.
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
