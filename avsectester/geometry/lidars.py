"""LiDAR calibration in canonical sensor and vehicle coordinates."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarCalibration:
    """A LiDAR-to-rig rigid transform and calibrated spin period in seconds.

    ``rig_from_lidar`` is a 4x4 rigid transform mapping sensor coordinates to vehicle
    coordinates, both with x forward, y left and z up, in meters. ``spin_period_s``
    describes one calibrated rotation. NuRec validates it against the scheduled acquisition
    interval, which sets the actual scan window. Fields are not validated at construction so
    recording can restore payload references without loading arrays. Consumers validate the
    transform and spin period when using them.
    """

    rig_from_lidar: np.ndarray
    spin_period_s: float


__all__ = [
    'LidarCalibration',
]
