# SPDX-FileCopyrightText: © 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Adapted from CARLA PythonAPI/examples/nvidia/nurec/projection_functions.py.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Convert geodetic WGS84 coordinates to Cartesian Earth-fixed and local tangent frames."""

import numpy as np

from .poses import make_pose, validated_vector


_SEMIMAJOR_AXIS_M = 6378137.
_FLATTENING = 1. / 298.257223563
_SEMIMINOR_AXIS_M = _SEMIMAJOR_AXIS_M * (1. - _FLATTENING)
_AXIS_RATIO_SQUARED = (_SEMIMINOR_AXIS_M / _SEMIMAJOR_AXIS_M) ** 2


def ecef_from_geodetic(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    """Return Earth-centered, Earth-fixed XYZ in meters from degrees and ellipsoidal altitude."""
    latitude, longitude, altitude = validated_vector(
        (latitude_deg, longitude_deg, altitude_m), "WGS84 coordinates",
    )
    if abs(latitude) > 90. or abs(longitude) > 180.:
        raise ValueError("WGS84 coordinates require latitude within ±90° and longitude within ±180°")
    lat, lon = np.deg2rad((latitude, longitude))
    slat, clat = np.sin(lat), np.cos(lat)
    radius = _SEMIMAJOR_AXIS_M / np.sqrt(1. - (1. - _AXIS_RATIO_SQUARED) * slat ** 2)
    return np.array([
        (radius + altitude) * clat * np.cos(lon),
        (radius + altitude) * clat * np.sin(lon),
        (radius * _AXIS_RATIO_SQUARED + altitude) * slat,
    ])


def enu_from_ecef(latitude_deg: float, longitude_deg: float, altitude_m: float) -> np.ndarray:
    """Return the 4x4 Earth-fixed to local east-north-up transform at a WGS84 origin."""
    origin = ecef_from_geodetic(latitude_deg, longitude_deg, altitude_m)
    lat, lon = np.deg2rad((float(latitude_deg), float(longitude_deg)))
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    rotation = np.array([
        [-slon, clon, 0.],
        [-slat * clon, -slat * slon, clat],
        [clat * clon, clat * slon, slat],
    ])
    return make_pose(rotation, -rotation @ origin)


__all__ = ['ecef_from_geodetic', 'enu_from_ecef']
