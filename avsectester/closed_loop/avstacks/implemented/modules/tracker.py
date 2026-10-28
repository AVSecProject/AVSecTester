"""Static stop-sign tracking from known-size ranges and bearing EKF updates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from avsectester.validation import require_positive_number

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState


@dataclass(frozen=True)
class SignTrack:
    """A stationary sign's ID, horizontal world position and 2×2 covariance."""

    track_id: int
    position: np.ndarray
    covariance: np.ndarray


class StopSignTracker:
    """Associate range measurements, then refine static sign positions by bearing EKF.

    Range is used only for initialization and association, with a default
    fractional standard deviation of 0.3. Bearing updates use a default standard
    deviation of 0.007 rad. The default 0.3 m position floor prevents
    overconfidence and duplicate tracks. Signs are static, so there is no
    process noise.
    """

    def __init__(
        self, range_sigma_fraction: float = .3, bearing_sigma_rad: float = .007,
        min_position_sigma_m: float = .3,
        max_association_mahalanobis_distance: float = 3.5,
    ) -> None:
        for name, value in (
            ("range_sigma_fraction", range_sigma_fraction),
            ("bearing_sigma_rad", bearing_sigma_rad),
            ("min_position_sigma_m", min_position_sigma_m),
            ("max_association_mahalanobis_distance", max_association_mahalanobis_distance),
        ):
            require_positive_number(value, name)
        self.range_sigma_fraction = range_sigma_fraction
        self.bearing_sigma_rad = bearing_sigma_rad
        self.min_position_sigma_m = min_position_sigma_m
        self.max_association_mahalanobis_distance = max_association_mahalanobis_distance
        self._tracks: list[SignTrack] = []

    def reset(self) -> None:
        """Forget all signs and restart track IDs at zero."""
        self._tracks = []

    @property
    def tracks(self) -> tuple[SignTrack, ...]:
        """Return detached snapshots of the horizontal world tracks."""
        return tuple(SignTrack(track.track_id, track.position.copy(), track.covariance.copy())
                     for track in self._tracks)

    def update(
        self, vehicle_state: PlanarEgoState, camera_position_rig: np.ndarray,
        sign_positions_rig: Sequence[np.ndarray],
    ) -> None:
        """Associate rig-frame measurements and update matched tracks by bearing.

        camera_position_rig is the validated camera transform's translation.
        Only its horizontal coordinates and the vehicle's x, y and yaw are used.
        Range initializes and associates tracks, but never updates their position
        after association; only the camera-to-sign bearing enters the EKF.
        """
        try:
            pose = np.array([vehicle_state.x, vehicle_state.y, vehicle_state.yaw], dtype=float)
        except AttributeError as error:
            raise ValueError("vehicle_state must provide x, y and yaw") from error
        if not np.isfinite(pose).all():
            raise ValueError("vehicle_state x, y and yaw must be finite")
        camera = np.asarray(camera_position_rig, dtype=float)
        if camera.shape != (3,) or not np.isfinite(camera).all():
            raise ValueError("camera_position_rig must contain three finite rig coordinates")
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        camera_world = rotation @ camera[:2] + pose[:2]
        for point in sign_positions_rig:
            point = np.asarray(point, dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Each sign position must contain three finite rig coordinates")
            measurement = rotation @ point[:2] + pose[:2]
            delta = measurement - camera_world
            distance = float(np.linalg.norm(delta))
            if distance < 1e-9:
                raise ValueError("A sign measurement must be horizontally separated from the camera")
            direction = delta / distance
            perpendicular = np.array([-direction[1], direction[0]])
            radial_sigma = math.hypot(self.range_sigma_fraction * distance, self.min_position_sigma_m)
            lateral_sigma = math.hypot(self.bearing_sigma_rad * distance, self.min_position_sigma_m)
            measurement_covariance = (radial_sigma ** 2 * np.outer(direction, direction)
                                      + lateral_sigma ** 2 * np.outer(perpendicular, perpendicular))
            associations = []
            for track in self._tracks:
                residual = measurement - track.position
                squared_distance = float(residual @ np.linalg.solve(
                    track.covariance + measurement_covariance, residual))
                associations.append(squared_distance)
            if (not associations or min(associations)
                    > self.max_association_mahalanobis_distance ** 2):
                self._tracks.append(SignTrack(len(self._tracks), measurement, measurement_covariance))
                continue
            index = int(np.argmin(associations))
            track = self._tracks[index]
            delta = track.position - camera_world
            rho = float(np.linalg.norm(delta))
            if rho < 1e-9:
                raise ValueError("A tracked sign must be horizontally separated from the camera")
            bearing = math.atan2(direction[1], direction[0])
            expected_bearing = math.atan2(delta[1], delta[0])
            residual = (bearing - expected_bearing + math.pi) % (2 * math.pi) - math.pi
            if residual <= -math.pi:
                residual += 2 * math.pi
            jacobian = np.array([-delta[1], delta[0]]) / rho ** 2
            variance = (self.bearing_sigma_rad ** 2
                        + (self.min_position_sigma_m / max(rho, 1.)) ** 2)
            innovation = float(jacobian @ track.covariance @ jacobian + variance)
            gain = track.covariance @ jacobian / innovation
            position = track.position + gain * residual
            covariance = (np.eye(2) - np.outer(gain, jacobian)) @ track.covariance
            # Preserve symmetry against floating-point roundoff in the EKF update.
            covariance = (covariance + covariance.T) / 2
            self._tracks[index] = SignTrack(track.track_id, position, covariance)


__all__ = [
    'SignTrack',
    'StopSignTracker',
]
