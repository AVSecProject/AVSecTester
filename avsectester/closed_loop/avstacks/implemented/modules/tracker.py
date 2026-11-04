"""Static sign tracks and scored visible-face vehicle tracking in horizontal world coordinates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np

from avsectester.validation import (
    require_nonnegative_number, require_positive_integer, require_positive_number,
)

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState


@dataclass(frozen=True)
class SignTrack:
    """A stationary sign's ID, horizontal world position and 2×2 covariance."""

    track_id: int
    position: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class VehicleTrack:
    """A horizontal world vehicle state and the width of its observed visible face.

    Position (2,) is in meters and velocity (2,) is in m/s. ``half_width_m`` is half the visible
    face's lateral span in meters. ``hits`` counts associated measurements, including the
    initial measurement.
    """

    track_id: int
    position: np.ndarray
    velocity: np.ndarray
    half_width_m: float
    hits: int = 1


@dataclass(frozen=True)
class _VehicleRecord:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    half_width_m: float
    hits: int = 1
    missed_updates: int = 0


class VehicleTracker:
    """Track scored visible faces with a constant-velocity Kalman filter in world x/y.

    State is [x, y, vx, vy]. Acceleration noise uses independent planar acceleration with
    standard deviation ``acceleration_noise_mps2`` and displacement/velocity factors dt²/2
    and dt. Isotropic position noise has standard deviation equal to the larger of
    ``min_measurement_noise_m`` and the rig-horizontal range times
    ``measurement_noise_fraction``. Initial velocity is zero with standard deviation
    ``initial_velocity_sigma_mps``. Initial position covariance is the measurement covariance.

    High-score measurements associate with all predicted tracks, including lost tracks.
    Low-score measurements then associate only with previously associated tracks unmatched
    in the first stage. Each stage greedily minimizes squared Mahalanobis innovation up to
    ``association_gate_chi2``. Only unmatched high-score measurements create tracks.
    Scores below ``low_score_threshold`` are ignored. Thresholds must satisfy
    0 <= low_score_threshold <= high_score_threshold <= 1.

    ``min_hits`` measurements confirm a track. Only confirmed tracks associated in the latest
    update are exposed. Lost tracks retain predictions for reacquisition until
    ``max_missed_updates`` consecutive unassociated updates delete them.
    """

    def __init__(
        self, acceleration_noise_mps2: float = 2., min_measurement_noise_m: float = .3,
        measurement_noise_fraction: float = .05, initial_velocity_sigma_mps: float = 5.,
        association_gate_chi2: float = 9.21, min_hits: int = 2,
        max_missed_updates: int = 10, high_score_threshold: float = .5,
        low_score_threshold: float = .1,
    ) -> None:
        for name, value in (
            ("acceleration_noise_mps2", acceleration_noise_mps2),
            ("min_measurement_noise_m", min_measurement_noise_m),
            ("initial_velocity_sigma_mps", initial_velocity_sigma_mps),
            ("association_gate_chi2", association_gate_chi2),
        ):
            require_positive_number(value, name)
        require_nonnegative_number(measurement_noise_fraction, "measurement_noise_fraction")
        require_positive_integer(min_hits, "min_hits")
        require_positive_integer(max_missed_updates, "max_missed_updates")
        require_nonnegative_number(high_score_threshold, "high_score_threshold")
        require_nonnegative_number(low_score_threshold, "low_score_threshold")
        if not low_score_threshold <= high_score_threshold <= 1:
            raise ValueError(
                "Score thresholds must satisfy 0 <= low_score_threshold <= high_score_threshold <= 1")
        self.acceleration_noise_mps2 = acceleration_noise_mps2
        self.min_measurement_noise_m = min_measurement_noise_m
        self.measurement_noise_fraction = measurement_noise_fraction
        self.initial_velocity_sigma_mps = initial_velocity_sigma_mps
        self.association_gate_chi2 = association_gate_chi2
        self.min_hits = min_hits
        self.max_missed_updates = max_missed_updates
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.reset()

    def reset(self) -> None:
        """Forget all vehicles and timestamps, restarting track IDs at zero."""
        self._tracks: list[_VehicleRecord] = []
        self._next_id = 0
        self._last_t: float | None = None

    @property
    def tracks(self) -> tuple[VehicleTrack, ...]:
        """Detached confirmed snapshots associated in the latest update, excluding lost tracks."""
        return tuple(VehicleTrack(record.track_id, record.state[:2].copy(),
                                  record.state[2:].copy(), record.half_width_m, record.hits)
                     for record in self._tracks
                     if record.hits >= self.min_hits and record.missed_updates == 0)

    def update(
        self, vehicle_state: PlanarEgoState, t: float,
        face_points_rig: Sequence[np.ndarray], scores: Sequence[float],
    ) -> None:
        """Predict and associate scored left/center/right visible-face points at timestamp t.

        Each measurement is a finite (3, 3) array, ordered left, center, right. The ego's
        planar x/y/yaw transforms its horizontal coordinates to the world. Equal timestamps
        skip prediction but allow measurement updates. Missed tracks retain predicted states.
        Scores must match the number of measurements and be finite values in [0, 1].
        Backward timestamps and invalid measurements or scores raise ValueError without
        updating tracks.
        """
        if isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t):
            raise ValueError("t must be a finite timestamp")
        if self._last_t is not None and t < self._last_t:
            raise ValueError("t must be nondecreasing")
        try:
            pose = np.array([vehicle_state.x, vehicle_state.y, vehicle_state.yaw], dtype=float)
        except AttributeError as error:
            raise ValueError("vehicle_state must provide x, y and yaw") from error
        if not np.isfinite(pose).all():
            raise ValueError("vehicle_state x, y and yaw must be finite")
        points = []
        for point in face_points_rig:
            point = np.asarray(point, dtype=float)
            if point.shape != (3, 3) or not np.isfinite(point).all():
                raise ValueError("Each visible-face measurement must be a finite (3, 3) array")
            points.append(point)
        score_values = np.asarray(scores, dtype=float)
        if (score_values.shape != (len(points),) or not np.isfinite(score_values).all()
                or (score_values < 0).any() or (score_values > 1).any()):
            raise ValueError("scores must match the measurements and contain finite values in [0, 1]")
        faces = np.asarray(points, dtype=float).reshape(-1, 3, 3)
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        world = faces[:, :, :2] @ rotation.T + pose[:2]
        if not np.isfinite(world).all():
            raise ValueError("Vehicle world face points must be finite")
        measurements = world[:, 1]
        half_widths = np.linalg.norm(world[:, 0] - world[:, 2], axis=1) / 2
        sigmas = np.maximum(self.min_measurement_noise_m,
                            self.measurement_noise_fraction * np.linalg.norm(faces[:, 1, :2], axis=1))
        covariances = sigmas[:, None, None] ** 2 * np.eye(2)

        dt = 0. if self._last_t is None else float(t) - self._last_t
        transition = np.eye(4)
        transition[:2, 2:] = np.eye(2) * dt
        acceleration = np.vstack((np.eye(2) * dt ** 2 / 2, np.eye(2) * dt))
        process_noise = self.acceleration_noise_mps2 ** 2 * acceleration @ acceleration.T
        predictions = [(transition @ record.state,
                        transition @ record.covariance @ transition.T + process_noise)
                       for record in self._tracks]

        costs = np.full((len(predictions), len(measurements)), np.inf)
        for index, (state, covariance) in enumerate(predictions):
            innovations = measurements - state[:2]
            innovation_covariances = covariance[:2, :2] + covariances
            solved = np.linalg.solve(innovation_covariances, innovations[..., None])[..., 0]
            costs[index] = np.einsum("ij,ij->i", innovations, solved)
        matches: dict[int, int] = {}
        used_measurements: set[int] = set()
        high = score_values >= self.high_score_threshold
        low = (score_values >= self.low_score_threshold) & ~high
        stages = (
            (np.ones(len(predictions), dtype=bool), high),
            (np.array([record.missed_updates == 0 for record in self._tracks], dtype=bool), low),
        )
        for available_tracks, available_measurements in stages:
            eligible = np.flatnonzero(
                (costs <= self.association_gate_chi2)
                & available_tracks[:, None] & available_measurements[None, :])
            for pair in eligible[np.argsort(costs.flat[eligible], kind="stable")]:
                track_index, measurement_index = divmod(int(pair), len(measurements))
                if track_index not in matches and measurement_index not in used_measurements:
                    matches[track_index] = measurement_index
                    used_measurements.add(measurement_index)

        updated = []
        for index, (record, (state, covariance)) in enumerate(zip(self._tracks, predictions)):
            if index not in matches:
                missed = record.missed_updates + 1
                if missed < self.max_missed_updates:
                    updated.append(_VehicleRecord(record.track_id, state, covariance,
                                                  record.half_width_m, record.hits, missed))
                continue
            measurement_index = matches[index]
            noise = covariances[measurement_index]
            innovation_covariance = covariance[:2, :2] + noise
            gain = np.linalg.solve(innovation_covariance, covariance[:2, :]).T
            state = state + gain @ (measurements[measurement_index] - state[:2])
            # Joseph form preserves covariance symmetry and positive semidefiniteness.
            residual = np.eye(4)
            residual[:, :2] -= gain
            covariance = residual @ covariance @ residual.T + gain @ noise @ gain.T
            covariance = (covariance + covariance.T) / 2
            updated.append(_VehicleRecord(record.track_id, state, covariance,
                                          float(half_widths[measurement_index]), record.hits + 1))
        next_id = self._next_id
        for index, position in enumerate(measurements):
            if high[index] and index not in used_measurements:
                covariance = np.zeros((4, 4))
                covariance[:2, :2] = covariances[index]
                covariance[2:, 2:] = np.eye(2) * self.initial_velocity_sigma_mps ** 2
                updated.append(_VehicleRecord(next_id, np.r_[position, 0., 0.], covariance,
                                              float(half_widths[index])))
                next_id += 1
        self._tracks = updated
        self._next_id = next_id
        self._last_t = float(t)


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
        after association. Only the camera-to-sign bearing enters the EKF.
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
    'VehicleTrack',
    'VehicleTracker',
]
