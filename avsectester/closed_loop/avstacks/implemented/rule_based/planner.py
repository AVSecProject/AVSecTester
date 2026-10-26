"""Route-following cruise and constant-deceleration stops at tracked signs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from avsectester.data_types import Control, Observation
from avsectester.validation import require_nonnegative_number, require_positive_number

from .tracker import SignTrack


class _Route:
    """Arc-length parameterization with endpoint tangent continuation."""

    def __init__(self, points: np.ndarray | None, length: float) -> None:
        if points is None:
            points = np.array([[0., 0., 0.], [length, 0., 0.]])
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("route must contain at least two finite (x, y, z) points")
        points = points[:, :2]
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
        self.points = points[keep]
        if len(self.points) < 2:
            raise ValueError("route must have nonzero horizontal length")
        self.vectors = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.axes = self.vectors / self.lengths[:, None]
        self.arc = np.r_[0., np.cumsum(self.lengths)]
        self.origin_s = self.project(np.zeros(2))[0]

    def project(self, point: np.ndarray) -> tuple[float, float]:
        fractions = np.sum((point - self.points[:-1]) * self.vectors, axis=1) / self.lengths ** 2
        clipped = np.clip(fractions, 0., 1.)
        # Keep a signed projection at the ends, so signs behind the ego are ignored.
        if len(fractions) == 1:
            clipped = fractions
        else:
            clipped[0] = min(1., fractions[0])
            clipped[-1] = max(0., fractions[-1])
        candidates = self.points[:-1] + clipped[:, None] * self.vectors
        index = int(np.argmin(np.linalg.norm(point - candidates, axis=1)))
        # Lateral relevance is measured to the finite supplied route, while the
        # signed arc projection keeps the behind/ahead distinction at its ends.
        nearest = self.points[:-1] + np.clip(fractions, 0., 1.)[:, None] * self.vectors
        lateral = float(np.min(np.linalg.norm(point - nearest, axis=1)))
        return float(self.arc[index] + clipped[index] * self.lengths[index]), lateral

    def point(self, arc: float) -> tuple[np.ndarray, float]:
        index = int(np.clip(np.searchsorted(self.arc, arc, side="right") - 1, 0, len(self.axes) - 1))
        point = self.points[index] + (arc - self.arc[index]) * self.axes[index]
        yaw = math.atan2(self.axes[index, 1], self.axes[index, 0])
        return np.array([point[0], point[1], 0.]), yaw


class StopSignPlanner:
    """Plan rig-frame trajectories from bearing-EKF sign tracks.

    Track positions are supplied by a detector, known-size monocular ranging and
    bearing EKF tracking. Served track IDs are ignored after their timed stop.
    Cruise uses the IDM free-flow term and stopping uses latched constant
    deceleration to the target. The route is parameterized by
    horizontal arc length. Beyond its ends the endpoint tangent continues; this
    extends the trajectory horizon without choosing a new navigation branch.
    With no route, the planner follows a straight line ahead.

    The default stop distance of 5.5 m is based on 32 selected NuRec segments:
    map geometry and ego front overhang put the front at least 0.5 m before
    the line in at least 90% of them. This geometric criterion excludes
    monocular ranging error.

    Before braking, IDM's free-flow term is
    max_acceleration_mps2 * (1 - (v / cruise_speed)**acceleration_exponent).
    No virtual leader or following-gap interaction is used. For a stop target
    at remaining distance d, required = v**2/(2*d) when d > 0 and
    max_deceleration_mps2 otherwise. Braking starts when required reaches
    comfortable_deceleration_mps2, then remains latched for that target.
    While latched, acceleration is -required, bounded by max_deceleration_mps2.
    The latch clears when the target changes, disappears or becomes STOPPED.
    Forecast integration uses its own latch; future braking in the trajectory
    does not activate the current decision's latch. Late detections can lead
    to overshooting the target because deceleration is capped.

    A sign is relevant only within sign_lookahead_m along the route and
    max_sign_lateral_offset_m of it. The latter defaults to 10 m: the 32
    selected NuRec segments have offsets of 2.81–7.23 m; adding 2 m to the
    maximum and rounding up gives 10 m. Other approaches' stop signs inside
    this lateral range may still trigger a stop.
    """

    def __init__(
        self, stop_before_sign_m: float = 5.5, stop_duration_s: float = 2.,
        stop_speed_mps: float = .1, stop_position_tolerance_m: float = 2.,
        sign_lookahead_m: float = 60., max_sign_lateral_offset_m: float = 10.,
        cruise_speed_mps: float | None = None,
        min_cruise_speed_mps: float = 5., max_acceleration_mps2: float = 1.,
        comfortable_deceleration_mps2: float = 3., max_deceleration_mps2: float = 6.,
        acceleration_exponent: float = 4.,
        trajectory_horizon_s: float = 5., trajectory_step_s: float = .1,
    ) -> None:
        for name, value in (
            ("stop_before_sign_m", stop_before_sign_m), ("stop_duration_s", stop_duration_s),
            ("stop_speed_mps", stop_speed_mps), ("stop_position_tolerance_m", stop_position_tolerance_m),
        ):
            require_nonnegative_number(value, name)
        if cruise_speed_mps is not None:
            require_positive_number(cruise_speed_mps, "cruise_speed_mps")
        for name, value in (
            ("sign_lookahead_m", sign_lookahead_m),
            ("max_sign_lateral_offset_m", max_sign_lateral_offset_m),
            ("min_cruise_speed_mps", min_cruise_speed_mps),
            ("trajectory_horizon_s", trajectory_horizon_s), ("trajectory_step_s", trajectory_step_s),
            ("max_acceleration_mps2", max_acceleration_mps2),
            ("comfortable_deceleration_mps2", comfortable_deceleration_mps2),
            ("acceleration_exponent", acceleration_exponent),
            ("max_deceleration_mps2", max_deceleration_mps2),
        ):
            require_positive_number(value, name)
        if trajectory_horizon_s < trajectory_step_s:
            raise ValueError("trajectory_horizon_s must be at least trajectory_step_s")
        if round(trajectory_step_s * 1e6) <= 0:
            raise ValueError("trajectory_step_s must round to a positive integer number of microseconds")
        self.stop_before_sign_m = stop_before_sign_m
        self.stop_duration_s = stop_duration_s
        self.stop_speed_mps = stop_speed_mps
        self.stop_position_tolerance_m = stop_position_tolerance_m
        self.sign_lookahead_m = sign_lookahead_m
        self.max_sign_lateral_offset_m = max_sign_lateral_offset_m
        self.cruise_speed_mps = cruise_speed_mps
        self.min_cruise_speed_mps = min_cruise_speed_mps
        self.max_acceleration_mps2 = max_acceleration_mps2
        self.comfortable_deceleration_mps2 = comfortable_deceleration_mps2
        self.max_deceleration_mps2 = max_deceleration_mps2
        self.acceleration_exponent = acceleration_exponent
        self.trajectory_horizon_s = trajectory_horizon_s
        self.trajectory_step_s = trajectory_step_s
        self._processed: set[int] = set()
        self._braking = False
        self._active: int | None = None
        self._state = "CRUISE"
        self._stopped_at: float | None = None
        self._v0: float | None = None
        self._last_t: float | None = None
        self._log: dict[str, Any] = {}

    @property
    def component_log(self) -> Mapping[str, Any]:
        """State, target track ID, target distance and acceleration/speed at the first planned step.

        The controller follows the complete trajectory and may brake in advance;
        acceleration_mps2 can therefore be positive while the physical ego slows.
        """
        return dict(self._log)

    def reset(self, observation: Observation) -> None:
        """Forget served track IDs and initialize cruise speed from the observation."""
        require_nonnegative_number(observation.ego_speed, "ego_speed")
        if not math.isfinite(observation.t):
            raise ValueError("observation.t must be finite")
        self._processed = set()
        self._active = None
        self._braking = False
        self._state = "CRUISE"
        self._stopped_at = None
        self._v0 = (max(self.min_cruise_speed_mps, observation.ego_speed) if self.cruise_speed_mps is None
                    else self.cruise_speed_mps)
        self._last_t = None
        self._log = {}

    def _required_deceleration(self, speed: float, remaining: float) -> float:
        return speed ** 2 / (2 * remaining) if remaining > 0 else self.max_deceleration_mps2

    def _acceleration(self, speed: float, remaining: float | None,
                      braking: bool) -> tuple[float, bool]:
        assert self._v0 is not None
        acceleration = self.max_acceleration_mps2 * (
            1 - (speed / self._v0) ** self.acceleration_exponent)
        if remaining is not None:
            required = self._required_deceleration(speed, remaining)
            braking = braking or required >= self.comfortable_deceleration_mps2
            if braking:
                acceleration = -required
        return float(np.clip(acceleration, -self.max_deceleration_mps2,
                             self.max_acceleration_mps2)), braking

    def step(self, observation: Observation, tracks: Sequence[SignTrack]) -> Control:
        """Use horizontal world tracks to return an absolute-time rig-frame plan."""
        if self._v0 is None:
            raise RuntimeError("Reset StopSignPlanner before stepping")
        require_nonnegative_number(observation.ego_speed, "ego_speed")
        if not math.isfinite(observation.t) or (
            self._last_t is not None and observation.t < self._last_t
        ):
            raise ValueError("observation.t must be finite and nondecreasing")
        state = observation.vehicle_state
        try:
            pose = np.array([state.x, state.y, state.yaw], dtype=float)
        except AttributeError as error:
            raise ValueError("StopSignPlanner requires vehicle_state with x, y and yaw") from error
        if not np.isfinite(pose).all():
            raise ValueError("vehicle_state x, y and yaw must be finite")
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        route = _Route(observation.route, max(self.sign_lookahead_m, self._v0 * self.trajectory_horizon_s * 2))
        distances = {}
        for track in tracks:
            if track.track_id in self._processed:
                continue
            world = np.asarray(track.position, dtype=float)
            if world.shape != (2,) or not np.isfinite(world).all():
                raise ValueError("Each track position must contain two finite world coordinates")
            rig = rotation.T @ (world - pose[:2])
            projected, lateral = route.project(rig)
            arc = projected - route.origin_s
            if 0 <= arc <= self.sign_lookahead_m and lateral <= self.max_sign_lateral_offset_m:
                distances[track.track_id] = arc - self.stop_before_sign_m
        previous_target = self._active
        if self._state == "STOPPED":
            assert self._stopped_at is not None
            if observation.t - self._stopped_at >= self.stop_duration_s:
                if self._active is not None:
                    self._processed.add(self._active)
                    distances.pop(self._active, None)
                self._active = None
                self._state = "CRUISE"
                self._stopped_at = None
        if self._state != "STOPPED":
            self._active = min(distances, key=distances.__getitem__) if distances else None
            self._state = "STOPPING" if self._active is not None else "CRUISE"
            if (self._active is not None and observation.ego_speed < self.stop_speed_mps
                    and distances[self._active] <= self.stop_position_tolerance_m):
                self._state = "STOPPED"
                self._stopped_at = observation.t
        distance_to_stop = None if self._active is None else distances.get(self._active)
        if self._state != "STOPPING" or self._active != previous_target:
            self._braking = False
        if self._state == "STOPPING" and distance_to_stop is not None:
            self._braking = self._braking or (
                self._required_deceleration(observation.ego_speed, distance_to_stop)
                >= self.comfortable_deceleration_mps2)
        # A forecast may reach its braking threshold later in the horizon. Keep
        # that hypothetical latch local, separate from the actual decision state.
        simulated_braking = self._braking
        origin_us = round(observation.t * 1e6)
        trajectory = [((0., 0., 0.), (1., 0., 0., 0.), origin_us)]
        speed, progress = float(observation.ego_speed), 0.
        acceleration, first_speed = 0., 0.
        elapsed = 0.
        horizon_us = round(self.trajectory_horizon_s * 1e6)
        step_us = round(self.trajectory_step_s * 1e6)
        offsets = list(range(step_us, horizon_us + 1, step_us))
        if offsets[-1] != horizon_us:
            offsets.append(horizon_us)
        for step_index, offset_us in enumerate(offsets, 1):
            time = offset_us / 1e6
            dt = time - elapsed
            if self._state == "STOPPED":
                position, yaw = np.zeros(3), 0.
                next_speed, a = 0., 0.
            else:
                remaining = None if distance_to_stop is None else distance_to_stop - progress
                a, simulated_braking = self._acceleration(speed, remaining, simulated_braking)
                next_speed = max(0., speed + a * dt)
                progress += (speed + next_speed) * dt / 2
                position, yaw = route.point(route.origin_s + progress)
            if step_index == 1:
                acceleration, first_speed = a, next_speed
            trajectory.append(((float(position[0]), float(position[1]), float(position[2])),
                               (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)),
                               origin_us + offset_us))
            speed, elapsed = next_speed, time
        self._last_t = observation.t
        self._log = {
            "state": self._state, "target_track_id": self._active, "braking": self._braking,
            "distance_to_stop_m": distance_to_stop,
            "acceleration_mps2": acceleration, "planned_speed_mps": first_speed,
        }
        return Control(trajectory=trajectory)
