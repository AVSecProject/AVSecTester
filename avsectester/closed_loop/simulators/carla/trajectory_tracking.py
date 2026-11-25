# Adapted from CARLA PythonAPI/examples/av_stacks/alpamayo/alpamayo_controller.py.
#
# MIT License
#
# Copyright (c) 2024 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
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

"""Pure-pursuit steering and longitudinal speed control for CARLA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from avsectester.data_types import Control
from avsectester.validation import require_nonnegative_number, require_positive_number


@dataclass
class TrajectoryTracker:
    """Track a right-handed trajectory with pure pursuit and speed PID.

    Positions are in meters, heading is in radians, and waypoint timestamps are absolute integer
    microseconds. CARLA actuator steer is the negative of the right-handed steering request,
    so positive output turns right. The segment ending at or just after the tick endpoint
    determines the speed from its horizontal distance and timestamp difference. Beyond the
    horizon, use the last segment. PID and steering-rate limits use the actual tick duration.
    """

    lookahead_m: float = 3.0
    speed_kp: float = 0.35
    speed_ki: float = 0.04
    speed_kd: float = 0.02
    speed_integral_limit: float = 5.0
    stop_target_speed_mps: float = 0.25
    hold_speed_mps: float = 0.15
    minimum_stop_brake: float = 0.25
    hold_brake: float = 1.0
    steering_gain: float = 1.0
    max_steer_rate_per_s: float = 1.0
    _previous_steer: float = field(init=False, default=0.0, repr=False, compare=False)
    _speed_integral: float = field(init=False, default=0.0, repr=False, compare=False)
    _previous_speed_mps: float | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in self.identity:
            require_nonnegative_number(getattr(self, name), name)
        require_positive_number(self.lookahead_m, "lookahead_m")
        if self.minimum_stop_brake > 1.0 or self.hold_brake > 1.0:
            raise ValueError("Stop and hold brakes must be in [0, 1]")

    @property
    def identity(self) -> dict[str, float]:
        """Return the tuning without controller history."""
        return {item.name: getattr(self, item.name) for item in fields(self) if item.init}

    def reset(self) -> None:
        """Clear steering-rate and longitudinal PID history."""
        self._previous_steer = 0.0
        self._speed_integral = 0.0
        self._previous_speed_mps = None

    def _longitudinal_command(self, target_speed: float, speed: float, dt: float) -> tuple[float, float]:
        stopping = target_speed <= self.stop_target_speed_mps
        if stopping:
            target_speed = 0.0
            self._speed_integral = min(self._speed_integral, 0.0)

        error = target_speed - speed
        derivative = (0.0 if self._previous_speed_mps is None
                      else -(speed - self._previous_speed_mps) / dt)
        self._previous_speed_mps = speed
        limit = self.speed_integral_limit
        candidate = float(np.clip(self._speed_integral + error * dt, -limit, limit))
        effort = self.speed_kp * error + self.speed_ki * candidate + self.speed_kd * derivative
        if not ((effort > 1.0 and error > 0.0) or (effort < -1.0 and error < 0.0)):
            self._speed_integral = candidate
        effort = float(np.clip(self.speed_kp * error + self.speed_ki * self._speed_integral
                               + self.speed_kd * derivative, -1.0, 1.0))

        if stopping:
            brake = self.hold_brake if speed <= self.hold_speed_mps else max(
                -effort, self.minimum_stop_brake)
            return 0.0, brake
        return max(0.0, effort), max(0.0, -effort)

    def command(
        self, path: np.ndarray, waypoint_times_us: np.ndarray, position: np.ndarray,
        yaw_rad: float, speed_mps: float, time_us: int, dt: float = 0.1,
        *, wheelbase_m: float, max_wheel_angle_deg: float,
    ) -> Control:
        """Return a CARLA command for one tick using the measured vehicle geometry.

        The wheelbase in meters and maximum wheel angle in degrees must be positive.
        """
        path = np.asarray(path, dtype=float)
        position = np.asarray(position, dtype=float)
        times = np.asarray(waypoint_times_us)
        require_positive_number(dt, "dt")
        require_positive_number(wheelbase_m, "wheelbase_m")
        require_positive_number(max_wheel_angle_deg, "max_wheel_angle_deg")
        require_nonnegative_number(speed_mps, "speed_mps")
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError("path must contain at least two finite XYZ positions")
        if times.shape != (len(path),) or times.dtype.kind not in "iu" or np.any(times[1:] <= times[:-1]):
            raise ValueError("Waypoint timestamps must be increasing integers matching path")
        if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(yaw_rad):
            raise ValueError("Current position and heading must be finite")

        distances = np.linalg.norm(path[:, :2] - position[:2], axis=1)
        nearest = int(np.argmin(distances))
        candidates = np.flatnonzero(distances[nearest:] >= self.lookahead_m)
        target = nearest + int(candidates[0]) if len(candidates) else len(path) - 1
        dx, dy = path[target, :2] - position[:2]
        lateral = -math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
        curvature = 2.0 * lateral / max(float(dx * dx + dy * dy), 1e-6)
        wheel_angle = math.atan(wheelbase_m * curvature)
        requested = float(np.clip(-self.steering_gain * wheel_angle
                                  / math.radians(max_wheel_angle_deg), -1.0, 1.0))
        delta = self.max_steer_rate_per_s * dt
        steer = float(np.clip(requested, self._previous_steer - delta, self._previous_steer + delta))
        self._previous_steer = steer

        end = time_us + round(dt * 1_000_000)
        right = min(len(times) - 1, max(1, int(np.searchsorted(times, end))))
        elapsed = (int(times[right]) - int(times[right - 1])) / 1_000_000
        target_speed = float(np.linalg.norm(path[right, :2] - path[right - 1, :2])) / elapsed
        throttle, brake = self._longitudinal_command(target_speed, speed_mps, dt)
        return Control(throttle=throttle, steer=steer, brake=brake)


__all__ = ["TrajectoryTracker"]
