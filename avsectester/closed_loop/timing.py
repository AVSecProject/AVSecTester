"""Integer-microsecond clocks and command selection matching AlpaSim runtime."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any
import math

from avsectester.validation import require_nonnegative_integer, require_positive_integer, timestamp_to_microseconds


def seconds_to_microseconds(seconds: float) -> int:
    """Convert a configured duration, rejecting precision below one microsecond."""
    if isinstance(seconds, bool) or not isinstance(seconds, Real) or not math.isfinite(seconds):
        raise ValueError("Time must be a finite number of seconds")
    value = float(seconds) * 1_000_000
    rounded = timestamp_to_microseconds(seconds)
    if not math.isclose(value, rounded, rel_tol=0., abs_tol=1e-6):
        raise ValueError("Time must have integer-microsecond precision")
    return rounded




@dataclass(frozen=True)
class SensorTiming:
    """Independent frame interval and shutter duration, with the first frame ending at zero."""
    frame_interval_us: int | None = None
    shutter_duration_us: int = 0

    def __post_init__(self) -> None:
        if self.frame_interval_us is not None:
            require_positive_integer(self.frame_interval_us, "frame_interval_us")
        require_nonnegative_integer(self.shutter_duration_us, "shutter_duration_us")

    def interval_us(self, decision_period_us: int) -> int:
        """Resolve an unspecified frame interval to the decision period."""
        require_positive_integer(decision_period_us, "decision_period_us")
        return decision_period_us if self.frame_interval_us is None else self.frame_interval_us

    def capture_time_us(self, index: int, decision_period_us: int) -> int:
        """Frame end relative to the run origin, including negative recorded context indices."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Frame index must be an integer")
        return index * self.interval_us(decision_period_us)

    def frame_range_us(self, index: int, decision_period_us: int) -> range:
        """Half-open shutter interval, matching Clock.ith_trigger with first_end_us=0."""
        end = self.capture_time_us(index, decision_period_us)
        return range(end - self.shutter_duration_us, end)

    def latest_index_at_or_before(self, time_us: int, decision_period_us: int) -> int:
        """Latest frame index whose end is not after the requested offset."""
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("Frame time must be integer microseconds")
        return time_us // self.interval_us(decision_period_us)

    def capture_index_at(self, time_us: int, decision_period_us: int) -> int:
        """Index of an exact frame end, rejecting times off this sensor's clock."""
        index = self.latest_index_at_or_before(time_us, decision_period_us)
        if self.capture_time_us(index, decision_period_us) != time_us:
            raise ValueError("Time is off the sensor frame clock")
        return index


@dataclass(frozen=True)
class TimingCapabilities:
    """Static simulator support for exposure, intermediate poses and recorded motion."""
    captures_between_decisions: bool = False
    motion_reports: bool = False
    recorded_ego_motion: bool = False
    camera_exposure: bool = False
    base_tick_us: int | None = None

    def __post_init__(self) -> None:
        if self.captures_between_decisions and not self.motion_reports:
            raise ValueError("Intermediate captures require motion reports")
        if self.base_tick_us is not None:
            require_positive_integer(self.base_tick_us, "base_tick_us")


@dataclass(frozen=True)
class HistoryRequirement:
    """Recorded context supplied at decision zero, using the closed-loop pose reporting clock."""
    duration_us: int = 0
    samples: int = 1

    def __post_init__(self) -> None:
        require_nonnegative_integer(self.duration_us, "duration_us")
        require_positive_integer(self.samples, "samples")


@dataclass(frozen=True)
class TimingConfiguration:
    """AlpaSim timing parameters with defaults preserving the original decision loop."""
    decision_period_us: int | None = None
    sensors: Mapping[str, SensorTiming] = field(default_factory=dict, hash=False)
    pose_reporting_interval_us: int = 0
    planner_delay_us: int = 0
    force_gt_duration_us: int = 0
    skip_driver_during_force_gt: bool = False

    def __post_init__(self) -> None:
        if self.decision_period_us is not None:
            require_positive_integer(self.decision_period_us, "decision_period_us")
        for name in ("pose_reporting_interval_us", "planner_delay_us", "force_gt_duration_us"):
            require_nonnegative_integer(getattr(self, name), name)
        if not isinstance(self.skip_driver_during_force_gt, bool):
            raise TypeError("skip_driver_during_force_gt must be boolean")
        if not isinstance(self.sensors, Mapping):
            raise TypeError("sensors must map sensor IDs to SensorTiming")
        for sensor, timing in self.sensors.items():
            if not isinstance(sensor, str) or not sensor:
                raise ValueError("Sensor timing IDs must be nonempty strings")
            if not isinstance(timing, SensorTiming):
                raise TypeError("Sensor timing values must be SensorTiming")
        object.__setattr__(self, "sensors", dict(self.sensors))

    def validate_period(self, decision_period_us: int) -> None:
        """Validate resolved frame intervals against a declared decision period."""
        require_positive_integer(decision_period_us, "decision_period_us")
        if self.force_gt_duration_us % decision_period_us:
            raise ValueError("force_gt_duration_us must be a multiple of decision_period_us")
        for clock in self.sensors.values():
            clock.interval_us(decision_period_us)

    def reporting_timestamps(self, start_us: int, end_us: int) -> tuple[int, ...]:
        """Intermediate controller reports as in alpasim_controller.system, excluding endpoints."""
        return _reporting_timestamps(start_us, end_us, self.pose_reporting_interval_us)

    def is_force_gt(self, decision_time_us: int, origin_us: int, decision_period_us: int) -> bool:
        """Whether the controller target lies in AlpaSim's inclusive force_gt_period."""
        target = decision_time_us + decision_period_us
        return target in range(origin_us, origin_us + self.force_gt_duration_us + 1)

    def is_handoff(self, decision_time_us: int, origin_us: int) -> bool:
        """Whether a decision starts exactly at the end of a positive force_gt interval."""
        return self.force_gt_duration_us > 0 and decision_time_us == origin_us + self.force_gt_duration_us


def _reporting_timestamps(start_us: int, end_us: int, interval_us: int) -> tuple[int, ...]:
    return tuple(range(start_us + interval_us, end_us, interval_us)) if interval_us else ()


class DelayBuffer:
    """AlpaSim's oldest-ready command rule, retaining a lone item before its delay expires."""
    def __init__(self, delay_us: int) -> None:
        require_nonnegative_integer(delay_us, "delay_us")
        self.delay_us = delay_us
        self._pending: deque[tuple[int, Any]] = deque()

    def add(self, obj: Any, timestamp_us: int) -> None:
        """Append an item at a strictly increasing production timestamp."""
        if self._pending and timestamp_us <= self._pending[-1][0]:
            raise ValueError("Timestamps must be in strictly ascending order")
        self._pending.append((timestamp_us, obj))

    def item_at(self, timestamp_us: int) -> tuple[int | None, Any]:
        """Return the oldest retained item after expiring eligible predecessors."""
        while len(self._pending) > 1 and timestamp_us - self._pending[1][0] >= self.delay_us:
            self._pending.popleft()
        return self._pending[0] if self._pending else (None, None)

    def at(self, timestamp_us: int) -> Any:
        """Read the selected command without its production timestamp."""
        return self.item_at(timestamp_us)[1]


@dataclass(frozen=True)
class InitialHistory:
    """Recorded payload and pose offsets, with the earliest source needed for exposure."""

    payload_times_us: Mapping[str, tuple[int, ...]]
    pose_times_us: tuple[int, ...]
    source_start_us: int


def initial_history_times(sensor_timings: Mapping[str, SensorTiming], history: HistoryRequirement,
                          decision_period_us: int, *, pose_reporting_interval_us: int = 0
                          ) -> InitialHistory:
    """Recorded frame ends and pose-clock offsets needed before the run origin.

    Frame clocks supply samples minus one payloads. Pose context uses the closed-loop
    decision and reporting timestamps, including the pose preceding its required boundary.
    """
    require_positive_integer(decision_period_us, "decision_period_us")
    require_nonnegative_integer(pose_reporting_interval_us, "pose_reporting_interval_us")
    payloads = {sensor: tuple(clock.capture_time_us(index, decision_period_us)
                             for index in range(1 - history.samples, 0))
                for sensor, clock in sensor_timings.items()}
    boundary = min(-history.duration_us,
                   *(time for times in payloads.values() for time in times), 0)
    source_start = min((boundary, *(clock.frame_range_us(index, decision_period_us).start
        for clock in sensor_timings.values() for index in range(1 - history.samples, 1))))
    if boundary == 0:
        return InitialHistory(payloads, (), source_start)
    block, remainder = divmod(boundary, decision_period_us)
    first = block * decision_period_us
    if 0 < pose_reporting_interval_us < decision_period_us:
        first += remainder // pose_reporting_interval_us * pose_reporting_interval_us
    poses = tuple(time for start in range(block * decision_period_us, 0, decision_period_us)
                  for time in (start, *_reporting_timestamps(start, start + decision_period_us,
                                                             pose_reporting_interval_us))
                  if time >= first)
    return InitialHistory(payloads, poses, min(source_start, *poses))


__all__ = [
    "seconds_to_microseconds", "SensorTiming", "TimingCapabilities",
    "HistoryRequirement", "TimingConfiguration", "DelayBuffer", "InitialHistory", "initial_history_times",
]
