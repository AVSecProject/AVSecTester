"""Integer-microsecond clocks and command selection matching AlpaSim runtime."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, NoReturn
import math

from avsectester.validation import (
    require_nonnegative_integer, require_positive_integer, timestamp_to_microseconds,
)


class _ReadOnlyMapping(dict):
    """Read-only insertion-ordered mapping that pickles, deep-copies and serializes as a dict.

    A shallow copy returns a writable dict, like ``dict.copy``, so snapshot tools that copy a
    mapping and then fill it can record its contents.
    """

    def _deny_mutation(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("Sensor timing mappings are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = _deny_mutation
    setdefault = update = __ior__ = _deny_mutation

    def __reduce__(self) -> tuple[type[_ReadOnlyMapping], tuple[dict[Any, Any]]]:
        return type(self), (dict(self),)

    def __copy__(self) -> dict[Any, Any]:
        return dict(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> _ReadOnlyMapping:
        return type(self)(deepcopy(dict(self), memo))


def seconds_to_microseconds(seconds: float) -> int:
    """Convert a configured duration, rejecting precision below one microsecond."""
    if isinstance(seconds, bool) or not isinstance(seconds, Real) or not math.isfinite(seconds):
        raise ValueError("Time must be a finite number of seconds")
    value = float(seconds) * 1_000_000
    rounded = timestamp_to_microseconds(seconds)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-6):
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

    def resolved_frame_interval_us(self, decision_period_us: int) -> int:
        """Resolve an unspecified frame interval to the decision period."""
        require_positive_integer(decision_period_us, "decision_period_us")
        return decision_period_us if self.frame_interval_us is None else self.frame_interval_us

    def frame_end_us(self, index: int, decision_period_us: int) -> int:
        """Frame end relative to the run origin, including negative recorded context indices."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Frame index must be an integer")
        return index * self.resolved_frame_interval_us(decision_period_us)

    def shutter_range_us(self, index: int, decision_period_us: int) -> range:
        """Half-open shutter interval ending at the frame's scheduled end time."""
        end = self.frame_end_us(index, decision_period_us)
        return range(end - self.shutter_duration_us, end)

    def latest_frame_index(self, time_us: int, decision_period_us: int) -> int:
        """Latest frame index whose end is not after the requested offset."""
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("Frame time must be integer microseconds")
        return time_us // self.resolved_frame_interval_us(decision_period_us)

    def frame_index_at(self, time_us: int, decision_period_us: int) -> int:
        """Index of an exact frame end, rejecting times off this sensor's clock."""
        index = self.latest_frame_index(time_us, decision_period_us)
        if self.frame_end_us(index, decision_period_us) != time_us:
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
    """Scheduling parameters with decision-rate frames and zero delays by default."""
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
        object.__setattr__(self, "sensors", _ReadOnlyMapping(self.sensors))


def reporting_timestamps(start_us: int, end_us: int, interval_us: int) -> tuple[int, ...]:
    """Intermediate controller reports as in alpasim_controller.system, excluding endpoints."""
    return tuple(range(start_us + interval_us, end_us, interval_us)) if interval_us else ()


def validate_decision_period(timing: TimingConfiguration, period: int) -> None:
    """Validate resolved frame intervals against a declared decision period."""
    require_positive_integer(period, "decision_period_us")
    if timing.force_gt_duration_us % period:
        raise ValueError("force_gt_duration_us must be a multiple of decision_period_us")
    for clock in timing.sensors.values():
        clock.resolved_frame_interval_us(period)


def is_force_gt(timing: TimingConfiguration, decision_time_us: int, origin_us: int, period: int) -> bool:
    """Whether the controller target lies in AlpaSim's inclusive force_gt_period."""
    target = decision_time_us + period
    return target in range(origin_us, origin_us + timing.force_gt_duration_us + 1)


def is_handoff(timing: TimingConfiguration, decision_time_us: int, origin_us: int) -> bool:
    """Whether a decision starts exactly at the end of a positive force_gt interval."""
    return (timing.force_gt_duration_us > 0
            and decision_time_us == origin_us + timing.force_gt_duration_us)


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
    payloads = {
        sensor: tuple(clock.frame_end_us(index, decision_period_us)
                      for index in range(1 - history.samples, 0))
        for sensor, clock in sensor_timings.items()
    }
    boundary = min(-history.duration_us, *(time for times in payloads.values() for time in times), 0)
    source_start = min((boundary, *(clock.shutter_range_us(index, decision_period_us).start
                                   for clock in sensor_timings.values()
                                   for index in range(1 - history.samples, 1))))
    if boundary == 0:
        return InitialHistory(payloads, (), source_start)
    block, remainder = divmod(boundary, decision_period_us)
    first = block * decision_period_us
    if 0 < pose_reporting_interval_us < decision_period_us:
        first += remainder // pose_reporting_interval_us * pose_reporting_interval_us
    poses = tuple(
        time for start in range(block * decision_period_us, 0, decision_period_us)
        for time in (start, *reporting_timestamps(
            start, start + decision_period_us, pose_reporting_interval_us))
        if time >= first
    )
    return InitialHistory(payloads, poses, min(source_start, *poses))


__all__ = [
    "seconds_to_microseconds", "SensorTiming", "TimingCapabilities", "HistoryRequirement",
    "TimingConfiguration", "InitialHistory", "initial_history_times",
]
