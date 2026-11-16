"""Frame delivery and pose interpolation on independent integer clocks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from typing import Any

from avsectester.closed_loop.timing import HistoryRequirement, SensorTiming, TimingConfiguration, initial_history_times
from avsectester.data_types import DeliveredAcquisition, Observation, TimedStateEstimate, interpolate_state_estimate


@dataclass(frozen=True)
class PendingAcquisition:
    time_us: int
    t: float
    frame: int
    data: Any
    calibration: Any
    has_calibration: bool
    capture_id: int
    frame_start_us: int


class AcquisitionBuffer:
    def __init__(self, sensors: frozenset[str], period_us: int, origin_us: int,
                 timing: TimingConfiguration, history: HistoryRequirement) -> None:
        self.period_us, self.origin_us, self.history = period_us, origin_us, history
        self.pose_reporting_interval_us = timing.pose_reporting_interval_us
        self.timings = {sensor: timing.sensors.get(sensor, SensorTiming()) for sensor in sorted(sensors)}
        self.pending: dict[str, deque[PendingAcquisition]] = {sensor: deque() for sensor in sorted(sensors)}
        self.states: deque[TimedStateEstimate] = deque()
        self._support: deque[TimedStateEstimate] = deque()
        self._dropped: set[str] = set()

    def index_at(self, sensor: str, time_us: int) -> int:
        return self.timings[sensor].capture_index_at(time_us - self.origin_us, self.period_us)

    def due_between(self, start_us: int, end_us: int) -> dict[int, tuple[str, ...]]:
        result: dict[int, list[str]] = {}
        for sensor, clock in self.timings.items():
            first = clock.latest_index_at_or_before(start_us - self.origin_us, self.period_us)
            first += 1
            last = clock.latest_index_at_or_before(end_us - self.origin_us, self.period_us)
            for index in range(first, last + 1):
                time = self.origin_us + clock.capture_time_us(index, self.period_us)
                result.setdefault(time, []).append(sensor)
        return {time: tuple(sensors) for time, sensors in sorted(result.items())}

    def past_times(self) -> tuple[dict[int, tuple[str, ...]], frozenset[int]]:
        history = initial_history_times(self.timings, self.history, self.period_us,
                                        pose_reporting_interval_us=self.pose_reporting_interval_us)
        payloads, poses = history.payload_times_us, history.pose_times_us
        result: dict[int, list[str]] = {time: [] for time in poses}
        for sensor, times in payloads.items():
            for time in times:
                result.setdefault(time, []).append(sensor)
        return {time: tuple(sensors) for time, sensors in sorted(result.items())}, frozenset(poses)

    def append_state(self, observation: Observation, time_us: int) -> TimedStateEstimate:
        state = TimedStateEstimate(time_us, observation.t, observation.vehicle_state, observation.ego_speed)
        self.states.append(state)
        self._support.append(state)
        observation.state_estimates = (state,)
        return state

    def state_at(self, time_us: int) -> TimedStateEstimate:
        return interpolate_state_estimate(tuple(self._support), time_us)

    def append(self, observation: Observation, time_us: int, capture_id: int,
               sensor_ids: Sequence[str], *, frame_ranges: Mapping[str, range]) -> None:
        for sensor in sensor_ids:
            if sensor not in observation.sensor_data:
                self._dropped.add(sensor)
                continue
            self.pending[sensor].append(PendingAcquisition(
                time_us, observation.t, observation.frame, observation.sensor_data[sensor],
                observation.calibration.get(sensor), sensor in observation.calibration,
                capture_id, frame_ranges[sensor].start))

    def update_capture(self, observation: Observation, time_us: int, capture_id: int) -> None:
        for sensor, pending in self.pending.items():
            if not pending or pending[-1].capture_id != capture_id or pending[-1].time_us != time_us:
                continue
            acquisition = pending.pop()
            if sensor not in observation.sensor_data:
                self._dropped.add(sensor)
                continue
            pending.append(replace(acquisition, data=observation.sensor_data[sensor],
                calibration=observation.calibration.get(sensor),
                has_calibration=sensor in observation.calibration))

    def take_ready(self, current: Observation, time_us: int) -> Observation:
        data = {key: value for key, value in current.sensor_data.items() if key not in self.pending}
        calibration = {key: value for key, value in current.calibration.items() if key not in self.pending}
        deliveries = {}
        for sensor, pending in self.pending.items():
            ready = []
            while pending and pending[0].time_us <= time_us:
                acquisition = pending.popleft()
                ready.append(DeliveredAcquisition(acquisition.time_us, acquisition.t, acquisition.frame,
                    acquisition.data, acquisition.calibration, self.state_at(acquisition.time_us),
                    acquisition.capture_id, frame_start_us=acquisition.frame_start_us))
            if ready:
                deliveries[sensor] = tuple(ready)
                data[sensor] = ready[-1].data
                if acquisition.has_calibration:
                    calibration[sensor] = ready[-1].calibration
        states = []
        while self.states and self.states[0].time_us <= time_us:
            states.append(self.states.popleft())
        # Keep only the interpolation boundary, not delivered frame payloads.
        while len(self._support) > 1 and self._support[1].time_us <= time_us:
            self._support.popleft()
        data = {sensor: data[sensor] for sensor in dict.fromkeys((*current.sensor_data, *data)) if sensor in data}
        calibration = {sensor: calibration[sensor] for sensor in dict.fromkeys((*current.calibration, *calibration))
                       if sensor in calibration}
        # A due sensor whose every frame was removed is delivered as an empty tuple.
        dropped = sorted(self._dropped - deliveries.keys())
        self._dropped.clear()
        return replace(current, sensor_data=data, calibration=calibration,
                       deliveries={**{sensor: deliveries[sensor] for sensor in data if sensor in deliveries},
                                   **{sensor: () for sensor in dropped}},
                       state_estimates=tuple(states))



