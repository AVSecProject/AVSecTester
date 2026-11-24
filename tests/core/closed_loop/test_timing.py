"""Frame clocks, recorded context and scheduling configuration."""
from copy import deepcopy
from dataclasses import asdict
import json
import pickle

import pytest

from avsectester.closed_loop.timing import (
    HistoryRequirement, SensorTiming, TimingCapabilities, TimingConfiguration,
    initial_history_times, seconds_to_microseconds,
)


def test_seconds_to_microseconds():
    assert seconds_to_microseconds(.1 + .2) == 300000
    assert seconds_to_microseconds(-.1) == -100000
    with pytest.raises(ValueError, match="integer-microsecond precision"):
        seconds_to_microseconds(.0000001)


def test_sensor_timing():
    assert SensorTiming().resolved_frame_interval_us(100000) == 100000
    clock = SensorTiming(33000, 17000)
    assert clock.resolved_frame_interval_us(100000) == 33000
    assert clock.frame_end_us(-2, 100000) == -66000
    assert clock.shutter_range_us(2, 100000) == range(49000, 66000)
    assert clock.shutter_range_us(-1, 100000) == range(-50000, -33000)
    assert clock.latest_frame_index(-33001, 100000) == -2
    assert clock.frame_index_at(99000, 100000) == 3
    with pytest.raises(ValueError, match="off the sensor frame clock"):
        clock.frame_index_at(100000, 100000)


def test_initial_history_times():
    history = initial_history_times({"front": SensorTiming(33000, 17000)},
        HistoryRequirement(duration_us=155000, samples=3), 100000,
        pose_reporting_interval_us=40000)
    assert history.payload_times_us == {"front": (-66000, -33000)}
    assert history.pose_times_us == (-160000, -120000, -100000, -60000, -20000)
    assert history.source_start_us == -160000
    exposed = initial_history_times({"front": SensorTiming(100000, 17000)},
                                    HistoryRequirement(samples=2), 100000)
    assert exposed.pose_times_us == (-100000,)
    assert exposed.source_start_us == -117000


def test_timing_configuration():
    clocks = {"right": SensorTiming(50000), "front": SensorTiming(33000)}
    configured = TimingConfiguration(sensors=clocks)
    clocks["front"] = SensorTiming(50000)
    assert configured.sensors["front"].frame_interval_us == 33000
    assert list(configured.sensors) == ["right", "front"]
    with pytest.raises(TypeError):
        configured.sensors["front"] = SensorTiming(50000)
    assert pickle.loads(pickle.dumps(configured)) == configured
    assert deepcopy(configured) == configured
    assert json.loads(json.dumps(asdict(configured)))["sensors"] == {
        "right": {"frame_interval_us": 50000, "shutter_duration_us": 0},
        "front": {"frame_interval_us": 33000, "shutter_duration_us": 0}}
    assert hash(configured) == hash(TimingConfiguration(sensors={
        "right": SensorTiming(50000), "front": SensorTiming(33000)}))
    with pytest.raises(ValueError, match="planner_delay_us"):
        TimingConfiguration(planner_delay_us=-1)


def test_timing_capabilities():
    with pytest.raises(ValueError, match="Intermediate captures require motion reports"):
        TimingCapabilities(captures_between_decisions=True)
