"""Independent frame clocks and delays match AlpaSim's runtime rules."""
import pickle

import pytest

from avsectester.closed_loop.timing import (
    HistoryRequirement, SensorTiming, TimingCapabilities, TimingConfiguration,
    seconds_to_microseconds,
)


def test_seconds_to_microseconds():
    assert seconds_to_microseconds(.1 + .2) == 300000
    assert seconds_to_microseconds(-.1) == -100000
    with pytest.raises(ValueError):
        seconds_to_microseconds(.0000001)


def test_sensor_timing():
    assert SensorTiming().frame_interval_us is None
    assert SensorTiming(33000, 17000).shutter_duration_us == 17000
    with pytest.raises(ValueError, match="frame_interval_us"):
        SensorTiming(0)


def test_timing_configuration():
    clocks = {"front": SensorTiming(33000)}
    configured = TimingConfiguration(sensors=clocks)
    clocks["front"] = SensorTiming(50000)
    assert configured.sensors["front"].frame_interval_us == 33000
    assert pickle.loads(pickle.dumps(configured)) == configured
    assert hash(configured) == hash(TimingConfiguration(sensors={"front": SensorTiming(33000)}))
    assert HistoryRequirement(150000, 3).samples == 3
    with pytest.raises(ValueError, match="planner_delay_us"):
        TimingConfiguration(planner_delay_us=-1)


def test_timing_capabilities():
    assert not TimingCapabilities().captures_between_decisions
    assert TimingCapabilities(captures_between_decisions=True, motion_reports=True).motion_reports
    with pytest.raises(ValueError, match="Intermediate captures require motion reports"):
        TimingCapabilities(captures_between_decisions=True)
