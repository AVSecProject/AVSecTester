"""Independent frame clocks and delays match AlpaSim's runtime rules."""
import pickle

import pytest

from avsectester.closed_loop.timing import (
    DelayBuffer, HistoryRequirement, SensorTiming, TimingCapabilities, TimingConfiguration,
    initial_history_times, seconds_to_microseconds,
)


def test_seconds_to_microseconds():
    assert seconds_to_microseconds(.1 + .2) == 300000
    assert seconds_to_microseconds(-.1) == -100000
    with pytest.raises(ValueError):
        seconds_to_microseconds(.0000001)


def test_sensor_timing():
    clock = SensorTiming(33000, 17000)
    assert [(clock.frame_range_us(i, 100000).start, clock.frame_range_us(i, 100000).stop)
            for i in range(4)] == [(-17000, 0), (16000, 33000), (49000, 66000), (82000, 99000)]
    assert clock.capture_time_us(-1, 100000) == -33000
    assert clock.latest_index_at_or_before(100000, 100000) == 3
    assert SensorTiming().interval_us(100000) == 100000
    with pytest.raises(ValueError, match="off the sensor frame clock"):
        clock.capture_index_at(100000, 100000)


def test_timing_configuration():
    clocks = {"front": SensorTiming(33000)}
    configured = TimingConfiguration(sensors=clocks)
    clocks["front"] = SensorTiming(50000)
    assert configured.sensors["front"].frame_interval_us == 33000
    assert pickle.loads(pickle.dumps(configured)) == configured
    assert hash(configured) == hash(TimingConfiguration(sensors={"front": SensorTiming(33000)}))
    timing = TimingConfiguration(100000, {"front": SensorTiming(33000, 17000)},
        pose_reporting_interval_us=30000, planner_delay_us=25000, force_gt_duration_us=200000)
    assert timing.reporting_timestamps(100000, 200000) == (130000, 160000, 190000)
    assert [timing.is_force_gt(t, 0, 100000) for t in (0, 100000, 200000)] == [True, True, False]
    assert timing.is_handoff(200000, 0)
    assert TimingConfiguration().reporting_timestamps(0, 100000) == ()
    with pytest.raises(ValueError, match="planner_delay_us"):
        TimingConfiguration(planner_delay_us=-1)
    with pytest.raises(ValueError, match="force_gt_duration_us must be a multiple"):
        TimingConfiguration(force_gt_duration_us=150000).validate_period(100000)


def test_initial_history_times():
    history = initial_history_times({"front": SensorTiming(33000, 17000)},
        HistoryRequirement(150000, 3), 100000)
    assert history.payload_times_us == {"front": (-66000, -33000)}
    assert history.pose_times_us == (-200000, -100000)
    assert history.source_start_us == -200000
    empty = initial_history_times({}, HistoryRequirement(), 100000)
    assert empty.payload_times_us == {} and empty.pose_times_us == () and empty.source_start_us == 0
    default = initial_history_times({"front": SensorTiming()}, HistoryRequirement(), 100000)
    assert default.payload_times_us == {"front": ()} and default.pose_times_us == ()
    exposed = initial_history_times({"front": SensorTiming(33000, 17000)}, HistoryRequirement(), 100000)
    assert exposed.pose_times_us == () and exposed.source_start_us == -17000
    assert initial_history_times({"front": SensorTiming(33000)}, HistoryRequirement(150000, 3),
        100000, pose_reporting_interval_us=40000).pose_times_us == (-160000, -120000, -100000, -60000, -20000)



def test_delay_buffer():
    buffer = DelayBuffer(150000)
    assert buffer.item_at(0) == (None, None)
    buffer.add("a", 0)
    assert buffer.item_at(0) == (0, "a")
    buffer.add("b", 100000)
    buffer.add("c", 200000)
    assert buffer.item_at(200000) == (0, "a")
    assert buffer.item_at(250000) == (100000, "b")
    with pytest.raises(ValueError, match="strictly ascending"):
        buffer.add("d", 200000)


def test_alpasim_timing_reference(monkeypatch):
    from pathlib import Path

    source = Path("/opt/alpasim/src/runtime")
    if source.exists():
        monkeypatch.syspath_prepend(str(source))
    clock_type = pytest.importorskip("alpasim_runtime.types").Clock
    official_buffer = pytest.importorskip("alpasim_runtime.delay_buffer").DelayBuffer(73000)
    clock = clock_type(interval_us=33000, duration_us=17000, start_us=-17000, first_end_us=0)
    ours = SensorTiming(33000, 17000)
    assert [ours.frame_range_us(index, 100000) for index in range(8)] == [
        clock.ith_trigger(index).time_range_us for index in range(8)]
    buffer = DelayBuffer(73000)
    for time in (0, 100000, 200000, 300000):
        official_buffer.add(time, time)
        buffer.add(time, time)
        assert buffer.at(time) == official_buffer.at(time)


def test_timing_capabilities():
    assert not TimingCapabilities().captures_between_decisions
    assert TimingCapabilities(captures_between_decisions=True, motion_reports=True).motion_reports
    with pytest.raises(ValueError, match="Intermediate captures require motion reports"):
        TimingCapabilities(captures_between_decisions=True)
