"""Validate CARLA steering and timestamp-based speed targets without a server."""

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla.trajectory_tracking import TrajectoryTracker


def test_trajectory_tracker_commands():
    tracker = TrajectoryTracker(speed_kp=0.1, speed_ki=0., speed_kd=0.)
    times = np.array([0, 100000, 300000])
    path = np.array([[0., 0., 0.], [.2, 0., 0.], [1.2, 0., 0.]])
    position = np.zeros(3)
    geometry = {"wheelbase_m": 2.8, "max_wheel_angle_deg": 70.}

    straight = tracker.command(path, times, position, 0., 0., 0, dt=.05, **geometry)
    assert straight.steer == 0. and straight.throttle == pytest.approx(.2)
    elapsed = tracker.command(path, times, position, 0., 0., 100000, dt=.05, **geometry)
    beyond = tracker.command(path, times, position, 0., 0., 400000, dt=.05, **geometry)
    assert elapsed.throttle == beyond.throttle == pytest.approx(.5)
    path[-1, 1] = 1.
    assert tracker.command(path, times, position, 0., 0., 0, dt=.05, **geometry).steer < 0.
    hold = tracker.command(np.zeros_like(path), times, position, 0., 0., 0, dt=.05, **geometry)
    assert hold.brake == 1. and hold.throttle == 0.
    assert not tracker.identity.keys() & geometry.keys()


@pytest.mark.parametrize("field", ["wheelbase_m", "max_wheel_angle_deg"])
def test_trajectory_tracker_rejects_nonpositive_geometry(field):
    geometry = {"wheelbase_m": 2.8, "max_wheel_angle_deg": 70., field: 0.}
    path = np.array([[0., 0., 0.], [1., 0., 0.]])

    with pytest.raises(ValueError, match=rf"{field} must be a positive number"):
        TrajectoryTracker().command(path, np.array([0, 100000]), path[0], 0., 0., 0, **geometry)
