"""Physical ego integrators and the optional AlpaSim controller use public state/command APIs."""

import numpy as np
import pytest

from avsectester.closed_loop.simulators.ego_dynamics import (
    AlpaSimController, KinematicBicycle, PlanarEgoState, TrajectoryFollower,
)
from avsectester.data_types import Control


@pytest.mark.parametrize("dynamics,command,position,speed,yaw,yaw_rate", [
    pytest.param(KinematicBicycle(), Control(throttle=.5), [10., 5.015], .15,
                 np.pi / 2, 0., id="kinematic-bicycle"),
    pytest.param(TrajectoryFollower(),
                 Control(trajectory=[((2., 0., 0.), (1., 0., 0., 0.), 200000)]),
                 [10., 6.], 10., np.pi / 2, 0., id="trajectory-follower"),
    pytest.param(TrajectoryFollower(),
                 Control(trajectory=[((0., 0., 0.), (1., 0., 0., 0.), 0),
                                     ((2., 0., 0.), (1., 0., 0., 0.), 200000)]),
                 [10., 6.], 10., np.pi / 2, 0., id="trajectory-follower-with-t0"),
    pytest.param(TrajectoryFollower(),
                 Control(trajectory=[((2., 2., 0.), (1., 0., 0., 0.), 200000)]),
                 [9., 6.], np.sqrt(2) * 10, 3 * np.pi / 4, np.pi / .4,
                 id="trajectory-follower-turning"),
])
def test_step(dynamics, command, position, speed, yaw, yaw_rate):
    initial = PlanarEgoState(x=10., y=5., yaw=np.pi / 2)
    dynamics.reset(initial)
    advanced = dynamics.step(initial, command, .1)
    np.testing.assert_allclose([advanced.x, advanced.y], position, atol=1e-12)
    assert advanced.speed == pytest.approx(speed)
    assert advanced.yaw == pytest.approx(yaw)
    assert advanced.t == pytest.approx(.1)
    assert advanced.yaw_rate == pytest.approx(yaw_rate)
    assert advanced.acceleration == pytest.approx(speed / .1)
    assert initial.x == 10. and initial.speed == 0.


def _decelerating_plan(state, acceleration):
    times = np.arange(51) * .1
    positions = state.speed * times + acceleration * times ** 2 / 2
    return Control(trajectory=[
        ((float(position), 0., 0.), (1., 0., 0., 0.), round((state.t + time) * 1e6))
        for time, position in zip(times, positions)
    ])


@pytest.mark.parametrize("acceleration,expected_speed,tolerance", [
    pytest.param(0., 10., .05, id="constant-speed"),
    pytest.param(-2., 8., .3, id="decelerating"),
])
def test_alpasim_controller_step(acceleration, expected_speed, tolerance):
    pytest.importorskip("alpasim_controller")
    dynamics = AlpaSimController()
    state = PlanarEgoState(x=10., y=5., yaw=np.pi / 2, speed=10.)
    dynamics.reset(state)
    speeds = []
    for _ in range(10):
        state = dynamics.step(state, _decelerating_plan(state, acceleration), .1)
        speeds.append(state.speed)
    assert state.speed == pytest.approx(expected_speed, abs=tolerance)
    np.testing.assert_allclose(speeds, 10 + acceleration * np.arange(1, 11) * .1,
                               atol=tolerance, rtol=0)
    assert state.t == pytest.approx(1.)
    assert state.x == pytest.approx(10., abs=.01) and state.y > 5.
    assert abs(state.lateral_speed) < .01 and abs(state.yaw_rate) < .01


def test_alpasim_controller_reset():
    pytest.importorskip("alpasim_controller")
    dynamics = AlpaSimController()
    first = PlanarEgoState(speed=10.)
    with pytest.raises(RuntimeError, match="Reset AlpaSimController before stepping"):
        dynamics.step(first, _decelerating_plan(first, 0.), .1)
    dynamics.reset(first)
    advanced = dynamics.step(first, _decelerating_plan(first, 0.), .1)
    assert advanced.speed == pytest.approx(10., abs=.05)
    second = PlanarEgoState(x=40., y=-2., yaw=np.pi / 2, speed=3., t=-.5)
    dynamics.reset(second)
    valid = _decelerating_plan(second, 0.)
    missing_origin = Control(trajectory=valid.trajectory[1:])
    with pytest.raises(ValueError, match="must start at t0"):
        dynamics.step(second, missing_origin, .1)
    advanced = dynamics.step(second, valid, .1)
    assert advanced.speed == pytest.approx(3., abs=.05)
    assert advanced.t == pytest.approx(-.4)
    assert advanced.x == pytest.approx(40., abs=.01)
    assert advanced.y == pytest.approx(-1.7, abs=.01)
    with pytest.raises(ValueError, match="latest propagated state"):
        dynamics.step(second, valid, .1)
