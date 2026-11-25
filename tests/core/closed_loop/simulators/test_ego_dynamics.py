"""Physical ego integrators and the optional AlpaSim controller use public state/command APIs."""

import numpy as np
import pytest

from avsectester.closed_loop.simulators.ego_dynamics import (
    AlpaSimController, EgoMotion, KinematicBicycle, PlanarEgoState, TrajectoryFollower,
)
from avsectester.data_types import Control
from tests.fakes import decelerating_command


def test_ego_motion():
    start = PlanarEgoState(yaw=np.deg2rad(179.))
    final = PlanarEgoState(x=10., yaw=np.deg2rad(-179.), t=.1)
    motion = EgoMotion((start, final))
    middle = motion.at(50000)
    assert middle.x == 5. and middle.t == .05
    assert middle.yaw == pytest.approx(np.pi)
    middle.x = 99.
    assert motion.at(50000).x == 5.
    with pytest.raises(ValueError, match="outside"):
        motion.at(200000)
    with pytest.raises(ValueError, match="strictly increasing"):
        EgoMotion((start, start))


def test_advance():
    dynamics = TrajectoryFollower()
    command = Control(trajectory=[((0., 0., 0.), (1., 0., 0., 0.), 0),
                                  ((1., 0., 0.), (1., 0., 0., 0.), 100000)])
    initial = PlanarEgoState()
    dynamics.reset(initial)
    motion = dynamics.advance(initial, command, 100000, (25000, 50000, 75000))
    assert [round(state.t * 1e6) for state in motion.states] == [0, 25000, 50000, 75000, 100000]
    assert motion.at(50000).x == pytest.approx(.5)
    assert motion.at(100000) == dynamics.step(initial, command, .1)
    assert initial == PlanarEgoState()
    with pytest.raises(ValueError, match="within the motion"):
        dynamics.advance(initial, command, 100000, (200000,))


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


def test_positive_bicycle_steer_turns_right():
    state = KinematicBicycle().step(PlanarEgoState(speed=5.), Control(steer=.5), .1)
    assert state.yaw < 0 and state.yaw_rate < 0 and state.y < 0


def test_trajectory_follower_delayed_reference():
    dynamics = TrajectoryFollower()
    initial = PlanarEgoState(x=10.)
    dynamics.reset(initial)
    command = Control(trajectory=[
        ((0., 0., 0.), (1., 0., 0., 0.), 0),
        ((2., 0., 0.), (1., 0., 0., 0.), 200000)])
    pose = dynamics.advance(initial, command, 100000).states[-1]
    assert pose.x == 11.
    assert dynamics.advance(pose, command, 100000).states[-1].x == 12.
    expired = dynamics.advance(dynamics.advance(pose, command, 100000).states[-1], command, 100000)
    assert expired.states[-1].x == 12. and expired.states[-1].speed == 0.

    dynamics.reset(PlanarEgoState())
    state = PlanarEgoState()
    for index in range(10):
        command = Control(trajectory=[
            ((0., 0., 0.), (1., 0., 0., 0.), index * 100000),
            ((1., .05, 0.), (1., 0., 0., 0.), (index + 1) * 100000)])
        state = dynamics.advance(state, command, 100000).states[-1]
        assert state.t == (index + 1) / 10
    assert state.speed == pytest.approx(np.hypot(1., .05) / .1, rel=1e-9)


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
        state = dynamics.step(state, decelerating_command(state.speed, state.t, acceleration), .1)
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
        dynamics.step(first, decelerating_command(first.speed, first.t, 0.), .1)
    dynamics.reset(first)
    advanced = dynamics.step(first, decelerating_command(first.speed, first.t, 0.), .1)
    assert advanced.speed == pytest.approx(10., abs=.05)
    second = PlanarEgoState(x=40., y=-2., yaw=np.pi / 2, speed=3., t=-.5)
    dynamics.reset(second)
    valid = decelerating_command(second.speed, second.t, 0.)
    advanced = dynamics.step(second, valid, .1)
    assert advanced.speed == pytest.approx(3., abs=.05)
    assert advanced.t == pytest.approx(-.4)
    assert advanced.x == pytest.approx(40., abs=.01)
    assert advanced.y == pytest.approx(-1.7, abs=.01)
    with pytest.raises(ValueError, match="latest propagated state"):
        dynamics.step(second, valid, .1)


def test_alpasim_controller_advance():
    pytest.importorskip("alpasim_controller")
    dynamics = AlpaSimController()
    initial = PlanarEgoState(x=10., speed=10.)
    command = decelerating_command(10., 0., 0.)
    dynamics.reset(initial)
    motion = dynamics.advance(initial, command, 100000, (33333, 66666, 99999))
    assert [round(state.t * 1e6) for state in motion.states] == [0, 33333, 66666, 99999, 100000]
    assert 10. < motion.at(33333).x < motion.at(66666).x < motion.at(100000).x
    current = motion.at(100000)
    next_state = dynamics.step(current, command, .1)
    assert next_state.t == .2 and next_state.x == pytest.approx(12., abs=.01)
    advanced = dynamics.step(next_state, decelerating_command(10., next_state.t, 0.), .1)
    assert advanced.x == pytest.approx(13., abs=.01)
    expired = dynamics.step(advanced, Control(trajectory=command.trajectory[:2]), .1)
    assert expired.t == .4 and expired.x >= advanced.x
