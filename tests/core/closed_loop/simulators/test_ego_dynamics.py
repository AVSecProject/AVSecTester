"""Ego motion implementations share a public step contract with hand-calculated values."""

import numpy as np
import pytest

from avsectester.closed_loop.simulators.ego_dynamics import EgoPose, KinematicBicycle, TrajectoryFollower
from avsectester.data_types import Control


@pytest.mark.parametrize("dynamics,command,position,speed", [
    pytest.param(KinematicBicycle(), Control(throttle=.5), [10., 5.015], .15,
                 id="kinematic-bicycle"),
    pytest.param(TrajectoryFollower(),
                 Control(trajectory=[((2., 0., 0.), (1., 0., 0., 0.), 200000)]),
                 [10., 6.], 10., id="trajectory-follower"),
])
def test_step(dynamics, command, position, speed):
    advanced = dynamics.step(EgoPose(x=10., y=5., yaw=np.pi / 2), command, .1)
    np.testing.assert_allclose([advanced.x, advanced.y], position, atol=1e-12)
    assert advanced.speed == pytest.approx(speed)
    assert advanced.yaw == pytest.approx(np.pi / 2)
    assert advanced.t == pytest.approx(.1)
