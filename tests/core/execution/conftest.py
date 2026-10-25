"""Synthetic longitudinal recordings and optional controller cases for closed-loop tests."""


import pytest

from avsectester.closed_loop.simulators.ego_dynamics import AlpaSimController, TrajectoryFollower


@pytest.fixture
def trajectory_follower_deceleration():
    return TrajectoryFollower(), 1., .02


@pytest.fixture
def alpasim_controller_deceleration():
    pytest.importorskip("alpasim_controller")
    return AlpaSimController(), 2., .3


@pytest.fixture(params=["trajectory_follower_deceleration", "alpasim_controller_deceleration"])
def deceleration_case(request):
    return request.getfixturevalue(request.param)
