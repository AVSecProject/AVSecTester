"""Tiny calibrated cameras shared by numerical insertion tests."""
import numpy as np
import pytest
from avsectester.geometry.cameras import FThetaCamera, PinholeCamera


@pytest.fixture(params=[
    pytest.param((
        PinholeCamera(np.array([[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]), 100, 80),
        PinholeCamera(np.array([[50., 0., 0.], [0., 50., 40.], [0., 0., 1.]]), 100, 80),
        ), id="pinhole"),
    pytest.param((
        FThetaCamera(50, 40, (0, 50), 100, 80, (0, .02), max_angle=1.5),
        FThetaCamera(0, 40, (0, 50), 100, 80, (0, .02), max_angle=1.5),
        ), id="ftheta"),
])
def camera_pair(request):
    from copy import deepcopy
    return deepcopy(request.param)


@pytest.fixture
def camera(camera_pair):
    return camera_pair[0]


@pytest.fixture
def optical_from_world():
    return np.array([[0., -1., 0., 0.], [0., 0., -1., 0.],
                     [1., 0., 0., 0.], [0., 0., 0., 1.]])
