"""Calibrated pinhole and f-theta lenses for the same numerical rendering contract."""
import numpy as np
import pytest

from avsectester.geometry.cameras import BODY_FROM_OPTICAL, FThetaCamera, PinholeCamera


@pytest.fixture(params=[
    pytest.param(PinholeCamera(np.array([[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]), 100, 80),
                 id="pinhole"),
    pytest.param(FThetaCamera(50, 40, (0, 50), 100, 80, (0, .02), max_angle=1.5), id="ftheta"),
])
def camera(request):
    return request.param


@pytest.fixture
def optical_from_world():
    return np.linalg.inv(BODY_FROM_OPTICAL)
