"""Box projection bounds hold for both lens models, including curved f-theta edges."""

import numpy as np
import pytest

from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.projection import project_box


CAMERAS = [
    pytest.param(PinholeCamera(np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]), 2, 2),
                 1 / 3, id="pinhole"),
    pytest.param(FThetaCamera(1., 1., (0., 1.), 2, 2, (0., 1.), max_angle=1.5),
                 np.arctan(1 / 3), id="ftheta"),
]


@pytest.mark.parametrize("camera,box_half", CAMERAS)
def test_project_box(camera, box_half):
    pose = np.eye(4)
    pose[2, 3] = 4.
    # The nearest box corners have optical coordinates (+/-1, +/-1, 3).
    bounds = project_box(camera, np.eye(4), pose, (2., 2., 2.))
    np.testing.assert_allclose(bounds,
                               [1 - box_half, 1 - box_half, 1 + box_half, 1 + box_half], atol=.01)
    pose[2, 3] = -4.
    assert project_box(camera, np.eye(4), pose, (2., 2., 2.)) is None
    pose[2, 3] = .1
    np.testing.assert_allclose(project_box(camera, np.eye(4), pose, (2., 2., 2.)), [0., 0., 2., 2.])
    pose[:3, 3] = (100., 0., 4.)
    assert project_box(camera, np.eye(4), pose, (2., 2., 2.)) is None
    with pytest.raises(ValueError, match="nonnegative"):
        project_box(camera, np.eye(4), pose, (-1., 2., 2.))


def test_project_box_curved_edges():
    camera = FThetaCamera(100., 100., (0., 100.), 200, 200, (0., .01), max_angle=1.5)
    pose = np.eye(4)
    pose[2, 3] = 2.
    bounds = project_box(camera, np.eye(4), pose, (2., 2., 2.))
    assert bounds is not None
    # The nearest edge midpoints extend farther than the corners in an f-theta image.
    expected_min = 100. - 100. * np.pi / 4
    expected_max = 100. + 100. * np.pi / 4
    assert bounds[0] <= expected_min and bounds[1] <= expected_min
    assert bounds[2] >= expected_max and bounds[3] >= expected_max
    edge = np.column_stack((np.linspace(-1., 1., 1001), np.ones(1001), np.ones(1001)))
    pixels = camera.project(edge)
    assert np.all(pixels >= bounds[:2])
    assert np.all(pixels <= bounds[2:])

    pose[:3, 3] = (-7.3, .6, .01)
    assert project_box(camera, np.eye(4), pose, (4., 2., 1.5)) is None
    pose[:3, 3] = (-1.5, .1, 1.)
    bounds = project_box(camera, np.eye(4), pose, (1., 1., 3.))
    assert bounds is not None
    visible_edge = np.column_stack((np.full(1001, -1.), np.full(1001, .1),
                                   np.linspace(.001, 2.5, 1001)))
    pixels = camera.project(visible_edge)
    in_image = ((pixels >= 0) & (pixels <= [camera.width, camera.height])).all(axis=1)
    assert in_image.any()
    assert np.all(pixels[in_image] >= bounds[:2])
    assert np.all(pixels[in_image] <= bounds[2:])
