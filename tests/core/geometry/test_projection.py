"""Plane and box projection use the same pixel-centre geometry for both lens models."""

import numpy as np
import pytest

from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.projection import plane_sampling_grid, project_box, visible_plane_mask


CAMERAS = [
    pytest.param(PinholeCamera(np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]), 2, 2),
                 .5, 1 / 3, id="pinhole"),
    pytest.param(FThetaCamera(1., 1., (0., 1.), 2, 2, (0., 1.), max_angle=1.5),
                 np.tan(np.sqrt(.5)) / np.sqrt(2), np.arctan(1 / 3),
                 id="ftheta"),
]
PLANE = np.array([[-2., -2., 2.], [2., -2., 2.], [2., 2., 2.], [-2., 2., 2.]])


@pytest.mark.parametrize("camera,grid_half,box_half", CAMERAS)
def test_plane_sampling_grid(camera, grid_half, box_half):
    grid, mask = plane_sampling_grid(camera, np.eye(4), PLANE)
    # Each pixel centre is half a pixel from the optical axis. The pinhole slope is .5;
    # the f-theta slope is tan(sqrt(.5)) / sqrt(2), from its unit angle-to-radius mapping.
    expected = [[[-grid_half, -grid_half], [grid_half, -grid_half]],
                [[-grid_half, grid_half], [grid_half, grid_half]]]
    np.testing.assert_allclose(grid, expected, atol=1e-7)
    np.testing.assert_array_equal(mask, np.ones((1, 2, 2), np.float32))
    behind = PLANE.copy()
    behind[:, 2] = -2.
    rejected_grid, rejected_mask = plane_sampling_grid(camera, np.eye(4), behind)
    np.testing.assert_array_equal(rejected_grid, np.full((2, 2, 2), 2., np.float32))
    assert not rejected_mask.any()
    with pytest.raises(ValueError, match="nondegenerate"):
        plane_sampling_grid(camera, np.eye(4), np.zeros((4, 3)))


@pytest.mark.parametrize("camera,grid_half,box_half", CAMERAS)
def test_project_box(camera, grid_half, box_half):
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


@pytest.mark.parametrize("camera,grid_half,box_half", CAMERAS)
def test_visible_plane_mask(camera, grid_half, box_half):
    depth = np.array([[1., 3.], [1., 3.]])
    visible = visible_plane_mask(camera, np.eye(4), depth, PLANE, .01)
    np.testing.assert_array_equal(visible, [[False, True], [False, True]])
    np.testing.assert_array_equal(visible_plane_mask(camera, np.eye(4),
                                  np.full((2, 2), np.inf), PLANE, .01), np.ones((2, 2), bool))
    full = visible_plane_mask(camera, np.eye(4), np.full((2, 2), np.inf), PLANE, .01)
    bounded = visible_plane_mask(camera, np.eye(4), np.full((2, 2), np.inf), PLANE, .01,
                                 bounds=(1, -2, 4, 2))
    np.testing.assert_array_equal(bounded[:, 1:], full[:, 1:])
    assert not bounded[:, :1].any()
    assert not visible_plane_mask(camera, np.eye(4), depth, PLANE, .01,
                                  bounds=(-4, 0, -1, 2)).any()
    with pytest.raises(ValueError, match="integer"):
        visible_plane_mask(camera, np.eye(4), depth, PLANE, .01, bounds=(.5, 0, 1, 2))
    with pytest.raises(ValueError, match="image shape"):
        visible_plane_mask(camera, np.eye(4), np.ones((1, 2)), PLANE, .01)


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
