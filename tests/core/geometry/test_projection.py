"""Box projection bounds hold for both lens models, including curved f-theta edges."""

import numpy as np
import pytest

from avsectester.geometry.cameras import FThetaCamera, PinholeCamera, WindshieldDistortion, camera_ray_mask
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

    # A windshield deflects the rays before the lens projects them; bounds follow the deflection.
    windshield = WindshieldDistortion((.05, 1., .1, 0., .2, 0.), (.03, 0., 0., 1., 0., .3),
                                      (0., 1., 0., 0., 0., 0.), (0., 0., 0., 1., 0., 0.))
    deflected = FThetaCamera(100., 100., (0., 100.), 200, 200, (0., .01), max_angle=1.5, windshield=windshield)
    pose[:3, 3] = (.3, -.2, 3.)
    bounds = np.asarray(project_box(deflected, np.eye(4), pose, (1., 1., 1.)))
    corners = np.array([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) + pose[:3, 3]
    fractions = np.linspace(0., 1., 1001)[:, None]
    samples = np.concatenate([corners[a] * (1 - fractions) + corners[b] * fractions
                              for a in range(8) for b in range(8) if a < b and bin(a ^ b).count("1") == 1])
    pixels = deflected.project(samples)
    assert np.all(pixels >= bounds[:2] - 1e-9) and np.all(pixels <= bounds[2:] + 1e-9)
    np.testing.assert_allclose(bounds, np.concatenate((pixels.min(axis=0), pixels.max(axis=0))), atol=.5)
    # Boxes reaching beyond the lens's field, where the windshield model does not hold, or
    # crossing the camera plane, still bound their imaged edges.
    for centre, extent, imaged in (((3., 0., .6), (1., 1., .8), False), ((-2., .3, .5), (1.5, 1.5, 4.), True)):
        pose[:3, 3] = centre
        bounds = project_box(deflected, np.eye(4), pose, extent)
        corners = np.array([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * extent + centre
        samples = np.concatenate([corners[a] * (1 - fractions) + corners[b] * fractions
                                  for a in range(8) for b in range(8) if a < b and bin(a ^ b).count("1") == 1])
        pixels = deflected.project(samples[camera_ray_mask(deflected, samples)])
        pixels = pixels[((pixels >= 0) & (pixels <= [200, 200])).all(axis=1)]
        assert bool(len(pixels)) == imaged
        if imaged:
            lower, upper = np.asarray(bounds[:2]), np.asarray(bounds[2:])
            assert np.all(pixels >= lower - 1e-9) and np.all(pixels <= upper + 1e-9)
            # Bounds of a box crossing the camera plane are loose, but not by a large image share.
            assert np.abs(np.r_[lower - pixels.min(axis=0), upper - pixels.max(axis=0)]).max() < 40.

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
