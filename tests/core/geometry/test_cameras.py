"""Pinhole and polynomial fisheye cameras share projection and inverse-ray numerics."""
import numpy as np
import pytest

from avsectester.geometry.cameras import (
    BODY_FROM_OPTICAL, FThetaCamera, PinholeCamera, make_pose, planar_rig_pose, transform,
)


CAMERAS = [PinholeCamera(np.array([[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]), 100, 80),
           FThetaCamera(50, 40, (0, 50), 100, 80, (0, .02), max_angle=1.5)]


@pytest.mark.parametrize("camera,delta", [
    pytest.param(CAMERAS[0], 10., id="pinhole"),
    pytest.param(CAMERAS[1], np.arctan(.2) * 50, id="ftheta"),
])
def test_project(camera, delta):
    points = np.array([[0., 0., 5.], [1., 0., 5.], [0., -1., 5.]])
    pixels = camera.project(points)
    np.testing.assert_allclose(pixels[0], [50, 40])
    np.testing.assert_allclose(pixels[1], [50 + delta, 40])
    np.testing.assert_allclose(pixels[2], [50, 40 - delta])
    np.testing.assert_allclose(camera.project(points * 2), pixels)


@pytest.mark.parametrize("camera", CAMERAS, ids=["pinhole", "ftheta"])
def test_unproject(camera):
    points = np.array([[0., 0., 1.], [.3, -.2, 1.], [-.7, .5, 1.]])
    rays = camera.unproject(camera.project(points))
    np.testing.assert_allclose(rays, points / np.linalg.norm(points, axis=1, keepdims=True), atol=1e-10)


def test_camera_world_transform():
    np.testing.assert_array_equal(transform(BODY_FROM_OPTICAL, [[1., 2., 3.]]), [[3., -1., -2.]])
    world_from_rig = planar_rig_pose(10., 5., np.pi / 2)
    rig_from_sensor = make_pose(np.eye(3), (1., 0., 2.))
    world_from_sensor = world_from_rig @ rig_from_sensor
    np.testing.assert_allclose(transform(world_from_sensor, [[0., 0., 0.]]), [[10., 6., 2.]])
    point = np.array([[1., 2., 3.]])
    np.testing.assert_allclose(transform(np.linalg.inv(world_from_sensor),
                                        transform(world_from_sensor, point)), point)
