"""Plane and box projection use the same pixel-centre geometry for both lens models."""

import numpy as np
import pytest

from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.poses import ActorPose, rotation_matrix
from avsectester.geometry.projection import box_ray_distances, plane_sampling_grid, project_box, visible_plane_mask


CAMERAS = [
    pytest.param(PinholeCamera(np.array([[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]]), 2, 2),
                 .5, 1 / 3, id="pinhole"),
    pytest.param(FThetaCamera(1., 1., (0., 1.), 2, 2, (0., 1.), max_angle=1.5),
                 np.tan(np.sqrt(.5)) / np.sqrt(2), np.arctan(np.sqrt(2) / 3) / np.sqrt(2),
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
    np.testing.assert_allclose(project_box(camera, np.eye(4), pose, (2., 2., 2.)),
                               [1 - box_half, 1 - box_half, 1 + box_half, 1 + box_half])
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


def test_box_ray_distances():
    transform = np.eye(4)
    transform[:3, 3] = (0, 0, 5)
    rays = np.array([[0., 0., 1.], [1., 0., 0.], [0., 0., -1.]])
    distances = box_ray_distances(ActorPose(transform, (2, 2, 2)), rays, np.eye(4))
    np.testing.assert_array_equal(distances, [4., np.inf, np.inf])
    transform[:3, :3] = rotation_matrix((0, 0, 45))
    rotated = box_ray_distances(ActorPose(transform, (4, 2, 2)), rays, np.eye(4))
    assert rotated[0] == pytest.approx(4.)
    np.testing.assert_array_equal(box_ray_distances(ActorPose(np.eye(4), (2, 2, 2)),
                                                   rays, np.eye(4)), [1, 1, 1])
    with pytest.raises(ValueError, match="extent"):
        box_ray_distances(ActorPose(transform), rays, np.eye(4))

