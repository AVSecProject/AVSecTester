"""Ray distances and top-down overlap use full dimensions about the box centre."""

import numpy as np
import pytest

from avsectester.geometry.boxes import box_ray_distances, oriented_boxes_overlap
from avsectester.geometry.poses import make_pose, rotation_matrix


def test_box_ray_distances():
    pose = make_pose(np.eye(3), (0., 0., 5.))
    rays = np.array([[0., 0., 1.], [1., 0., 0.], [0., 0., -1.]])
    distances = box_ray_distances(pose, (2., 2., 2.), rays, np.eye(4))
    np.testing.assert_array_equal(distances, [4., np.inf, np.inf])
    rotated = make_pose(rotation_matrix((0., 0., 45.)), (0., 0., 5.))
    assert box_ray_distances(rotated, (4., 2., 2.), rays, np.eye(4))[0] == pytest.approx(4.)
    np.testing.assert_array_equal(box_ray_distances(np.eye(4), (2., 2., 2.), rays, np.eye(4)),
                                  [1., 1., 1.])
    np.testing.assert_array_equal(box_ray_distances(pose, (0., 2., 2.), rays, np.eye(4)),
                                  [np.inf, np.inf, np.inf])
    with pytest.raises(ValueError, match="extent"):
        box_ray_distances(pose, (-1., 2., 2.), rays, np.eye(4))


def test_oriented_boxes_overlap():
    first = np.eye(4)
    second = make_pose(np.eye(3), (3., 0., 10.))
    assert oriented_boxes_overlap(first, (4., 2., 2.), second, (2., 2., 2.))
    second[0, 3] = 3.1
    assert not oriented_boxes_overlap(first, (4., 2., 2.), second, (2., 2., 2.))
    second = make_pose(rotation_matrix((0., 0., 90.)), (3.5, 0., 0.))
    assert oriented_boxes_overlap(first, (4., 2., 2.), second, (1., 4., 2.))
    assert oriented_boxes_overlap(first, (0., 0., 0.), first, (0., 0., 0.))
    second[0, 3] = .1
    assert not oriented_boxes_overlap(first, (0., 0., 0.), second, (0., 0., 0.))
    with pytest.raises(ValueError, match="nonnegative"):
        oriented_boxes_overlap(first, (-1., 2., 2.), second, (2., 2., 2.))
