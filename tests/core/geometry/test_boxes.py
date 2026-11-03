"""Ray distances and top-down overlap use full dimensions about the box centre."""

import numpy as np
import pytest

from avsectester.geometry.boxes import (
    box_ray_distances, oriented_boxes_distance, oriented_boxes_overlap,
)
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


def test_oriented_boxes_distance():
    first = np.eye(4)
    shifted = make_pose(rotation_matrix((20., -10., 0.)), (5., 6., 20.))
    rotated = make_pose(rotation_matrix((0., 0., 45.)), (4., 0., 20.))
    contact_rotation = rotation_matrix((0., 0., -178.5))
    cases = [
        # Horizontal gaps of 3 m and 4 m give a 5 m corner-to-corner distance.
        (first, (2., 2., 1.), shifted, (2., 2., 8.), 5.),
        # The rotated square's left vertex meets the first square's right edge at y=0.
        (first, (2., 2., 1.), rotated, (2., 2., 1.), 3.-np.sqrt(2.)),
        (first, (2., 2., 1.), make_pose(np.eye(3), (2., 0., 0.)), (2., 2., 1.), 0.),
        (first, (2., 2., 1.), make_pose(np.eye(3), (1., 0., 0.)), (2., 2., 1.), 0.),
        (first, (0., 0., 0.), make_pose(np.eye(3), (3., 4., 9.)), (0., 0., 0.), 5.),
        (first, (0., 0., 0.), make_pose(np.eye(3), (3., 2., 0.)), (4., 0., 0.), np.sqrt(5.)),
        (first, (4., 0., 0.), make_pose(np.eye(3), (0., 3., 0.)), (4., 0., 0.), 3.),
        (first, (4., 0., 0.), make_pose(rotation_matrix((0., 0., 90.)), (0., 0., 0.)),
         (4., 0., 0.), 0.),
        (first, (0., 0., 0.), first, (0., 0., 0.), 0.),
        (make_pose(contact_rotation, (0., 0., 0.)), (4., 2., 1.),
         make_pose(contact_rotation, contact_rotation[:, 0]*3.), (2., 2., 1.), 0.),
    ]
    for pose_a, extent_a, pose_b, extent_b, expected in cases:
        distance = oriented_boxes_distance(pose_a, extent_a, pose_b, extent_b)
        assert distance == pytest.approx(expected)
        assert oriented_boxes_distance(pose_b, extent_b, pose_a, extent_a) == pytest.approx(expected)
        assert oriented_boxes_overlap(pose_a, extent_a, pose_b, extent_b) == (distance == 0.)
    with pytest.raises(ValueError, match="nonnegative"):
        oriented_boxes_distance(first, (-1., 2., 2.), first, (2., 2., 2.))
    invalid = np.eye(4)
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match="rigid"):
        oriented_boxes_distance(invalid, (2., 2., 2.), first, (2., 2., 2.))
