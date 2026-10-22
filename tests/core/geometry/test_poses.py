"""Canonical rigid transforms, interpolation and handedness have explicit numerical expectations."""

import numpy as np
import pytest

from avsectester.geometry.poses import (
    MIRROR_Y, interpolate_transform, make_pose, mirror_y_points, mirror_y_transform,
    quaternion_to_rotation, rotation_matrix, rotation_to_quaternion, transform,
    validated_transform, validated_vector,
)


def test_rotation_matrix():
    examples = [((90, 0, 0), [0, 1, 0], [0, 0, 1]),
                ((0, 90, 0), [1, 0, 0], [0, 0, -1]),
                ((0, 0, 90), [1, 0, 0], [0, 1, 0])]
    for angles, basis, expected in examples:
        actual = rotation_matrix(angles)
        np.testing.assert_allclose(actual @ basis, expected, atol=1e-12)
        np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)
    with pytest.raises(ValueError, match="three finite"):
        rotation_matrix((0, np.nan, 0))


def test_quaternion_to_rotation():
    np.testing.assert_allclose(quaternion_to_rotation(np.sqrt(.5), 0., 0., np.sqrt(.5)),
                               rotation_matrix((0., 0., 90.)), atol=1e-12)
    np.testing.assert_array_equal(quaternion_to_rotation(2., 0., 0., 0.), np.eye(3))
    with pytest.raises(ValueError, match="nonzero"):
        quaternion_to_rotation(0., 0., 0., 0.)


def test_rotation_to_quaternion():
    for angles in ((180., 0., 0.), (0., 179.999999, 0.), (37., -52., 179.999999), (0., 0., 0.)):
        rotation = rotation_matrix(angles)
        quaternion = rotation_to_quaternion(rotation)
        assert np.linalg.norm(quaternion) == pytest.approx(1.)
        np.testing.assert_allclose(quaternion_to_rotation(*quaternion), rotation, atol=1e-12)
    with pytest.raises(ValueError, match="orthonormal"):
        rotation_to_quaternion(np.zeros((3, 3)))


def test_interpolate_transform():
    first = make_pose(rotation_matrix((0., 0., 170.)), (0., 2., 4.))
    second = make_pose(rotation_matrix((0., 0., -170.)), (4., 6., 8.))
    midpoint = interpolate_transform(first, second, .5)
    np.testing.assert_allclose(midpoint[:3, :3], rotation_matrix((0., 0., 180.)), atol=1e-12)
    np.testing.assert_allclose(midpoint[:3, 3], [2., 4., 6.])
    np.testing.assert_allclose(interpolate_transform(first, second, 0.), first, atol=1e-12)
    np.testing.assert_allclose(interpolate_transform(first, second, 1.), second, atol=1e-12)
    with pytest.raises(ValueError, match="between zero and one"):
        interpolate_transform(first, second, 1.1)


def test_make_pose_and_transform():
    pose = make_pose(rotation_matrix((0., 0., 90.)), (10., 5., 1.))
    np.testing.assert_allclose(transform(pose, np.array([[1., 0., 2.]])), [[10., 6., 3.]])
    points = np.array([[1., 2., 3.], [-3., 2., 7.]])
    np.testing.assert_allclose(transform(np.linalg.inv(pose), transform(pose, points)), points)


def test_mirror_y():
    points = np.array([[1., 2., 3.], [-4., -5., 6.]])
    np.testing.assert_array_equal(mirror_y_points(points), [[1., -2., 3.], [-4., 5., 6.]])
    np.testing.assert_array_equal(mirror_y_points(points[0]), [1., -2., 3.])
    pose = make_pose(rotation_matrix((20., -10., 45.)), (7., 8., 9.))
    reflected = mirror_y_transform(pose)
    np.testing.assert_allclose(reflected, MIRROR_Y @ pose @ MIRROR_Y)
    np.testing.assert_allclose(transform(reflected, mirror_y_points(points)),
                               mirror_y_points(transform(pose, points)))
    np.testing.assert_array_equal(mirror_y_transform(reflected), pose)
    with pytest.raises(ValueError, match="dimension of three"):
        mirror_y_points([1., 2.])


def test_validated_transform_and_vector():
    pose = np.eye(4)
    validated = validated_transform(pose)
    pose[0, 3] = 4.
    assert validated[0, 3] == 0.
    assert not validated.flags.writeable
    with pytest.raises(ValueError, match="rigid"):
        validated_transform(np.zeros((4, 4)))
    np.testing.assert_array_equal(validated_vector([1., 2., 3.], "point"), [1., 2., 3.])
    with pytest.raises(ValueError, match="three finite"):
        validated_vector([1., np.nan, 3.], "point")
