"""Rigid poses and box-centre transforms have explicit canonical numerical expectations."""
import numpy as np
import pytest

from avsectester.datasets import ActorState
from avsectester.geometry.poses import ActorPose, rotation_matrix, rotation_to_carla


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


def test_rotation_to_carla():
    for pitch, yaw, roll in [(90., 0., 0.), (-90., 0., 0.), (90., 37., 23.),
                              (-90., -57., 17.), (20., 110., -35.), (89.99, 37., 23.)]:
        matrix = rotation_matrix((-roll, -pitch, yaw))
        native_pitch, native_yaw, native_roll = rotation_to_carla(matrix)
        np.testing.assert_allclose(rotation_matrix((-native_roll, -native_pitch, native_yaw)),
                                   matrix, atol=1e-10)
    with pytest.raises(ValueError, match="right-handed"):
        rotation_to_carla(np.diag((1., 1., -1.)))


def test_actor_pose_and_box_transform():
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix((0., 0., 90.))
    transform[:3, 3] = (10., 5., 1.)
    actor = ActorState(transform, extent=(4., 2., 1.5), box_center=(2., 0., .75),
                       box_rotation=(0., 0., 90.))
    box = actor.box_pose()
    np.testing.assert_allclose(box.transform[:3, 3], [10., 7., 1.75])
    np.testing.assert_allclose(box.transform[:3, :3], rotation_matrix((0., 0., 180.)), atol=1e-12)
    np.testing.assert_allclose(actor.box_transform(), box.transform)
    assert box.extent == (4., 2., 1.5)
    assert ActorPose(np.eye(4)).extent is None
    with pytest.raises(ValueError, match="rigid"):
        ActorPose(np.zeros((4, 4)))
