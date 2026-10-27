"""Ground contact fits the road while rejecting implausible corrections."""
import numpy as np
import pytest

from avsectester.closed_loop.simulators.ground_contact import constrain_to_ground
from avsectester.geometry.mesh import MeshRaycaster
from avsectester.geometry.poses import make_pose, rotation_matrix


def ground(pitch, height=0.0):
    vertices = np.array([[-20., -20., 0.], [20., -20., 0.],
                         [20., 20., 0.], [-20., 20., 0.]])
    vertices[:, 2] = height + np.tan(np.deg2rad(pitch)) * vertices[:, 0]
    return MeshRaycaster(vertices, np.array([[0, 1, 2], [0, 2, 3]]))


def test_constrain_to_ground():
    for pitch in (-5., 5.):
        pose = make_pose(np.eye(3), (0., 1., .75))
        updated, status = constrain_to_ground(ground(pitch), pose, (4., 2., 1.5))
        assert status == "successful_update"
        np.testing.assert_allclose(updated[:3, :3], rotation_matrix((0, -pitch, 0)), atol=1e-12)
        np.testing.assert_allclose(updated[:3, 3], pose[:3, 3], atol=1e-12)
        bottom = np.array([[-2., -1., -.75], [2., -1., -.75],
                           [2., 1., -.75], [-2., 1., -.75]])
        world_bottom = bottom @ updated[:3, :3].T + updated[:3, 3]
        residual = world_bottom[:, 2] - np.tan(np.deg2rad(pitch)) * world_bottom[:, 0]
        # The official centroid-and-rotation fit leaves a small second-order contact residual.
        np.testing.assert_allclose(residual, -.75 * (1 / np.cos(np.deg2rad(pitch)) - 1), atol=1e-12)
        assert np.max(np.abs(residual)) < .003
        np.testing.assert_array_equal(pose, make_pose(np.eye(3), (0., 1., .75)))
        with pytest.raises(ValueError, match="positive box dimensions"):
            constrain_to_ground(ground(pitch), pose, (4., 0., 1.5))

    pose = make_pose(np.eye(3), (0., 1., .75))
    for surface, expected in [(ground(0., 3.), "high_translation"),
                              (ground(15.), "high_rotation"),
                              (ground(-15.), "high_rotation")]:
        updated, status = constrain_to_ground(surface, pose, (4., 2., 1.5))
        assert status == expected
        np.testing.assert_array_equal(updated, pose)

    empty = MeshRaycaster(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    pose = make_pose(np.eye(3), (0., 1., .75))
    updated, status = constrain_to_ground(empty, pose, (4., 2., 1.5))
    assert status == "insufficient_intersections"
    np.testing.assert_array_equal(updated, pose)
    distant, status = constrain_to_ground(ground(0, 101.0), pose, (4., 2., 1.5))
    assert status == "insufficient_intersections"
    np.testing.assert_array_equal(distant, pose)
    narrow = MeshRaycaster(np.array([[-2.1, -2., 0.], [-1.9, -2., 0.],
                                      [-1.9, 2., 0.], [-2.1, 2., 0.]]),
                           np.array([[0, 1, 2], [0, 2, 3]]))
    above = make_pose(np.eye(3), (0., 0., 1.0))
    updated, status = constrain_to_ground(narrow, above, (4., 2., 1.5))
    assert status == "insufficient_intersections"  # Only one grid column: four hits.
    np.testing.assert_array_equal(updated, above)
