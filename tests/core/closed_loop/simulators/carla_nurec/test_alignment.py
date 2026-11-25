"""Reconstruction and right-handed CARLA coordinates share a rigid alignment."""

import numpy as np

from avsectester.closed_loop.simulators.carla_nurec.alignment import SceneAlignment
from avsectester.geometry.poses import make_pose


def test_scene_alignment():
    alignment = SceneAlignment.from_scene((0., 0., 0.), make_pose(np.eye(3), [6378138., 2., 3.]))
    pose = make_pose(np.eye(3), [3., 4., 5.])
    transformed = alignment.to_carla(pose)
    np.testing.assert_array_equal(transformed, [[0., 1., 0., 6.], [0., 0., 1., 8.],
                                                [1., 0., 0., 4.], [0., 0., 0., 1.]])
    np.testing.assert_array_equal(alignment.from_carla(transformed), pose)
    vector = [1., 2., 3.]
    np.testing.assert_array_equal(alignment.vector_to_carla(vector), [2., 3., 1.])
    np.testing.assert_array_equal(alignment.vector_from_carla([2., 3., 1.]), vector)
