"""CARLA rotations preserve native Euler conventions, including pitch singularities."""

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla.conversions import rotation_to_carla
from avsectester.geometry.poses import rotation_matrix


def test_rotation_to_carla():
    for pitch, yaw, roll in [(90., 0., 0.), (-90., 0., 0.), (90., 37., 23.),
                              (-90., -57., 17.), (20., 110., -35.), (89.99, 37., 23.)]:
        matrix = rotation_matrix((-roll, -pitch, yaw))
        native_pitch, native_yaw, native_roll = rotation_to_carla(matrix)
        np.testing.assert_allclose(rotation_matrix((-native_roll, -native_pitch, native_yaw)),
                                   matrix, atol=1e-10)
    with pytest.raises(ValueError, match="right-handed"):
        rotation_to_carla(np.diag((1., 1., -1.)))
