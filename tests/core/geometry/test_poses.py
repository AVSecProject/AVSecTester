"""Canonical pose rotations use right-handed x-forward, y-left, z-up axes."""
import numpy as np
import pytest

from avsectester.geometry.poses import rotation_matrix, rotation_to_carla


@pytest.mark.parametrize("angles", [(90, 0, 0), (0, 90, 0), (0, 0, 90)])
def test_rotation_convention_matches_positive_right_hand_rule(angles):
    basis = np.array([0, 1, 0]) if angles[0] else np.array([1, 0, 0])
    expected = [0, 0, 1] if angles[0] else ([0, 0, -1] if angles[1] else [0, 1, 0])
    np.testing.assert_allclose(rotation_matrix(angles) @ basis, expected, atol=1e-12)


@pytest.mark.parametrize("pitch,yaw,roll", [
    (90., 0., 0.), (-90., 0., 0.), (90., 37., 23.), (-90., -57., 17.),
    (20., 110., -35.), (-20., -120., 65.), (89.99, 37., 23.),
])
def test_carla_rotation_conversion_round_trips_at_pitch_singularities(pitch, yaw, roll):
    matrix = rotation_matrix((-roll, -pitch, yaw))
    actual_pitch, actual_yaw, actual_roll = rotation_to_carla(matrix)
    actual = rotation_matrix((-actual_roll, -actual_pitch, actual_yaw))
    np.testing.assert_allclose(actual, matrix, atol=1e-10)


@pytest.mark.parametrize("rotation", [np.zeros((3, 3)), np.eye(4), np.diag((1., 1., -1.))])
def test_carla_rotation_conversion_rejects_non_rigid_rotations(rotation):
    with pytest.raises(ValueError, match="right-handed"):
        rotation_to_carla(rotation)
