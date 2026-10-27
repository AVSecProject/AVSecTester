"""Optional mesh queries preserve metric distances independently of lens and winding."""

import numpy as np
import pytest

from avsectester.geometry.mesh import MeshRaycaster, camera_depth


VERTICES = np.array([[-10., -10., 5.], [10., -10., 5.],
                     [10., 10., 5.], [-10., 10., 5.]])
FACES = np.array([[0, 1, 2], [0, 2, 3]])


def test_mesh_raycaster():
    vertices, faces = VERTICES.copy(), FACES.copy()
    raycaster = MeshRaycaster(vertices, faces)
    vertices[:] = 0.
    faces[:] = 0
    np.testing.assert_allclose(raycaster.distances(np.zeros((1, 3)), [[0., 0., 1.]]), [5.])
    with pytest.raises(ValueError, match="finite coordinates"):
        MeshRaycaster(np.full((3, 3), np.nan), [[0, 1, 2]])
    with pytest.raises(ValueError, match="invalid vertex index"):
        MeshRaycaster(VERTICES, [[0, 1, 4]])


def test_distances():
    raycaster = MeshRaycaster(VERTICES, FACES)
    origins = np.array([[0., 0., 0.], [0., 0., 10.], [20., 0., 0.], [0., 0., 0.]])
    directions = np.array([[0., 0., 2.], [0., 0., -4.], [0., 0., 1.], [1., 0., 0.]])
    # The two opposite directions hit opposite sides of the same triangular surface.
    np.testing.assert_array_equal(raycaster.distances(origins, directions), [5., 5., np.inf, np.inf])
    offset = np.array([1e8, -2e8, 3e8])
    np.testing.assert_allclose(MeshRaycaster(VERTICES + offset, FACES).distances(
        origins + offset, directions), [5., 5., np.inf, np.inf], rtol=0, atol=1e-6)
    empty = MeshRaycaster(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    assert np.isposinf(empty.distances(origins, directions)).all()
    with pytest.raises(ValueError, match="nonzero"):
        raycaster.distances(np.zeros((1, 3)), np.zeros((1, 3)))


def test_camera_depth(tiny_camera):
    camera = tiny_camera
    far = MeshRaycaster(VERTICES, FACES)
    near_vertices = VERTICES.copy()
    near_vertices[:, 2] = 2.
    near = MeshRaycaster(near_vertices, FACES)
    np.testing.assert_allclose(camera_depth(camera, np.eye(4), [far]), np.full((2, 2), 5.),
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(camera_depth(camera, np.eye(4), [far, near]),
                               np.full((2, 2), 2.), rtol=0, atol=1e-6)
    assert np.isposinf(camera_depth(camera, np.eye(4), [])).all()
    with pytest.raises(ValueError, match="4x4"):
        camera_depth(camera, np.eye(3), [far])
