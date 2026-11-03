"""Mesh Gaussian geometry and colors at the public NRE asset boundary."""

import numpy as np

from avsectester.closed_loop.simulators.nurec import NuRecRenderer
from avsectester.rendering import MeshAsset, MeshPart, PlacedObject
from tests.fakes import NuRecRpcBoundary

TRACK_FROM_PLY = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def _rows(paths):
    path, = paths
    return np.frombuffer(path.read_bytes().split(b"end_header\n", 1)[1], "<f4").reshape(-1, 14)


def _points_and_colors(rows, extent):
    points = rows[:, :3] @ TRACK_FROM_PLY.T * max(extent) + [0., 0., extent[2] / 2]
    return points, (rows[:, 3:6] * .28209479177387814 + .5) * 255


def test_mesh_scene_asset(monkeypatch, tmp_path):
    from scipy.spatial.transform import Rotation

    NuRecRpcBoundary().install(monkeypatch)
    vertices = np.array([[0., -.03, 0.], [0., .03, 0.], [0., .03, .06], [0., -.03, .06]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 0, 1]], np.int64)
    uv = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    # Red above blue. Transparent green replaces the blue on the right.
    texture = np.zeros((64, 64, 4), np.uint8)
    texture[:32] = [200, 0, 0, 255]
    texture[32:, :32] = [0, 0, 200, 255]
    texture[32:, 32:] = [0, 200, 0, 0]
    textured = MeshAsset((MeshPart(vertices, faces, uv=uv, texture=texture),),
                         (0., .06, .06), "object")
    triangle = np.array([[0., -.03, 0.], [0., .03, 0.], [0., -.03, .06]])
    tiny = np.array([[0., .02, .05], [0., .021, .05], [0., .02, .051]])
    colored = MeshAsset((
        MeshPart(triangle, np.array([[0, 1, 2]]), vertex_colors=np.array(
            [[20, 30, 40, 255], [80, 30, 100, 255], [20, 90, 100, 255]], np.uint8)),
        MeshPart(tiny, np.array([[0, 1, 2]]), color=np.array([0, 0, 240, 255], np.uint8)),
    ), (0., .06, .06), "object")
    pose = np.eye(4)
    pose[:3, 3] = [3., 0., .5]
    renderer = NuRecRenderer(cameras=["front"], asset_directory=tmp_path)
    try:
        renderer.load_scene("scene")
        placed = PlacedObject("textured", textured, pose)
        renderer.scene_object_tracks((placed,))
        rows = _rows(list(tmp_path.glob("*.ply")))
        points, rgb = _points_and_colors(rows, textured.extent)
        np.testing.assert_allclose(points[:, 0], 0., atol=1e-8)
        assert np.all(np.abs(points[:, 1]) <= .03 + 1e-8)
        assert np.all((points[:, 2] >= -1e-8) & (points[:, 2] <= .06 + 1e-8))
        assert .9 <= len(rows) / (.75 * .06 * .06 / .005**2) <= 1.2
        np.testing.assert_allclose(np.exp(rows[:, 7:10]) * .06,
                                   np.tile([.001, .005, .005], (len(rows), 1)), rtol=1e-6)
        rotations = Rotation.from_quat(rows[:, [11, 12, 13, 10]]).as_matrix()
        np.testing.assert_allclose(rotations[:, :, 0] @ TRACK_FROM_PLY.T,
                                   np.tile([1., 0., 0.], (len(rows), 1)), atol=1e-6)
        top, bottom = points[:, 2] > .035, points[:, 2] < .025
        assert top.any() and bottom.any()
        assert not np.any(bottom & (points[:, 1] > .005))
        np.testing.assert_allclose(rgb[top], np.tile([200., 0., 0.], (top.sum(), 1)), atol=1e-4)
        np.testing.assert_allclose(rgb[bottom], np.tile([0., 0., 200.], (bottom.sum(), 1)),
                                   atol=1e-4)
        known = set(tmp_path.glob("*.ply"))
        renderer.render_object_coverage(np.eye(4), 0, "front", placed, ())
        copies = set(tmp_path.glob("*.ply")) - known
        assert len(copies) == 2
        for path in copies:
            copy = _rows([path])
            np.testing.assert_array_equal(copy[:, :3], rows[:, :3])
            np.testing.assert_array_equal(copy[:, 6:], rows[:, 6:])

        known = set(tmp_path.glob("*.ply"))
        renderer.scene_object_tracks((PlacedObject("colored", colored, pose),))
        points, rgb = _points_and_colors(_rows(set(tmp_path.glob("*.ply")) - known),
                                         colored.extent)
        small = np.all(np.abs(rgb - [0., 0., 240.]) < 1e-3, axis=1)
        assert small.sum() == 1 and (~small).any()
        np.testing.assert_allclose(points[small], tiny.mean(axis=0)[None], atol=1e-7)
        u, v = (points[~small, 1] + .03) / .06, points[~small, 2] / .06
        np.testing.assert_allclose(
            rgb[~small], np.column_stack((20 + 60 * u, 30 + 60 * v, 40 + 60 * u + 60 * v)),
            atol=1e-4)
    finally:
        renderer.close()
