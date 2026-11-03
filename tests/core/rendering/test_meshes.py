"""Mesh file geometry, materials and object-coordinate contracts through public APIs."""
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from avsectester.rendering import MeshAsset, MeshPart, load_mesh_asset


def test_mesh_part():
    vertices = np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]])
    faces = np.array([[0, 1, 2]])
    rgba = np.array([10, 20, 30, 255], np.uint8)
    part = MeshPart(vertices, faces, color=rgba)
    vertices[:] = 9
    rgba[:] = 0
    np.testing.assert_array_equal(part.vertices, [[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]])
    np.testing.assert_array_equal(part.color, [10, 20, 30, 255])
    assert part.vertices.dtype == np.float64 and part.faces.dtype == np.int64
    assert not part.vertices.flags.writeable and not part.color.flags.writeable
    with pytest.raises(ValueError, match="invalid vertex indices"):
        replace(part, faces=np.array([[0, 1, 3]]))
    with pytest.raises(ValueError, match="exactly one color source"):
        replace(part, vertex_colors=np.tile(part.color, (3, 1)))
    with pytest.raises(ValueError, match="finite"):
        replace(part, vertices=np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="shape"):
        MeshPart(part.vertices, part.faces, uv=np.zeros((2, 2)),
                 texture=np.full((2, 2, 4), 255, np.uint8))


def test_mesh_asset():
    part = MeshPart(np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]]),
                    np.array([[0, 1, 2]]), color=np.array([10, 20, 30, 255], np.uint8))
    asset = MeshAsset([part], [0, 2, 2], "vehicle")
    assert asset.parts == (part,) and asset.extent == (0., 2., 2.)
    near_boundary = replace(part, vertices=part.vertices + [5e-7, 0., 0.])
    assert MeshAsset((near_boundary,), asset.extent, asset.category).extent == asset.extent
    with pytest.raises(ValueError, match="inside"):
        MeshAsset((replace(part, vertices=part.vertices + [2e-6, 0., 0.]),),
                  asset.extent, asset.category)
    with pytest.raises(ValueError, match="inside"):
        replace(asset, extent=(0., 1., 2.))
    with pytest.raises(ValueError, match="nonempty"):
        replace(asset, category="")
    with pytest.raises(ValueError, match="at least one"):
        replace(asset, parts=())


def test_load_mesh_asset(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    vertices = np.array([[1., 2., 3.], [5., 2., 3.], [1., 4., 6.]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    mesh.visual.vertex_colors = [[10, 20, 30, 255], [40, 50, 60, 255], [70, 80, 90, 255]]
    path = tmp_path / "source.PLY"
    mesh.export(path, file_type="ply")
    loaded = load_mesh_asset(path, category="vehicle", scale=2., forward_axis="+y", up_axis="+z")
    assert loaded.extent == (4., 8., 6.)
    np.testing.assert_array_equal(loaded.parts[0].vertices,
                                  [[-2., 4., 0.], [-2., -4., 0.], [2., 4., 6.]])
    np.testing.assert_array_equal(loaded.parts[0].vertex_colors, mesh.visual.vertex_colors)
    preserved = load_mesh_asset(path, category="vehicle", scale=.5, origin="file")
    np.testing.assert_array_equal(preserved.parts[0].vertices, vertices * .5)
    assert preserved.extent == (5., 4., 3.)
    stl = tmp_path / "plain.STL"
    mesh.export(stl, file_type="stl")
    plain, = load_mesh_asset(stl, category="object", color=[80, 90, 100]).parts
    np.testing.assert_array_equal(plain.color, [80, 90, 100, 255])

    (tmp_path / "faces.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 2\nproperty list uchar int vertex_indices\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n"
        "element edge 1\nproperty int vertex1\nproperty int vertex2\nend_header\n"
        "0 -1 0\n0 1 0\n0 1 2\n0 -1 2\n3 0 1 2 255 0 0 255\n3 0 2 3 0 0 255 255\n0 1\n")
    faces, = load_mesh_asset(tmp_path / "faces.ply", category="object").parts
    np.testing.assert_array_equal(faces.vertex_colors,
                                  np.repeat([[255, 0, 0, 255], [0, 0, 255, 255]], 3, axis=0))
    (tmp_path / "plain.mtl").write_text("newmtl painted\nKd 0.2 0.4 0.6\nd 1.0\n")
    (tmp_path / "plain.obj").write_text(
        "mtllib plain.mtl\nv 0 -1 0\nv 0 1 0\nv 0 0 2\nusemtl painted\nf 1 2 3\n")
    uniform, = load_mesh_asset(tmp_path / "plain.obj", category="object").parts
    np.testing.assert_array_equal(uniform.color, [51, 102, 153, 255])

    texture = np.array([[[240, 10, 20, 60], [10, 200, 30, 150], [40, 30, 180, 255]]], np.uint8)
    uv = np.array([[0., 0.], [1., 0.], [0., 1.]])
    surface = trimesh.Trimesh(
        vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 1.]], faces=[[0, 1, 2]], process=False)
    surface.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.SimpleMaterial(
            image=Image.fromarray(texture), diffuse=[255, 255, 255, 255]))
    surface.export(tmp_path / "textured.obj")
    textured, = load_mesh_asset(tmp_path / "textured.obj", category="object").parts
    np.testing.assert_array_equal(textured.texture, texture)
    np.testing.assert_array_equal(
        trimesh.visual.color.uv_to_color(textured.uv, Image.fromarray(textured.texture)),
        trimesh.visual.color.uv_to_color(uv, Image.fromarray(texture)))

    scene = trimesh.Scene()
    for offset, mode in enumerate(("OPAQUE", "MASK", "BLEND")):
        part = surface.copy()
        part.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(texture), baseColorFactor=[.5, 1., .25, 1.],
                alphaMode=mode, alphaCutoff=.5))
        transform = np.eye(4)
        transform[:3, 3] = [3. * offset, 0., 0.]
        scene.add_geometry(part, node_name=mode, geom_name=mode, transform=transform)
    (tmp_path / "scene").mkdir()
    scene.export(tmp_path / "scene/nodes.gltf")
    monkeypatch.chdir(tmp_path)
    parts = load_mesh_asset("scene/nodes.gltf", category="object", origin="file").parts
    assert len(parts) == 3
    by_x = sorted(parts, key=lambda item: item.vertices[:, 0].min())
    np.testing.assert_array_equal([item.vertices[:, 0].min() for item in by_x], [0., 3., 6.])
    for item, alpha in zip(by_x, ([255] * 3, [0, 255, 255], [60, 150, 255])):
        np.testing.assert_array_equal(
            item.texture[0, :, :3], np.round(texture[0, :, :3] * np.array([128, 255, 64]) / 255))
        np.testing.assert_array_equal(item.texture[0, :, 3], alpha)

    with pytest.raises(ValueError, match="no colors"):
        load_mesh_asset(stl, category="object")
    with pytest.raises(ValueError, match="orthogonal"):
        load_mesh_asset(path, category="object", forward_axis="-z", up_axis="+z")
    with pytest.raises(ValueError, match="Supported mesh formats"):
        load_mesh_asset(tmp_path / "mesh.fbx", category="object")
    trimesh.points.PointCloud(vertices).export(tmp_path / "gaussians.ply")
    with pytest.raises(ValueError, match="Gaussian PLY assets are not supported"):
        load_mesh_asset(tmp_path / "gaussians.ply", category="object")
