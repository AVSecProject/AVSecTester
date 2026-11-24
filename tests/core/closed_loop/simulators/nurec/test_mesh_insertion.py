"""Mesh-specific NuRec insertion: supported modes, LiDAR geometry, harmonization and depth."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.data_types import Control
from avsectester.geometry.cameras import camera_ray_mask
from avsectester.rendering import MeshAsset, MeshPart
from avsectester.rendering.harmonizers import ExposureHarmonizer
from tests.fakes import NuRecRpcBoundary


def _asset(rgba):
    vertices = np.array([[0., -.08, 0.], [0., .08, 0.],
                         [0., .08, .16], [0., -.08, .16]])
    part = MeshPart(
        vertices, np.array([[0, 1, 2], [0, 2, 3]], np.int64),
        uv=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        texture=rgba)
    return MeshAsset((part,), (0., .16, .16), "object")


def _pose(view, distance=.02):
    optical_from_object = np.array([
        [0., 1., 0., 0.], [0., 0., -1., .08],
        [-1., 0., 0., distance], [0., 0., 0., 1.]])
    return np.linalg.inv(view.cam_from_world) @ optical_from_object


def _start(dataset):
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    return replace(start, payload={
        **start.payload, "ego": replace(start.payload["ego"], velocity=(0., 0., 0.))})


def _simulator(tmp_path, *, object_insertion="scene", depth="none", lidar=None):
    return NuRecSimulator(
        NuRecRenderer(cameras=["front"], lidar=lidar, asset_directory=tmp_path),
        object_insertion=object_insertion, depth=depth, physics_update_mode="none",
        route_generator="none", min_track_duration_s=0, time_step=.1)


def test_mesh_scene_insertion(nurec_longitudinal_dataset, monkeypatch, tmp_path):
    start = _start(nurec_longitudinal_dataset)
    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.hidden = np.zeros((12, 16), bool)
    boundary.hidden[:, :8] = True
    asset = _asset(np.full((64, 64, 4), [230, 10, 90, 255], np.uint8))
    with _simulator(tmp_path / "composite", object_insertion="composite").create_runtime() as runtime:
        runtime.reset(start)
        with pytest.raises(TypeError, match="NuRecSimulator does not support asset type MeshAsset"):
            runtime.insert_object("mesh", asset, _pose(runtime.sensor_view("front")))
        assert not runtime.state().objects
    lidar = "lidar_gt_top_p128"
    with _simulator(tmp_path, lidar=lidar).create_runtime() as runtime:
        runtime.reset(start)
        pose = _pose(runtime.sensor_view("front"))
        runtime.insert_object("mesh", asset, pose)
        runtime.step(Control())
        view = runtime.sensor_view("front")
        mask = view.object_masks["mesh"]
        assert mask.any() and not mask[:, :8].any()
        assert np.any(view.data[mask] != 30)
        observation = runtime.observe()
        world_from_lidar = runtime.state().ego.pose @ observation.calibration[lidar].rig_from_lidar
        # The first two points are the fake scan's native returns.
        points = observation.sensor_data[lidar].points[2:]
        assert len(points)
        world = points @ world_from_lidar[:3, :3].T + world_from_lidar[:3, 3]
        local = (world - pose[:3, 3]) @ pose[:3, :3]
        np.testing.assert_allclose(local[:, 0], 0., atol=1e-6)
        assert (np.abs(local[:, 1]) <= .08 + 1e-6).all()
        assert ((local[:, 2] >= -1e-6) & (local[:, 2] <= .16 + 1e-6)).all()


def test_mesh_harmonizer(nurec_longitudinal_dataset, monkeypatch, tmp_path):
    start = _start(nurec_longitudinal_dataset)
    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.hidden = np.zeros((12, 16), bool)
    boundary.hidden[:, :8] = True
    asset = _asset(np.full((64, 64, 4), [230, 10, 90, 255], np.uint8))
    with _simulator(tmp_path).create_runtime() as runtime:
        runtime.reset(start)
        pose = _pose(runtime.sensor_view("front"))
        runtime.insert_object("mesh", asset, pose)
        runtime.step(Control())
        raw = runtime.sensor_view("front")
        runtime.remove_object("mesh")
        runtime.insert_object("mesh", asset, pose,
                              harmonizer=ExposureHarmonizer(reference="under", feather_sigma=0))
        before = len(boundary.requests)
        runtime.step(Control())
        adjusted = runtime.sensor_view("front")
        mask = adjusted.object_masks["mesh"]
        assert mask.any() and adjusted.composition is None
        np.testing.assert_array_equal(adjusted.data[~mask], raw.data[~mask])
        assert np.any(adjusted.data[mask] != raw.data[mask])
        # Mesh assets have no nominal textures, so even a differentiable harmonizer renders
        # no nominal copy.
        track = adjusted.native["object_tracks"]["mesh"][0]
        assert {item.track_id for request in boundary.requests[before:]
                if hasattr(request, "image_quality") for item in request.dynamic_objects
                if item.track_id.startswith("avsectester_scene_")} == {track}


def test_mesh_depth(nurec_longitudinal_dataset, monkeypatch, tmp_path):
    from avsectester.geometry.mesh import MeshRaycaster

    try:
        MeshRaycaster.check_dependencies()
    except ImportError as error:
        pytest.skip(str(error))
    start = _start(nurec_longitudinal_dataset)
    NuRecRpcBoundary().install(monkeypatch)
    opaque = np.full((64, 64, 4), [230, 10, 90, 255], np.uint8)
    perforated = opaque.copy()
    perforated[:, :32, 3] = 0
    near, far = _asset(perforated), _asset(opaque)
    with _simulator(tmp_path, depth="mesh").create_runtime() as runtime:
        runtime.reset(start)
        clean = runtime.sensor_view("front")
        near_pose, far_pose = _pose(clean), _pose(clean, .04)
        runtime.insert_object("near", near, near_pose)
        runtime.insert_object("far", far, far_pose)
        runtime.step(Control())
        view = runtime.sensor_view("front")
        near_mask, far_mask = view.object_masks["near"], view.object_masks["far"]
        far_only = far_mask & ~near_mask
        assert near_mask.any() and far_only.any()
        # Both panels lie in planes perpendicular to the optical axis.
        np.testing.assert_allclose(view.depth[near_mask], .02, atol=1e-8)
        y, x = np.nonzero(far_only)
        rays = view.camera.unproject(np.column_stack((x + .5, y + .5)))
        valid = camera_ray_mask(view.camera, rays)
        xy = .04 * rays[valid, :2] / rays[valid, 2:]
        hits = np.zeros(len(x), dtype=bool)
        hits[valid] = (np.abs(xy) <= .08).all(axis=1)
        assert hits.any()
        np.testing.assert_allclose(view.depth[y[hits], x[hits]], .04, atol=1e-8)
        assert np.isposinf(view.depth[y[~hits], x[~hits]]).all()
        uncovered = ~(near_mask | far_mask)
        np.testing.assert_array_equal(view.depth[uncovered], clean.depth[uncovered])
