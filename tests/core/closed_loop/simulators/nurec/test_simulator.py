"""NuRec actor editing, depth support and checkpoint behavior through its public runtime."""

from dataclasses import replace
import json
import zipfile

import numpy as np
import pytest

from avsectester.closed_loop.simulators.ego_dynamics import TrajectoryFollower
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.nurec.renderer import pose_from_proto
from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.poses import rotation_matrix
from avsectester.data_types import Control
from tests.fakes import NuRecImageBoundary, NuRecRpcBoundary


def test_reset_requires_mesh(nurec_usdz_path, tmp_path, monkeypatch):
    archive_path = tmp_path / "without-ground.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(archive_path, "w") as target:
        for member in source.infolist():
            if member.filename != "mesh_ground.ply":
                target.writestr(member.filename, source.read(member.filename))
    dataset = NuRecDataset([archive_path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    boundary = NuRecRpcBoundary().install(monkeypatch)
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        with pytest.raises(ValueError, match=r"missing: mesh_ground\.ply"):
            runtime.reset(start)
        assert boundary.requests == []
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"]), depth="none").create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.sensor_view("front").depth is None
        assert boundary.last_request is not None


def test_checkpoint(monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    simulator = NuRecSimulator({"dt": .1, "ego0": {"x": 10., "y": 5., "yaw": np.pi / 2}},
                               NuRecRenderer(cameras=["front"]), TrajectoryFollower())
    with simulator.create_runtime() as runtime:
        runtime.reset()
        runtime.step(Control(trajectory=[((2., 0., 0.), (1., 0., 0., 0.), 200000)]))
        saved = runtime.checkpoint()
        runtime.step(Control())
        expected = runtime.state()
        runtime.restore(saved)
        runtime.step(Control())
        np.testing.assert_allclose(runtime.state().ego.pose, expected.ego.pose)


def test_set_actor_state(nurec_usdz_path, monkeypatch, make_object_in_front):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = NuRecDataset([nurec_usdz_path], time_step=.1)
    frames = dataset.frames("scene")
    start = dataset.scenario_start("scene", frames[2])
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        runtime.reset(start)
        recorded = dataset.frame_data("scene", frames[2]).actors
        request_actors = {item.track_id: item for item in boundary.last_request.dynamic_objects}
        assert request_actors.keys() == recorded.keys() == {"host", "brief"}
        for identity, state in recorded.items():
            pair = request_actors[identity].pose_pair
            # Round-trip the recorded centre pose, including its xyzw orientation, without
            # an extra half-height shift or quaternion component permutation.
            np.testing.assert_allclose(pose_from_proto(pair.start_pose), state.pose, atol=1e-12)
            np.testing.assert_allclose(pose_from_proto(pair.end_pose), state.pose, atol=1e-12)
            np.testing.assert_allclose(runtime.sensor_view("front").actors[identity].box_transform(),
                                       state.box_transform(), atol=1e-12)

        captured = runtime.sensor_view("front")
        asset, world_from_object = make_object_in_front(captured)
        runtime.insert_object("inserted-sign", asset, world_from_object)
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix((8., -12., 30.))
        pose[:3, 3] = [6., -1., 1.2]
        velocity = np.array([1., -.5, .2])
        overridden = replace(runtime.state().actors["brief"], pose=pose, velocity=tuple(velocity))
        runtime.set_actor_state("brief", overridden)
        np.testing.assert_allclose(runtime.state().actors["brief"].pose, pose)
        np.testing.assert_allclose(runtime.sensor_view("front").actors["brief"].box_transform(),
                                   captured.actors["brief"].box_transform())
        np.testing.assert_array_equal(runtime.observe().sensor_data["front"], captured.data)

        runtime.step(Control())
        state = runtime.state()
        expected = pose.copy()
        expected[:3, 3] += velocity * .1
        np.testing.assert_allclose(state.actors["brief"].pose, expected)
        assert state.actors["brief"].velocity == tuple(velocity)
        request_actors = {item.track_id: item for item in boundary.last_request.dynamic_objects}
        assert request_actors.keys() == {"host", "brief"}
        assert "inserted-sign" in runtime.state().objects
        for identity, actor in state.actors.items():
            np.testing.assert_allclose(pose_from_proto(request_actors[identity].pose_pair.start_pose),
                                       actor.pose, atol=1e-12)
        np.testing.assert_allclose(state.actors["host"].pose,
                                   dataset.frame_data("scene", frames[3]).actors["host"].pose)
        np.testing.assert_allclose(runtime.sensor_view("front").actors["brief"].box_transform(),
                                   state.actors["brief"].box_transform())

        saved = runtime.checkpoint()
        runtime.step(Control())
        expected[:3, 3] += velocity * .1
        np.testing.assert_allclose(runtime.state().actors["brief"].pose, expected)
        # An overridden actor persists even after its recorded track disappears.
        assert "brief" not in dataset.frame_data("scene", frames[4]).actors
        assert {item.track_id for item in boundary.last_request.dynamic_objects} == {"host", "brief"}
        runtime.restore(saved)
        np.testing.assert_allclose(runtime.state().actors["brief"].pose, state.actors["brief"].pose)
        runtime.set_actor_state("brief", replace(state.actors["brief"], velocity=(0., 0., 0.)))
        runtime.step(Control())
        np.testing.assert_allclose(runtime.state().actors["brief"].pose, state.actors["brief"].pose)

        runtime.reset(start)
        np.testing.assert_allclose(runtime.state().actors["brief"].pose, recorded["brief"].pose)
        with pytest.raises(KeyError, match="Unknown NuRec actor 'missing'; available:"):
            runtime.set_actor_state("missing", overridden)
        with pytest.raises(ValueError, match="ego is controlled by the driving system"):
            runtime.set_actor_state("ego", overridden)
        with pytest.raises(ValueError, match="require a finite world velocity"):
            runtime.set_actor_state("brief", replace(overridden, velocity=None))
        with pytest.raises(NotImplementedError, match="cannot control actors"):
            runtime.control_actor("brief", Control())


@pytest.mark.parametrize("camera, ground_depth", [
    (PinholeCamera(np.array([[8., 0., 8.], [0., 8., 6.], [0., 0., 1.]]), 16, 12), 3.2),
    (FThetaCamera(8., 6., (0., 8.), 16, 12, (0., .125), max_angle=1.5),
     1.8 * np.hypot(.5, 4.5) / (4.5 * np.tan(np.hypot(.5, 4.5) / 8.))),
], ids=["pinhole", "f-theta"])
def test_depth(nurec_dataset, camera, ground_depth):
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    simulator = NuRecSimulator({"dt": .1}, NuRecImageBoundary(camera))
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        view = runtime.sensor_view("front")
        # Camera height is 1.8 m; the cube's front is x=4 m. The lower ray
        # reaches the ground before the cube, while the upper corner misses both.
        assert view.depth.shape == (12, 16)
        np.testing.assert_allclose(view.depth[6, 8], 4., atol=1e-6, rtol=0)
        np.testing.assert_allclose(view.depth[10, 8], ground_depth, atol=1e-6, rtol=0)
        assert np.isposinf(view.depth[0, 0])
        view.depth[6, 8] = 100.
        assert runtime.sensor_view("front").depth[6, 8] == pytest.approx(4., abs=1e-6)

        pose = np.eye(4)
        pose[:3, 3] = [3., 0., 1.]
        runtime.set_actor_state("host", replace(runtime.state().actors["host"],
                                                pose=pose, velocity=(0., 0., 0.)))
        # Moving an actor changes the next capture, preserving the current one.
        assert runtime.sensor_view("front").depth[6, 8] == pytest.approx(4., abs=1e-6)
        runtime.step(Control())
        np.testing.assert_allclose(runtime.sensor_view("front").depth[6, 8], .8,
                                   atol=1e-6, rtol=0)
    with NuRecSimulator({"dt": .1}, NuRecImageBoundary(camera), depth="none").create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.sensor_view("front").depth is None
    with pytest.raises(ValueError, match="depth must be 'mesh' or 'none'"):
        NuRecSimulator({"dt": .1}, NuRecImageBoundary(camera), depth="unsupported")


def test_depth_without_default_camera(nurec_usdz_path, tmp_path, monkeypatch):
    archive_path = tmp_path / "side_only.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source:
        metadata = json.loads(source.read("rig_trajectories.json"))
        previous = next(iter(metadata["camera_calibrations"]))
        replacement = "side@scene"
        metadata["camera_calibrations"] = {
            replacement: metadata["camera_calibrations"][previous]}
        for trajectory in metadata["rig_trajectories"]:
            for field in ("cameras_frame_timestamps_us", "cameras_frame_T_rig_worlds"):
                trajectory[field] = {replacement: trajectory[field][previous]}
        with zipfile.ZipFile(archive_path, "w") as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(metadata) if name == "rig_trajectories.json"
                                else source.read(name))
    dataset = NuRecDataset([str(archive_path)], camera_id="side", time_step=.1)
    assert list(dataset.raw_metadata("scene")["rig_trajectories.json"][
        "camera_calibrations"]) == ["side@scene"]
    camera, _ = dataset.camera_geometry("scene")
    np.testing.assert_allclose(camera.unproject([[8., 6.]]), [[0., 0., 1.]])
    NuRecRpcBoundary().install(monkeypatch)
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        runtime.reset(dataset.scenario_start("scene", dataset.frames("scene")[0]))
        np.testing.assert_allclose(runtime.sensor_view("front").depth[6, 8], 4., atol=1e-6)
