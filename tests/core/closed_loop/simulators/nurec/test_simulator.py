"""NuRec actor editing, depth support and checkpoint behavior through its public runtime."""

from dataclasses import replace
import json
import zipfile

import numpy as np
import pytest

from avsectester.closed_loop.simulators.ego_dynamics import TrajectoryFollower
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.closed_loop.simulators.nurec.renderer import pose_from_proto
from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.poses import rotation_matrix
from avsectester.data_types import Control
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.rendering import ObjectAsset
from tests.fakes import ConstantAVStack, NuRecImageBoundary, NuRecRpcBoundary, NuRecTrafficImageBoundary


def test_reset_requires_mesh(nurec_usdz_path, tmp_path, monkeypatch):
    archive_path = tmp_path / "without-ground.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(archive_path, "w") as target:
        for member in source.infolist():
            if member.filename != "mesh_ground.ply":
                target.writestr(member.filename, source.read(member.filename))
    dataset = NuRecDataset([archive_path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    boundary = NuRecRpcBoundary().install(monkeypatch)
    with NuRecSimulator(NuRecRenderer(cameras=["front"]), min_track_duration_s=0, time_step=.1).create_runtime() as runtime:
        with pytest.raises(ValueError, match=r"missing: mesh_ground\.ply"):
            runtime.reset(start)
        assert boundary.requests == []
    with NuRecSimulator(NuRecRenderer(cameras=["front"]), depth="none", min_track_duration_s=0, time_step=.1).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.sensor_view("front").depth is None
        assert boundary.last_request is not None


def test_reset_initializes_dynamics(nurec_dataset, monkeypatch):
    from tests.fakes import ResetRecordingDynamics

    NuRecRpcBoundary().install(monkeypatch)
    frames = nurec_dataset.frames("scene")
    start = nurec_dataset.scenario_start("scene", frames[2])
    recorded = []
    for position, original in enumerate(start.recorded_ego):
        pose = original.pose.copy()
        pose[:3, :3] = rotation_matrix((0., 0., position * 2.))
        local_velocity = np.array([3. + position * .2, .4, 0.])
        recorded.append(replace(original, pose=pose,
                                velocity=tuple(pose[:3, :3] @ local_velocity)))
    start = replace(start, recorded_ego=tuple(recorded),
                    payload={**start.payload, "ego": recorded[2]})
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]),
                               ResetRecordingDynamics(), physics_update_mode="none",
                               route_generator="none", min_track_duration_s=0, time_step=.1)
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        initial = runtime.native["dynamics"].initial_state
        assert initial.speed == pytest.approx(3.4)
        assert initial.lateral_speed == pytest.approx(.4)
        assert initial.yaw_rate == pytest.approx(np.deg2rad(2.) / .1)
        assert initial.acceleration == pytest.approx(2.)
        assert initial.t == 0.
        np.testing.assert_allclose(runtime.state().ego.velocity, recorded[2].velocity)
        runtime.step(Control(throttle=.2))
        runtime.reset(start)
        assert runtime.state().vehicle_state == initial
        assert runtime.native["dynamics"].initial_state == initial


def test_checkpoint(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), TrajectoryFollower(),
                               route_generator="none", physics_update_mode="none", min_track_duration_s=0, time_step=.1)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
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
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), object_occlusion="geometry", min_track_duration_s=0, time_step=.1)
    with simulator.create_runtime() as runtime:
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
    simulator = NuRecSimulator(NuRecImageBoundary(camera), min_track_duration_s=0, time_step=.1)
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
    with NuRecSimulator(NuRecImageBoundary(camera), depth="none", min_track_duration_s=0, time_step=.1).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.sensor_view("front").depth is None
    with pytest.raises(ValueError, match="depth must be 'mesh' or 'none'"):
        NuRecSimulator(NuRecImageBoundary(camera), depth="unsupported", min_track_duration_s=0, time_step=.1)


def test_rendered_probe_occlusion(nurec_dataset, tmp_path, monkeypatch, make_object_in_front):
    from avsectester.closed_loop.hooks import Hook, HookSet
    from avsectester.rendering import PlacedObject, plan_composite
    from tests.fakes import ConstantAVStack

    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.asset_directories = {"/server/assets": str(tmp_path)}
    boundary.scene_tracks = {"host", "brief"}
    boundary.hidden = np.zeros((12, 16), bool)
    boundary.hidden[:, :8] = True  # The reconstruction covers the left half of the image.
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[2])
    renderer = NuRecRenderer(cameras=["front"], probe_directory=str(tmp_path),
                             server_probe_directory="/server/assets")
    # Rendered probes are the default object occlusion.
    simulator = NuRecSimulator(renderer, physics_update_mode="none", depth="none", min_track_duration_s=0, time_step=.1)
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        asset, world_from_object = make_object_in_front(runtime.sensor_view("front"))
        # A dense texture places probe Gaussians in every covered pixel of this small image.
        (surface,) = asset.surfaces
        asset = replace(asset, surfaces=(replace(surface, texture=np.full((256, 256, 4), [230, 10, 90, 255], np.uint8)),))
        runtime.insert_object("sign", asset, world_from_object)

        def check_capture():
            view = runtime.sensor_view("front")
            expected = plan_composite(view.camera, view.cam_from_world, (PlacedObject("sign", asset, world_from_object),),
                                      scene_visibility=~boundary.hidden)
            assert not expected.visible_mask[:, :8].any() and expected.visible_mask.any()
            np.testing.assert_array_equal(view.composition.plan.visible_mask, expected.visible_mask)
            np.testing.assert_array_equal(expected.apply(view.composition.background).image, view.data)
            visibility = runtime.object_visibility("sign")["front"]
            assert visibility.occlusion_source == "scene_visibility"
            assert visibility.visible_pixels == expected.visible_mask.sum()
            return view

        runtime.step(Control())
        view = check_capture()
        with pytest.raises(ValueError, match="'rendered_probes' occlusion"):
            view.composite_plan((PlacedObject("other", asset, world_from_object),))
        edits = [request for request in boundary.requests if hasattr(request, "insert")]
        assert len(edits) == 1 and len(edits[0].insert) == 2
        for track in edits[0].insert:
            assert track.asset_id == f"/server/assets/{track.id}.ply"
            assert (tmp_path / f"{track.id}.ply").read_bytes().startswith(b"ply\n")

        # Inserted probes persist on the server; a server that dropped them gets them again.
        runtime.step(Control())
        check_capture()
        assert len([request for request in boundary.requests if hasattr(request, "insert")]) == 1
        boundary.tracks.clear()
        runtime.step(Control())
        check_capture()
        assert len([request for request in boundary.requests if hasattr(request, "insert")]) == 2

        setup = HookSet([Hook("simulator.setup", lambda value, context: context.simulator_runtime.insert_object(
            "start-sign", asset, world_from_object))])
        with setup.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            # Objects present at the start are hidden by the reconstruction in past captures too.
            for observation in runtime.observe_past(2):
                image = observation.sensor_data["front"]
                assert np.all(image[:, :8] == 30) and np.any(image[:, 8:] != 30)

    # The occlusion tolerance moves only the probes, toward the camera along the ray to the box
    # centre; the composed capture keeps the object's pose.
    from avsectester.closed_loop.simulators.nurec.renderer import pose_from_proto

    def capture(tolerance):
        tolerant = NuRecSimulator(renderer, physics_update_mode="none", depth="none",
                                  occlusion_tolerance_m=tolerance, min_track_duration_s=0, time_step=.1)
        assert tolerant.identity["occlusion_tolerance_m"] == tolerance
        with tolerant.create_runtime() as runtime:
            runtime.reset(start)
            runtime.insert_object("sign", asset, world_from_object)
            runtime.step(Control())
            data = runtime.sensor_view("front").data
        request = boundary.last_request
        probe = next(item for item in request.dynamic_objects if item.track_id.startswith("avsectester_probe_"))
        return (data, pose_from_proto(request.sensor_pose.start_pose)[:3, 3],
                pose_from_proto(probe.pose_pair.start_pose)[:3, 3])

    data, camera_centre, probe_centre = capture(0.)
    moved_data, _, moved_centre = capture(.25)
    ray = probe_centre - camera_centre
    np.testing.assert_allclose(moved_centre, probe_centre - .25 * ray / np.linalg.norm(ray), atol=1e-6)
    np.testing.assert_array_equal(moved_data, data)
    assert NuRecSimulator(renderer, min_track_duration_s=0, time_step=.1).occlusion_tolerance_m == .15

    unconfigured = NuRecSimulator(NuRecRenderer(cameras=["front"]), physics_update_mode="none",
                                  depth="none", min_track_duration_s=0, time_step=.1)
    with unconfigured.create_runtime() as runtime:
        runtime.reset(start)
        runtime.insert_object("sign", asset, world_from_object)
        runtime.step(Control())
        with pytest.raises(ValueError, match="probe_directory"):
            runtime.sensor_view("front")
    with pytest.raises(ValueError, match="object_occlusion must be"):
        NuRecSimulator(renderer, object_occlusion="unsupported", min_track_duration_s=0, time_step=.1)


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
    with NuRecSimulator(NuRecRenderer(cameras=["front"]), min_track_duration_s=0, time_step=.1).create_runtime() as runtime:
        runtime.reset(dataset.scenario_start("scene", dataset.frames("scene")[0]))
        np.testing.assert_allclose(runtime.sensor_view("front").depth[6, 8], 4., atol=1e-6)


def test_route(nurec_dataset, monkeypatch):
    from avsectester.data_types import Unavailable

    NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    with NuRecSimulator(NuRecRenderer(cameras=["front"]), min_track_duration_s=0, time_step=.1).create_runtime() as runtime:
        runtime.reset(start)
        route = runtime.observe().route
        assert route.shape == (20, 3)
        np.testing.assert_allclose(route[0], [0., 0., 0.])
        runtime.step(Control(throttle=.2))
        np.testing.assert_allclose(runtime.observe().route[0], [0., 0., 0.], atol=1e-12)
        missing = replace(start, scenario_map=Unavailable("No map"))
        with pytest.raises(ValueError, match="recorded.*none"):
            runtime.reset(missing)


def test_validate_start(nurec_dataset):
    from avsectester.closed_loop.simulators import UnsupportedStartError

    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    # A recording that turns back on itself has no valid navigation route.
    last = start.recorded_ego[-1]
    reversed_pose = last.pose.copy()
    reversed_pose[0, 3] -= 1.
    folded = replace(start, recorded_ego=(*start.recorded_ego, replace(last, pose=reversed_pose)))
    for kind in ("map", "recorded"):
        simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), route_generator=kind, min_track_duration_s=0, time_step=.1)
        simulator.validate_start(start)
        with pytest.raises(UnsupportedStartError, match="folds back"):
            simulator.validate_start(folded)
    NuRecSimulator(NuRecRenderer(cameras=["front"]), route_generator="none", min_track_duration_s=0, time_step=.1).validate_start(folded)


@pytest.mark.parametrize("mode,height", [("ego_only", .05), ("none", .3)])
def test_ground_contact(mode, height, nurec_dataset, monkeypatch):
    from avsectester.execution.runner import ClosedLoopRunner, RunRequest
    from tests.fakes import ConstantAVStack

    NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]),
                               physics_update_mode=mode, ground_mesh="mesh_ground", min_track_duration_s=0, time_step=.1)
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(1, start=start))
        assert log.steps[0].physical_post.ego.pose[2, 3] == pytest.approx(height)
        termination = next(event for event in log.events if event.kind == "termination")
        assert termination.data["ground_contact_failures"] == {}


def test_ground_contact_failure(nurec_dataset, monkeypatch):
    from avsectester.execution.runner import ClosedLoopRunner, RunRequest
    from tests.fakes import ConstantAVStack

    NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    ego = start.payload["ego"]
    pose = ego.pose.copy()
    pose[0, 3] = 1000.
    start = replace(start, payload={**start.payload, "ego": replace(ego, pose=pose)})
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), route_generator="none", min_track_duration_s=0, time_step=.1)
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(1, start=start))
        termination = next(event for event in log.events if event.kind == "termination")
        assert termination.data["ground_contact_failures"] == {"insufficient_intersections": 1}
        assert log.steps[0].physical_post.ego.pose[2, 3] == pytest.approx(.3)


def test_observe_past(nurec_dataset, monkeypatch, make_object_in_front):
    from avsectester.closed_loop.hooks import Hook, HookSet
    from tests.fakes import ConstantAVStack

    boundary = NuRecRpcBoundary().install(monkeypatch)
    frames = nurec_dataset.frames("scene")
    start = nurec_dataset.scenario_start("scene", frames[3])
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]),
                               physics_update_mode="none", object_occlusion="geometry", min_track_duration_s=0, time_step=.1)
    with simulator.create_runtime() as runtime:
        hooks = HookSet([Hook("render.post", lambda image, context: image + 7)])
        with hooks.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            asset, world_from_object = make_object_in_front(runtime.sensor_view("front"))
            runtime.insert_object("new-sign", asset, world_from_object)
            host = runtime.state().actors["host"]
            pose = host.pose.copy()
            pose[1, 3] = 20.
            runtime.set_actor_state("host", replace(host, pose=pose, velocity=(0., 0., 0.)))
            runtime.step(Control())
            original = runtime.observe()
            physical = runtime.state()
            request = runtime.render_request("front")
            past = runtime.observe_past(3)
            assert runtime.can_observe_past
            assert [observation.frame for observation in past] == [-3, -2, -1]
            np.testing.assert_allclose([observation.t for observation in past], [-.3, -.2, -.1])
            rendered = [item for item in boundary.requests if hasattr(item, "frame_start_us")][-3:]
            for observation, native, frame in zip(past, rendered, frames[:3]):
                recorded = nurec_dataset.frame_data("scene", frame)
                assert native.frame_start_us == round(frame.timestamp * 1e6)
                np.testing.assert_allclose(pose_from_proto(native.sensor_pose.start_pose),
                    recorded.ego.pose @ pose_from_proto(boundary.mount), atol=1e-12)
                assert observation.vehicle_state.x == pytest.approx(recorded.ego.pose[0, 3])
                assert observation.ego_speed == pytest.approx(np.linalg.norm(recorded.ego.velocity))
                # Past captures bypass render hooks, actor overrides and framework insertions.
                np.testing.assert_array_equal(observation.sensor_data["front"], 30)
                actors = {item.track_id: item for item in native.dynamic_objects}
                assert actors.keys() == recorded.actors.keys()
                for identity, actor in recorded.actors.items():
                    np.testing.assert_allclose(pose_from_proto(actors[identity].pose_pair.start_pose),
                                               actor.pose, atol=1e-12)
                np.testing.assert_allclose(observation.route[0], [0., 0., 0.], atol=1e-12)
            np.testing.assert_array_equal(runtime.observe().sensor_data["front"], original.sensor_data["front"])
            np.testing.assert_allclose(runtime.observe().route, original.route)
            np.testing.assert_allclose(runtime.state().ego.pose, physical.ego.pose)
            np.testing.assert_allclose(runtime.state().actors["host"].pose, physical.actors["host"].pose)
            assert runtime.state().objects.keys() == {"new-sign"}
            assert runtime.clock.t == physical.t and runtime.clock.frame == physical.frame
            np.testing.assert_array_equal(runtime.render_request("front").pose, request.pose)
            runtime.reset(nurec_dataset.scenario_start("scene", frames[0]))
            with pytest.raises(ValueError, match="only 0 past decisions; 1 required"):
                runtime.observe_past(1)
        setup = HookSet([Hook("simulator.setup", lambda value, context: context.simulator_runtime.insert_object(
            "start-sign", asset, world_from_object))])
        with setup.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            # A world object present at the start is composed into the recorded past captures.
            for observation in runtime.observe_past(3):
                image = observation.sensor_data["front"]
                assert np.any(image != 30) and np.all(image[0, 0] == 30)


@pytest.fixture
def traffic_world(traffic_recording):
    from types import SimpleNamespace

    pytest.importorskip("alpasim_grpc")
    from alpasim_grpc.v0 import common_pb2 as common, traffic_pb2 as traffic
    from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic
    from tests.fakes import NuRecTrafficImageBoundary, TrafficServiceBoundary

    def response(request):
        t = request.time_query_us
        return traffic.TrafficReturn(object_trajectory_updates=[
            traffic.ObjectTrajectoryUpdate(object_id=key, trajectory=common.Trajectory(poses=[
                common.PoseAtTime(timestamp_us=t, pose=common.Pose(
                    vec=common.Vec3(x=x, y=2., z=1.), quat=common.Quat(w=1.)))]))
            for key, x in [("host", 40 + t / 100000), ("static", 88.),
                           ("expired", 100.), ("unknown", -20.), ("EGO", -30.)]])
    camera = PinholeCamera(np.array([[20., 0., 16.], [0., 20., 12.], [0., 0., 1.]]), 32, 24)
    boundary = NuRecTrafficImageBoundary(camera)
    frames = traffic_recording.frames("scene")
    start = traffic_recording.scenario_start("scene", frames[15])
    with TrafficServiceBoundary(simulate_response=response) as service:
        simulator = NuRecSimulator(boundary, depth="none", route_generator="none",
                                   physics_update_mode="none", traffic=AlpaSimTraffic(service.endpoint), time_step=.1)
        yield SimpleNamespace(dataset=traffic_recording, simulator=simulator, boundary=boundary,
                              service=service, frames=frames, start=start)


def test_reset_traffic(traffic_world):
    world = traffic_world
    with world.simulator.create_runtime(7) as runtime:
        runtime.reset(world.start)
        request = world.service.sessions[-1]
        assert request.scene_id == "scene" and request.random_seed == 7
        assert request.handover_time_us == 1600000
        assert [obj.object_id for obj in request.logged_object_trajectories] == [
            "EGO", "host", "static", "fallback", "expired"]
        ego = request.logged_object_trajectories[0]
        np.testing.assert_allclose(pose_from_proto(ego.trajectory.poses[15].pose),
                                   world.start.recorded_ego[15].box_transform())
        assert (ego.aabb.size_x, ego.aabb.size_y, ego.aabb.size_z) == world.start.recorded_ego[15].extent
        assert request.logged_object_trajectories[2].is_static
        assert not request.logged_object_trajectories[1].is_static
        assert set(runtime.state().actors) == {"host", "static", "fallback", "parked-brief", "expired"}
        first_session = request.session_uuid
        runtime.reset(world.start)
        assert world.service.closed[0].session_uuid == first_session
    assert len(world.service.closed) == 2
    world.boundary.fail_render = True
    with pytest.raises(RuntimeError, match="image service"):
        with world.simulator.create_runtime(7) as runtime:
            runtime.reset(world.start)
    assert world.service.closed[-1].session_uuid == world.service.sessions[-1].session_uuid


def test_traffic_world_recording(traffic_world, tmp_path):
    from avsectester.attack import AttackSegment
    from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
    from avsectester.closed_loop.hooks import Hook, HookSet
    from avsectester.execution.services import CandidateRequest, ClosedLoopServices
    from avsectester.execution.runner import RecordingPolicy
    from avsectester.rendering import ObjectAsset
    from avsectester.runlog import RunLog
    from tests.fakes import ConstantAVStack

    world = traffic_world
    panel = ObjectAsset.from_panel(np.full((8, 8, 4), 255, np.uint8),
                                  category="panel", width_m=1., mount_height_m=0., post_count=0)
    pose = np.diag([-1., -1., 1., 1.])
    pose[:3, 3] = [20., 0., 1.]
    setup = HookSet([Hook("simulator.setup", lambda value, context:
        context.simulator_runtime.insert_object("panel", panel, pose))])
    with world.simulator.create_runtime() as runtime, setup.session(runtime, ConstantAVStack()).activate():
        runtime.reset(world.start)
        runtime.step(Control())
        runtime.sensor_view("front")
        update = world.service.requests[-1]
        assert update.time_query_us == 1600000
        assert [item.object_id for item in update.object_trajectory_updates] == ["EGO"]
        np.testing.assert_allclose(pose_from_proto(update.object_trajectory_updates[0].trajectory.poses[0].pose),
                                   runtime.state().ego.box_transform())
    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), past_steps=0)
    segment = AttackSegment("scene", world.frames[15], world.frames[17])
    with ClosedLoopServices(world.dataset, world.simulator, adapter) as services, services.session(0) as session:
        log = session.rollout(CandidateRequest(segment, 0, 2, {}, RecordingPolicy(), output_path=tmp_path/"rollout"))
    assert log.status == "completed"
    loaded = RunLog.read(tmp_path/"rollout")
    assert len(loaded.steps) == 2
    assert "brief" not in loaded.steps[0].physical_post.actors
    state = loaded.steps[0].physical_post
    np.testing.assert_allclose(state.actors["host"].pose[:3, 3], [56., 2., 1.])
    np.testing.assert_allclose(state.actors["host"].pose, log.steps[0].physical_post.actors["host"].pose)
    # Captures and probes receive the same current world poses.
    matching = [objects["host"] for time, objects in world.boundary.rendered if time == 1600000]
    assert matching
    np.testing.assert_allclose(matching[-1], state.actors["host"].pose)
    matching_probes = [objects["host"] for time, objects in world.boundary.probes if time == 1600000]
    assert matching_probes
    np.testing.assert_allclose(matching_probes[-1], state.actors["host"].pose)



def test_traffic_velocity(traffic_world):
    with traffic_world.simulator.create_runtime() as runtime:
        runtime.reset(traffic_world.start)
        runtime.step(Control())
        assert runtime.state().actors["host"].velocity == pytest.approx((345., 20., 0.))


def test_traffic_recorded_fallback(traffic_world):
    with traffic_world.simulator.create_runtime() as runtime:
        runtime.reset(traffic_world.start)
        runtime.step(Control())
        expected = traffic_world.dataset.actor_states("scene", 1600000)["fallback"]
        np.testing.assert_array_equal(runtime.state().actors["fallback"].pose, expected.pose)
        assert runtime.state().actors["fallback"].pose[0, 3] == pytest.approx(31.6)


def test_traffic_parked_and_expired(traffic_world):
    world = traffic_world
    with world.simulator.create_runtime(7) as runtime:
        runtime.reset(world.start)
        np.testing.assert_allclose(runtime.state().actors["static"].pose,
                                   world.dataset.actor_states("scene", 1500000)["static"].pose)
        runtime.step(Control())
        np.testing.assert_allclose(runtime.state().actors["static"].pose,
                                   world.dataset.actor_states("scene", 1600000)["static"].pose)
        view = runtime.sensor_view("front")
        np.testing.assert_allclose(dict(view.native["dynamic_objects"])["host"], runtime.state().actors["host"].pose)
        for _ in range(15):
            runtime.step(Control())
        state = runtime.state()
        assert state.t == pytest.approx(1.6)
        assert "static" not in state.actors
        assert "static" not in world.dataset.actor_states("scene", 3100000)
        assert "static" not in dict(runtime.sensor_view("front").native["dynamic_objects"])
        assert state.actors["expired"].extent == (6., 3., 3.)
        assert state.actors["expired"].category == "vehicle"
        assert state.actors["expired"].pose[0, 3] == 100.


def test_set_traffic_actor_state(traffic_world):
    world = traffic_world
    with world.simulator.create_runtime(7) as runtime:
        runtime.reset(world.start)
        actor = runtime.state().actors["host"]
        with pytest.raises(ValueError, match="Filtered"):
            runtime.set_actor_state("brief", actor)
        with pytest.raises(ValueError, match="NRE cannot move parked"):
            runtime.set_actor_state("static", actor)
        pose = actor.pose.copy()
        pose[:3, 3] = [70., 0., 1.]
        velocity = np.array([1., -.5, .2])
        runtime.set_actor_state("host", replace(actor, pose=pose, velocity=tuple(velocity)))
        runtime.step(Control())
        assert runtime.state().actors["host"].pose[0, 3] == pytest.approx(70.1)
        runtime.step(Control())
        assert runtime.state().actors["host"].pose[0, 3] == pytest.approx(70.2)
        runtime.set_actor_state("host", replace(runtime.state().actors["host"], velocity=(0., 0., 0.)))
        runtime.step(Control())
        track = next(item for item in world.service.requests[-1].object_trajectory_updates if item.object_id == "host")
        assert [point.timestamp_us for point in track.trajectory.poses] == list(range(0, 1800001, 100000))
        assert track.trajectory.poses[-1].pose.vec.x == pytest.approx(70.2)


def test_traffic_start_requirements(traffic_world):
    from avsectester.closed_loop.simulators.base import UnsupportedStartError
    from avsectester.execution.services import ClosedLoopServices

    world = traffic_world
    assert world.simulator.past_steps == 15
    with ClosedLoopServices(world.dataset, world.simulator) as services, services.session(7) as session:
        assert not session.can_start_at(14)
        assert session.can_start_at(15)
    with pytest.raises(UnsupportedStartError, match="15 past"):
        world.simulator.validate_start(world.dataset.scenario_start("scene", world.frames[14]))
    with pytest.raises(ValueError, match="0.1"):
        NuRecSimulator(world.boundary, depth="none", physics_update_mode="none",
                       traffic=world.simulator.traffic, time_step=.2)


def test_traffic_checkpoint(traffic_world):
    world = traffic_world
    with world.simulator.create_runtime() as runtime:
        runtime.reset(world.start)
        with pytest.raises(RuntimeError, match="checkpoint"):
            runtime.checkpoint()


def test_recorded_actor_filtering(traffic_recording):
    camera = PinholeCamera(np.array([[20., 0., 16.], [0., 20., 12.], [0., 0., 1.]]), 32, 24)
    boundary = NuRecTrafficImageBoundary(camera)
    frames = traffic_recording.frames("scene")
    start = traffic_recording.scenario_start("scene", frames[15])
    simulator = NuRecSimulator(boundary, depth="none", route_generator="none",
                               physics_update_mode="none", time_step=.1)
    assert simulator.identity["min_track_duration_s"] == 3.
    panel = ObjectAsset.from_panel(np.full((8, 8, 4), 255, np.uint8),
                                   category="panel", width_m=1., mount_height_m=0., post_count=0)
    pose = np.diag([-1., -1., 1., 1.])
    pose[:3, 3] = [20., 0., 1.]
    setup = HookSet([Hook("simulator.setup", lambda value, context:
        context.simulator_runtime.insert_object("panel", panel, pose))])
    with simulator.create_runtime() as runtime, setup.session(runtime, ConstantAVStack()).activate():
        runtime.reset(start)
        actors = runtime.state().actors
        assert "brief" not in actors and "parked-brief" in actors
        assert "brief" not in runtime.sensor_view("front").actors
        assert boundary.rendered[-1][1]["brief"][2, 3] == -99.
        assert boundary.probes[-1][1]["brief"][2, 3] == -99.
        runtime.observe_past(1)
        assert boundary.rendered[-1][0] == 1400000
        assert boundary.rendered[-1][1]["brief"][2, 3] == -99.
        assert boundary.probes[-1][1]["brief"][2, 3] == -99.
        with pytest.raises(ValueError, match="Filtered"):
            runtime.set_actor_state("brief", actors["host"])
        with pytest.raises(ValueError, match="NRE cannot move parked"):
            runtime.set_actor_state("parked-brief", actors["host"])
        runtime.set_actor_state("host", replace(actors["host"], velocity=(0., 0., 0.)))
        for key, state in actors.items():
            np.testing.assert_allclose(state.pose, traffic_recording.actor_states("scene", 1500000)[key].pose)
    unfiltered = NuRecSimulator(boundary, depth="none", route_generator="none",
                                physics_update_mode="none", min_track_duration_s=0, time_step=.1)
    assert unfiltered.identity["min_track_duration_s"] == 0
    with unfiltered.create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.state().actors.keys() == traffic_recording.actor_states("scene", 1500000).keys()
        assert runtime.sensor_view("front").actors.keys() == runtime.state().actors.keys()
        assert boundary.rendered[-1][1]["brief"][2, 3] == 1.
        runtime.set_actor_state("brief", replace(runtime.state().actors["brief"], velocity=(0., 0., 0.)))


def test_decision_time_step():
    renderer = NuRecRenderer(cameras=["front"])
    simulator = NuRecSimulator(renderer, time_step=.2, depth="none", physics_update_mode="none")
    assert simulator.time_step == simulator.decision_time_step == .2
    assert simulator.identity["time_step"] == .2
    with pytest.raises(ValueError, match="dataset.time_step"):
        NuRecSimulator(renderer)


def test_camera_aliases(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    aliases = {"front": "forward"}
    simulator = NuRecSimulator(NuRecRenderer(cameras=["front"]), time_step=.1,
                               camera_aliases=aliases, depth="none", route_generator="none",
                               physics_update_mode="none", min_track_duration_s=0)
    assert simulator.identity["camera_aliases"] == aliases
    assert "config" not in simulator.identity
    aliases["front"] = "changed"
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.camera_sensors == {"forward"}
        assert set(runtime.observe().sensor_data) == {"forward"}
        assert set(runtime.observe().calibration) == {"forward"}
        view = runtime.sensor_view("forward")
        assert view.sensor == runtime.sensor_view("front").sensor == "forward"
        assert runtime.sensor_geometry("forward").sensor == "forward"
        np.testing.assert_array_equal(view.data, runtime.observe().sensor_data["forward"])
        runtime.step(Control())
        assert runtime.sensor_view("forward").sensor == "forward"


@pytest.mark.parametrize(("constructor", "options"), [
    (NuRecSimulator, {"renderer": NuRecRenderer(cameras=["front"]), "time_step": .1,
                      "depth": "none", "physics_update_mode": "none"}),
    (RecordedReplaySimulator, {}),
])
def test_camera_alias_validation(constructor, options):
    configured = constructor(**options, camera_aliases={"front": "forward"})
    assert configured.camera_aliases == {"front": "forward"}
    with pytest.raises(ValueError, match="nonempty native camera names"):
        constructor(**options, camera_aliases={"front": ""})
    with pytest.raises(ValueError, match="unique output names"):
        constructor(**options, camera_aliases={"front": "same", "side": "same"})
