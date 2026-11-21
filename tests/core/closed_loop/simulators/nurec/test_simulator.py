"""NuRec world composition, capture, routes and traffic protocol through public runtimes."""

from dataclasses import replace
from io import BytesIO
import json
import zipfile
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.attack.insertion import InsertObject
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.closed_loop.simulators import UnsupportedStartError
from avsectester.closed_loop.simulators.ego_dynamics import AlpaSimController, TrajectoryFollower
from avsectester.closed_loop.simulators.ground_contact import constrain_to_ground
from avsectester.closed_loop.simulators.nurec import AlpaSimTraffic, NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.timing import HistoryRequirement, SensorTiming, TimingConfiguration
from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.cameras import FThetaCamera, PinholeCamera
from avsectester.geometry.lidars import LidarCalibration
from avsectester.geometry.mesh import MeshRaycaster
from avsectester.geometry.poses import interpolate_transform, rotation_matrix
from avsectester.data_types import Control, PointCloud, Unavailable
from avsectester.execution.runner import RecordingPolicy
from avsectester.execution.services import CandidateRequest, ClosedLoopServices, RenderRequest
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.rendering import ObjectAsset, PlacedObject, plan_composite
from avsectester.rendering.harmonizers import ClassicHarmonizer, ExposureHarmonizer
from avsectester.runlog import RunLog
from avsectester.validation import timestamp_to_microseconds
from tests.fakes import (
    CAMERA, ConstantAVStack, NuRecImageBoundary, NuRecRpcBoundary, NuRecTrafficImageBoundary,
    ResetRecordingDynamics, TrafficServiceBoundary, proto_world_pose,
)


@pytest.fixture
def make_nurec_simulator():
    """Construct a synthetic NuRec world with a 10 Hz timeline and no track filtering."""
    return partial(NuRecSimulator, min_track_duration_s=0, time_step=.1)


def test_motion_reports(nurec_dataset, make_nurec_simulator):
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    simulator = make_nurec_simulator(NuRecImageBoundary(CAMERA), physics_update_mode="none", depth="none")
    seen = []

    def remember_clock(value, context):
        seen.append((context.stage, context.simulator_runtime.clock))
        return value

    hooks = HookSet([Hook("simulator.step.pre", remember_clock),
                     Hook("simulator.step.post", remember_clock)])
    with simulator.create_runtime() as runtime, hooks.session(runtime, ConstantAVStack()).activate():
        with pytest.raises(RuntimeError, match="Reset"):
            runtime.clock
        runtime.reset(start)
        initial = runtime.state()
        request = runtime.render_request("front")
        runtime.advance_interval(100000, Control(throttle=.3), report_times_us=(25000, 50000, 75000))
        np.testing.assert_array_equal(runtime.render_request("front").pose, request.pose)
        reports = {}
        for time in (25000, 50000, 75000, 100000):
            runtime.capture_at(time, ())
            reports[time] = runtime.state()
        assert list(reports) == [25000, 50000, 75000, 100000]
        middle = reports[50000]
        assert middle.t == .05 and middle.vehicle_state.t == .05
        assert middle.ego_speed == pytest.approx(initial.ego_speed + .9 * .05)
        assert middle.ego.pose[0, 3] < reports[100000].ego.pose[0, 3]
        reports[100000].ego.pose[0, 3] = 99.
        assert runtime.state().ego.pose[0, 3] != 99.
        runtime.capture_at(100000)
        assert runtime.clock.time_us == 100000


def test_camera_exposure(nurec_longitudinal_dataset, monkeypatch, make_nurec_simulator):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]),
                                    physics_update_mode="none", depth="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={
            "front": SensorTiming(frame_interval_us=33000, shutter_duration_us=17000)},
            pose_reporting_interval_us=40000))
        runtime.reset(start)
        initial = boundary.last_request
        assert (initial.frame_start_us, initial.frame_end_us) == (283000, 300000)
        np.testing.assert_allclose(proto_world_pose(initial.sensor_pose.start_pose),
            dataset.ego_state("scene", 283000).pose @ proto_world_pose(boundary.mount), atol=1e-12)
        runtime.advance_interval(100000, Control(throttle=.2), report_times_us=(40000, 80000))
        runtime.capture_at(16000, ())
        beginning = runtime.state()
        runtime.capture_at(33000, ("front",))
        end = runtime.state()
        request = boundary.last_request
        assert (request.frame_start_us, request.frame_end_us) == (316000, 333000)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.start_pose),
            beginning.ego.pose @ proto_world_pose(boundary.mount), atol=1e-12)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.end_pose),
            end.ego.pose @ proto_world_pose(boundary.mount), atol=1e-12)
        for actor in request.dynamic_objects:
            np.testing.assert_allclose(proto_world_pose(actor.pose_pair.start_pose),
                                      beginning.actors[actor.track_id].pose, atol=1e-12)
            np.testing.assert_allclose(proto_world_pose(actor.pose_pair.end_pose),
                                      end.actors[actor.track_id].pose, atol=1e-12)
        np.testing.assert_allclose(runtime.sensor_geometry("front").cam_from_world,
            np.linalg.inv(end.ego.pose @ proto_world_pose(boundary.mount)), atol=1e-12)
        runtime.capture_at(66000, ("front",))
        runtime.capture_at(99000, ("front",))
        runtime.capture_at(100000, ())
        assert runtime.observe().sensor_data == {}

    later_start = dataset.scenario_start("scene", dataset.frames("scene")[6])
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={
            "front": SensorTiming(frame_interval_us=1000000, shutter_duration_us=500000)}))
        runtime.reset(later_start)
        for index in range(1, 11):
            runtime.advance_interval(index * 100000, Control())
            runtime.capture_at(index * 100000, ("front",) if index == 10 else ())
        assert (boundary.last_request.frame_start_us, boundary.last_request.frame_end_us) == (1100000, 1600000)


def test_clock(nurec_usdz_path, tmp_path, make_nurec_simulator):
    path = tmp_path / "long-clock.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source:
        metadata = json.loads(source.read("rig_trajectories.json"))
        recording = metadata["rig_trajectories"][0]
        camera = next(iter(metadata["camera_calibrations"]))
        pose = np.eye(4).tolist()
        recording["cameras_frame_timestamps_us"][camera] = [
            [index * 100000, index * 100000 + 1] for index in range(1002)]
        recording["cameras_frame_T_rig_worlds"][camera] = [[pose, pose] for _ in range(1002)]
        with zipfile.ZipFile(path, "w") as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(metadata) if name == "rig_trajectories.json"
                                else source.read(name))
    dataset = NuRecDataset([path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    simulator = make_nurec_simulator(NuRecImageBoundary(CAMERA), physics_update_mode="none",
                                    depth="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        for index in range(1, 1001):
            runtime.step(Control())
            assert runtime.clock.time_us == index * 100000
            assert runtime.state().t == runtime.observe().vehicle_state.t == index * 100000 / 1e6


def test_recorded_startup(nurec_longitudinal_dataset, make_nurec_simulator, tmp_path):
    pytest.importorskip("alpasim_controller")
    dataset = nurec_longitudinal_dataset
    frames = dataset.frames("scene")
    start = dataset.scenario_start("scene", frames[3])
    simulator = make_nurec_simulator(NuRecImageBoundary(CAMERA), AlpaSimController(),
                                    physics_update_mode="none", depth="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(force_gt_duration_us=200000))
        runtime.reset(start)
        issued = Control(trajectory=[
            ((0., 0., 0.), (1., 0., 0., 0.), 0),
            ((30., 0., 0.), (1., 0., 0., 0.), 3000000),
        ])
        runtime.advance_interval(100000, runtime.recorded_reference(0, 100000),
            report_times_us=(50000,), force_gt=True, coerce_dynamic_state=True)
        reports = {}
        for time in (50000, 100000):
            runtime.capture_at(time, ())
            reports[time] = runtime.state()
        expected_middle = dataset.ego_state("scene", 350000).pose
        np.testing.assert_array_equal(reports[50000].ego.pose, expected_middle)
        runtime.capture_at(100000)
        np.testing.assert_array_equal(runtime.state().ego.pose, start.recorded_ego[4].pose)
        runtime.advance_interval(200000, runtime.recorded_reference(100000, 200000),
                                 force_gt=True, coerce_dynamic_state=True)
        runtime.capture_at(200000)
        np.testing.assert_array_equal(runtime.state().ego.pose, start.recorded_ego[5].pose)
        runtime.advance_interval(300000, issued, coerce_dynamic_state=True)
        runtime.capture_at(300000)
        assert runtime.state().ego.pose[0, 3] == pytest.approx(6., abs=.01)
        assert runtime.state().ego_speed == pytest.approx(10., abs=.05)
    with zipfile.ZipFile(dataset.usdz_path("scene")) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    metadata = json.loads(entries["rig_trajectories.json"])
    transforms = metadata["rig_trajectories"][0]["cameras_frame_T_rig_worlds"]
    for camera_poses in transforms.values():
        for exposure in camera_poses[3]:
            pose = np.asarray(exposure)
            pose[:3, :3] = rotation_matrix((10., 20., 0.))
            pose[2, 3] = 1.
            exposure[:] = pose.tolist()
    path = tmp_path / "startup-slope.usdz"
    with zipfile.ZipFile(path, "w") as target:
        for name, payload in entries.items():
            target.writestr(name, json.dumps(metadata) if name == "rig_trajectories.json" else payload)
    dataset = NuRecDataset([path], time_step=.2)
    start = dataset.scenario_start("scene", dataset.frames("scene")[1])
    simulator = NuRecSimulator(NuRecImageBoundary(CAMERA), time_step=.2,
                               physics_update_mode="none", depth="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(force_gt_duration_us=200000))
        runtime.reset(start)
        runtime.advance_interval(200000, None, report_times_us=(100000,), force_gt=True)
        expected = dataset.ego_state("scene", 300000).pose
        runtime.capture_at(100000, ())
        np.testing.assert_array_equal(runtime.state().ego.pose, expected)


@pytest.fixture
def traffic_recording(nurec_longitudinal_dataset, tmp_path):
    source_path = nurec_longitudinal_dataset.usdz_paths[0]
    target_path = tmp_path / "traffic.usdz"
    with zipfile.ZipFile(source_path) as source:
        rig = json.loads(source.read("rig_trajectories.json"))
        recording = rig["rig_trajectories"][0]
        camera = next(iter(recording["cameras_frame_timestamps_us"]))
        times, poses = [], []
        for i in range(41):
            pose = np.eye(4)
            pose[:3, 3] = [i, 0., .3]
            times.append([i * 100000, i * 100000 + 1])
            poses.append([pose.tolist(), pose.tolist()])
        recording["cameras_frame_timestamps_us"][camera] = times
        recording["cameras_frame_T_rig_worlds"][camera] = poses
        tracks = {"scene": {"tracks_data": {
            "tracks_id": ["host", "static", "fallback", "brief", "parked-brief", "expired"],
            "tracks_label_class": ["automobile", "pedestrian", "truck", "pedestrian", "automobile", "truck"],
            "tracks_flags": ["DYNAMIC|CONTROLLABLE", "NONE", "CONTROLLABLE", "CONTROLLABLE", "NONE", "CONTROLLABLE"],
            "tracks_timestamps_us": [[0, 4000000], [0, 3000000], [0, 4000000], [1400000, 1800000],
                                     [1400000, 1800000], [-2000000, 1600000]],
            "tracks_poses": [
                [[20, 0, 1, 0, 0, 0, 1], [24, 0, 1, 0, 0, 0, 1]],
                [[8, 2, 1, 0, 0, 0, 1], [8, 2, 1, 0, 0, 0, 1]],
                [[30, 0, 1, 0, 0, 0, 1], [34, 0, 1, 0, 0, 0, 1]],
                [[7, 2, 1, 0, 0, 0, 1], [7, 2, 1, 0, 0, 0, 1]],
                [[7, -2, 1, 0, 0, 0, 1], [7, -2, 1, 0, 0, 0, 1]],
                [[30, 4, 1, 0, 0, 0, 1], [30, 4, 1, 0, 0, 0, 1]]]},
            "cuboidtracks_data": {"cuboids_dims": [[4, 2, 2], [1, 1, 2], [6, 3, 3], [1, 1, 2], [2, 1, 2], [6, 3, 3]]}}}
        with zipfile.ZipFile(target_path, "w") as target:
            for name in source.namelist():
                payload = source.read(name)
                if name == "rig_trajectories.json":
                    payload = json.dumps(rig)
                elif name == "sequence_tracks.json":
                    payload = json.dumps(tracks)
                target.writestr(name, payload)
    return NuRecDataset([target_path], time_step=.1)


def test_reset(nurec_usdz_path, nurec_dataset, tmp_path, monkeypatch, make_nurec_simulator):
    archive_path = tmp_path / "without-ground.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(archive_path, "w") as target:
        for member in source.infolist():
            if member.filename != "mesh_ground.ply":
                target.writestr(member.filename, source.read(member.filename))
    dataset = NuRecDataset([archive_path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    boundary = NuRecRpcBoundary().install(monkeypatch)
    with make_nurec_simulator(NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        with pytest.raises(ValueError, match=r"missing: mesh_ground\.ply"):
            runtime.reset(start)
        assert boundary.requests == []
    with make_nurec_simulator(NuRecRenderer(cameras=["front"]), depth="none").create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.sensor_view("front").depth is None
        assert boundary.last_request is not None

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
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]),
                               ResetRecordingDynamics(), physics_update_mode="none",
                               route_generator="none")
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

    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = NuRecDataset([nurec_usdz_path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[-2])
    origin = start.payload["ego"].pose.copy()
    origin[:3, :3] = rotation_matrix((8., -12., 30.))
    start = replace(start, payload={**start.payload, "ego": replace(start.payload["ego"], pose=origin)})
    with make_nurec_simulator(NuRecRenderer(cameras=["front"]),
                        route_generator="none", physics_update_mode="none", time_step=.1).create_runtime() as runtime:
        runtime.reset(start)
        np.testing.assert_allclose(runtime.state().ego.pose, origin, atol=1e-12)
        np.testing.assert_allclose(proto_world_pose(boundary.last_request.sensor_pose.start_pose),
                                   origin @ proto_world_pose(boundary.mount), atol=1e-12)
        runtime.step(Control())
        assert runtime.sensor_view("front").native["timestamp_us"] == 500000
        assert (boundary.last_request.frame_start_us, boundary.last_request.frame_end_us) == (500000, 500001)
        with pytest.raises(ValueError, match="terminal recorded support"):
            runtime.step(Control())
        with pytest.raises(ValueError, match="NuRec requires a ScenarioStart"):
            runtime.reset()


def test_route(nurec_dataset, monkeypatch, make_nurec_simulator):
    NuRecRpcBoundary().install(monkeypatch)
    with make_nurec_simulator(NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
        runtime.reset(start)
        assert runtime.observe().route.shape == (20, 3)
        runtime.step(Control(throttle=.2))
        np.testing.assert_allclose(runtime.observe().route[0], [0., 0., 0.], atol=1e-12)


def test_camera_aliases(nurec_dataset, monkeypatch, make_nurec_simulator):
    NuRecRpcBoundary().install(monkeypatch)
    aliases = {"front": "forward"}
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]), time_step=.1,
                               camera_aliases=aliases, depth="none", route_generator="none",
                               physics_update_mode="none", min_track_duration_s=0)
    assert simulator.identity["camera_aliases"] == aliases
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


def test_camera_without_geometry(nurec_dataset, make_nurec_simulator):
    renderer = NuRecImageBoundary(CAMERA, provide_geometry=False)
    simulator = make_nurec_simulator(renderer, camera_aliases={"front": "forward"},
        depth="none", route_generator="none", physics_update_mode="none")
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.sensors == {"forward"} and runtime.camera_sensors == frozenset()
        np.testing.assert_array_equal(runtime.observe().sensor_data["forward"], 30)
        runtime.step(Control())
        assert runtime.observe().sensor_data["forward"].shape == (CAMERA.height, CAMERA.width, 3)
        assert runtime.sensor_view("forward").camera is None
        with pytest.raises(ValueError, match="no camera geometry"):
            runtime.sensor_geometry("forward")


def test_resolution(nurec_dataset, monkeypatch, make_nurec_simulator):
    NuRecRpcBoundary(
        returned_resolution_hw=lambda time_us: None if time_us == 0 else (7, 9)
    ).install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    renderer = NuRecRenderer(cameras=["front"])
    with make_nurec_simulator(renderer, route_generator="none",
                               physics_update_mode="none").create_runtime() as runtime:
        runtime.reset(start)
        view = runtime.sensor_view("front")
        assert view.data.shape == (12, 16, 3) and view.depth.shape == (12, 16)
        assert runtime.sensor_geometry("front").camera == view.camera
        with pytest.raises(ValueError, match="NRE returned image dimensions that are not native"):
            runtime.step(Control())


def test_checkpoint(nurec_dataset, monkeypatch, make_nurec_simulator):
    NuRecRpcBoundary().install(monkeypatch)
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]), TrajectoryFollower(),
                               route_generator="none", physics_update_mode="none")
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        runtime.step(Control(trajectory=[((2., 0., 0.), (1., 0., 0., 0.), 200000)]))
        actor = runtime.state().actors["host"]
        runtime.set_actor_state("host", replace(actor, velocity=(1., 0., 0.)))
        saved = runtime.checkpoint()
        runtime.step(Control())
        expected = runtime.state()
        runtime.restore(saved)
        runtime.step(Control())
        np.testing.assert_allclose(runtime.state().ego.pose, expected.ego.pose)
        np.testing.assert_allclose(runtime.state().actors["host"].pose, expected.actors["host"].pose)
        end = runtime.clock.time_us + runtime.decision_period_us
        runtime.advance_interval(end, Control())
        with pytest.raises(RuntimeError, match="pending motion"):
            runtime.checkpoint()
        runtime.capture_at(end, ())


def test_moving_object_checkpoint(nurec_dataset, make_object_in_front, make_nurec_simulator):
    dataset = nurec_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    simulator = make_nurec_simulator(NuRecImageBoundary(CAMERA), depth="none",
                                     object_occlusion="geometry", physics_update_mode="none")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        asset, pose = make_object_in_front(runtime.sensor_view("front"))

        def setup(value, context):
            context.simulator_runtime.insert_object("sign", asset, pose, velocity=(2., 0., 0.))

        def move(value, context):
            world = context.simulator_runtime
            target = pose.copy()
            target[1, 3] += .4 + world.clock.time_us / 1e6
            world.move_object("sign", target)

        hooks = HookSet([Hook("simulator.setup", setup), Hook("simulator.step.pre", move)])
        with hooks.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            runtime.step(Control())
            checkpoint = runtime.checkpoint()
            saved = runtime.state().objects["sign"]
            recorded = runtime.observe_recorded_at(-100000).physical_state.objects["sign"]
            runtime.step(Control())
            expected = runtime.state().objects["sign"]
        runtime.reset(start)
        runtime.restore(checkpoint)
        restored = runtime.state().objects["sign"]
        np.testing.assert_array_equal(restored.pose, saved.pose)
        assert restored.velocity == saved.velocity
        past = runtime.observe_recorded_at(-100000).physical_state.objects["sign"]
        np.testing.assert_array_equal(past.pose, recorded.pose)
        assert past.velocity == recorded.velocity
        with hooks.session(runtime, ConstantAVStack()).activate():
            runtime.step(Control())
        actual = runtime.state().objects["sign"]
        np.testing.assert_array_equal(actual.pose, expected.pose)
        assert actual.velocity == expected.velocity


def test_set_actor_state(nurec_usdz_path, monkeypatch, make_object_in_front, make_nurec_simulator):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = NuRecDataset([nurec_usdz_path], time_step=.1)
    frames = dataset.frames("scene")
    start = dataset.scenario_start("scene", frames[2])
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]), object_occlusion="geometry")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        recorded = dataset.frame_data("scene", frames[2]).actors
        request_actors = {item.track_id: item for item in boundary.last_request.dynamic_objects}
        assert request_actors.keys() == recorded.keys() == {"host", "brief"}
        for identity, state in recorded.items():
            pair = request_actors[identity].pose_pair
            # Round-trip the recorded centre pose, including its xyzw orientation, without
            # an extra half-height shift or quaternion component permutation.
            np.testing.assert_allclose(proto_world_pose(pair.start_pose), state.pose, atol=1e-12)
            np.testing.assert_allclose(proto_world_pose(pair.end_pose), state.pose, atol=1e-12)
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
            np.testing.assert_allclose(proto_world_pose(request_actors[identity].pose_pair.start_pose),
                                       actor.pose, atol=1e-12)
        np.testing.assert_allclose(state.actors["host"].pose,
                                   dataset.frame_data("scene", frames[3]).actors["host"].pose)
        np.testing.assert_allclose(runtime.sensor_view("front").actors["brief"].box_transform(),
                                   state.actors["brief"].box_transform())

        runtime.step(Control())
        expected[:3, 3] += velocity * .1
        np.testing.assert_allclose(runtime.state().actors["brief"].pose, expected)
        # An overridden actor persists even after its recorded track disappears.
        assert "brief" not in dataset.frame_data("scene", frames[4]).actors
        assert {item.track_id for item in boundary.last_request.dynamic_objects} == {"host", "brief"}
        runtime.set_actor_state("brief", replace(runtime.state().actors["brief"], velocity=(0., 0., 0.)))
        runtime.step(Control())
        np.testing.assert_allclose(runtime.state().actors["brief"].pose, expected)

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
def test_depth(nurec_dataset, camera, ground_depth, make_nurec_simulator):
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    simulator = make_nurec_simulator(NuRecImageBoundary(camera))
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
    with make_nurec_simulator(NuRecImageBoundary(camera), depth="none").create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.sensor_view("front").depth is None
    mount = np.array([[0., 0., 1., 0.], [-1., 0., 0., .5], [0., -1., 0., 1.5], [0., 0., 0., 1.]])
    side = NuRecImageBoundary(camera, camera_id="side", rig_from_camera=mount)
    with make_nurec_simulator(side).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.camera_sensors == {"side"}
        assert runtime.sensor_view("side").depth[6, 8] == pytest.approx(4., abs=1e-6)
    with pytest.raises(ValueError, match="depth must be 'mesh' or 'none'"):
        make_nurec_simulator(NuRecImageBoundary(camera), depth="unsupported")


def test_rendered_probe_occlusion(nurec_dataset, tmp_path, monkeypatch, make_object_in_front, make_nurec_simulator):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.asset_directories = {"/server/assets": str(tmp_path)}
    boundary.scene_tracks = {"host", "brief"}
    boundary.hidden = np.zeros((12, 16), bool)
    boundary.hidden[:, :8] = True  # The reconstruction covers the left half of the image.
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[2])
    renderer = NuRecRenderer(cameras=["front"], asset_directory=str(tmp_path),
                             server_asset_directory="/server/assets")
    # Rendered probes are the default object occlusion.
    simulator = make_nurec_simulator(renderer, physics_update_mode="none", depth="none")
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
        # Probe assets are present and can be used again after the service loses its edits.
        edits = [request for request in boundary.requests if hasattr(request, "insert")]
        assert edits and edits[-1].insert
        assets = {f"/server/assets/{path.name}": path for path in tmp_path.glob("avsectester_asset_*.ply")}
        for track in edits[-1].insert:
            assert track.asset_id in assets
            blob = assets[track.asset_id].read_bytes()
            assert blob.startswith(b"ply\n")
            rows = np.frombuffer(blob.split(b"end_header\n", 1)[1], "<f4").reshape(-1, 14)
            assert len(rows) == 40000
            np.testing.assert_allclose(np.exp(rows[:, 7:10]) * max(asset.extent),
                                       np.tile([.001, .009, .009], (len(rows), 1)), rtol=1e-6)
        uploads = len(edits)
        runtime.step(Control())
        assert len([request for request in boundary.requests if hasattr(request, "insert")]) == uploads
        boundary.tracks.clear()
        runtime.step(Control())
        check_capture()

        setup = HookSet([Hook("simulator.setup", lambda value, context: context.simulator_runtime.insert_object(
            "start-sign", asset, world_from_object))])
        with setup.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            # Objects present at the start are hidden by the reconstruction in past captures too.
            for observation in tuple(runtime.observe_recorded_at(-i * 100000).observation for i in range(2, 0, -1)):
                image = observation.sensor_data["front"]
                assert np.all(image[:, :8] == 30) and np.any(image[:, 8:] != 30)


def test_occlusion_tolerance(nurec_dataset, monkeypatch, make_object_in_front, tmp_path, make_nurec_simulator):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[2])
    boundary.asset_directories = {"/server/assets": str(tmp_path)}
    renderer = NuRecRenderer(cameras=["front"], asset_directory=str(tmp_path),
                             server_asset_directory="/server/assets")
    with make_nurec_simulator(renderer, physics_update_mode="none", depth="none",
                        min_track_duration_s=0, time_step=.1).create_runtime() as runtime:
        runtime.reset(start)
        asset, world_from_object = make_object_in_front(runtime.sensor_view("front"))
    # The occlusion tolerance moves only the probes, toward the camera along the ray to the box
    # centre; the composed capture keeps the object's pose.

    def capture(tolerance):
        tolerant = make_nurec_simulator(renderer, physics_update_mode="none", depth="none",
                                  occlusion_tolerance_m=tolerance)
        assert tolerant.identity["occlusion_tolerance_m"] == tolerance
        with tolerant.create_runtime() as runtime:
            runtime.reset(start)
            runtime.insert_object("sign", asset, world_from_object)
            runtime.step(Control())
            data = runtime.sensor_view("front").data
        request = boundary.last_request
        probe = next(item for item in request.dynamic_objects if item.track_id.startswith("avsectester_probe_"))
        return (data, proto_world_pose(request.sensor_pose.start_pose)[:3, 3],
                proto_world_pose(probe.pose_pair.start_pose)[:3, 3])

    data, camera_centre, probe_centre = capture(0.)
    moved_data, _, moved_centre = capture(.25)
    ray = probe_centre - camera_centre
    np.testing.assert_allclose(moved_centre, probe_centre - .25 * ray / np.linalg.norm(ray), atol=1e-6)
    np.testing.assert_array_equal(moved_data, data)
    assert make_nurec_simulator(renderer).occlusion_tolerance_m == .15

    unconfigured = make_nurec_simulator(NuRecRenderer(cameras=["front"]), physics_update_mode="none",
                                  depth="none")
    with unconfigured.create_runtime() as runtime:
        runtime.reset(start)
        runtime.insert_object("sign", asset, world_from_object)
        runtime.step(Control())
        with pytest.raises(ValueError, match="asset_directory"):
            runtime.sensor_view("front")
    with pytest.raises(ValueError, match="object_occlusion must be"):
        make_nurec_simulator(renderer, object_occlusion="unsupported")


def test_validate_start(nurec_dataset, make_nurec_simulator):
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    camera_only = make_nurec_simulator(NuRecRenderer(cameras=["front"]))
    assert camera_only.observation_lead_steps == 0
    assert camera_only.past_steps == 0
    camera_only.validate_start(start)
    # A recording that turns back on itself has no valid navigation route.
    last = start.recorded_ego[-1]
    reversed_pose = last.pose.copy()
    reversed_pose[0, 3] -= 1.
    folded = replace(start, recorded_ego=(*start.recorded_ego, replace(last, pose=reversed_pose)))
    for kind in ("map", "recorded"):
        make_nurec_simulator(NuRecRenderer(cameras=["front"]), route_generator=kind).validate_start(start)
        # A scenario has one recording per simulator, so the folded recording needs its own.
        simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]), route_generator=kind)
        for _ in range(2):
            with pytest.raises(UnsupportedStartError, match="folds back"):
                simulator.validate_start(folded)
    make_nurec_simulator(NuRecRenderer(cameras=["front"]), route_generator="none").validate_start(folded)

    missing = replace(start, scenario_map=Unavailable("No map"))
    with pytest.raises(UnsupportedStartError, match="recorded.*none"):
        make_nurec_simulator(NuRecRenderer(cameras=["front"])).validate_start(missing)


@pytest.mark.parametrize("mode,height", [("ego_only", .05), ("none", .3)], ids=["grounded", "unchanged"])
def test_ground_contact(mode, height, nurec_dataset, monkeypatch, make_nurec_simulator):
    NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]), physics_update_mode=mode,
                               ground_mesh="mesh_ground")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        runtime.step(Control())
        assert runtime.state().ego.pose[2, 3] == pytest.approx(height)
        assert runtime.termination_details["ground_contact_failures"] == {}


def test_ground_contact_failure(nurec_dataset, monkeypatch, make_nurec_simulator):
    NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    ego = start.payload["ego"]
    pose = ego.pose.copy()
    pose[0, 3] = 1000.
    distant = replace(start, payload={**start.payload, "ego": replace(ego, pose=pose)})
    with make_nurec_simulator(NuRecRenderer(cameras=["front"]), route_generator="none",
                        min_track_duration_s=0, time_step=.1).create_runtime() as runtime:
        runtime.reset(distant)
        runtime.step(Control())
        assert runtime.state().ego.pose[2, 3] == pytest.approx(.3)
        assert runtime.termination_details["ground_contact_failures"] == {"insufficient_intersections": 1}


def test_observe_recorded_at(nurec_dataset, monkeypatch, make_object_in_front, make_nurec_simulator):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    frames = nurec_dataset.frames("scene")
    start = nurec_dataset.scenario_start("scene", frames[3])
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]),
                               physics_update_mode="none", object_occlusion="geometry")
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
            request_count = len(boundary.requests)
            pose_capture = runtime.observe_recorded_at(-100000, ())
            pose_only = pose_capture.observation
            assert pose_capture.sensor_geometry == {} and pose_capture.object_visibility == {}
            assert pose_only.sensor_data == {} and pose_only.calibration == {}
            assert pose_only.t == -.1
            assert len(boundary.requests) == request_count
            recorded_truth = runtime.observe_recorded_at(-100000).physical_state
            recorded_x = recorded_truth.actors["host"].pose[0, 3]
            recorded_truth.actors["host"].pose[0, 3] = 999.
            runtime.observe_recorded_at(-200000).physical_state
            assert runtime.observe_recorded_at(-100000).physical_state.actors["host"].pose[0, 3] == recorded_x
            past = tuple(runtime.observe_recorded_at(-i * 100000).observation for i in range(3, 0, -1))
            assert runtime.can_observe_past
            assert [observation.frame for observation in past] == [-3, -2, -1]
            np.testing.assert_allclose([observation.t for observation in past], [-.3, -.2, -.1])
            rendered = [item for item in boundary.requests if hasattr(item, "frame_start_us")]
            assert len(rendered) >= 3
            rendered = rendered[-3:]
            for observation, native, frame in zip(past, rendered, frames[:3]):
                recorded = nurec_dataset.frame_data("scene", frame)
                assert native.frame_start_us == round(frame.timestamp * 1e6)
                np.testing.assert_allclose(proto_world_pose(native.sensor_pose.start_pose),
                    recorded.ego.pose @ proto_world_pose(boundary.mount), atol=1e-12)
                assert observation.vehicle_state.x == pytest.approx(recorded.ego.pose[0, 3])
                assert observation.ego_speed == pytest.approx(np.linalg.norm(recorded.ego.velocity))
                # Past captures bypass render hooks, actor overrides and framework insertions.
                np.testing.assert_array_equal(observation.sensor_data["front"], 30)
                actors = {item.track_id: item for item in native.dynamic_objects}
                assert actors.keys() == recorded.actors.keys()
                for identity, actor in recorded.actors.items():
                    np.testing.assert_allclose(proto_world_pose(actors[identity].pose_pair.start_pose),
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
            with pytest.raises(ValueError, match="precedes available source history"):
                runtime.observe_recorded_at(-100000)
        setup = HookSet([Hook("simulator.setup", lambda value, context: context.simulator_runtime.insert_object(
            "start-sign", asset, world_from_object))])
        with setup.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            # A world object present at the start is composed into the recorded past captures.
            for observation in tuple(runtime.observe_recorded_at(-i * 100000).observation for i in range(3, 0, -1)):
                image = observation.sensor_data["front"]
                assert np.any(image != 30) and np.all(image[0, 0] == 30)


@pytest.fixture
def traffic_world(traffic_recording):
    pytest.importorskip("alpasim_grpc")
    from alpasim_grpc.v0 import common_pb2 as common, traffic_pb2 as traffic

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
    with pytest.raises(UnsupportedStartError, match="15 past decision steps"):
        world.simulator.validate_start(world.dataset.scenario_start("scene", world.frames[14]))
    with pytest.raises(ValueError, match="whole multiples of the traffic step"):
        NuRecSimulator(world.boundary, traffic=world.simulator.traffic, time_step=.15)
    with world.simulator.create_runtime(7) as runtime:
        runtime.reset(world.start)
        request = world.service.sessions[-1]
        assert set(runtime.state().actors) == {"host", "static", "fallback", "parked-brief", "expired"}
        assert request.scene_id == "scene"
        assert request.random_seed == 7 and request.handover_time_us == 1600000
        trajectories = request.logged_object_trajectories
        assert trajectories[0].object_id == "EGO"
        assert {item.object_id for item in trajectories} == {"EGO", "host", "static", "fallback", "expired"}
        ego = trajectories[0]
        assert (ego.aabb.size_x, ego.aabb.size_y, ego.aabb.size_z) == runtime.state().ego.extent
        assert len(ego.trajectory.poses) == len(world.start.recorded_ego)
        for point, recorded in zip(ego.trajectory.poses, world.start.recorded_ego):
            np.testing.assert_allclose(proto_world_pose(point.pose), recorded.box_transform(), atol=1e-12)
        assert {item.object_id for item in trajectories if item.is_static} == {"static"}
        with pytest.raises(RuntimeError, match="checkpoint"):
            runtime.checkpoint()
        first_session = request.session_uuid
        runtime.reset(world.start)
        assert world.service.closed[0].session_uuid == first_session
    assert len(world.service.closed) == 2
    world.boundary.fail_render = True
    with pytest.raises(RuntimeError, match="image service"):
        with world.simulator.create_runtime(7) as runtime:
            runtime.reset(world.start)
    assert world.service.closed[-1].session_uuid == world.service.sessions[-1].session_uuid


def test_intermediate_traffic_snapshot(traffic_world, tmp_path):
    source_path = traffic_world.dataset.usdz_paths[0]
    path = tmp_path / "longer.usdz"
    with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(path, "w") as target:
        metadata = json.loads(source.read("rig_trajectories.json"))
        recording = metadata["rig_trajectories"][0]
        for camera, times in recording["cameras_frame_timestamps_us"].items():
            times.append([4000000, 4000001])
            transforms = recording["cameras_frame_T_rig_worlds"][camera]
            transforms.append(transforms[-1])
        for name in source.namelist():
            payload = source.read(name)
            if name == "rig_trajectories.json":
                payload = json.dumps(metadata)
            elif name == "sequence_tracks.json":
                tracks = json.loads(payload)
                data = tracks["scene"]["tracks_data"]
                data["tracks_timestamps_us"][0] = [0, 3050000, 4000000]
                data["tracks_poses"][0] = [[20, 0, 1, 0, 0, 0, 1],
                                          [26, 0, 1, 0, 0, 0, 1], [24, 0, 1, 0, 0, 0, 1]]
                data["tracks_timestamps_us"][1] = [0, 3050000, 4000000]
                data["tracks_poses"][1] = [[8, 2, 1, 0, 0, 0, 1],
                                          [9, 2, 1, 0, 0, 0, 1], [8, 2, 1, 0, 0, 0, 1]]
                payload = json.dumps(tracks)
            target.writestr(name, payload)
    dataset = NuRecDataset([path], time_step=.2)
    start = dataset.scenario_start("scene", dataset.frames("scene")[15])
    simulator = NuRecSimulator(traffic_world.boundary, traffic=traffic_world.simulator.traffic,
        time_step=.2, depth="none", route_generator="none", physics_update_mode="none")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={"front": SensorTiming(frame_interval_us=100000)}))
        runtime.reset(start)
        runtime.advance_interval(200000, Control(), report_times_us=(100000,))
        runtime.capture_at(100000, ("front",))
        assert runtime.state().actors["host"].pose[0, 3] == 71.
        host = runtime.state().actors["host"]
        pose = host.pose.copy()
        pose[0, 3] = 99.
        runtime.set_actor_state("host", replace(host, pose=pose))
        assert runtime.sensor_view("front").actors["host"].pose[0, 3] == 71.
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(force_gt_duration_us=200000,
            sensors={"front": SensorTiming(frame_interval_us=50000)}))
        runtime.reset(start)
        assert traffic_world.service.sessions[-1].handover_time_us == 3400000
        runtime.advance_interval(200000, Control(), report_times_us=(50000, 100000, 150000),
                                 force_gt=True)
        for timestamp in (50000, 100000, 150000):
            first = dataset.actor_states("scene", 3000000)["host"].pose
            last = dataset.actor_states("scene", 3200000)["host"].pose
            expected = interpolate_transform(first, last, timestamp / 200000)
            runtime.capture_at(timestamp, ())
            np.testing.assert_array_equal(runtime.state().actors["host"].pose, expected)
            np.testing.assert_array_equal(runtime.state().actors["static"].pose,
                dataset.actor_states("scene", 3000000 + timestamp)["static"].pose)
    recorded_traffic = NuRecSimulator(traffic_world.boundary, time_step=.2, depth="none",
        route_generator="none", physics_update_mode="none")
    with recorded_traffic.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(force_gt_duration_us=200000))
        runtime.reset(start)
        runtime.advance_interval(200000, Control(), force_gt=True)
        expected = interpolate_transform(dataset.actor_states("scene", 3000000)["host"].pose,
                                         dataset.actor_states("scene", 3200000)["host"].pose, .25)
        runtime.capture_at(50000, ())
        np.testing.assert_array_equal(runtime.state().actors["host"].pose, expected)
        np.testing.assert_array_equal(runtime.state().actors["static"].pose,
                                     dataset.actor_states("scene", 3050000)["static"].pose)


def test_step_traffic(traffic_world, tmp_path):
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
    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), history=HistoryRequirement(samples=1))
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

    with traffic_world.simulator.create_runtime() as runtime:
        runtime.reset(traffic_world.start)
        runtime.step(Control())
        assert runtime.state().actors["host"].velocity == pytest.approx((345., 20., 0.))
        request = world.service.requests[-1]
        assert request.time_query_us == 1600000
        assert [item.object_id for item in request.object_trajectory_updates] == ["EGO"]
        update = request.object_trajectory_updates[0]
        assert update.object_id == "EGO" and update.trajectory.poses[-1].timestamp_us == request.time_query_us
        np.testing.assert_allclose(proto_world_pose(update.trajectory.poses[-1].pose),
                                   runtime.state().ego.box_transform(), atol=1e-12)

    with traffic_world.simulator.create_runtime() as runtime:
        runtime.reset(traffic_world.start)
        runtime.step(Control())
        expected = traffic_world.dataset.actor_states("scene", 1600000)["fallback"]
        np.testing.assert_array_equal(runtime.state().actors["fallback"].pose, expected.pose)
        assert runtime.state().actors["fallback"].pose[0, 3] == pytest.approx(31.6)

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


def test_set_actor_state_with_traffic(traffic_world):
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
        update = next(item for item in world.service.requests[-1].object_trajectory_updates if item.object_id == "host")
        assert update.trajectory.poses[-1].timestamp_us == world.service.requests[-1].time_query_us
        assert update.trajectory.poses[-1].pose.vec.x == pytest.approx(70.2)
        runtime.set_actor_state("host", replace(runtime.state().actors["host"], velocity=(0., 0., 0.)))
        runtime.step(Control())
        assert runtime.state().actors["host"].pose[0, 3] == pytest.approx(70.2)
        update = next(item for item in world.service.requests[-1].object_trajectory_updates if item.object_id == "host")
        assert [point.timestamp_us for point in update.trajectory.poses] == list(
            range(0, 1800001, 100000))
        assert update.trajectory.poses[-1].timestamp_us == world.service.requests[-1].time_query_us
        assert update.trajectory.poses[-1].pose.vec.x == pytest.approx(70.2)
        np.testing.assert_array_equal(dict(runtime.sensor_view("front").native["dynamic_objects"])["host"],
                                      runtime.state().actors["host"].pose)


def test_recorded_actor_filtering(traffic_recording, make_nurec_simulator):
    camera = PinholeCamera(np.array([[20., 0., 16.], [0., 20., 12.], [0., 0., 1.]]), 32, 24)
    boundary = NuRecTrafficImageBoundary(camera)
    frames = traffic_recording.frames("scene")
    start = traffic_recording.scenario_start("scene", frames[15])
    simulator = make_nurec_simulator(boundary, depth="none", route_generator="none",
                               physics_update_mode="none", min_track_duration_s=3.)
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
        runtime.observe_recorded_at(-100000)
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
    unfiltered = make_nurec_simulator(boundary, depth="none", route_generator="none",
                                physics_update_mode="none")
    assert unfiltered.identity["min_track_duration_s"] == 0
    with unfiltered.create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.state().actors.keys() == traffic_recording.actor_states("scene", 1500000).keys()
        assert runtime.sensor_view("front").actors.keys() == runtime.state().actors.keys()
        assert boundary.rendered[-1][1]["brief"][2, 3] == 1.
        runtime.set_actor_state("brief", replace(runtime.state().actors["brief"], velocity=(0., 0., 0.)))


def test_nurec_simulator():
    renderer = NuRecRenderer(cameras=["front"])
    simulator = NuRecSimulator(renderer, time_step=.2, depth="none", physics_update_mode="none")
    assert simulator.time_step == simulator.decision_time_step == .2
    assert simulator.identity["time_step"] == .2
    with pytest.raises(ValueError, match="dataset.time_step"):
        NuRecSimulator(renderer)


def test_render_hooks(nurec_dataset, monkeypatch, make_nurec_simulator):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    physical = start.payload["ego"].pose.copy()

    def move_camera(request, context):
        pose = request.pose.copy()
        pose[1, 3] += 2.
        return replace(request, pose=pose)

    hooks = HookSet([Hook("render.pre", move_camera),
                     Hook("render.post", lambda image, context: image + 7)])
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"]), depth="none",
                               route_generator="none", physics_update_mode="none",
                               min_track_duration_s=0, time_step=.1)
    with simulator.create_runtime() as runtime, hooks.session(runtime, ConstantAVStack()).activate():
        runtime.reset(start)
        np.testing.assert_array_equal(runtime.state().ego.pose, physical)
        expected = physical.copy()
        expected[1, 3] += 2.
        np.testing.assert_allclose(proto_world_pose(boundary.last_request.sensor_pose.start_pose),
                                   expected @ proto_world_pose(boundary.mount))
        np.testing.assert_array_equal(runtime.observe().sensor_data["front"],
                                      np.full((12, 16, 3), 37, np.uint8))


@pytest.fixture
def lidar_recording(nurec_longitudinal_dataset, tmp_path):
    with zipfile.ZipFile(nurec_longitudinal_dataset.usdz_path("scene")) as source:
        entries = {name: source.read(name) for name in source.namelist()}

    def recording(scan_period_us):
        metadata = json.loads(entries["rig_trajectories.json"])
        metadata["rig_trajectories"][0]["lidars_frame_timestamps_us"] = {
            "lidar_gt_top_p128@scene": [[0, scan_period_us]]}
        path = tmp_path / f"scan-{scan_period_us}.usdz"
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in entries.items():
                archive.writestr(name, json.dumps(metadata) if name == "rig_trajectories.json" else payload)
        return NuRecDataset([path], time_step=.1)

    return recording


def test_lidar(lidar_recording, monkeypatch, make_nurec_simulator):
    dataset = lidar_recording(100500)
    frames = dataset.frames("scene")
    start = dataset.scenario_start("scene", frames[3])
    lidar = "lidar_gt_top_p128"
    boundary = NuRecRpcBoundary().install(monkeypatch)
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"], lidar=lidar),
                                     depth="none", physics_update_mode="none", route_generator="none")
    assert simulator.past_steps == 0
    assert simulator.observation_lead_steps == 1
    with pytest.raises(UnsupportedStartError,
                       match="NuRec LiDAR requires 1 recorded decision step before the start"):
        simulator.validate_start(dataset.scenario_start("scene", frames[0]))
    simulator.validate_start(dataset.scenario_start("scene", frames[1]))
    stages = []

    def rendering(value, context):
        stages.append((context.stage, context.sensor))
        if context.stage == "render.post" and context.sensor == lidar:
            return PointCloud(value.points + np.array([0., .5, 0.], np.float32), value.intensity)
        return value

    hooks = HookSet([Hook("render.pre", rendering), Hook("render.post", rendering)])
    with simulator.create_runtime() as runtime, hooks.session(runtime, ConstantAVStack()).activate():
        runtime.reset(start)
        observation = runtime.observe()
        assert list(observation.sensor_data) == ["front", lidar]
        assert runtime.sensors == {"front", lidar} and runtime.camera_sensors == {"front"}
        cloud = observation.sensor_data[lidar]
        assert isinstance(cloud, PointCloud) and cloud.intensity is None
        np.testing.assert_allclose(cloud.points, [[4., 1.5, -1.95], [8., -1.5, -1.]])
        calibration = observation.calibration[lidar]
        assert isinstance(calibration, LidarCalibration) and calibration.spin_period_s == .1005
        expected_mount = np.eye(4)
        expected_mount[:3, 3] = [1.17, 0., 1.95]
        np.testing.assert_array_equal(calibration.rig_from_lidar, expected_mount)
        request = boundary.lidar_requests[-1]
        assert (request.frame_start_us, request.frame_end_us) == (200000, 300000)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.start_pose),
                                  dataset.ego_state("scene", 200000).pose @ expected_mount)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.end_pose),
                                  runtime.state().ego.pose @ expected_mount)
        actors_start = dataset.actor_states("scene", 200000)
        actors_end = dataset.actor_states("scene", 300000)
        assert {item.track_id for item in request.dynamic_objects} == actors_start.keys() & actors_end.keys()
        for item in request.dynamic_objects:
            np.testing.assert_allclose(proto_world_pose(item.pose_pair.start_pose), actors_start[item.track_id].pose)
            np.testing.assert_allclose(proto_world_pose(item.pose_pair.end_pose), actors_end[item.track_id].pose)
        view = runtime.sensor_view(lidar)
        assert view.camera is None and view.cam_from_world is None
        assert view.native["timestamp_us"] == 300000
        assert {identity for identity, _ in view.native["dynamic_objects"]} == actors_end.keys()
        np.testing.assert_array_equal(view.data.points, cloud.points)
        np.testing.assert_allclose(view.sensor_from_world,
                                  np.linalg.inv(runtime.state().ego.pose @ expected_mount))
        with pytest.raises(ValueError, match="has no camera geometry"):
            runtime.sensor_geometry(lidar)
        runtime.step(Control(throttle=.2))
        request = boundary.lidar_requests[-1]
        assert (request.frame_start_us, request.frame_end_us) == (300000, 400000)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.start_pose),
                                  dataset.ego_state("scene", 300000).pose @ expected_mount)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.end_pose),
                                  runtime.state().ego.pose @ expected_mount)
        assert stages == [("render.pre", "front"), ("render.post", "front"),
                          ("render.post", lidar)] * 2


def test_lidar_recorded_history(lidar_recording, monkeypatch, make_nurec_simulator):
    dataset = lidar_recording(100500)
    lidar = "lidar_gt_top_p128"
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    boundary = NuRecRpcBoundary().install(monkeypatch)
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"], lidar=lidar),
        depth="none", physics_update_mode="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        runtime.step(Control())
        past = runtime.observe_recorded_at(-100000, (lidar,)).observation
        assert isinstance(past.sensor_data[lidar], PointCloud)
        np.testing.assert_allclose(past.sensor_data[lidar].points[:, 1], [1., -2.])
        request = boundary.lidar_requests[-1]
        assert (request.frame_start_us, request.frame_end_us) == (100000, 200000)
        mount = past.calibration[lidar].rig_from_lidar
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.start_pose),
                                  dataset.ego_state("scene", 100000).pose @ mount)
        np.testing.assert_allclose(proto_world_pose(request.sensor_pose.end_pose),
                                  dataset.ego_state("scene", 200000).pose @ mount)


def test_lidar_scan_clock(lidar_recording, monkeypatch, make_nurec_simulator):
    dataset = lidar_recording(33333)
    lidar = "lidar_gt_top_p128"
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    boundary = NuRecRpcBoundary().install(monkeypatch)
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"], lidar=lidar),
        depth="none", physics_update_mode="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={lidar: SensorTiming(33333)}))
        runtime.reset(start)
        runtime.advance_interval(100000, Control(), report_times_us=(33333, 66667))
        for timestamp in (33333, 66666, 99999):
            runtime.capture_at(timestamp, (lidar,))
        assert [(item.frame_start_us, item.frame_end_us) for item in boundary.lidar_requests] == [
            (266667, 300000), (300000, 333333), (333333, 366666), (366666, 399999)]
        runtime.observe_recorded_at(-33333, (lidar,))
        request = boundary.lidar_requests[-1]
        assert (request.frame_start_us, request.frame_end_us) == (233334, 266667)
    inconsistent = lidar_recording(102000)
    with simulator.create_runtime() as runtime:
        with pytest.raises(ValueError, match="acquisition period within 1%"):
            runtime.reset(inconsistent.scenario_start("scene", inconsistent.frames("scene")[3]))


def test_lidar_history_requirement(lidar_recording, monkeypatch, make_nurec_simulator):
    dataset = lidar_recording(200000)
    lidar = "lidar_gt_top_p128"
    boundary = NuRecRpcBoundary().install(monkeypatch)
    simulator = make_nurec_simulator(NuRecRenderer(cameras=["front"], lidar=lidar),
        depth="none", physics_update_mode="none", route_generator="none")
    with pytest.raises(ValueError, match="does not accept shutter_duration_us"):
        simulator.validate_timing(TimingConfiguration(sensors={lidar: SensorTiming(100000, 1000)}))
    timing = TimingConfiguration(sensors={lidar: SensorTiming(200000)})
    assert simulator.observation_lead_steps_for(timing) == 2
    with ClosedLoopServices(dataset, simulator, timing=timing) as services, services.session(0) as session:
        assert not session.can_start_at(1)
        assert session.can_start_at(2)
    early = dataset.scenario_start("scene", dataset.frames("scene")[1])
    with pytest.raises(UnsupportedStartError, match="requires 2 recorded"):
        simulator.validate_start_timing(early, timing)
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(timing)
        runtime.reset(dataset.scenario_start("scene", dataset.frames("scene")[2]))
        request = boundary.lidar_requests[-1]
        assert request.frame_end_us - request.frame_start_us == 200000


def test_composite_shared_asset(nurec_longitudinal_dataset, monkeypatch,
                               make_nurec_simulator, tmp_path):
    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    start = replace(start, payload={**start.payload,
                    "ego": replace(start.payload["ego"], velocity=(0., 0., 0.))})
    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.scene_tracks = set(dataset.track_trajectories("scene"))
    renderer = NuRecRenderer(cameras=["front"], asset_directory=tmp_path)
    simulator = make_nurec_simulator(renderer, object_insertion="composite",
                                     depth="none", physics_update_mode="none",
                                     route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        clean = runtime.sensor_view("front")
        asset = ObjectAsset.from_panel(
            np.full((256, 256, 4), [230, 10, 90, 255], np.uint8),
            category="sign", width_m=2., mount_height_m=0., post_count=0)
        objects = []
        for object_id, horizontal in (("left", -1.5), ("right", 1.5)):
            optical_from_object = np.array([
                [0., 1., 0., horizontal], [0., 0., -1., 1.],
                [-1., 0., 0., 4.], [0., 0., 0., 1.]])
            pose = np.linalg.inv(clean.cam_from_world) @ optical_from_object
            objects.append(PlacedObject(object_id, asset, pose))
        expected = plan_composite(clean.camera, clean.cam_from_world, objects).apply(clean.data)
        assert all(expected.object_masks[placed.object_id].any() for placed in objects)
        assert not np.any(expected.object_masks["left"] & expected.object_masks["right"])
        first, second = objects
        runtime.insert_object(first.object_id, first.asset, first.world_from_object)
        runtime.step(Control())
        first_mask = runtime.sensor_view("front").object_masks[first.object_id].copy()
        np.testing.assert_array_equal(first_mask, expected.object_masks[first.object_id])
        assert len(boundary.insertion_counts) == 2
        assert set(boundary.insertion_counts.values()) == {1}
        runtime.insert_object(second.object_id, second.asset, second.world_from_object)
        runtime.step(Control())
        view = runtime.sensor_view("front")
        np.testing.assert_array_equal(view.object_masks[first.object_id], first_mask)
        for placed in objects:
            mask = expected.object_masks[placed.object_id]
            np.testing.assert_array_equal(view.object_masks[placed.object_id], mask)
            assert runtime.object_visibility(
                placed.object_id)["front"].visible_pixels == int(mask.sum())
        np.testing.assert_array_equal(view.data, expected.image)
        assert len(boundary.insertion_counts) == 4
        assert set(boundary.insertion_counts.values()) == {1}
        edits = [request for request in boundary.requests if hasattr(request, "insert")]
        assert len({track.asset_id for request in edits for track in request.insert}) == 2
        assert all(len(request.insert) == 1 for request in edits)


def test_scene_track_deduplication(nurec_longitudinal_dataset, monkeypatch,
                                   make_nurec_simulator, tmp_path):
    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    start = replace(start, payload={**start.payload,
                    "ego": replace(start.payload["ego"], velocity=(0., 0., 0.))})
    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.scene_tracks = set(dataset.track_trajectories("scene"))
    simulator = make_nurec_simulator(
        NuRecRenderer(cameras=["front"], asset_directory=tmp_path),
        object_insertion="scene", depth="none", physics_update_mode="none",
        route_generator="none")
    harmonizer = ExposureHarmonizer(reference="under", feather_sigma=0)
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        clean = runtime.sensor_view("front")
        texture = np.full((256, 256, 4), [230, 10, 90, 255], np.uint8)
        asset = ObjectAsset.from_panel(
            texture, category="sign", width_m=2., mount_height_m=0., post_count=0)
        surface = asset.surfaces[0]
        asset = replace(asset, surfaces=(replace(
            surface, nominal_texture=np.full_like(texture, [180, 180, 180, 255])),))
        objects = []
        for object_id, horizontal in (("left", -1.5), ("right", 1.5)):
            optical_from_object = np.array([
                [0., 1., 0., horizontal], [0., 0., -1., 1.],
                [-1., 0., 0., 4.], [0., 0., 0., 1.]])
            pose = np.linalg.inv(clean.cam_from_world) @ optical_from_object
            placed = PlacedObject(object_id, asset, pose, harmonizer)
            objects.append(placed)
            runtime.insert_object(object_id, asset, pose, harmonizer=harmonizer)
        expected = plan_composite(clean.camera, clean.cam_from_world, objects).apply(clean.data)
        runtime.step(Control())
        initial = runtime.sensor_view("front")
        for placed in objects:
            mask = expected.object_masks[placed.object_id]
            assert mask.any()
            np.testing.assert_array_equal(initial.object_masks[placed.object_id], mask)
            assert runtime.object_visibility(
                placed.object_id)["front"].visible_pixels == int(mask.sum())
        left_mask = initial.object_masks["left"].copy()
        left_pixels = initial.data[left_mask].copy()
        assert len(boundary.insertion_counts) == 6
        assert set(boundary.insertion_counts.values()) == {1}
        changed = replace(asset, surfaces=(replace(
            asset.surfaces[0], texture=np.full_like(texture, [10, 230, 90, 255])),))
        runtime.remove_object("right")
        runtime.insert_object("right", changed, objects[1].world_from_object, harmonizer=harmonizer)
        before = len(boundary.requests)
        runtime.step(Control())
        updated = runtime.sensor_view("front")
        np.testing.assert_array_equal(updated.object_masks["left"], left_mask)
        np.testing.assert_array_equal(updated.data[left_mask], left_pixels)
        assert np.any(updated.data[updated.object_masks["right"]]
                      != initial.data[initial.object_masks["right"]])
        assert len(boundary.insertion_counts) == 7
        assert set(boundary.insertion_counts.values()) == {1}
        active = {
            item.track_id for request in boundary.requests[before:]
            if hasattr(request, "dynamic_objects") for item in request.dynamic_objects
            if item.track_id.startswith("avsectester_")}
        assert len(active) == 6
        boundary.tracks.clear()
        boundary.inserted_tracks.clear()
        boundary.stored_trajectories.clear()
        runtime.step(Control())
        restarted = runtime.sensor_view("front")
        np.testing.assert_array_equal(restarted.data, updated.data)
        for placed in objects:
            np.testing.assert_array_equal(
                restarted.object_masks[placed.object_id], updated.object_masks[placed.object_id])
        assert boundary.insertion_counts == {
            key: 1 + int(key in active) for key in boundary.insertion_counts}
        edits = [request for request in boundary.requests if hasattr(request, "insert")]
        assert all(len(request.insert) == 1 for request in edits)


def test_scene_insertion(nurec_longitudinal_dataset, monkeypatch, make_nurec_simulator, tmp_path):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    boundary.hidden = np.zeros((12, 16), bool)
    boundary.hidden[:, :8] = True
    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    start = replace(start, payload={**start.payload,
                    "ego": replace(start.payload["ego"], velocity=(0., 0., 0.))})
    renderer = NuRecRenderer(cameras=["front"], lidar="lidar_gt_top_p128",
                             asset_directory=str(tmp_path))
    simulator = make_nurec_simulator(renderer, object_insertion="scene", depth="none",
                                     physics_update_mode="none", route_generator="none")
    assert simulator.identity["object_insertion"] == "scene"
    assert "object_insertion" not in make_nurec_simulator(renderer).identity
    with pytest.raises(ValueError, match="object_insertion must be 'composite' or 'scene'"):
        make_nurec_simulator(renderer, object_insertion="invalid")
    with pytest.raises(ValueError, match="requires renderer.asset_directory"):
        make_nurec_simulator(NuRecRenderer(cameras=["front"]), object_insertion="scene")
    post = []

    def rendered(value, context):
        post.append((context.sensor, value))
        return value

    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        view = runtime.sensor_view("front")
        # This face covers the complete miniature camera, so its silhouette is explicit.
        asset = ObjectAsset.from_panel(np.full((256, 256, 4), [230, 10, 90, 255], np.uint8),
                                       category="sign", width_m=16.)
        optical_from_object = np.array([[0., 1., 0., 0.], [0., 0., -1., 8.],
                                         [-1., 0., 0., 2.], [0., 0., 0., 1.]])
        pose = np.linalg.inv(view.cam_from_world) @ optical_from_object
        hooks = HookSet([
            Hook("simulator.setup", lambda value, context:
                 context.simulator_runtime.insert_object("sign", asset, pose)),
            Hook("render.post", rendered),
        ])
        with hooks.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            view = runtime.sensor_view("front")
            mask = view.object_masks["sign"]
            np.testing.assert_array_equal(mask, ~boundary.hidden)
            assert view.composition is None and view.object_occlusion == "scene_render"
            assert view.occlusion_tolerance_m == 0.
            assert runtime.object_visibility("sign")["front"].visible_pixels == int(mask.sum())
            assert runtime.object_visibility("sign")["front"].occlusion_source == "scene_render"
            np.testing.assert_array_equal(view.data[~mask], 30)
            assert np.any(view.data[mask] != 30)
            np.testing.assert_array_equal(post[-2][1], view.data)
            assert isinstance(post[-1][1], PointCloud) and len(post[-1][1].points) > 2
            track, track_pose = view.native["object_tracks"]["sign"]
            assert track.startswith("avsectester_scene_")
            assert all(not identity.startswith("avsectester_scene_")
                       for identity, _ in view.native["actor_tracks"])
            request = boundary.lidar_requests[-1]
            scene_objects = [item for item in request.dynamic_objects if item.track_id == track]
            assert len(scene_objects) == 1
            np.testing.assert_allclose(proto_world_pose(scene_objects[0].pose_pair.start_pose), track_pose)
            np.testing.assert_allclose(proto_world_pose(scene_objects[0].pose_pair.end_pose), track_pose)
            with pytest.raises(ValueError, match="object_masks"):
                view.composite_plan((PlacedObject("sign", asset, pose),))
            past = runtime.observe_recorded_at(-100000).observation
            assert np.any(past.sensor_data["front"][:, 8:] != 30)
            assert track in {item.track_id for item in boundary.lidar_requests[-1].dynamic_objects}
        checkpoint = runtime.checkpoint()
        initial_image = view.data.copy()
        runtime.remove_object("sign")
        runtime.step(Control())
        assert "sign" not in runtime.state().objects
        assert track not in {item.track_id for item in boundary.last_request.dynamic_objects}
        np.testing.assert_array_equal(runtime.observe().sensor_data["front"], 30)
        runtime.restore(checkpoint)
        restored = runtime.sensor_view("front")
        np.testing.assert_array_equal(restored.data, initial_image)
        np.testing.assert_array_equal(restored.object_masks["sign"], mask)
        runtime.remove_object("sign")
        surface = asset.surfaces[0]
        changed = replace(asset, surfaces=(replace(surface,
                          texture=np.full_like(surface.texture, [10, 230, 90, 255])),))
        runtime.insert_object("sign", changed, pose)
        runtime.step(Control())
        changed_view = runtime.sensor_view("front")
        assert changed_view.native["object_tracks"]["sign"][0] != track
        assert np.any(changed_view.data[mask] != initial_image[mask])
        # Later edits do not replace objects belonging to the recorded start.
        runtime.observe_recorded_at(-100000)
        assert track in {item.track_id for item in boundary.lidar_requests[-1].dynamic_objects}
        boundary.hidden = None
        runtime.step(Control())
        np.testing.assert_array_equal(runtime.sensor_view("front").object_masks["sign"], True)


def test_moving_composite_insertion(nurec_dataset, make_nurec_simulator, make_object_in_front):
    dataset = nurec_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    simulator = make_nurec_simulator(NuRecImageBoundary(CAMERA), depth="none",
                                     object_occlusion="geometry", physics_update_mode="none")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={
            "front": SensorTiming(shutter_duration_us=50000),
        }))
        runtime.reset(start)
        clean = runtime.sensor_view("front")
        asset, pose = make_object_in_front(clean)
        runtime.insert_object("sign", asset, pose)
        target = pose.copy()
        target[1, 3] += .8

        def move(value, context):
            context.simulator_runtime.move_object("sign", target)

        hooks = HookSet([Hook("simulator.step.pre", move)])
        with hooks.session(runtime, ConstantAVStack()).activate():
            runtime.step(Control())
        view = runtime.sensor_view("front")
        end_only = plan_composite(view.camera, view.cam_from_world,
                                  (PlacedObject("sign", asset, target),)).apply(clean.data)
        np.testing.assert_array_equal(view.data, end_only.image)
        np.testing.assert_array_equal(view.object_masks["sign"], end_only.object_masks["sign"])
        midpoint = interpolate_transform(pose, target, .5)
        shutter_start = plan_composite(view.camera, view.cam_from_world,
                                       (PlacedObject("sign", asset, midpoint),)).apply(clean.data)
        assert np.any(view.data != shutter_start.image)


def test_moving_scene_insertion(nurec_longitudinal_dataset, monkeypatch, make_nurec_simulator,
                                make_object_in_front, tmp_path):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    start = replace(start, payload={**start.payload,
                    "ego": replace(start.payload["ego"], velocity=(0., 0., 0.))})
    lidar = "lidar_gt_top_p128"
    renderer = NuRecRenderer(cameras=["front"], lidar=lidar, asset_directory=tmp_path)
    simulator = make_nurec_simulator(renderer, object_insertion="scene", depth="none",
                                     physics_update_mode="none", route_generator="none")
    timing = TimingConfiguration(sensors={"front": SensorTiming(shutter_duration_us=50000)})
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(timing)
        runtime.reset(start)
        asset, pose = make_object_in_front(runtime.sensor_view("front"))
        surface = asset.surfaces[0]
        asset = replace(asset, surfaces=(replace(surface, nominal_texture=np.full_like(
            surface.texture, [180, 180, 180, 255])),))
        target = pose.copy()
        target[1, 3] += .4
        stationary = pose.copy()
        stationary[1, 3] += 1.5

        def setup(value, context):
            world = context.simulator_runtime
            world.insert_object("moving", asset, pose, velocity=(2., 0., 0.),
                                harmonizer=ExposureHarmonizer(reference="under", feather_sigma=0))
            world.insert_object("stationary", asset, stationary)

        def move(value, context):
            context.simulator_runtime.move_object("moving", target)

        hooks = HookSet([Hook("simulator.setup", setup), Hook("simulator.step.pre", move)])
        with hooks.session(runtime, ConstantAVStack()).activate():
            runtime.reset(start)
            origin = timestamp_to_microseconds(start.frame.timestamp)
            view = runtime.sensor_view("front")
            moving_track, initial_track_pose = view.native["object_tracks"]["moving"]
            static_track, static_track_pose = view.native["object_tracks"]["stationary"]

            def find(rgb_end, track_id):
                return [item for request in boundary.requests
                        if hasattr(request, "camera_intrinsics") and request.frame_end_us == rgb_end
                        and {moving_track, static_track} <= {obj.track_id for obj in request.dynamic_objects}
                        for item in request.dynamic_objects if item.track_id == track_id][-1]

            def check(object_track, start_xyz, end_xyz):
                expected_start = initial_track_pose.copy()
                expected_start[:3, 3] += start_xyz
                expected_end = initial_track_pose.copy()
                expected_end[:3, 3] += end_xyz
                np.testing.assert_allclose(proto_world_pose(object_track.pose_pair.start_pose),
                                           expected_start, atol=1e-12)
                np.testing.assert_allclose(proto_world_pose(object_track.pose_pair.end_pose),
                                           expected_end, atol=1e-12)

            check(find(origin, moving_track), [-.1, 0., 0.], [0., 0., 0.])
            static = find(origin, static_track)
            np.testing.assert_array_equal(proto_world_pose(static.pose_pair.start_pose),
                                          proto_world_pose(static.pose_pair.end_pose))
            np.testing.assert_allclose(proto_world_pose(static.pose_pair.end_pose), static_track_pose)

            runtime.observe_recorded_at(-100000)
            check(find(origin - 100000, moving_track), [-.3, 0., 0.], [-.2, 0., 0.])
            scan = boundary.lidar_requests[-1]
            check(next(item for item in scan.dynamic_objects if item.track_id == moving_track),
                  [-.4, 0., 0.], [-.2, 0., 0.])

            runtime.step(Control())
            view = runtime.sensor_view("front")
            assert view.object_masks["moving"].any()
            end = origin + 100000
            check(find(end, moving_track), [0., .2, 0.], [0., .4, 0.])
            scan = boundary.lidar_requests[-1]
            check(next(item for item in scan.dynamic_objects if item.track_id == moving_track),
                  [0., 0., 0.], [0., .4, 0.])

            # A coverage probe substitutes the moving object's track and retains the other object.
            requests = [request for request in boundary.requests if hasattr(request, "camera_intrinsics")
                        and request.frame_end_us == end]
            probes = [item for request in requests
                      if static_track in {item.track_id for item in request.dynamic_objects}
                      for item in request.dynamic_objects if item.track_id.startswith("avsectester_coverage_")]
            assert probes
            for probe in probes:
                check(probe, [0., .2, 0.], [0., .4, 0.])
            # Nominal rendering substitutes a different texture and follows the same pose pair.
            nominal = [item for request in requests for item in request.dynamic_objects
                       if item.track_id.startswith("avsectester_scene_")
                       and item.track_id not in {moving_track, static_track}]
            assert nominal
            for item in nominal:
                check(item, [0., .2, 0.], [0., .4, 0.])


@pytest.mark.parametrize("harmonizer", [
    ExposureHarmonizer(reference="under", feather_sigma=0),
    ClassicHarmonizer(poisson=False),
], ids=["tensor", "numpy"])
def test_scene_harmonizer(harmonizer, nurec_longitudinal_dataset, monkeypatch,
                          make_nurec_simulator, make_object_in_front, tmp_path):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    start = replace(start, payload={**start.payload,
                    "ego": replace(start.payload["ego"], velocity=(0., 0., 0.))})
    renderer = NuRecRenderer(cameras=["front"], asset_directory=str(tmp_path))
    simulator = make_nurec_simulator(renderer, object_insertion="scene", depth="none",
                                     physics_update_mode="none", route_generator="none")
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        asset, pose = make_object_in_front(runtime.sensor_view("front"))
        surface = asset.surfaces[0]
        texture = np.full((256, 256, 4), [230, 10, 90, 255], np.uint8)
        asset = replace(asset, surfaces=(replace(surface, texture=texture,
                         nominal_texture=np.full_like(texture, [180, 180, 180, 255])),))
        runtime.insert_object("sign", asset, pose)
        before = len(boundary.requests)
        runtime.step(Control())
        raw = runtime.sensor_view("front")
        assert raw.object_masks["sign"].any()
        # Without a harmonizer, every camera request contains an object or its coverage copy.
        requests = [item for item in boundary.requests[before:]
                    if hasattr(item, "image_quality")]
        assert requests
        assert all(any(item.track_id.startswith(("avsectester_scene_", "avsectester_coverage_"))
                       for item in request.dynamic_objects) for request in requests)
        runtime.remove_object("sign")
        runtime.insert_object("sign", asset, pose, harmonizer=harmonizer)
        before = len(boundary.requests)
        runtime.step(Control())
        adjusted = runtime.sensor_view("front")
        # Only differentiable harmonizers render the object with its nominal texture.
        track = adjusted.native["object_tracks"]["sign"][0]
        nominal = {item.track_id for request in boundary.requests[before:]
                   if hasattr(request, "image_quality") for item in request.dynamic_objects
                   if item.track_id.startswith("avsectester_scene_") and item.track_id != track}
        assert bool(nominal) == getattr(harmonizer, "differentiable", False)
        mask = adjusted.object_masks["sign"]
        np.testing.assert_array_equal(adjusted.data[~mask], raw.data[~mask])
        assert np.any(adjusted.data[mask] != raw.data[mask])
        assert adjusted.composition is None


def test_scene_render_without_actors(nurec_longitudinal_dataset, monkeypatch,
                                     make_nurec_simulator, make_object_in_front, tmp_path):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    path = tmp_path / "without-actors.usdz"
    with zipfile.ZipFile(nurec_longitudinal_dataset.usdz_paths[0]) as source:
        tracks = json.loads(source.read("sequence_tracks.json"))
        tracks["scene"]["tracks_data"] = {name: [] for name in tracks["scene"]["tracks_data"]}
        tracks["scene"]["cuboidtracks_data"]["cuboids_dims"] = []
        with zipfile.ZipFile(path, "w") as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(tracks) if name == "sequence_tracks.json"
                                else source.read(name))
    dataset = NuRecDataset([path], time_step=.1)
    frame = dataset.frames("scene")[3]
    renderer = NuRecRenderer(cameras=["front"], lidar="lidar_gt_top_p128",
                             asset_directory=str(tmp_path / "assets"))
    simulator = make_nurec_simulator(renderer, object_insertion="scene", depth="none",
                                     physics_update_mode="none", route_generator="none")
    with ClosedLoopServices(dataset, simulator) as services, services.session(0) as session:
        plain = session.render(RenderRequest("scene", frame, "front"))
        assert not boundary.last_request.dynamic_objects
        assert not boundary.lidar_requests[-1].dynamic_objects
        asset, pose = make_object_in_front(plain.view)
        surface = asset.surfaces[0]
        asset = replace(asset, surfaces=(replace(
            surface, texture=np.full((256, 256, 4), [230, 10, 90, 255], np.uint8)),))
        inserted = session.render(RenderRequest("scene", frame, "front", actions=(
            InsertObject("sign", asset, pose),
        )))
        assert np.any(inserted.view.data != plain.view.data)
        assert len(inserted.observation.sensor_data["lidar_gt_top_p128"].points) > 2
        # A fresh render has no actors or objects even though NRE retains prior scene tracks.
        clean = session.render(RenderRequest("scene", frame, "front"))
        np.testing.assert_array_equal(clean.view.data, plain.view.data)
        np.testing.assert_array_equal(clean.observation.sensor_data["lidar_gt_top_p128"].points,
                                      plain.observation.sensor_data["lidar_gt_top_p128"].points)
        assert clean.object_visibility == {}
        assert not boundary.last_request.dynamic_objects
        assert not boundary.lidar_requests[-1].dynamic_objects


def test_force_gt_physics_trajectory(nurec_longitudinal_dataset, tmp_path):
    import trimesh

    path = tmp_path / "turn.usdz"
    with zipfile.ZipFile(nurec_longitudinal_dataset.usdz_paths[0]) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    recording = json.loads(entries["rig_trajectories.json"])
    native = recording["rig_trajectories"][0]
    for camera, stamps in native["cameras_frame_timestamps_us"].items():
        poses = []
        for stamp in stamps:
            angle = stamp[0] / 1000000
            pose = np.eye(4)
            pose[:3, :3] = rotation_matrix((0., 0., np.rad2deg(angle)))
            pose[:3, 3] = (2 * np.sin(angle), 2 * (1 - np.cos(angle)),
                             .3 if stamp[0] >= 300000 else 10.)
            poses.append([pose.tolist(), pose.tolist()])
        native["cameras_frame_T_rig_worlds"][camera] = poses
    entries["rig_trajectories.json"] = json.dumps(recording).encode()
    with zipfile.ZipFile(path, "w") as target:
        for name, data in entries.items():
            target.writestr(name, data)
    dataset = NuRecDataset([path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    origin = timestamp_to_microseconds(start.frame.timestamp)
    duration = 300000
    end = timestamp_to_microseconds(dataset.frames("scene")[-1].timestamp) - origin
    times = sorted({-17000, 0, duration, end,
        *(timestamp_to_microseconds(t) - origin for t in dataset.camera_timestamps("scene")
          if -17000 <= timestamp_to_microseconds(t) - origin <= end), *range(0, end + 1, 100000)})
    ego = [dataset.ego_state("scene", origin + t) for t in times]
    mesh = trimesh.load(BytesIO(entries["mesh_ground.ply"]), file_type="ply", process=False)
    raycaster = MeshRaycaster(mesh.vertices, mesh.faces)
    blended = []
    for timestamp, state in zip(times, ego):
        box, _status = constrain_to_ground(raycaster, state.box_transform(), state.extent)
        corrected = box @ np.linalg.inv(np.linalg.inv(state.pose) @ state.box_transform())
        alpha = float(np.float32(np.clip(timestamp / duration, 0., 1.)))
        blended.append(interpolate_transform(state.pose, corrected, alpha))

    def interpolate(stamps, poses, timestamp):
        index = int(np.searchsorted(stamps, timestamp))
        if stamps[index] == timestamp:
            return poses[index]
        fraction = (timestamp - stamps[index - 1]) / (stamps[index] - stamps[index - 1])
        return interpolate_transform(poses[index - 1], poses[index], fraction)

    timing = TimingConfiguration(force_gt_duration_us=duration, pose_reporting_interval_us=50000,
                                 sensors={"front": SensorTiming(33000, 17000)})
    simulator = NuRecSimulator(NuRecImageBoundary(CAMERA), TrajectoryFollower(), time_step=.1,
        min_track_duration_s=0., depth="none", route_generator="none",
        physics_update_mode="ego_only", ground_mesh="mesh_ground")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(timing)
        runtime.reset(start)
        assert runtime.termination_details["ground_contact_failures"] == {}
        for index in range(3):
            begin, end = index * 100000, (index + 1) * 100000
            nodes = (begin, begin+50000, end)
            poses = [interpolate(times, blended, timestamp) for timestamp in nodes]
            runtime.advance_interval(end, runtime.recorded_reference(begin, end),
                                     report_times_us=(begin+50000,), force_gt=True)
            for t in (begin+1000, begin+33000, begin+50000, begin+83000, end):
                expected_pose = interpolate(nodes, poses, t)
                runtime.capture_at(t, ("front",) if t == begin+33000 else ())
                state = runtime.state()
                np.testing.assert_allclose(state.ego.pose, expected_pose, atol=2e-6)
                if t == begin+33000:
                    np.testing.assert_allclose(runtime.render_request("front").pose,
                                               expected_pose, atol=2e-6)
            runtime.capture_at(end, ())
