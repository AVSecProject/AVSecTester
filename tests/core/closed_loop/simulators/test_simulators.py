"""Simulator components share capture/state geometry contracts at native boundaries."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import zipfile

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.frame_data import Initialization
from avsectester.closed_loop.services import ClosedLoopServices, RenderRequest, SensorRequest
from avsectester.closed_loop.simulators.carla import CarlaSimulator, SensorAlignmentError, decode_carla_depth, rotation_to_carla
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator, TrajectoryFollower
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.datasets import ActorState
from avsectester.datasets.carla import CarlaDataset, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.cameras import BODY_FROM_OPTICAL, FThetaCamera, PinholeCamera, pose_from_proto
from avsectester.geometry.poses import rotation_matrix
from tests.fakes import CarlaNativeBoundary, NuRecImageBoundary, NuRecRpcBoundary


@pytest.fixture
def carla_case(carla_dataset, monkeypatch):
    CarlaNativeBoundary().install(monkeypatch)
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0)
    frames = carla_dataset.frames("scene")
    with simulator.create_runtime(seed=3) as runtime:
        sequence = runtime.generate_frame_data(carla_dataset.scenario_start("scene", frames[0]), frames)
    start = carla_dataset.scenario_start("scene", frames[2])
    initialization = Initialization(sequence.at(frames[2]), sequence.resolved_scenario, sequence.provenance)
    initial = carla_dataset.scenario_start("scene", frames[0])
    initial_state = Initialization(sequence.at(frames[0]), sequence.resolved_scenario, sequence.provenance)
    return carla_dataset, simulator, start, initialization, (12, 16, 3), initial, initial_state


@pytest.fixture
def nurec_case(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    simulator = NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"]))
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[1])
    initial = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    return nurec_dataset, simulator, start, None, (12, 16, 3), initial, None


@pytest.fixture
def recorded_case(nuscenes_dataset):
    simulator = RecordedReplaySimulator()
    start = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[1])
    initial = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[0])
    return nuscenes_dataset, simulator, start, None, (80, 100, 3), initial, None


@pytest.fixture(params=["carla_case", "nurec_case", "recorded_case"])
def simulator_case(request):
    return request.getfixturevalue(request.param)


def test_simulators(simulator_case):
    _dataset, simulator, start, initialization, expected_shape, _initial, _initial_state = simulator_case
    with simulator.create_runtime(seed=3) as runtime:
        runtime.reset(start, initialization)
        original = runtime.observe()
        state = runtime.state()
        assert state.ego is not None and "ego" not in state.actors
        view = runtime.sensor_view("front")
        assert runtime.camera_sensors <= runtime.sensors
        assert view.sensor in runtime.sensors
        assert view.data.shape == expected_shape
        assert view.time == original.t and view.frame == original.frame
        assert view.cam_from_world.shape == (4, 4)
        assert view.sensor_from_world.shape == (4, 4)
        np.testing.assert_allclose(view.sensor_from_world, BODY_FROM_OPTICAL @ view.cam_from_world)
        assert view.camera.project(np.array([[0., 0., 4.]])).shape == (1, 2)
        runtime.step(Control(throttle=.1))
        advanced = runtime.observe()
        assert advanced.frame == state.frame + 1
        assert advanced.t - state.t == pytest.approx(.1)
        assert runtime.state().frame == advanced.frame
        with pytest.raises(ValueError):
            runtime.sensor_view("missing-camera")
    with pytest.raises(RuntimeError, match="Reset"):
        runtime.observe()


def test_sensor_data_views(simulator_case, assert_camera_views_equal):
    dataset, simulator, start, _initialization, _expected_shape, _initial, _initial_state = simulator_case
    frames = dataset.frames(start.scenario_id)[:3]
    with ClosedLoopServices(dataset, simulator) as services:
        with services.session(3) as session:
            samples = [session.sensor_data(SensorRequest(start.scenario_id, frame, ("front",),
                                                        views=True)) for frame in frames]
        for sample in samples:
            with services.session(3) as session:
                rendered = session.render(RenderRequest(start.scenario_id, sample.frame, "front"))
            assert_camera_views_equal(sample.views["front"], rendered.view)


@pytest.mark.parametrize("simulator_case", ["carla_case", "recorded_case"], indirect=True)
def test_insert_object(simulator_case, assert_inserted_world_object):
    _dataset, simulator, _start, _initialization, _shape, initial, initial_state = simulator_case
    with simulator.create_runtime(seed=3) as runtime:
        original, inserted, _expected = assert_inserted_world_object(runtime, initial, initial_state)
        assert original.depth is None and inserted.depth is None


def test_insert_object_with_depth(nurec_case, assert_inserted_world_object):
    _dataset, simulator, _start, _initialization, _shape, initial, initial_state = nurec_case
    with simulator.create_runtime(seed=3) as runtime:
        original, inserted, expected = assert_inserted_world_object(runtime, initial, initial_state)
        assert original.depth is not None
        np.testing.assert_array_equal(inserted.depth, expected.depth)


def test_decode_carla_depth():
    for metres in (0., 17.25):
        encoded = round(metres / 1000 * (2**24 - 1))
        bgra = np.array([[[encoded >> 16, (encoded >> 8) & 255, encoded & 255, 255]]], np.uint8)
        image = SimpleNamespace(height=1, width=1, raw_data=bgra.tobytes())
        assert decode_carla_depth(image)[0, 0] == pytest.approx(metres, abs=.00003)


def test_carla_capture_alignment(monkeypatch, carla_dataset):
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["sensors"] = [{"type": "CarlaLidar", "name": "lidar"}]
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    CarlaNativeBoundary().install(monkeypatch)
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start)
        observation = runtime.observe()
        view = runtime.sensor_view("front")
        assert view.frame == observation.frame == runtime.clock.frame
        np.testing.assert_allclose(view.ego.transform[:3, 3], [.2, 0., .7])
        assert view.ego.extent == (4., 2., 1.4)
        lidar = runtime.sensor_view("lidar")
        assert runtime.sensors == {"front", "lidar"} and runtime.camera_sensors == {"front"}
        assert lidar.camera is None and lidar.cam_from_world is None
        assert lidar.frame == observation.frame and lidar.time == observation.t
        np.testing.assert_array_equal(np.frombuffer(lidar.data.raw_data, np.float32).reshape(-1, 4),
                                      np.array([[5., -2., .5, .8]], np.float32))
        ego_x = runtime.state().ego.pose[0, 3]
        np.testing.assert_allclose(lidar.sensor_from_world, [
            [0., -1., 0., -2.], [1., 0., 0., -(ego_x + 1.)],
            [0., 0., 1., -1.5], [0., 0., 0., 1.],
        ], atol=1e-12)
        assert observation.sensor_data["lidar"].flipy is True
        state = runtime.state()
        assert state.traffic_lights == {"signal-1": "red"}
        traffic_light = next(actor for actor in runtime.native["world"].get_actors()
                             if actor.type_id == "traffic.traffic_light")
        traffic_light.set_state("Green")
        assert runtime.state().traffic_lights == {"signal-1": "green"}
        assert state.traffic_lights == {"signal-1": "red"}
    CarlaNativeBoundary(frame_offset=10).install(monkeypatch)
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        with pytest.raises(SensorAlignmentError, match="ahead"):
            runtime.reset(start)


def test_carla_world_settings(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
    dataset = carla_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start, world={"weather": {"precipitation": 60.},
            "road_friction": {"tire_friction": .7}, "map_layers": {"unload": ["Buildings"]}})
        native = runtime.native
        assert native["world"].weather.precipitation == 60.
        assert native["world"].layers == [("unload", "Buildings")]
        assert native["ego"].actor.get_physics_control().wheels[0].tire_friction == .7
        with pytest.raises(ValueError, match="support"):
            runtime.reset(start, world={"unknown": True})


def test_carla_scene_choices_follow_the_run_seed(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"].update(spawn="random", vehicle="random")
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    origins = []
    for _ in range(2):
        with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime(seed=17) as runtime:
            runtime.reset(start)
            origins.append((runtime.native["ego"].actor.type_id, runtime.state().ego.pose.copy()))
    assert origins[0][0] == origins[1][0]
    np.testing.assert_allclose(origins[0][1], origins[1][1])


def test_carla_actor_control_and_placement(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["npcs"] = [{"type": "CarlaMobileActor", "spawn": 2, "npc_type": "vehicle.b"}]
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario, motion={
        **recipe.motion, "npc:0": PrescribedMotion(constant_control=Control(throttle=1.)),
    })})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start)
        native = runtime.native["actors"]["npc:0"]
        runtime.control_actor("npc:0", Control(throttle=.4, steer=.2))
        runtime.step(Control())
        # The attack's command replaces the recipe's prescribed throttle on later ticks.
        assert (native.control.throttle, native.control.brake, native.control.steer) == (.4, 0., .2)

        pose = np.eye(4)
        pose[:3, 3] = (12., -3., 1.2)
        runtime.set_actor_state("npc:0", replace(runtime.state().actors["npc:0"],
                                                 pose=pose, velocity=(2., .5, 0.)))
        # CARLA's y axis points to the right of the canonical frame.
        location, velocity = native.get_transform().location, native.get_velocity()
        assert (location.x, location.y, location.z) == pytest.approx((12., 3., 1.2))
        assert (velocity.x, velocity.y, velocity.z) == pytest.approx((2., -.5, 0.))
        with pytest.raises(ValueError, match="ego is controlled by the driving system"):
            runtime.control_actor("ego", Control())
        with pytest.raises(KeyError, match="Unknown CARLA actor 'missing'"):
            runtime.set_actor_state("missing", ActorState(pose))


def test_nurec_recorded_support(nurec_usdz_path, monkeypatch):
    boundary = NuRecRpcBoundary().install(monkeypatch)
    dataset = NuRecDataset([nurec_usdz_path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[-2])
    origin = start.payload["ego"].pose.copy()
    origin[:3, :3] = rotation_matrix((8., -12., 30.))
    start = replace(start, payload={**start.payload, "ego": replace(start.payload["ego"], pose=origin)})
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.native["renderer"].origin_timestamp_us == 400000
        assert runtime.native["renderer"].origin_bounds == (0., .5)
        np.testing.assert_allclose(runtime.state().ego.pose, origin, atol=1e-12)
        np.testing.assert_allclose(pose_from_proto(boundary.last_request.sensor_pose.start_pose),
                                   origin @ pose_from_proto(boundary.mount), atol=1e-12)
        runtime.step(Control())
        assert runtime.sensor_view("front").native["timestamp_us"] == 500000
        assert (boundary.last_request.frame_start_us, boundary.last_request.frame_end_us) == (500000, 500001)
        with pytest.raises(ValueError, match="terminal recorded support"):
            runtime.step(Control())
        runtime.reset()
        assert runtime.native["renderer"].origin_bounds is None
        assert runtime.observe().t == 0.


def test_nurec_reset_requires_mesh(nurec_usdz_path, tmp_path, monkeypatch):
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


def test_nurec_trajectory(monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    simulator = NuRecSimulator({"dt": .1, "ego0": {"x": 10., "y": 5., "yaw": np.pi / 2}},
                               NuRecRenderer(cameras=["front"]), TrajectoryFollower())
    with simulator.create_runtime() as runtime:
        runtime.reset()
        runtime.step(Control(trajectory=[((2., 0., 0.), (1., 0., 0., 0.), 200000)]))
        state = runtime.state()
        np.testing.assert_allclose(state.ego.pose[:2, 3], [10., 6.], atol=1e-12)
        assert state.ego_speed == pytest.approx(10.)
        saved = runtime.checkpoint()
        runtime.step(Control())
        expected = runtime.state()
        runtime.restore(saved)
        runtime.step(Control())
        np.testing.assert_allclose(runtime.state().ego.pose, expected.ego.pose)


def test_nurec_set_actor_state(nurec_usdz_path, monkeypatch, make_object_in_front):
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
            np.testing.assert_allclose(runtime.sensor_view("front").actors[identity].transform,
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
        np.testing.assert_allclose(runtime.sensor_view("front").actors["brief"].transform,
                                   captured.actors["brief"].transform)
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
        np.testing.assert_allclose(runtime.sensor_view("front").actors["brief"].transform,
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
def test_nurec_depth(nurec_dataset, camera, ground_depth):
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
        NuRecSimulator({"dt": .1}, depth="unsupported")


def test_nurec_depth_without_default_camera(nurec_usdz_path, tmp_path, monkeypatch):
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
    assert list(dataset.raw_metadata(str(archive_path))["rig_trajectories.json"][
        "camera_calibrations"]) == ["side@scene"]
    camera, _ = dataset.camera_geometry(str(archive_path))
    np.testing.assert_allclose(camera.unproject([[8., 6.]]), [[0., 0., 1.]])
    NuRecRpcBoundary().install(monkeypatch)
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        runtime.reset(dataset.scenario_start("scene", dataset.frames("scene")[0]))
        np.testing.assert_allclose(runtime.sensor_view("front").depth[6, 8], 4., atol=1e-6)


def test_recorded_terminal_frame(nuscenes_dataset):
    frames = nuscenes_dataset.frames("scene")
    with RecordedReplaySimulator().create_runtime() as runtime:
        runtime.reset(nuscenes_dataset.scenario_start("scene", frames[1]))
        runtime.step(Control(throttle=1.))
        assert runtime.observe().frame == frames[-1].position
        assert runtime.observe().sensor_data["front"][0, 0, 0] == 52
        assert runtime.state().ego.extent is None
        with pytest.raises(ValueError, match="terminal recorded support"):
            runtime.step(Control(throttle=1.))

def test_rotation_to_carla():
    for pitch, yaw, roll in [(90., 0., 0.), (-90., 0., 0.), (90., 37., 23.),
                              (-90., -57., 17.), (20., 110., -35.), (89.99, 37., 23.)]:
        matrix = rotation_matrix((-roll, -pitch, yaw))
        native_pitch, native_yaw, native_roll = rotation_to_carla(matrix)
        np.testing.assert_allclose(rotation_matrix((-native_roll, -native_pitch, native_yaw)),
                                   matrix, atol=1e-10)
    with pytest.raises(ValueError, match="right-handed"):
        rotation_to_carla(np.diag((1., 1., -1.)))
