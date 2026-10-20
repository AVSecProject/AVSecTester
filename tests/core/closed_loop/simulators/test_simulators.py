"""Simulator components share capture/state geometry contracts at native boundaries."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.frame_data import Initialization
from avsectester.closed_loop.simulators.carla import CarlaSimulator, SensorAlignmentError, decode_carla_depth
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator, TrajectoryFollower
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.datasets import ActorState
from avsectester.datasets.carla import CarlaDataset, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset
from avsectester.geometry.cameras import pose_from_proto
from avsectester.geometry.poses import rotation_matrix
from tests.fakes import CarlaNativeBoundary, NuRecRpcBoundary


@pytest.fixture
def carla_case(carla_dataset, monkeypatch):
    CarlaNativeBoundary().install(monkeypatch)
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0)
    frames = carla_dataset.frames("scene")
    with simulator.create_runtime(seed=3) as runtime:
        sequence = runtime.generate_frame_data(carla_dataset.scenario_start("scene", frames[0]), frames)
    start = carla_dataset.scenario_start("scene", frames[2])
    initialization = Initialization(sequence.at(frames[2]), sequence.resolved_scenario, sequence.provenance)
    return simulator, start, initialization, (12, 16, 3)


@pytest.fixture
def nurec_case(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    simulator = NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"]))
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[1])
    return simulator, start, None, (12, 16, 3)


@pytest.fixture
def recorded_case(nuscenes_dataset):
    simulator = RecordedReplaySimulator()
    start = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[1])
    return simulator, start, None, (80, 100, 3)


@pytest.fixture(params=["carla_case", "nurec_case", "recorded_case"])
def simulator_case(request):
    return request.getfixturevalue(request.param)


def test_simulators(simulator_case):
    simulator, start, initialization, expected_shape = simulator_case
    with simulator.create_runtime(seed=3) as runtime:
        runtime.reset(start, initialization)
        original = runtime.observe()
        state = runtime.state()
        assert state.ego is not None and "ego" not in state.actors
        view = runtime.sensor_view("front")
        assert view.data.shape == expected_shape
        assert view.time == original.t and view.frame == original.frame
        assert view.cam_from_world.shape == (4, 4)
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


def test_decode_carla_depth():
    for metres in (0., 17.25):
        encoded = round(metres / 1000 * (2**24 - 1))
        bgra = np.array([[[encoded >> 16, (encoded >> 8) & 255, encoded & 255, 255]]], np.uint8)
        image = SimpleNamespace(height=1, width=1, raw_data=bgra.tobytes())
        assert decode_carla_depth(image)[0, 0] == pytest.approx(metres, abs=.00003)


def test_carla_capture_alignment(monkeypatch, carla_dataset):
    dataset = carla_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    CarlaNativeBoundary().install(monkeypatch)
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start)
        observation = runtime.observe()
        view = runtime.sensor_view("front")
        assert view.frame == observation.frame == runtime.clock.frame
        np.testing.assert_allclose(view.ego.transform[:3, 3], [.2, 0., .7])
        assert view.ego.extent == (4., 2., 1.4)
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


def test_nurec_actor_interpolation(nurec_usdz_path, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    dataset = NuRecDataset([nurec_usdz_path], time_step=.1)
    start = dataset.scenario_start("scene", dataset.frames("scene")[1])
    with NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"])).create_runtime() as runtime:
        runtime.reset(start)
        view = runtime.sensor_view("front")
        np.testing.assert_allclose(view.actors["host"].transform[:3, 3], [8.4, 0., 1.])
        runtime.step(Control())
        np.testing.assert_allclose(runtime.sensor_view("front").actors["host"].transform[:3, 3], [8.8, 0., 1.])


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
