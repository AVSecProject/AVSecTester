"""CARLA capture alignment, native settings, deterministic preparation and actor control."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.carla.sensors import SensorAlignmentError
from avsectester.closed_loop.simulators.base import Initialization
from avsectester.closed_loop.timing import HistoryRequirement, SensorTiming, TimingConfiguration
from avsectester.datasets.carla import CarlaDataset, PrescribedMotion
from avsectester.data_types import ActorState, Control
from tests.fakes import CarlaNativeBoundary


def test_capture_alignment(monkeypatch, carla_dataset):
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["sensors"] = [{"type": "CarlaLidar", "name": "lidar", "sensor_tick": .2}]
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    with pytest.raises(ValueError, match="sensor_tick"):
        CarlaSimulator({"rate": 10}).validate_start(dataset.scenario_start("scene", dataset.frames("scene")[0]))
    scenario["ego"]["sensors"][0]["sensor_tick"] = 0.
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    CarlaNativeBoundary().install(monkeypatch, None)
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start)
        observation = runtime.observe()
        assert runtime.replay_scenario["ego"]["sensors"][0]["sensor_tick"] == 0.
        view = runtime.sensor_view("front")
        assert view.frame == observation.frame == runtime.clock.frame
        np.testing.assert_allclose(view.ego.box_transform()[:3, 3], [.2, 0., .7])
        assert view.ego.extent == (4., 2., 1.4)
        lidar = runtime.sensor_view("lidar")
        capture_us = runtime.clock.time_us
        assert runtime.frame_range_us("top_lidar", capture_us) == range(capture_us - 100000, capture_us)
        assert runtime.frame_range_us("front_camera", capture_us) == range(capture_us, capture_us)
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
        # Payload order follows the recipe even when the camera packet arrives after the LiDAR.
        manager = runtime.native["ego"].sensor_data_manager
        polls = []
        has_data = manager.has_data

        def camera_arrives_late(name):
            polls.append(name)
            return has_data(name) and (name != "front" or polls.count("front") > 1)

        monkeypatch.setattr(manager, "has_data", camera_arrives_late)
        runtime.step(Control())
        assert list(runtime.observe().sensor_data) == ["front", "lidar"]
    CarlaNativeBoundary(frame_offset=10).install(monkeypatch, None)
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        with pytest.raises(SensorAlignmentError, match="ahead"):
            runtime.reset(start)


def test_traffic_light_state(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start)
        state = runtime.state()
        assert state.traffic_lights == {"signal-1": "red"}
        traffic_light = next(actor for actor in runtime.native["world"].get_actors()
                             if actor.type_id == "traffic.traffic_light")
        traffic_light.set_state("Green")
        assert runtime.state().traffic_lights == {"signal-1": "green"}
        assert state.traffic_lights == {"signal-1": "red"}


def test_world_settings(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    dataset = carla_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        runtime.reset(start, world={"weather": {"precipitation": 60.},
            "road_friction": {"tire_friction": .7}, "map_layers": {"unload": ["Buildings"]},
            "lighting": {"vehicle_lights": {"ego": ["Position", "LowBeam"]}}})
        native = runtime.native
        assert native["world"].weather.precipitation == 60.
        import carla
        assert native["ego"].actor.light_state == (carla.VehicleLightState.Position
                                                   | carla.VehicleLightState.LowBeam)
        assert native["world"].layers == [("unload", "Buildings")]
        assert native["ego"].actor.get_physics_control().wheels[0].tire_friction == .7
        with pytest.raises(ValueError, match="road_friction.tire_friction"):
            runtime.reset(start, world={"road_friction": {"tire_friction": 0}})


def test_scene_choices_follow_the_run_seed(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
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


def test_control_actor(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
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

        with pytest.raises(ValueError, match="ego is controlled by the driving system"):
            runtime.control_actor("ego", Control())


def test_set_actor_state(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
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
        pose = np.eye(4)
        pose[:3, 3] = (12., -3., 1.2)
        runtime.set_actor_state("npc:0", replace(runtime.state().actors["npc:0"],
                                                 pose=pose, velocity=(2., .5, 0.)))
        # CARLA's y axis points to the right of the canonical frame.
        location, velocity = native.get_transform().location, native.get_velocity()
        assert (location.x, location.y, location.z) == pytest.approx((12., 3., 1.2))
        assert (velocity.x, velocity.y, velocity.z) == pytest.approx((2., -.5, 0.))
        with pytest.raises(KeyError, match="Unknown CARLA actor 'missing'"):
            runtime.set_actor_state("missing", ActorState(pose))


@pytest.fixture
def two_tick_start(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    simulator = CarlaSimulator({"rate": 20}, decision_period_us=100000, settle_iters=0)
    frames = carla_dataset.frames("scene")
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={"front": SensorTiming(50000)}))
        source = runtime.generate_frame_data(carla_dataset.scenario_start("scene", frames[0]), frames[:4])
    start = carla_dataset.scenario_start("scene", frames[2])
    initialization = Initialization(source.at(frames[2]),
                                     source.resolved_scenario, source.provenance)
    return simulator, start, initialization, source.at(frames[3])


def test_native_tick_schedule(two_tick_start):
    simulator, start, initialization, _ = two_tick_start
    assert simulator.decision_ticks == 2
    with simulator.create_runtime() as runtime:
        runtime.reset(start, initialization)
        before = runtime.state()
        with pytest.raises(ValueError, match="base ticks"):
            runtime.advance_interval(runtime.clock.time_us + 100000, Control(),
                                     report_times_us=(runtime.clock.time_us + 25000,))
        assert runtime.state().frame == before.frame
        assert "pipeline" not in runtime.replay_scenario["ego"]
        np.testing.assert_array_equal(runtime.state().ego.pose, before.ego.pose)
    with pytest.raises(ValueError, match="client.rate.*integer-microsecond"):
        CarlaSimulator({"rate": 60})


def test_recorded_tick_history(two_tick_start):
    simulator, start, initialization, _ = two_tick_start
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={"front": SensorTiming(50000)}),
                                 history=HistoryRequirement(duration_us=100000, samples=2))
        runtime.reset(start, initialization)
        pose_only = runtime.observe_recorded_at(-100000, ()).observation
        assert pose_only.sensor_data == {} and pose_only.calibration == {}
        assert pose_only.vehicle_state is not None
        pose_only.vehicle_state.position.x[0] = 999.
        assert runtime.observe_recorded_at(-100000, ()).observation.vehicle_state.position.x[0] != 999.
        acquisition = runtime.observe_recorded_at(-50000, ("front_camera",))
        assert acquisition.observation.sensor_data.keys() == {"front"}
        assert acquisition.sensor_geometry["front"].sensor == "front"
        with pytest.raises(ValueError, match="insufficient replayed sensor history"):
            runtime.observe_recorded_at(-50000, ("front",))


def test_force_gt_replays_recipe(two_tick_start):
    simulator, start, initialization, recorded_end = two_tick_start
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(force_gt_duration_us=100000))
        runtime.reset(start, initialization)
        origin = runtime.clock.time_us
        runtime.advance_interval(origin + 100000, None, force_gt=True)
        runtime.capture_at(origin + 100000)
        np.testing.assert_allclose(runtime.state().ego.pose, recorded_end.ego.pose)


def test_intermediate_tick_capture(two_tick_start):
    simulator, start, initialization, _ = two_tick_start
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration())
        runtime.reset(start, initialization)
        origin = runtime.clock.time_us
        before = runtime.state()
        runtime.advance_interval(origin + 100000, Control(throttle=.3),
                                 report_times_us=(origin + 50000,))
        middle = runtime.capture_at(origin + 50000, ())
        expected = runtime.state()
        assert middle.sensor_data == {} and middle.calibration == {}
        assert middle.t == pytest.approx(before.t + .05)
        assert middle.frame == before.frame + 1
        assert middle.t == expected.t and middle.frame == expected.frame
        assert middle.ego_speed == expected.ego_speed
        np.testing.assert_array_equal(middle.vehicle_state.position.x, expected.vehicle_state.position.x)
        runtime.capture_at(origin + 100000)


def test_capture_selection_uses_native_tick_schedule(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["sensors"] = [{"type": "CarlaLidar", "name": "lidar"}]
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    simulator = CarlaSimulator({"rate": 20}, decision_period_us=100000, settle_iters=0)
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={"front": SensorTiming(50000)}))
        runtime.reset(start)
        assert "sensor_tick" not in runtime.replay_scenario["ego"]["sensors"][0]
        origin = runtime.clock.time_us
        runtime.advance_interval(origin + 100000, Control(), report_times_us=(origin + 50000,))
        intermediate = runtime.capture_at(origin + 50000, ("front",))
        assert set(intermediate.sensor_data) == {"front"}
        logical_name = "front_camera"
        assert runtime.canonical_sensor_id(logical_name) == "front"
        assert runtime.sensor_geometry(logical_name).sensor == "front"
        assert runtime.sensor_view(logical_name).sensor == "front"
        assert runtime.capture_at(origin + 50000, (logical_name,)).sensor_data.keys() == {"front"}
        with pytest.raises(ValueError, match="base ticks"):
            runtime.capture_at(origin + 75000, ("front",))
        with pytest.raises(ValueError, match="not due.*lidar"):
            runtime.capture_at(origin + 50000, ("lidar",))
        endpoint = runtime.capture_at(origin + 100000, ("front", "lidar"))
        assert set(endpoint.sensor_data) == {"front", "lidar"}
        assert endpoint.frame == intermediate.frame + 1


def test_capture_without_due_sensors(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    simulator = CarlaSimulator({"rate": 20}, decision_period_us=100000, settle_iters=0)
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={"front": SensorTiming(150000)}))
        runtime.reset(start)
        origin = runtime.clock.time_us
        runtime.advance_interval(origin + 100000, Control())
        with pytest.raises(ValueError, match="not due"):
            runtime.capture_at(origin + 50000, ("front",))
        assert runtime.clock.time_us == origin
        assert runtime.capture_at(origin + 100000, ()).sensor_data == {}
        runtime.advance_interval(origin + 200000, Control())
        acquisition = runtime.capture_at(origin + 150000, ("front_camera",))
        assert acquisition.sensor_data.keys() == {"front"}
        assert acquisition.t == runtime.state().t
        assert runtime.capture_at(origin + 200000, ()).sensor_data == {}
