"""CARLA capture alignment, native settings, deterministic preparation and actor control."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.carla.sensors import SensorAlignmentError
from avsectester.closed_loop.simulators.base import Initialization
from avsectester.closed_loop.timing import HistoryRequirement, SensorTiming, TimingConfiguration
from avsectester.datasets.carla import CarlaDataset, PrescribedMotion
from avsectester.data_types import ActorState, Control, Observation
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
    with CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.,
                        ground_settle_s=0.).create_runtime() as runtime:
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
            "road_friction": {"friction_force_multiplier": .7},
            "map_layers": {"unload": ["Buildings"]},
            "lighting": {"vehicle_lights": {"ego": ["Position", "LowBeam"]}}})
        native = runtime.native
        assert native["world"].weather.precipitation == 60.
        import carla
        assert native["ego"].actor.light_state == (carla.VehicleLightState.Position
                                                   | carla.VehicleLightState.LowBeam)
        assert native["world"].layers == [("unload", "Buildings")]
        assert native["ego"].actor.get_physics_control().wheels[0].friction_force_multiplier == .7
        with pytest.raises(ValueError, match="road_friction supports only"):
            runtime.reset(start, world={"road_friction": {"tire_friction": .7}})
    scenario = deepcopy(dataset.native("scene").scenario)
    scenario["client"]["map_name"] = "Town10HD_Opt"
    unlayered = CarlaDataset({"scene": replace(dataset.native("scene"), scenario=scenario)})
    start = unlayered.scenario_start("scene", unlayered.frames("scene")[0])
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        with pytest.raises(ValueError, match="unsupported.*Town10HD_Opt"):
            runtime.reset(start, world={"map_layers": {"unload": ["Buildings"]}})


@pytest.mark.parametrize("module_name", ["carla", "carla.libcarla"])
def test_copy_native_sdk_capture(module_name):
    def cannot_copy(self, memo):
        raise TypeError("Native SDK captures cannot be deep-copied")

    capture_type = type("NativeCapture", (), {
        "__module__": module_name, "__deepcopy__": cannot_copy, "raw_data": b"capture"})
    capture = capture_type()
    wrapper = SimpleNamespace(data=capture, edits=[0])
    observation = Observation(0., 0, {"native": capture, "wrapped": wrapper})
    with CarlaSimulator({}).create_runtime() as runtime:
        copied = runtime.copy_observation(observation)
    assert copied.sensor_data["native"] is capture
    assert copied.sensor_data["wrapped"].data is capture
    copied.sensor_data["wrapped"].edits.append(1)
    assert wrapper.edits == [0]


def test_camera_warmup(monkeypatch, carla_dataset):
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    results = []
    for warmup, expected_ticks in ((0., 0), (.2, 2)):
        CarlaNativeBoundary().install(monkeypatch, None)
        simulator = CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=warmup)
        assert simulator.camera_warmup_ticks == expected_ticks
        with simulator.create_runtime() as runtime:
            runtime.reset(start)
            report = runtime.initialization_report
            assert report["deployment_ticks"] == 0
            assert report["actual"]["scenario_t0"] == pytest.approx(runtime.observe().t)
            assert report["actual"]["actual_scenario_time"] == 0.
            results.append(runtime.clock)
    assert results[1].frame == results[0].frame + 2
    assert results[1].t == pytest.approx(results[0].t + .2)

    CarlaNativeBoundary(include_rgb=False).install(monkeypatch, None)
    with CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=.2).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.camera_sensors == set()
        assert runtime.clock.frame == results[0].frame
    with pytest.raises(ValueError, match="camera_warmup_s"):
        CarlaSimulator({"rate": 10}, camera_warmup_s=-1.)
    with pytest.raises(ValueError, match="whole number"):
        CarlaSimulator({"rate": 10}, camera_warmup_s=.15)


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


def test_cached_scenarios_across_maps(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = carla_dataset.native("scene")
    cached = []
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.)
    with simulator.create_runtime() as runtime:
        for name in ("Town03_Opt", "Town10HD_Opt"):
            scenario = deepcopy(recipe.scenario)
            scenario["client"]["map_name"] = name
            dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
            start = dataset.scenario_start("scene", dataset.frames("scene")[0])
            source = runtime.generate_frame_data(start, (start.frame,))
            resolved = deepcopy(source.resolved_scenario)
            assert resolved["client"]["map_name"] == name
            # Saved native names may also be asset-relative.
            resolved["client"]["map_name"] = f"Carla/Maps/{name}"
            cached.append((start, Initialization(source.at(start.frame), resolved, source.provenance)))
        for start, initialization in reversed(cached):
            runtime.reset(start, initialization)
            assert runtime.replay_scenario["client"]["map_name"] in {"Town03_Opt", "Town10HD_Opt"}


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
        runtime.step(Control())
        # CARLA's y axis points to the right of the canonical frame.
        location, velocity = native.get_transform().location, native.get_velocity()
        assert (location.x, location.y, location.z) == pytest.approx((12.2, 3., 1.2))
        assert (velocity.x, velocity.y, velocity.z) == pytest.approx((2., -.5, 0.))
        with pytest.raises(KeyError, match="Unknown CARLA actor 'missing'"):
            runtime.set_actor_state("missing", ActorState(pose))


def test_synchronous_actor_updates(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    with CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.).create_runtime() as runtime:
        runtime.reset(start, world={"road_friction": {"friction_force_multiplier": .7}})
        native = runtime.native
        first = len(native["world"].command_events)
        runtime.step(Control(throttle=.5))
        events = native["world"].command_events
        assert [event[0] for event in events[first:]] == ["batch", "tick"]
        assert all(event[1] and event[2] is False for event in events if event[0] == "batch")
        assert native["world"].direct_actor_updates == []
        assert native["ego"].actor.get_control().throttle == .5
        frame = runtime.clock.frame
        native["client"].command_error = "actor rejected command"
        with pytest.raises(RuntimeError, match="actor rejected command"):
            runtime.step(Control(throttle=.3))
        assert native["world"].get_snapshot().frame == frame


def test_ground_settling(monkeypatch, carla_dataset):
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    CarlaNativeBoundary(ground_height=-.3, ground_offset=(.4, -.3, 7.)).install(monkeypatch, None)
    results = []
    for duration in (0., .2):
        simulator = CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.,
                                   ground_settle_s=duration)
        with simulator.create_runtime() as runtime:
            runtime.reset(start)
            results.append((runtime.clock.frame, runtime.state().ego.pose.copy(),
                            runtime.native["ego"].actor.enable_gravity))
            assert simulator.identity["ground_settle_s"] == duration
            born = runtime.replay_scenario["ego"]["spawn_transform"]["location"]
            assert born["x"] == 0. and born["y"] == 0.
    assert results[1][0] == results[0][0] + 2
    np.testing.assert_allclose(results[0][1][:2, 3], (0., 0.))
    np.testing.assert_allclose(results[1][1][:3, 3], (.4, .3, -.3))
    pose = results[1][1]
    assert np.rad2deg(np.arctan2(pose[1, 0], pose[0, 0])) == pytest.approx(-7.)
    assert results[0][2] is False and results[1][2] is True
    with pytest.raises(ValueError, match="whole number"):
        CarlaSimulator({"rate": 10}, ground_settle_s=.15)


def test_realtime_pacing(monkeypatch, carla_dataset):
    class Clock:
        def __init__(self):
            self.now, self.waits = 0., []

        def monotonic(self):
            return self.now

        def sleep(self, duration):
            self.waits.append(duration)
            self.now += duration

    clock = Clock()
    monkeypatch.setattr("avsectester.closed_loop.simulators.carla.simulator.time", clock)
    CarlaNativeBoundary().install(monkeypatch, None)
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.,
                               external_startup_s=.2, realtime_factor=2.)
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        assert clock.waits == []
        assert "realtime" not in runtime.initialization_report
        assert "realtime" not in runtime.termination_details
        runtime.reset(start, closed_loop_run=True)
        stats = runtime.initialization_report["realtime"]
        assert stats["ticks"] == 4
        assert stats["elapsed_wall_s"] == pytest.approx(.2)
        assert stats["actual_realtime_factor"] == pytest.approx(2.)
        assert stats["late_ticks"] == 0 and stats["max_lateness_s"] == 0.
        np.testing.assert_allclose(clock.waits, [.05, .05, .05, .05])
        clock.now += .1
        runtime.step(Control())
        assert runtime.initialization_report["realtime"]["late_ticks"] == 0
        assert runtime.initialization_report["realtime"]["ticks"] == 4
        final = runtime.termination_details["realtime"]
        assert final["late_ticks"] == 1 and final["ticks"] == 5
        assert final["max_lateness_s"] == pytest.approx(.05)
    assert runtime.termination_details["realtime"] == final
    with pytest.raises(ValueError, match="positive and finite"):
        CarlaSimulator({}, realtime_factor=0.)


def test_external_ego_control(monkeypatch, carla_dataset, make_object_in_front):
    clock = SimpleNamespace(monotonic=lambda: 0., sleep=lambda duration: None)
    monkeypatch.setattr("avsectester.closed_loop.simulators.carla.simulator.time", clock)
    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["blueprint_attributes"] = {"role_name": "ego", "ros2_ackermann_control": True}
    trajectory = []
    for index in range(recipe.decision_steps + 1):
        pose = np.eye(4)
        pose[0, 3] = index * .1
        trajectory.append(ActorState(pose, (1., 0., 0.)))
    datasets = (
        CarlaDataset({"scene": replace(recipe, scenario=scenario)}),
        CarlaDataset({"scene": replace(recipe, scenario=scenario,
            motion={"ego": PrescribedMotion(trajectory=tuple(trajectory))})}),
    )
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.,
                               ego_control="ros2", realtime_factor=1.)
    assert not simulator.can_start_runs_after_origin
    assert CarlaSimulator({}).can_start_runs_after_origin
    initializations = []
    for dataset in datasets:
        frames = dataset.frames("scene")
        start = dataset.scenario_start("scene", frames[0])
        with simulator.create_runtime() as runtime:
            source = runtime.generate_frame_data(start, frames[:2])
            initialization = Initialization(source.at(frames[0]), source.resolved_scenario, source.provenance)
            initializations.append(initialization)
            runtime.reset(start, initialization)
            native = runtime.native
            assert runtime.command_modes == frozenset()
            assert runtime.can_insert_objects is True
            asset, transform = make_object_in_front(runtime.sensor_view("front"))
            runtime.insert_object("offline", asset, transform)
            runtime.remove_object("offline")
            assert native["ego"].actor.attributes["role_name"] == "ego"
            runtime.reset(start, initialization, closed_loop_run=True)
            native = runtime.native
            assert not runtime.can_insert_objects
            for edit in (lambda: runtime.insert_object("framework_only", asset, transform),
                         lambda: runtime.remove_object("framework_only")):
                with pytest.raises(NotImplementedError, match="Framework-composited objects do not reach"):
                    edit()
            first = len(native["world"].command_events)
            runtime.step(None)
            queued = [command for event in native["world"].command_events[first:] if event[0] == "batch"
                      for command in event[1]]
            assert all(command.actor_id != native["ego"].actor.id for command in queued)
            assert runtime.decision_telemetry()["ego_control"]["throttle"] == 0.
    frames = datasets[0].frames("scene")
    start = datasets[0].scenario_start("scene", frames[0])
    initialization = initializations[0]
    CarlaNativeBoundary(external_ego_throttle=.5).install(monkeypatch, None)
    with simulator.create_runtime() as runtime:
        with pytest.raises(RuntimeError, match="external client controlled the ego during CARLA replay"):
            runtime.generate_frame_data(start, frames[:2])
        runtime.start_replay(start, initialization)
        assert "front" in runtime.observe().sensor_data
        with pytest.raises(RuntimeError, match="external client controlled the ego during CARLA replay"):
            runtime.advance_replay(1)
        with pytest.raises(RuntimeError, match="External controls before the CARLA run changed"):
            runtime.reset(start, initialization, closed_loop_run=True)
    with pytest.raises(ValueError, match="requires realtime_factor"):
        CarlaSimulator({}, ego_control="ros2")
    with pytest.raises(ValueError, match="role_name"):
        simulator.validate_start(carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0]))
    with pytest.raises(ValueError, match="exclusively controls the ego"):
        simulator.validate_timing(TimingConfiguration(force_gt_duration_us=100000))


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
