"""CARLA world lifecycle, native settings, sensor timing and ROS message boundaries."""

from copy import deepcopy
from dataclasses import replace
import gc
from threading import Event
from time import time_ns
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack.runtime import AttackRuntime, SensorAttackAction
from avsectester.closed_loop.avstacks.implemented.external_control import ExternalAVStack
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.ros_interception.declaration import InterceptedTopic, InterceptionDeclaration
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.carla.sensors import SensorAlignmentError
from avsectester.closed_loop.simulators.base import Initialization
from avsectester.closed_loop.timing import HistoryRequirement, SensorTiming, TimingConfiguration
from avsectester.datasets.carla import CarlaDataset, PrescribedMotion
from avsectester.data_types import ActorState, Control, Observation
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from tests.fakes import (
    CarlaNativeBoundary, ConstantAVStack, ControllableClock, MemoryDrivingSystem,
    RelayMemoryDrivingSystem,
)


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
    CarlaNativeBoundary().install(monkeypatch)
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
    CarlaNativeBoundary(frame_offset=10).install(monkeypatch)
    with CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.,
                        ground_settle_s=0.).create_runtime() as runtime:
        with pytest.raises(SensorAlignmentError, match="ahead"):
            runtime.reset(start)


def test_traffic_light_state(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
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
    CarlaNativeBoundary().install(monkeypatch)
    dataset = carla_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    client = {"rate": 10, "substepping": False, "max_substep_delta_time": .005, "max_substeps": 12}
    simulator = CarlaSimulator(client, settle_iters=0)
    assert all(simulator.identity["client"][name] == value for name, value in client.items())
    with simulator.create_runtime() as runtime:
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
        assert runtime.initialization_report["world_settings"] == {
            "synchronous_mode": True, "fixed_delta_seconds": .1,
            "substepping": False, "max_substep_delta_time": .005, "max_substeps": 12,
        }
        with pytest.raises(ValueError, match="road_friction supports only"):
            runtime.reset(start, world={"road_friction": {"tire_friction": .7}})
    scenario = deepcopy(dataset.native("scene").scenario)
    scenario["client"]["map_name"] = "Town10HD_Opt"
    unlayered = CarlaDataset({"scene": replace(dataset.native("scene"), scenario=scenario)})
    start = unlayered.scenario_start("scene", unlayered.frames("scene")[0])
    with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime() as runtime:
        with pytest.raises(ValueError, match="unsupported.*Town10HD_Opt"):
            runtime.reset(start, world={"map_layers": {"unload": ["Buildings"]}})
    CarlaNativeBoundary(world_settings_overrides={"substepping": True}).install(monkeypatch)
    with simulator.create_runtime() as runtime, pytest.raises(RuntimeError, match="substepping"):
        runtime.reset(start)
    with pytest.raises(ValueError, match="substepping must be a boolean"):
        CarlaSimulator({"substepping": 0})
    with pytest.raises(ValueError, match="max_substep_delta_time must be a positive number"):
        CarlaSimulator({"max_substep_delta_time": float("inf")})
    with pytest.raises(ValueError, match="max_substeps must be a positive integer"):
        CarlaSimulator({"max_substeps": 1.5})


def test_copy_native_sdk_capture():
    def cannot_copy(self, memo):
        raise TypeError("Native SDK captures cannot be deep-copied")

    capture_type = type("NativeCapture", (), {
        "__module__": "carla.libcarla", "__deepcopy__": cannot_copy, "raw_data": b"capture"})
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
        CarlaNativeBoundary().install(monkeypatch)
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

    CarlaNativeBoundary(include_rgb=False).install(monkeypatch)
    with CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=.2).create_runtime() as runtime:
        runtime.reset(start)
        assert runtime.camera_sensors == set()
        assert runtime.clock.frame == results[0].frame
    with pytest.raises(ValueError, match="camera_warmup_s"):
        CarlaSimulator({"rate": 10}, camera_warmup_s=-1.)
    with pytest.raises(ValueError, match="whole number"):
        CarlaSimulator({"rate": 10}, camera_warmup_s=.15)


def test_scene_choices_follow_the_run_seed(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"].update(spawn="random", vehicle="random")
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    origins = []
    for seed in (17, 17, 23, 29, 37):
        with CarlaSimulator({"rate": 10}, settle_iters=0).create_runtime(seed=seed) as runtime:
            runtime.reset(start)
            origins.append((runtime.native["ego"].actor.type_id, runtime.state().ego.pose.copy()))
    assert origins[0][0] == origins[1][0]
    np.testing.assert_array_equal(origins[0][1], origins[1][1])
    vehicle, pose = origins[0]
    assert any(kind != vehicle or not np.array_equal(transform, pose)
               for kind, transform in origins[2:])


def test_cached_scenarios_across_maps(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
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
            cached.append((name, start, Initialization(source.at(start.frame), resolved, source.provenance)))
        for expected_map, start, initialization in reversed(cached):
            runtime.reset(start, initialization)
            assert runtime.replay_scenario["client"]["map_name"] == expected_map
            assert runtime.native["world"].get_map().name == expected_map


def test_control_actor(monkeypatch, carla_dataset):
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

        with pytest.raises(ValueError, match="ego is controlled by the driving system"):
            runtime.control_actor("ego", Control())


def test_set_actor_state(monkeypatch, carla_dataset):
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
    CarlaNativeBoundary().install(monkeypatch)
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
    CarlaNativeBoundary(ground_height=-.3, ground_offset=(.4, -.3, 7.)).install(monkeypatch)
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


@pytest.fixture
def clock(monkeypatch):
    clock = ControllableClock()
    clock.install(monkeypatch, CarlaSimulator)
    return clock


def test_realtime_pacing(monkeypatch, carla_dataset, clock):
    CarlaNativeBoundary(tick_callback=lambda: clock.advance(.01),
                        batch_callback=lambda: clock.advance(.005)).install(monkeypatch)
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    simulator = CarlaSimulator({"rate": 20}, decision_period_us=100000, settle_iters=0,
                               camera_warmup_s=0., external_startup_s=.1, realtime_factor=2.)
    delays = iter((.037, 0., .086))
    delay_hook = Hook("simulator.step.pre", lambda value, context: clock.advance(next(delays)))
    frozen_before = gc.get_freeze_count()
    callbacks_before = tuple(gc.callbacks)
    with simulator.create_runtime() as runtime:
        runtime.reset(start)
        assert clock.waits == []
        assert gc.get_freeze_count() == frozen_before
        assert tuple(gc.callbacks) == callbacks_before
        assert "realtime" not in runtime.initialization_report
        assert "realtime" not in runtime.termination_details
        assert "pacing" not in runtime.interval_telemetry()
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(),
                                     RunRequest(3, start=start, hooks=HookSet((delay_hook,))))
        stats = runtime.initialization_report["realtime"]
        assert stats["gc_preexisting_frozen_objects"] == frozen_before
        assert stats["gc_frozen_objects"] > frozen_before
        assert stats["ticks"] == 4
        assert stats["elapsed_wall_s"] == pytest.approx(.09)
        assert stats["actual_realtime_factor"] == pytest.approx(.2 / .09)
        assert stats["behind_ticks"] == 0 and stats["max_behind_schedule_s"] == 0.
        assert stats["resynchronizations"] == 0
        world_tick = {"mean": .01, "p95": .01, "max": .01}
        command = {"mean": .005, "p95": .005, "max": .005}
        assert stats["world_tick_s"] == pytest.approx(world_tick)
        assert stats["command_s"] == pytest.approx(command)
        assert stats["framework_s"] == {"mean": 0., "p95": 0., "max": 0.}
        assert set(stats["gc_s"]) == {"mean", "p95", "max"}
        assert [len(step.telemetry["pacing"]) for step in log.steps] == [2, 2, 2]
        samples = [sample for step in log.steps for sample in step.telemetry["pacing"]]
        expected = [
            (.037, 0., .017, False),
            (0., 0., .007, False),
            (0., 0., 0., False),
            (0., .003, 0., False),
            (.086, 0., .066, True),
            (0., 0., 0., False),
        ]
        for sample, (framework, waited, behind, resynchronized) in zip(samples, expected):
            assert sample["resynchronized"] is resynchronized
            timing = {
                "framework_s": framework, "sleep_s": waited, "behind_schedule_s": behind,
                "resynchronized": resynchronized, "command_s": .005, "world_tick_s": .01,
            }
            assert {key: sample[key] for key in timing} == pytest.approx(timing)
            assert sample["gc_s"] >= 0.
            assert type(sample["full_collections"]) is int and sample["full_collections"] >= 0
        final = runtime.termination_details["realtime"]
        assert final["ticks"] == 10 and final["behind_ticks"] == 3
        assert final["resynchronizations"] == 1
        assert final["max_behind_schedule_s"] == pytest.approx(.066)
        assert final["actual_realtime_factor"] == pytest.approx(.5 / .306)
        assert final["command_s"] == pytest.approx(command)
        post = final["post_origin"]
        assert post["ticks"] == 6 and post["behind_ticks"] == 3
        assert post["resynchronizations"] == 1
        assert post["max_behind_schedule_s"] == pytest.approx(.066)
        assert post["elapsed_wall_s"] == pytest.approx(.216)
        assert post["actual_realtime_factor"] == pytest.approx(.3 / .216)
        assert post["world_tick_s"] == pytest.approx(world_tick)
        assert post["command_s"] == pytest.approx(command)
        assert post["framework_s"] == pytest.approx({"mean": .0205, "p95": .07375, "max": .086})
        gc_durations = [sample["gc_s"] for sample in samples]
        assert post["gc_s"] == pytest.approx({
            "mean": np.mean(gc_durations), "p95": np.percentile(gc_durations, 95),
            "max": max(gc_durations),
        })
        assert final["gc_s"]["max"] == max(stats["gc_s"]["max"], post["gc_s"]["max"])
        assert post["full_collections"] == sum(sample["full_collections"] for sample in samples)
        assert final["full_collections"] == stats["full_collections"] + post["full_collections"]
        termination = next(event for event in log.events if event.kind == "termination")
        assert termination.data["realtime"] == final
    assert runtime.termination_details["realtime"] == final
    assert gc.get_freeze_count() == 0
    assert tuple(gc.callbacks) == callbacks_before

    # Closing also releases objects that were frozen before the run.
    gc.freeze()
    try:
        preexisting = gc.get_freeze_count()
        with simulator.create_runtime() as runtime:
            runtime.reset(start, closed_loop_run=True)
            realtime = runtime.initialization_report["realtime"]
            # Frozen objects that are released leave the permanent generation.
            assert 0 < realtime["gc_preexisting_frozen_objects"] <= preexisting
            assert realtime["gc_frozen_objects"] >= gc.get_freeze_count() > 0
        assert gc.get_freeze_count() == 0
        assert tuple(gc.callbacks) == callbacks_before
    finally:
        gc.unfreeze()

    with pytest.raises(ValueError, match="positive and finite"):
        CarlaSimulator({}, realtime_factor=0.)


def test_external_ego_control(monkeypatch, carla_dataset, make_object_in_front, clock):
    # The server can retain a disabled TF setting from a previous hosted run.
    CarlaNativeBoundary(publish_tf=False).install(monkeypatch)
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
            assert native["world"].get_publish_tf() is True
            assert runtime.initialization_report["world_settings"]["publish_tf"] is True
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
    CarlaNativeBoundary(external_ego_throttle=.5).install(monkeypatch)
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
def hosted_start(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["blueprint_attributes"] = {"role_name": "ego", "ros2_autoware_control": True}
    scenario["ego_goal"] = {"x": 4., "y": -6., "yaw_deg": -10.}
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    frames = dataset.frames("scene")
    return dataset.scenario_start("scene", frames[0]), frames[:2]


def _hosted_simulator(system):
    return CarlaSimulator({"rate": 10, "connect_ip": "localhost", "connect_port": 24880},
                          ground_settle_s=0., camera_warmup_s=0., ego_control="ros2",
                          realtime_factor=100., external_driving_system=system)


def _imu_rig_start(start, *names):
    recipe = start.payload["recipe"]
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["ros_rig"] = [{
        "name": name, "parent": "ego", "blueprint": "sensor.other.imu", "ros": True,
        "blueprint_attributes": {"ros_topic_name": f"/{name}"},
        "transform": {"location": {"x": 1., "y": 0., "z": 0.},
                      "rotation": {"roll": 0., "pitch": 0., "yaw": 0.}},
    } for name in names]
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    return dataset.scenario_start("scene", dataset.frames("scene")[0])


def test_message_processing_while_carla_tick_is_blocked(hosted_start, tmp_path, monkeypatch):
    start = _imu_rig_start(hosted_start[0], "imu")
    topic = "/imu"
    kind = "sensor_msgs/msg/Imu"
    system = RelayMemoryDrivingSystem(tmp_path / "relay.sock", InterceptionDeclaration(
        (InterceptedTopic(topic, kind, ("/carla",), "imu"),)))
    delivered = Event()
    seen = []

    def tick():
        if system.sessions and system.sessions[-1].engaged_s is not None:
            system.peer.send({"kind": "message", "id": 0, "topic": topic, "type": kind,
                              "fields": {}, "received_ns": time_ns()})
            assert delivered.wait(2.), "Process the sensor message before the CARLA tick returns"

    class InspectImu(SensorAttackAction):
        sensor = "imu"
        active_for_us = 0

        def apply(self, value, view):
            assert view.sensor == self.sensor
            world_from_sensor = view.ego.pose.copy()
            world_from_sensor[:3, 3] += world_from_sensor[:3, 0]
            np.testing.assert_allclose(view.sensor_from_world, np.linalg.inv(world_from_sensor))
            result = replace(value, fields={"sample": 23})
            seen.append(result)
            delivered.set()
            return result

    class InspectImuAttack(AttackRuntime):
        def step(self, context):
            return (InspectImu(),)

    CarlaNativeBoundary(tick_callback=tick).install(monkeypatch)
    request = RunRequest(1, start=start, attacks=(InspectImuAttack(),))
    with _hosted_simulator(system).create_runtime() as runtime:
        ClosedLoopRunner().run(runtime, ExternalAVStack(), request)
        native = runtime.native
        imu = next(actor for actor in native["world"].get_actors() if actor.type_id == "sensor.other.imu")
        assert imu.attributes["ros_topic_name"] == "/avsectester/intercepted/imu"
    assert len(seen) == 1
    reply, = system.replies
    assert reply["fields"] == {"sample": 23}


def test_hosted_driver_lifecycle(hosted_start, monkeypatch):
    start, frames = hosted_start
    system = MemoryDrivingSystem(ready_at_s=.8)
    settings = _hosted_simulator(system)
    frozen_before = gc.get_freeze_count()
    callbacks_before = tuple(gc.callbacks)
    with settings.create_runtime() as runtime:
        source = runtime.generate_frame_data(start, frames)
        initialization = Initialization(source.at(frames[0]), source.resolved_scenario,
                                        source.provenance)
        runtime.start_replay(start, initialization)
        assert not system.sessions
        assert gc.get_freeze_count() == frozen_before
        assert tuple(gc.callbacks) == callbacks_before
        assert runtime.native["world"].get_publish_tf() is False
        runtime.reset(start, initialization, closed_loop_run=True)
        assert runtime.initialization_report["world_settings"]["publish_tf"] is False
        realtime = runtime.initialization_report["realtime"]
        assert realtime["gc_preexisting_frozen_objects"] == frozen_before
        assert realtime["gc_frozen_objects"] >= gc.get_freeze_count() > frozen_before
        session = system.sessions[-1]
        report = runtime.initialization_report["external_driving_system"]
        assert report["ready_s"] >= .8
        assert report["engaged_s"] == runtime.clock.t
        assert report["engaged_s"] > report["ready_s"]
        assert session.context.host == "localhost" and session.context.port == 24880
        assert session.context.ego_goal == {"x": 4., "y": -6., "yaw_deg": -10.}
        np.testing.assert_array_equal(session.pose, runtime.state().ego.pose)
        runtime.step(None)
        assert runtime.decision_telemetry()["external_driving_system"] == {"mode": "AUTONOMOUS"}
    assert runtime.termination_details["external_driving_system"]["stopped"]
    assert gc.get_freeze_count() == 0
    assert tuple(gc.callbacks) == callbacks_before
    assert settings.identity["external_driving_system"] == {
        "type": "MemoryDrivingSystem", "ready_at_s": .8,
    }
    failing_system = MemoryDrivingSystem(startup_error="Driver startup timed out")
    with pytest.raises(TimeoutError, match="Driver startup timed out"):
        with _hosted_simulator(failing_system).create_runtime() as failed:
            failed.reset(start, initialization, closed_loop_run=True)
    assert failed.termination_details["external_driving_system"]["stopped"]
    assert gc.get_freeze_count() == 0
    assert tuple(gc.callbacks) == callbacks_before

    wrong_ego = replace(initialization.expected.ego, velocity=(1., 0., 0.))
    wrong_origin = replace(initialization, expected=replace(initialization.expected, ego=wrong_ego))
    with settings.create_runtime() as failed:
        with pytest.raises(RuntimeError, match="External controls before the CARLA run changed"):
            failed.reset(start, wrong_origin, closed_loop_run=True)
        assert failed.initialization_report["realtime"]["gc_frozen_objects"] > 0
        assert gc.get_freeze_count() > 0
    assert gc.get_freeze_count() == 0
    assert tuple(gc.callbacks) == callbacks_before

    failing_system = MemoryDrivingSystem(run_error="Driver disconnected")
    with pytest.raises(RuntimeError, match="Driver disconnected"):
        with _hosted_simulator(failing_system).create_runtime() as failed:
            failed.reset(start, initialization, closed_loop_run=True)
            failed.step(None)
    assert failed.termination_details["external_driving_system"]["stopped"]
    resolved = deepcopy(initialization.resolved_scenario)
    resolved["client"]["reset_world"] = False
    with _hosted_simulator(system).create_runtime() as unreloaded:
        with pytest.raises(ValueError, match="requires client.reset_world"):
            unreloaded.reset(start, replace(initialization, resolved_scenario=resolved),
                             closed_loop_run=True)
    with pytest.raises(ValueError, match="external_startup_s applies only without"):
        CarlaSimulator({"rate": 10}, ego_control="ros2", realtime_factor=1.,
                       external_startup_s=1., external_driving_system=system)
    CarlaNativeBoundary(world_settings_overrides={"publish_tf": True}).install(monkeypatch)
    with settings.create_runtime() as runtime, pytest.raises(RuntimeError, match="publish_tf"):
        runtime.reset(start)


def test_traffic_light_origin_after_readiness_hold(hosted_start):
    start, frames = hosted_start
    system = MemoryDrivingSystem(ready_at_s=1.)
    with _hosted_simulator(system).create_runtime() as runtime:
        source = runtime.generate_frame_data(start, frames)
        expected = source.at(frames[0])
        reference = Initialization(expected, source.resolved_scenario, source.provenance)
        runtime.reset(start, reference, closed_loop_run=True)
        prepared = expected.provenance["traffic_lights"]
        assert runtime.clock.t >= 1.
        assert runtime.clock.t > expected.provenance["native_timestamp"]
        assert runtime.initialization_report["actual"]["traffic_lights"] == prepared
        altered = {**expected.provenance, "traffic_lights": {
            "signal-1": {**prepared["signal-1"], "elapsed_s": 2.},
        }}
        with pytest.raises(RuntimeError, match="traffic light 'signal-1' phase differs"):
            runtime.reset(start, replace(reference, expected=replace(expected, provenance=altered)),
                          closed_loop_run=True)


def test_ros_rig_uses_native_attachments_and_ticks(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    transform = {"location": {"x": .9, "y": 0., "z": 2.},
                 "rotation": {"pitch": 0., "yaw": 0., "roll": 0.}}
    rig = scenario["ego"]["ros_rig"] = [
        {"name": "base", "parent": "ego", "blueprint": "util.actor.empty",
         "blueprint_attributes": {"ros_name": "sensor_kit_base_link"},
         "transform": transform, "ros": False},
        {"name": "lidar", "parent": "base", "blueprint": "sensor.lidar.ray_cast",
         "blueprint_attributes": {"sensor_tick": .1, "ros_name": "lidar"},
         "transform": transform, "ros": True},
    ]
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    with CarlaSimulator({"rate": 10}).create_runtime() as runtime:
        runtime.reset(start)
        native = runtime.native
        world, ego = native["world"], native["ego"].actor
        base = next(actor for actor in world.get_actors() if actor.type_id == "util.actor.empty")
        lidar = next(actor for actor in world.get_actors() if actor.type_id == "sensor.lidar.ray_cast")
        assert base.attached_to.id == ego.id and lidar.attached_to.id == base.id
        assert base.attributes["ros_name"] == "sensor_kit_base_link" and not base.ros_enabled
        assert lidar.attributes == {"sensor_tick": "0.1", "ros_name": "lidar"}
        assert lidar.ros_enabled and lidar.get_transform().location.z == 2.
        assert runtime.sensors == {"front"} and "lidar" not in runtime.observe().sensor_data
        assert runtime.replay_scenario["ego"]["ros_rig"] == rig
    assert world.destroyed_ids == [lidar.id, base.id, ego.id]


@pytest.fixture
def two_tick_start(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
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
        acquisition = runtime.observe_recorded_at(-50000, ("front",))
        assert acquisition.observation.sensor_data.keys() == {"front"}
        assert acquisition.sensor_geometry["front"].sensor == "front"
        with pytest.raises(ValueError, match="insufficient replayed sensor history"):
            runtime.observe_recorded_at(-50000, ("front",))


def test_force_gt_replays_recipe(two_tick_start):
    simulator, start, initialization, recorded_end = two_tick_start
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(force_gt_duration_us=100000))
        runtime.reset(start, initialization)
        runs = runtime.initialization_report["motion"]["ego"]["control_runs"]
        assert [run["count"] for run in runs] == [4]
        controls = tuple(run["control"] for run in runs for _ in range(run["count"]))
        assert controls == start.payload["recipe"].motion["ego"].controls
        origin = runtime.clock.time_us
        runtime.advance_interval(origin + 100000, None, force_gt=True)
        runtime.capture_at(origin + 100000)
        np.testing.assert_allclose(runtime.state().ego.pose, recorded_end.ego.pose)


def test_capture_at_native_tick(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["sensors"] = [{"type": "CarlaLidar", "name": "lidar"}]
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    simulator = CarlaSimulator({"rate": 20}, decision_period_us=100000, settle_iters=0)
    with simulator.create_runtime() as runtime:
        runtime.configure_timing(TimingConfiguration(sensors={
            "front": SensorTiming(50000), "lidar": SensorTiming(150000),
        }))
        runtime.reset(start)
        before = runtime.state()
        origin = runtime.clock.time_us
        runtime.advance_interval(origin + 100000, Control(), report_times_us=(origin + 50000,))
        with pytest.raises(ValueError, match="base ticks"):
            runtime.capture_at(origin + 75000, ("front",))
        with pytest.raises(ValueError, match="not due.*lidar"):
            runtime.capture_at(origin + 50000, ("lidar",))
        assert runtime.clock.time_us == origin

        middle = runtime.capture_at(origin + 50000, ())
        assert middle.sensor_data == {} and middle.calibration == {}
        assert middle.t == pytest.approx(before.t + .05)
        assert middle.frame == before.frame + 1
        assert middle.ego_speed == runtime.state().ego_speed
        np.testing.assert_array_equal(
            middle.vehicle_state.position.x, runtime.state().vehicle_state.position.x)
        endpoint = runtime.capture_at(origin + 100000, ("front",))
        assert set(endpoint.sensor_data) == {"front"}
        assert endpoint.frame == middle.frame + 1
        runtime.advance_interval(origin + 200000, Control())
        both_due = runtime.capture_at(origin + 150000, ("front", "lidar"))
        assert set(both_due.sensor_data) == {"front", "lidar"}
        assert both_due.t == runtime.state().t
