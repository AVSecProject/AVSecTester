"""Selection and preparation use the same capabilities without borrowing active state."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack.runtime import (
    AttackRuntime, AVStackAttackEffect, SensorAttackEffect, WorldAttackEffect,
)
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.closed_loop.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.datasets.base import Dataset, FrameReference, FrameData, ScenarioStart, Unavailable
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.frame_data import FrameDataSequence
from avsectester.attack import AttackSegment
from avsectester.closed_loop.services import (
    CandidateRequest, ClosedLoopServices, ComponentDescriptor,
    DrivingAdapter, RenderRequest, ScorerDescriptor, SensorRequest,
)


class Data(Dataset):
    def __init__(self, kind="memory"):
        self.kind = kind
        self.refs = tuple(FrameReference("s", i, f"source-{i}", i * 0.1) for i in range(5))

    def scenarios(self):
        return ()

    def frames(self, scenario_id):
        assert scenario_id == "s"
        return self.refs

    def frame_data(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return Unavailable("recipe_only") if self.kind == "CarlaDataset" else FrameData(frame)

    def native(self, scenario_id):
        return {"recipe": "source data"}

    def scenario_start(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return ScenarioStart(self.kind, scenario_id, frame, self.native(scenario_id))


class World(SimulatorRuntime):
    def __init__(self, key=None, factory=None):
        super().__init__(factory if factory is not None else Factory(key))
        self.factory = self.simulator
        self.closed = False
        self.resource_key = key if key is not None else ("memory", id(self))
        self.resets = 0
        self.source = ScenarioStart("memory", "s", FrameReference("s", 0, "source-0", 0), {})

    def _initialize_world(self, source, initialization, environment):
        if source is not None:
            self.source = deepcopy(source)
        self.resets += 1
        self.i = 0

    def _check_initialization(self, initialization):
        pass

    def _observe(self):
        return Observation(self.source.frame.timestamp + self.i * 0.1,
                           self.source.frame.position + self.i,
                           {"front": np.full((2, 2, 3), self.i, dtype=np.uint8)})

    def _advance_native(self, control):
        self.i += 1

    def _state(self):
        observation = self._observe()
        return WorldSnapshot(observation.t, observation.frame)

    def generate_frame_data(self, source, frames, **kwargs):
        if self.factory.materialize is not None:
            return self.factory.materialize(source, frames)
        self.factory.generations += 1
        self.reset(source)
        return FrameDataSequence(tuple(FrameData(f) for f in frames),
                                 {"resolved": True}, {"seed": 7})

    def sensor_view(self, sensor):
        from avsectester.geometry.poses import ActorPose
        from avsectester.closed_loop.simulator import SensorView

        data = self.observe().sensor_data[sensor]
        return SensorView(sensor, data, None, np.eye(4), {}, ActorPose(np.eye(4)), 0.0, self.i,
                          native={"world": self})

    def close(self):
        self.closed = True


class Factory(Simulator):
    """The configured test simulator; it records every runtime it creates."""

    dataset_kinds = ("memory", "CarlaDataset")
    can_render = True

    def __init__(self, key=None):
        self.resource_key = key
        self.worlds = []
        self.seeds = []
        self.generations = 0
        self.materialize = None

    @property
    def identity(self):
        return {"simulator": "test", "version": 1}

    def create_runtime(self, seed=None):
        world = World(self.resource_key, self)
        self.worlds.append(world)
        self.seeds.append(seed)
        return world


class Driver(AVStack):
    def __init__(self):
        self.closed = False
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return Control(throttle=self.calls / 10)

    def close(self):
        self.closed = True


def services(kind="memory", key=None):
    data, adapter, drivers = Data(kind), Factory(key), []
    def factory(seed):
        assert seed == 7
        driver = Driver()
        drivers.append(driver)
        return driver
    return ClosedLoopServices(data, adapter, DrivingAdapter(factory)), adapter, drivers


def candidate(data, **kwargs):
    return CandidateRequest(AttackSegment("s", data.refs[1], data.refs[3]), 7, 2, {},
                            RecordingPolicy(("front",)), **kwargs)


class Fill(SensorAttackEffect):
    sensor = "front"

    def apply(self, data, view):
        assert view.data is not data or (data == view.data).all()
        return np.full_like(data, 123)


class Paint(WorldAttackEffect):
    """A world change applied at setup, visible in the capture, removed after the operation."""

    def apply(self, simulator, resources):
        simulator.i = 5
        resources.callback(setattr, simulator, "painted_removed", True)


def test_sensor_render_and_rollout_have_fresh_worlds_and_no_data_side_effects():
    svc, adapter, drivers = services()
    with svc.session(7) as scope:
        first = scope.sensor_data(SensorRequest("s", svc.dataset.refs[1], ("front",)))
        rendered = scope.render(RenderRequest("s", svc.dataset.refs[1], "front", (Fill(),)))
        logs = [scope.rollout(candidate(svc.dataset)) for _ in range(2)]
    # Each operation runs in a new runtime seeded with the session seed, and closes it.
    assert len(adapter.worlds) == 4 and all(w.closed for w in adapter.worlds)
    assert adapter.seeds == [7, 7, 7, 7]
    assert all(d.closed and d.calls == 2 for d in drivers)
    assert first.observation.sensor_data["front"].max() == 0
    assert rendered.observation.sensor_data["front"].min() == 123
    assert rendered.view.data.max() == 0 and rendered.view.native == {}
    for log in logs:
        assert [s.produced_command.throttle for s in log.steps] == [0.1, 0.2]
        assert log.steps[0].physical_pre.frame == 1
        assert log.steps[-1].physical_post.frame == 3
        assert log.metadata["candidate"]["roles"] == {}
    assert svc.dataset.native("s") == {"recipe": "source data"}


def test_render_applies_world_effects_before_the_capture_and_rejects_component_effects():
    svc, adapter, _ = services()
    with svc.session(7) as scope:
        rendered = scope.render(RenderRequest("s", svc.dataset.refs[1], "front", (Paint(), Fill())))
    assert rendered.view.data.min() == 5  # the world effect is visible in the capture
    assert rendered.observation.sensor_data["front"].min() == 123
    assert adapter.worlds[0].painted_removed and adapter.worlds[0].closed
    with pytest.raises(TypeError, match="world and sensor"):
        RenderRequest("s", svc.dataset.refs[1], "front",
                      (AVStackAttackEffect("command", lambda value, context: value),))
    heavy = Paint()
    heavy.setup_ticks = 1
    with svc.session(7) as scope, pytest.raises(ValueError, match="deployment tick"):
        scope.render(RenderRequest("s", svc.dataset.refs[1], "front", (heavy,)))


def test_candidate_bounds_and_seed_fail_before_allocating_resources():
    svc, adapter, _ = services()
    with svc.session(7) as scope:
        with pytest.raises(ValueError, match="seed"):
            scope.rollout(replace(candidate(svc.dataset), seed=8))
        bad = AttackSegment("s", svc.dataset.refs[1], FrameReference("s", 6, "6", 0.6))
        with pytest.raises(ValueError, match="terminal frame"):
            scope.rollout(CandidateRequest(bad, 7, 5, {}, RecordingPolicy()))
    assert adapter.worlds == []
    with pytest.raises(ValueError, match="decision_steps"):
        replace(candidate(svc.dataset), decision_steps=1)
    with pytest.raises(ValueError, match="fresh"):
        svc.session(7, history="reuse")


def test_preparation_cache_is_reused_and_does_not_change_dataset_behavior():
    svc, adapter, _ = services("CarlaDataset", ("test-server", 1))
    view = svc.scenario("s", seed=7)
    original = svc.dataset.native("s")
    assert isinstance(svc.dataset.frame_data("s", view.frames[1]), Unavailable)
    assert view.frame_data(view.frames[1]).frame == view.frames[1]
    view.frame_data(view.frames[3])
    with svc.session(7) as scope:
        scope.rollout(candidate(svc.dataset))
    assert adapter.generations == 1
    assert svc.dataset.native("s") == original
    assert isinstance(svc.dataset.frame_data("s", view.frames[1]), Unavailable)
    with svc.session(8) as scope:
        scope.frame_data("s", view.frames[1])
    assert adapter.generations == 2


def test_unclosed_unused_scope_does_not_reserve_a_server():
    svc, adapter, _ = services(key=("test-server", 2))
    unused = svc.session(7)
    with svc.session(7) as scope:
        scope.rollout(candidate(svc.dataset))
    ClosedLoopRunner().run(World(adapter.resource_key), Driver(), RunRequest(1))
    unused.close()


def test_active_runner_endpoint_is_not_available_to_selection():
    svc, adapter, _ = services(key=("test-server", 3))
    scope = svc.session(7)
    class SelectingDriver(Driver):
        def __call__(self, obs):
            with pytest.raises(RuntimeError, match="already in use"):
                scope.sensor_data(SensorRequest("s", svc.dataset.refs[0], ("front",)))
            return super().__call__(obs)
    ClosedLoopRunner().run(World(adapter.resource_key), SelectingDriver(), RunRequest(2))
    assert all(world.resets == 0 for world in adapter.worlds)
    scope.close()


def test_overlapping_operations_in_two_scopes_are_rejected():
    svc, adapter, _ = services(key=("test-server", 9))
    a, b = svc.session(7), svc.session(7)
    with a._operation() as resources:
        a._simulator_runtime(resources, svc.dataset.scenario_start("s", svc.dataset.refs[0]))
        with pytest.raises(RuntimeError, match="already in use"):
            b.rollout(candidate(svc.dataset))
    assert b.rollout(candidate(svc.dataset)).status == "completed"
    a.close()
    b.close()


def test_cancelled_candidate_stops_attack_closes_resources_and_releases_scope():
    svc, adapter, drivers = services(key=("test-server", 4))
    attacks, origins, assets = [], [], object()
    class Attack(AttackRuntime):
        def step(self, context):
            raise KeyboardInterrupt()
        def stop(self, context):
            self.stopped = True
    def factory(segment, artifacts):
        assert artifacts is assets
        origins.append(segment)
        runtime = Attack()
        attacks.append(runtime)
        return runtime
    with pytest.raises(KeyboardInterrupt) as error:
        with svc.session(7) as scope:
            scope.rollout(candidate(svc.dataset, attack=SimpleNamespace(create_runtime=factory),
                                    artifacts=assets))
    assert error.value.run_log.status == "cancelled"
    assert attacks[0].stopped and drivers[0].closed and adapter.worlds[0].closed
    assert origins[0].roles == {}  # assetless/hostless attacks use the same create_runtime path
    with svc.session(7) as scope:
        assert scope.rollout(candidate(svc.dataset)).status == "completed"


def test_attack_runtime_is_fresh_for_each_candidate_and_logged():
    svc, _, _ = services()
    runtimes = []
    class Attack(AttackRuntime):
        def __init__(self):
            self.count = 0
        def step(self, context):
            self.count += 1
            return (AVStackAttackEffect("command", lambda value, context: Control(brake=1)),)
    def factory(*args):
        runtime = Attack()
        runtimes.append(runtime)
        return runtime
    request = candidate(svc.dataset, attack=SimpleNamespace(create_runtime=factory))
    with svc.session(7) as scope:
        logs = [scope.rollout(request), scope.rollout(request)]
    assert len(runtimes) == 2 and all(r.count == 2 for r in runtimes)
    assert all(log.steps[0].executed_command.brake == 1 for log in logs)
    assert all(log.steps[0].effects for log in logs)


def test_reused_factory_object_is_rejected_without_closing_an_active_borrower():
    svc, _, _ = services()
    shared = Driver()
    svc.driving_adapter = DrivingAdapter(lambda seed: shared)
    with svc.session(7) as scope:
        scope.rollout(candidate(svc.dataset))
        shared.closed = False
        with pytest.raises(RuntimeError, match="reused"):
            scope.rollout(candidate(svc.dataset))
    assert not shared.closed


def test_create_runtime_returning_a_used_runtime_is_rejected():
    svc, adapter, _ = services()
    cached = World(factory=adapter)
    adapter.create_runtime = lambda seed=None: cached
    request = SensorRequest("s", svc.dataset.refs[1], ("front",))
    with svc.session(7) as scope:
        scope.sensor_data(request)
        with pytest.raises(RuntimeError, match="reused"):
            scope.sensor_data(request)


def test_components_keep_fresh_history_and_scorers_reuse_models_with_gradient_checks():
    svc, _, _ = services()
    builds = []
    def factory(seed):
        driver = Driver()
        builds.append(driver)
        return driver
    desc = ComponentDescriptor("counts", {"x": "int"}, {"y": "int"}, {"y": "int"})
    score = ScorerDescriptor("distance", {"x": "float"}, {"y": "float"}, {"y": "float"})
    svc.driving_adapter = DrivingAdapter(factory,
        components=((desc, lambda avstack, value: (avstack.calls, value)),),
        scorers=((score, lambda avstack, value, target: abs(value - target)),))
    with svc.session(7) as scope:
        assert scope.components() == (desc,)
        assert scope.scorers() == (score,)
        with pytest.raises(ValueError, match="gradients"):
            scope.score("distance", 2, 5, require_grad=True)
        assert not builds
        assert scope.evaluate("counts", 8).value == (0, 8)
        assert scope.evaluate("counts", 9).value == (0, 9)
        assert scope.score("distance", 2, 5).value == 3
        assert scope.score("distance", 4, 5).value == 1
    assert len(builds) == 3 and all(d.closed for d in builds)
    with pytest.raises(RuntimeError, match="closed"):
        scope.components()


def test_differentiable_score_preserves_graph_and_rejects_detached_result():
    torch = pytest.importorskip("torch")
    svc, _, _ = services()
    descriptor = ScorerDescriptor("square", {"x": "tensor"}, {"score": "tensor"},
                                  {"score": "tensor"}, ("x",))
    svc.driving_adapter = DrivingAdapter(lambda seed: Driver(), scorers=((
        descriptor, lambda avstack, value, target: (value - target).square().sum()),))
    x = torch.tensor([3.], requires_grad=True)
    with svc.session(7) as scope:
        result = scope.score("square", x, 1, require_grad=True)
        result.value.backward()
        assert x.grad.item() == 4
        with pytest.raises(ValueError, match="detached"):
            scope.score("square", x.detach(), 1, require_grad=True)


def test_failed_reference_generation_is_not_cached_and_resources_close():
    svc, adapter, _ = services("CarlaDataset")
    def fail(*args):
        raise ValueError("reference unavailable")
    adapter.materialize = fail
    with svc.session(7) as scope:
        for _ in range(2):
            with pytest.raises(ValueError, match="reference unavailable"):
                scope.frame_data("s", svc.dataset.refs[1])
    assert len(adapter.worlds) == 2 and all(world.closed for world in adapter.worlds)


def test_carla_runtime_uses_actual_connection_fields_and_detached_recipe():
    from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
    from avsectester.closed_loop.simulators.carla import CarlaSimulator

    recipe = CarlaRecipe({"client": {"map_name": "Town03"}, "ego": {}}, 1, .05, {"ego": PrescribedMotion((Control(),))})
    data = CarlaDataset({"s": recipe})
    source = data.scenario_start("s", data.frames("s")[0])
    config = {"connect_ip": "127.0.0.1", "connect_port": 2300}
    simulator_runtime = CarlaSimulator(config).create_runtime()
    assert simulator_runtime.resource_key == ("carla", "localhost", 2300)
    assert source.payload["recipe"].scenario["client"] == {"map_name": "Town03"}
    assert "weather" not in data.native("s").scenario
    with pytest.raises(ValueError, match="does not support world settings"):
        simulator_runtime.reset(source, world={"teleport": True})


def test_owned_resource_cleanup_failure_is_recorded_before_log_completion(tmp_path):
    svc, adapter, _ = services()
    class BadCloseDriver(Driver):
        def close(self):
            super().close()
            raise RuntimeError("driver resource release failed")
    svc.driving_adapter = DrivingAdapter(lambda seed: BadCloseDriver())
    with svc.session(7) as scope:
        with pytest.raises(RuntimeError, match="resource release failed") as error:
            scope.rollout(candidate(svc.dataset, output_path=tmp_path / "log"))
    from avsectester.runlog import RunLog
    saved = RunLog.read(tmp_path / "log")
    assert saved.status == error.value.run_log.status == "failed"
    assert len(saved.steps) == 2 and all(world.closed for world in adapter.worlds)
    failure = next(event for event in saved.events if event.kind == "failure")
    assert failure.data["message"] == "driver resource release failed"


def test_reference_failure_is_saved_before_any_driving_decision(tmp_path):
    from avsectester.runlog import RunLog
    svc, adapter, drivers = services("CarlaDataset")
    report = {"failures": ["missing aligned capture: ['front']"]}

    class ReferenceFailure(ValueError):
        def __init__(self):
            self.report = report
            super().__init__("reference preparation failed")

    def fail(*args):
        raise ReferenceFailure()
    adapter.materialize = fail
    with svc.session(7) as scope:
        with pytest.raises(ReferenceFailure):
            scope.rollout(candidate(svc.dataset, output_path=tmp_path / "log"))
    saved = RunLog.read(tmp_path / "log")
    assert saved.status == "failed" and saved.steps == [] and drivers == []
    assert saved.events[0].data["initialization_report"] == report


def test_simulator_runtime_closes_before_its_server_reservation_is_released(monkeypatch):
    from avsectester.closed_loop import resources

    key = ("carla", "localhost", 2000)
    svc, _, _ = services(key=key)
    held_during_close = []
    original_close = World.close

    def close(self):
        held_during_close.append(key in resources._owners)
        original_close(self)

    monkeypatch.setattr(World, "close", close)
    with svc.session(7) as scope:
        scope.sensor_data(SensorRequest("s", svc.dataset.refs[1], ("front",)))
    assert held_during_close == [True]
    assert key not in resources._owners


def test_forward_sensor_scan_reuses_runtime_and_restarts_for_world_operations():
    svc, simulator, drivers = services(key=("scan", 1))

    modes = []
    class ScannedWorld(World):
        def _check_initialization(self, initialization):
            modes.append(self._generating_frame_data)

        def _replay_tick(self):
            self.i += 1

    simulator.create_runtime = lambda seed=None: make(seed)
    def make(seed):
        world = ScannedWorld(simulator.resource_key, simulator)
        simulator.worlds.append(world)
        return world

    with svc.session(7) as scope:
        frames = svc.dataset.refs
        first = scope.sensor_data(SensorRequest("s", frames[0], ("front",)))
        second = scope.sensor_data(SensorRequest("s", frames[2], ("front",)))
        same = scope.sensor_data(SensorRequest("s", frames[2], ("front",)))
        assert len(simulator.worlds) == 1
        assert modes == [True]  # Scans must retain prescribed ego motion.
        assert not simulator.worlds[0].closed
        assert second.observation.frame == same.observation.frame == 2
        assert first.observation.sensor_data["front"].max() == 0
        assert second.observation.sensor_data["front"].max() == 2
        assert not drivers  # A scan never constructs a driving system.
        scope.render(RenderRequest("s", frames[2], "front"))
        assert len(simulator.worlds) == 2 and all(w.closed for w in simulator.worlds)
        assert modes == [True, False]  # The independent render uses an ordinary reset.
        scope.sensor_data(SensorRequest("s", frames[2], ("front",)))
        scope.sensor_data(SensorRequest("s", frames[1], ("front",)))
        assert len(simulator.worlds) == 4 and simulator.worlds[2].closed
    assert all(w.closed for w in simulator.worlds)


def test_services_close_builder_sessions_and_keep_selection_reasons():
    svc, simulator, _ = services()
    view = svc.scenario("s", seed=7)
    view.record_reason("The requested host is visible")
    assert view.reasons == ("The requested host is visible (count=1)",)
    with pytest.raises(ValueError, match="nonempty"):
        view.record_reason(" ")
    with pytest.raises(RuntimeError, match="builder failed"):
        with svc:
            scope = view.services.session(7)
            scope.sensor_data(SensorRequest("s", svc.dataset.refs[0], ("front",)))
            raise RuntimeError("builder failed")
    assert all(world.closed for world in simulator.worlds)
    with pytest.raises(RuntimeError, match="closed"):
        scope.sensor_data(SensorRequest("s", svc.dataset.refs[1], ("front",)))


def test_sensor_environment_is_identical_offline_and_online_at_absolute_frame_positions(monkeypatch):
    # Identical scene content at each position isolates the noise's frame-index contract.
    def capture(self):
        return Observation(self.source.frame.timestamp + self.i * .1,
                           self.source.frame.position + self.i,
                           {"front": np.full((16, 16, 3), 100, dtype=np.uint8)})

    monkeypatch.setattr(World, "_observe", capture)
    svc, _, _ = services()
    environment = {"sensors": [{"type": "GaussianNoise", "options": {"severity": .3},
                                "camera": "front"}]}

    class SensorRuntime(AttackRuntime):
        def step(self, context):
            return (Fill(),)

    segment = AttackSegment("s", svc.dataset.refs[1], svc.dataset.refs[3])
    with svc, svc.session(7) as session:
        offline = [session.render(RenderRequest("s", frame, "front", (Fill(),), environment))
                   .observation.sensor_data["front"] for frame in svc.dataset.refs[1:3]]
        online = session.rollout(CandidateRequest(
            segment, 7, 2, environment, RecordingPolicy(("front",)),
            attack=SimpleNamespace(create_runtime=lambda segment, artifacts: SensorRuntime()),
        ))
    for expected, record in zip(offline, online.steps):
        np.testing.assert_array_equal(record.model_observation.sensor_data["front"], expected)
    assert not np.array_equal(offline[0], offline[1])


def test_render_sensor_aliases_chain_and_write_the_canonical_output(monkeypatch):
    class AliasWorld(World):
        def _observe(self):
            return Observation(self.i*.1, self.i, {"camera0": np.full((2, 2, 3), 5, np.uint8)})
        def sensor_view(self, sensor):
            assert sensor in {"front", "camera0"}
            from avsectester.closed_loop.simulator import SensorView
            from avsectester.geometry.poses import ActorPose
            return SensorView("camera0", self._capture.sensor_data["camera0"].copy(),
                              None, np.eye(4), {}, ActorPose(np.eye(4)), self.i*.1, self.i)
    seen = []
    class Add(SensorAttackEffect):
        def __init__(self, sensor, increment):
            self.sensor, self.increment = sensor, increment
        def apply(self, data, view):
            seen.append((self.sensor, int(data.min()), int(view.data.min())))
            return data + self.increment
    svc, configured, _ = services()
    def factory(seed=None):
        world = AliasWorld(factory=configured)
        configured.worlds.append(world)
        return world
    monkeypatch.setattr(configured, "create_runtime", factory)
    with svc.session(7) as session:
        result = session.render(RenderRequest("s", svc.dataset.refs[0], "front",
                                (Add("front", 10), Add("camera0", 20))))
    assert seen == [("front", 5, 5), ("camera0", 15, 5)]
    assert set(result.observation.sensor_data) == {"camera0"}
    assert result.observation.sensor_data["camera0"].min() == 35


def test_render_returns_original_detached_view_after_in_place_sensor_effect(monkeypatch):
    class NativeHandle:
        def __deepcopy__(self, memo):
            raise AssertionError("A view's live native resources must never be deep-copied")

    original_view = World.sensor_view
    def sensor_view(self, sensor):
        view = original_view(self, sensor)
        return replace(view, native={**view.native, "sdk_handle": NativeHandle()})
    monkeypatch.setattr(World, "sensor_view", sensor_view)

    class InPlace(SensorAttackEffect):
        sensor = "front"
        def apply(self, data, view):
            assert not np.shares_memory(data, view.data)
            assert view.native["sdk_handle"] is not None
            data[:] = 70
            # Even an effect modifying its delivered view cannot alter the returned
            # evidence of the pre-effect capture or the runtime's cached observation.
            view.data[:] = 99
            return data

    svc, configured, _ = services()
    with svc.session(7) as session:
        result = session.render(RenderRequest("s", svc.dataset.refs[0], "front", (InPlace(),)))
    assert np.all(result.observation.sensor_data["front"] == 70)
    assert np.all(result.view.data == 0)
    assert result.view.native == {}
    assert np.all(configured.worlds[0].observe().sensor_data["front"] == 0)
    assert configured.worlds[0].closed


def test_components_stay_fresh_while_scoring_reuses_its_loaded_avstack():
    svc, _, drivers = services()
    component = ComponentDescriptor('echo', {'x': 'float'}, {'value': 'float'}, {'x': 'plain'})
    scorer = ScorerDescriptor('distance', {'x': 'float'}, {'value': 'float'}, {'x': 'plain'})
    svc.driving_adapter.component_bindings['echo'] = (component, lambda avstack, x: x)
    svc.driving_adapter.scorer_bindings['distance'] = (scorer, lambda avstack, x, target: x - target)
    with svc.session(7) as session:
        assert session.evaluate('echo', 3).value == 3
        assert session.score('distance', 5, 2).value == 3
        assert session.score('distance', 7, 2).value == 5
        assert len(drivers) == 2 and drivers[0].closed and not drivers[1].closed
        session.rollout(candidate(svc.dataset))
        assert len(drivers) == 3 and drivers[2].closed
    assert drivers[1].closed


def test_candidate_failure_before_runner_stops_created_runtime_without_expanding_budget(monkeypatch):
    svc, simulator, _ = services()
    stopped = []
    class NeedsTicks(AttackRuntime):
        setup_ticks = 1
        def stop(self, context):
            stopped.append(context.stage)
    attack = SimpleNamespace(create_runtime=lambda *_: NeedsTicks())
    with svc.session(7) as session:
        with pytest.raises(ValueError, match='deployment tick budget'):
            session.rollout(candidate(svc.dataset, attack=attack))
    assert stopped == ['stop'] and simulator.worlds == []


def test_stateful_component_evaluations_do_not_share_history():
    svc, _, drivers = services()
    descriptor = ComponentDescriptor('increment', {'x': 'int'}, {'value': 'int'}, {'x': 'plain'})
    def increment(avstack, value):
        avstack.calls += 1
        return avstack.calls
    svc.driving_adapter.component_bindings['increment'] = descriptor, increment
    with svc.session(7) as session:
        assert session.evaluate('increment', 1).value == 1
        assert session.evaluate('increment', 1).value == 1
    assert len(drivers) == 2 and all(avstack.closed for avstack in drivers)


def test_selection_reasons_are_counted_and_bounded():
    svc, _, _ = services()
    view = svc.scenario("s", seed=7)
    for _ in range(1000):
        view.record_reason("Host is not visible")
    assert view.reasons == ("Host is not visible (count=1000)",)
    for index in range(150):
        view.record_reason(f"Distinct user reason {index}")
    assert len(view.reasons) == 101
    assert view.reasons[-1] == "Other selection reasons (count=51)"


def test_session_checks_origin_against_the_shared_deployment_budget():
    from avsectester.closed_loop.simulator import DeploymentPlan

    svc, simulator, _ = services()
    svc.deployment = DeploymentPlan(2)
    simulator.capture_ticks = 1
    with svc.session(7) as session:
        assert [session.can_deploy_at(position) for position in range(5)] == [True, False, False, True, True]
    assert simulator.worlds == []


@pytest.mark.parametrize("failure_phase", ["simulator", "avstack"])
def test_candidate_resource_creation_failure_stops_an_already_created_attack(monkeypatch, failure_phase):
    svc, simulator, _ = services()
    stopped = []
    class TrackingAttack(AttackRuntime):
        def stop(self, context):
            stopped.append(context.stage)
    attack = SimpleNamespace(create_runtime=lambda *_: TrackingAttack())
    def fail(*args, **kwargs):
        raise RuntimeError("factory failed")
    with svc.session(7) as session:
        monkeypatch.setattr(session, "_simulator_runtime" if failure_phase == "simulator" else "_avstack", fail)
        with pytest.raises(RuntimeError, match="factory failed"):
            session.rollout(candidate(svc.dataset, attack=attack))
    assert stopped == ["stop"]
    assert all(world.closed for world in simulator.worlds)
