"""Selection and preparation use the same capabilities without borrowing active state."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack_runtime import AttackRuntime, StageAction
from avsectester.backend import AVStack
from avsectester.capture import CaptureBackend
from avsectester.closed_loop import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.datasets.base import Dataset, FrameRef, ReferenceFrame, SourceData, Unavailable
from avsectester.plane import Control, Observation
from avsectester.reference import ReferenceSequence
from avsectester.services import (
    CandidateOrigin, CandidateRequest, ClosedLoopServices, ComponentDescriptor,
    DrivingAdapter, RenderRequest, RenderResult, ScorerDescriptor, SensorRequest,
)


class Data(Dataset):
    def __init__(self, kind="memory"):
        self.kind = kind
        self.refs = tuple(FrameRef("s", i, f"source-{i}", i * 0.1) for i in range(5))

    def scenarios(self):
        return ()

    def frames(self, scenario_id):
        assert scenario_id == "s"
        return self.refs

    def reference(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return Unavailable("recipe_only") if self.kind == "carla" else ReferenceFrame(frame, None)

    def native(self, scenario_id):
        return {"recipe": "source data"}

    def source_data(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return SourceData(self.kind, scenario_id, frame, self.native(scenario_id))


class World(CaptureBackend):
    source_kinds = ("memory", "carla")
    can_render = True
    identity = {"simulator": "test", "version": 1}

    def __init__(self, key=None, factory=None):
        super().__init__()
        self.factory = factory
        self.closed = False
        self.resource_key = key if key is not None else ("memory", id(self))
        self.resets = 0
        self.source = SourceData("memory", "s", FrameRef("s", 0, "source-0", 0), {})

    def _initialize_world(self, source, initialization, environment):
        if source is not None:
            self.source = deepcopy(source)
        self.resets += 1
        self.i = 0
        self._remember_ground_truth(self._observe())

    def _check_initialization(self, initialization):
        pass

    def _observe(self):
        return Observation(self.source.start.timestamp + self.i * 0.1,
                           self.source.start.position + self.i,
                           {"front": np.full((2, 2, 3), self.i, dtype=np.uint8)})

    def _advance_native(self, control):
        self.i += 1
        self._remember_ground_truth(self._observe())

    def materialize_reference(self, source, frames, **kwargs):
        if self.factory.materialize is not None:
            return self.factory.materialize(source, frames)
        self.factory.materializations += 1
        self.reset_world(source)
        return ReferenceSequence(tuple(ReferenceFrame(f, None) for f in frames),
                                 {"resolved": True}, {"seed": 7})

    def render(self, observation, request):
        observation.sensor_data[request.sensor_id][:] = request.insertions[0]
        return RenderResult(request.frame, observation, {}, {"candidate": request.insertions[0]})

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, key=None):
        self.resource_key = key
        self.worlds = []
        self.materializations = 0
        self.materialize = None

    def __call__(self, source, seed):
        world = World(self.resource_key, self)
        self.worlds.append(world)
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
    return ClosedLoopServices(data, adapter, DrivingAdapter(factory), backend_type=World), adapter, drivers


def candidate(data, **kwargs):
    return CandidateRequest(CandidateOrigin("s", data.refs[1], data.refs[3]), 7, 2, {},
                            RecordingPolicy(("front",)), **kwargs)


def test_sensor_render_and_rollout_have_fresh_worlds_and_no_data_side_effects():
    svc, adapter, drivers = services()
    with svc.open_scope(7) as scope:
        first = scope.sensor_data(SensorRequest("s", svc.dataset.refs[1], ("front",)))
        rendered = scope.render(RenderRequest("s", svc.dataset.refs[1], "front", (123,)))
        logs = [scope.rollout(candidate(svc.dataset)) for _ in range(2)]
    assert len(adapter.worlds) == 4 and all(w.closed for w in adapter.worlds)
    assert all(d.closed and d.calls == 2 for d in drivers)
    assert first.observation.sensor_data["front"].max() == 0
    assert rendered.observation.sensor_data["front"].min() == 123
    assert rendered.evidence == {"candidate": 123}
    for log in logs:
        assert [s.produced_command.throttle for s in log.steps] == [0.1, 0.2]
        assert log.steps[0].physical_pre.frame == 1
        assert log.steps[-1].physical_post.frame == 3
        assert log.metadata["candidate"]["roles"] == {}
    assert svc.dataset.native("s") == {"recipe": "source data"}


def test_candidate_bounds_and_seed_fail_before_allocating_resources():
    svc, adapter, _ = services()
    with svc.open_scope(7) as scope:
        with pytest.raises(ValueError, match="seed"):
            scope.rollout(replace(candidate(svc.dataset), seed=8))
        bad = CandidateOrigin("s", svc.dataset.refs[1], FrameRef("s", 6, "6", 0.6))
        with pytest.raises(ValueError, match="support"):
            scope.rollout(CandidateRequest(bad, 7, 5, {}, RecordingPolicy()))
    assert adapter.worlds == []
    with pytest.raises(ValueError, match="decision_steps"):
        replace(candidate(svc.dataset), decision_steps=1)
    with pytest.raises(ValueError, match="fresh"):
        svc.open_scope(7, history="reuse")


def test_preparation_cache_is_reused_and_does_not_change_dataset_behavior():
    svc, adapter, _ = services("carla", ("test-server", 1))
    view = svc.scenario("s", seed=7)
    original = svc.dataset.native("s")
    assert isinstance(svc.dataset.reference("s", view.frames[1]), Unavailable)
    assert view.reference(view.frames[1]).frame == view.frames[1]
    view.reference(view.frames[3])
    with svc.open_scope(7) as scope:
        scope.rollout(candidate(svc.dataset))
    assert adapter.materializations == 1
    assert svc.dataset.native("s") == original
    assert isinstance(svc.dataset.reference("s", view.frames[1]), Unavailable)
    with svc.open_scope(8) as scope:
        scope.reference("s", view.frames[1])
    assert adapter.materializations == 2


def test_unclosed_unused_scope_does_not_reserve_a_server():
    svc, adapter, _ = services(key=("test-server", 2))
    unused = svc.open_scope(7)
    with svc.open_scope(7) as scope:
        scope.rollout(candidate(svc.dataset))
    ClosedLoopRunner().run(World(adapter.resource_key), Driver(), RunRequest(1))
    unused.close()


def test_active_runner_endpoint_is_not_available_to_selection():
    svc, adapter, _ = services(key=("test-server", 3))
    scope = svc.open_scope(7)
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
    a, b = svc.open_scope(7), svc.open_scope(7)
    with a._operation() as resources:
        a._backend(resources, svc.dataset.source_data("s", svc.dataset.refs[0]))
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
    def factory(attack, origin, artifacts):
        assert artifacts is assets
        origins.append(origin)
        runtime = Attack()
        attacks.append(runtime)
        return runtime
    with pytest.raises(KeyboardInterrupt) as error:
        with svc.open_scope(7) as scope:
            scope.rollout(candidate(svc.dataset, attack=SimpleNamespace(runtime_factory=factory),
                                    artifacts=assets))
    assert error.value.run_log.status == "cancelled"
    assert attacks[0].stopped and drivers[0].closed and adapter.worlds[0].closed
    assert origins[0].roles == {}  # assetless/hostless attacks use the same factory path
    with svc.open_scope(7) as scope:
        assert scope.rollout(candidate(svc.dataset)).status == "completed"


def test_attack_runtime_factory_is_fresh_for_each_candidate_and_logged():
    svc, _, _ = services()
    runtimes = []
    class Attack(AttackRuntime):
        def start(self, context):
            self.count = 0
            return ()
        def step(self, context):
            self.count += 1
            return (StageAction("command", Control(brake=1)),)
    def factory(*args):
        runtime = Attack()
        runtimes.append(runtime)
        return runtime
    request = candidate(svc.dataset, attack=SimpleNamespace(runtime_factory=factory))
    with svc.open_scope(7) as scope:
        logs = [scope.rollout(request), scope.rollout(request)]
    assert len(runtimes) == 2 and all(r.count == 2 for r in runtimes)
    assert all(log.steps[0].executed_command.brake == 1 for log in logs)
    assert all(log.steps[0].actions for log in logs)


def test_reused_factory_object_is_rejected_without_closing_an_active_borrower():
    svc, _, _ = services()
    shared = Driver()
    svc.driving_adapter = DrivingAdapter(lambda seed: shared)
    with svc.open_scope(7) as scope:
        scope.rollout(candidate(svc.dataset))
        shared.closed = False
        with pytest.raises(RuntimeError, match="reused"):
            scope.rollout(candidate(svc.dataset))
    assert not shared.closed


def test_components_and_scorers_get_fresh_state_and_gradients_fail_early():
    svc, _, _ = services()
    builds = []
    def factory(seed):
        driver = Driver()
        builds.append(driver)
        return driver
    desc = ComponentDescriptor("counts", {"x": "int"}, {"y": "int"}, {"y": "int"})
    score = ScorerDescriptor("distance", {"x": "float"}, {"y": "float"}, {"y": "float"})
    svc.driving_adapter = DrivingAdapter(factory,
        components=((desc, lambda stack, value: (stack.calls, value)),),
        scorers=((score, lambda stack, value, target: abs(value - target)),))
    with svc.open_scope(7) as scope:
        assert scope.components() == (desc,)
        assert scope.scorers() == (score,)
        with pytest.raises(ValueError, match="gradients"):
            scope.score("distance", 2, 5, require_grad=True)
        assert not builds
        assert scope.evaluate("counts", 8).value == (0, 8)
        assert scope.evaluate("counts", 9).value == (0, 9)
        assert scope.score("distance", 2, 5).value == 3
    assert len(builds) == 3 and all(d.closed for d in builds)
    with pytest.raises(RuntimeError, match="closed"):
        scope.components()


def test_differentiable_score_preserves_graph_and_rejects_detached_result():
    torch = pytest.importorskip("torch")
    svc, _, _ = services()
    descriptor = ScorerDescriptor("square", {"x": "tensor"}, {"score": "tensor"},
                                  {"score": "tensor"}, ("x",))
    svc.driving_adapter = DrivingAdapter(lambda seed: Driver(), scorers=((
        descriptor, lambda stack, value, target: (value - target).square().sum()),))
    x = torch.tensor([3.], requires_grad=True)
    with svc.open_scope(7) as scope:
        result = scope.score("square", x, 1, require_grad=True)
        result.value.backward()
        assert x.grad.item() == 4
        with pytest.raises(ValueError, match="detached"):
            scope.score("square", x.detach(), 1, require_grad=True)


def test_failed_reference_generation_is_not_cached_and_resources_close():
    svc, adapter, _ = services("carla")
    def fail(*args):
        raise ValueError("reference unavailable")
    adapter.materialize = fail
    with svc.open_scope(7) as scope:
        for _ in range(2):
            with pytest.raises(ValueError, match="reference unavailable"):
                scope.reference("s", svc.dataset.refs[1])
    assert len(adapter.worlds) == 2 and all(world.closed for world in adapter.worlds)


def test_car_source_factory_uses_actual_connection_fields_and_detached_recipe():
    from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
    from avsectester.simulators.carla import CarlaBackend

    recipe = CarlaRecipe({"client": {"synchronous": True, "rate": 20, "connect_port": 2000},
                          "ego": {}}, 1, .05, {"ego": PrescribedMotion((Control(),))})
    data = CarlaDataset({"s": recipe})
    source = data.source_data("s", data.frames("s")[0])
    config = deepcopy(recipe.scenario)
    config["client"].update(connect_ip="127.0.0.1", connect_port=2300)
    backend = CarlaBackend(config)
    assert backend.resource_key == ("carla", "localhost", 2300)
    assert source.payload["recipe"].scenario["client"]["connect_port"] == 2000
    assert "weather" not in data.native("s").scenario
    with pytest.raises(ValueError, match="weather only"):
        backend._resolve_scenario(source, None, {"teleport": True})


def test_owned_resource_cleanup_failure_is_recorded_before_log_completion(tmp_path):
    svc, adapter, _ = services()
    class BadCloseDriver(Driver):
        def close(self):
            super().close()
            raise RuntimeError("driver resource release failed")
    svc.driving_adapter = DrivingAdapter(lambda seed: BadCloseDriver())
    with svc.open_scope(7) as scope:
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
    svc, adapter, drivers = services("carla")
    report = {"failures": ["missing aligned capture: ['front']"]}

    class ReferenceFailure(ValueError):
        def __init__(self):
            self.report = report
            super().__init__("reference preparation failed")

    def fail(*args):
        raise ReferenceFailure()
    adapter.materialize = fail
    with svc.open_scope(7) as scope:
        with pytest.raises(ReferenceFailure):
            scope.rollout(candidate(svc.dataset, output_path=tmp_path / "log"))
    saved = RunLog.read(tmp_path / "log")
    assert saved.status == "failed" and saved.steps == [] and drivers == []
    assert saved.events[0].data["initialization_report"] == report


def test_backend_closes_before_its_server_reservation_is_released(monkeypatch):
    from avsectester import resources

    key = ("carla", "localhost", 2000)
    svc, _, _ = services(key=key)
    held_during_close = []
    original_close = World.close

    def close(self):
        held_during_close.append(key in resources._owners)
        original_close(self)

    monkeypatch.setattr(World, "close", close)
    with svc.open_scope(7) as scope:
        scope.sensor_data(SensorRequest("s", svc.dataset.refs[1], ("front",)))
    assert held_during_close == [True]
    assert key not in resources._owners
