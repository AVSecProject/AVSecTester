"""The public preparation/execution pipeline selects once and isolates every run."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack import Attack, AttackSegment
from avsectester.attack.runtime import AttackRuntime, AVStackAttackEffect
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.runner import RecordingPolicy
from avsectester.closed_loop.services import CandidateRequest, DrivingAdapter
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.datasets import ActorState, Dataset, FrameData, FrameReference, ScenarioDescriptor, ScenarioStart
from avsectester.experiment import Experiment, ExperimentConfiguration, PreparedExperiment, TreatmentConfiguration
from avsectester.runlog import RunLog


class Data(Dataset):
    def __init__(self, names=("s",)):
        self.timelines = {name: tuple(FrameReference(name, i, f"frame-{i}", i * .1)
                                      for i in range(6)) for name in names}

    def scenarios(self):
        return iter(ScenarioDescriptor(name, "memory") for name in self.timelines)

    def frames(self, scenario_id):
        return self.timelines[scenario_id]

    def frame_data(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return FrameData(frame, None, {name: ActorState(np.eye(4)) for name in ("a", "b")})

    def native(self, scenario_id):
        return {"scenario": scenario_id}

    def scenario_start(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return ScenarioStart("memory", scenario_id, frame, self.native(scenario_id))


class World(SimulatorRuntime):
    command_modes = frozenset({"actuator"})

    def __init__(self, simulator):
        super().__init__(simulator)
        self.closed = False

    def _initialize_world(self, source, initialization, environment):
        self.source, self.i = source, 0
        self.environment = environment
        self.simulator.starts.append(source.frame)

    def _observe(self):
        return Observation(self.source.frame.timestamp + self.i * .1,
                           self.source.frame.position + self.i,
                           {"front": np.full((2, 2, 3), self.i + 10, np.uint8)})

    def _state(self):
        obs = self._observe()
        return WorldSnapshot(obs.t, obs.frame, ego_speed=float(self.i))

    def _advance_native(self, command):
        self.i += 1

    def close(self):
        self.closed = True


class Factory(Simulator):
    dataset_kinds = ("memory",)

    def __init__(self):
        self.worlds, self.starts = [], []

    @property
    def identity(self):
        return {"simulator": "memory", "version": 1}

    def create_runtime(self, seed=None):
        world = World(self)
        self.worlds.append(world)
        return world


class Driver(AVStack):
    command_modes = frozenset({"actuator"})

    def __init__(self, seed):
        self.seed, self.calls, self.closed = seed, 0, False

    def __call__(self, observation):
        self.calls += 1
        return Control(throttle=self.calls / 10)

    def close(self):
        self.closed = True


class Online(AttackRuntime):
    def __init__(self, artifacts):
        self.artifacts, self.calls, self.stopped = artifacts, 0, False

    def step(self, context):
        self.calls += 1
        context.record_output("decision", self.calls)
        return (AVStackAttackEffect("command", self.change),)

    def change(self, value, context):
        return replace(value, brake=float(self.artifacts["brake"]))

    def stop(self, context):
        self.stopped = True


class Method(Attack):
    def __init__(self, *, roles=False, candidate=False):
        self.builders, self.offline_calls, self.runtimes = [], 0, []
        self.views, self.roles, self.candidate = [], roles, candidate
        self.assets = np.zeros((1, 1, 3), np.uint8)

    def build_metadata(self, view):
        self.builders.append(view.scenario_id)
        self.views.append(view)
        view.record_reason("Chosen by the method")
        refs = view.frames
        return (AttackSegment(view.scenario_id, refs[1], refs[3],
                              roles={"hosts": ("a", "b")} if self.roles else {},
                              selection_score=.75 if self.roles else None,
                              selection_rank=0 if self.roles else None,
                              user_data={"source": "method"}),)

    def run_offline(self, context):
        self.offline_calls += 1
        # Changed shape is intentional: the framework must not reselect based on artifacts.
        artifacts = {"brake": .8, "texture": np.ones((7, 11, 3), np.uint8)}
        if self.candidate:
            segment = context.metadata[0]
            with context.services.session(2) as session:
                self.candidate_log = session.rollout(CandidateRequest(
                    segment, 2, segment.decision_steps, {}, RecordingPolicy(), self, artifacts,
                ))
        return artifacts

    def create_runtime(self, segment, artifacts):
        online = Online(artifacts)
        self.runtimes.append(online)
        return online


def configuration(method=None, names=("s",), **options):
    method = method or Method()
    drivers = []

    def factory(seed):
        value = Driver(seed)
        drivers.append(value)
        return value

    config = ExperimentConfiguration(Data(names), Factory(), DrivingAdapter(factory),
                                     attacks={"method": method}, **options)
    return config, drivers


def test_prepare_once_then_fresh_runs_share_origin_artifacts_and_candidate_behavior():
    method = Method(roles=True, candidate=True)
    config, drivers = configuration(method, names=("first", "second"), seeds=(2, 7),
                                    treatments=(TreatmentConfiguration("clean"),
                                                TreatmentConfiguration("attack", "method"),
                                                TreatmentConfiguration("repeat", "method")))
    prepared = Experiment.prepare(config)
    assert method.builders == ["first", "second"] and method.offline_calls == 1
    assert len(prepared.metadata["method"]) == 2
    assert all(view.services._closed for view in method.views)
    assert prepared.artifacts["method"]["texture"].shape == (7, 11, 3)
    result = Experiment.execute(prepared)
    assert len(result.runs) == 12 and len(result.groups) == 4 and not result.failures
    assert method.builders == ["first", "second"] and method.offline_calls == 1
    assert len(method.runtimes) == 9  # One candidate plus eight attacked formal runs.
    assert all(value.stopped and value.calls == 2 for value in method.runtimes)
    assert all(value.artifacts is prepared.artifacts["method"] for value in method.runtimes)
    assert all(value.closed and value.calls == 2 for value in drivers)
    assert all(world.closed for world in config.simulator.worlds)
    for runs in result.groups.values():
        clean, attack, repeat = (runs[name] for name in ("clean", "attack", "repeat"))
        assert clean.steps[0].physical_pre.frame == attack.steps[0].physical_pre.frame == 1
        assert [r.produced_command.throttle for r in attack.steps] == [.1, .2]
        assert [r.executed_command.brake for r in clean.steps] == [0, 0]
        assert [r.executed_command.brake for r in attack.steps] == [.8, .8]
        assert [r.executed_command for r in attack.steps] == [r.executed_command for r in repeat.steps]
        assert [r.executed_command for r in attack.steps] == [r.executed_command for r in method.candidate_log.steps]


def test_saved_preparation_loads_without_selecting_or_optimizing_and_runs_identically(tmp_path):
    method = Method(roles=True)
    config, _ = configuration(method)
    prepared = Experiment.prepare(config)
    prepared.save(tmp_path / "prepared")
    restored = PreparedExperiment.load(config, tmp_path / "prepared")
    assert method.builders == ["s"] and method.offline_calls == 1
    assert restored.metadata == prepared.metadata
    np.testing.assert_array_equal(restored.artifacts["method"]["texture"],
                                  prepared.artifacts["method"]["texture"])
    direct, loaded = Experiment.execute(prepared), Experiment.execute(restored)
    assert [[r.executed_command for r in run.log.steps] for run in direct.runs] == [
        [r.executed_command for r in run.log.steps] for run in loaded.runs]


def test_persistent_runs_have_group_environment_metadata_and_reopen(tmp_path):
    config, _ = configuration(output=tmp_path,
                              recording=RecordingPolicy(("front",), ("raw", "model_input")))
    result = Experiment.execute(Experiment.prepare(config))
    assert not result.failures
    for run in result.runs:
        log = RunLog.read(tmp_path / "runs" / "method" / "segment-00000" / "seed-0" / run.treatment)
        assert log.status == "completed" and log.metadata["segment"]["start"]["position"] == 1
        assert log.metadata["environment"] == {"world": {}, "sensors": []}
        assert len(log.steps) == 2
        assert log.steps[0].raw_observation.sensor_data["front"].shape == (2, 2, 3)


def test_selection_rejections_and_failures_are_reported_and_strict_stops():
    class Mixed(Method):
        def build_metadata(self, view):
            view.record_reason(view.scenario_id)
            if view.scenario_id == "rejected":
                return ()
            if view.scenario_id == "broken":
                raise RuntimeError("Selection failed")
            return super().build_metadata(view)

    config, _ = configuration(Mixed(), names=("rejected", "broken", "good"))
    prepared = Experiment.prepare(config)
    assert [item.status for item in prepared.selections] == ["not_selected", "error", "selected"]
    assert prepared.selections[0].reasons == ("rejected",)
    assert "Selection failed" in prepared.selections[1].error
    assert prepared.metadata["method"][0].scenario_id == "good"
    assert Experiment.execute(prepared).selection_errors == (prepared.selections[1],)
    with pytest.raises(RuntimeError, match="Selection failed"):
        Experiment.prepare(replace(config, strict=True))


@pytest.mark.parametrize("case,match", [
    ("element", "AttackSegment"), ("scenario", "another scenario"),
    ("boundary", "outside"), ("roles", "absent at the start"),
])
def test_invalid_metadata_is_rejected_before_offline(case, match):
    class Invalid(Method):
        def build_metadata(self, view):
            refs = view.frames
            if case == "element":
                return ({"start": 0},)
            if case == "scenario":
                return (AttackSegment("other", FrameReference("other", 0, 0, 0),
                                      FrameReference("other", 1, 1, .1)),)
            if case == "boundary":
                return (AttackSegment(view.scenario_id, refs[1],
                                      FrameReference(view.scenario_id, 6, "past-end", .6)),)
            return (AttackSegment(view.scenario_id, refs[1], refs[2], roles={"host": ("missing",)}),)

    method = Invalid()
    config, _ = configuration(method, strict=True)
    with pytest.raises((TypeError, ValueError), match=match):
        Experiment.prepare(config)
    assert method.offline_calls == 0


def test_no_asset_host_placement_or_scores_required_and_explicit_cap_is_logged():
    class Minimal(Attack):
        def build_metadata(self, view):
            return (AttackSegment(view.scenario_id, view.frames[0], view.frames[5]),)

    config, _ = configuration(Minimal(), decision_steps=2)
    prepared = Experiment.prepare(config)
    assert prepared.artifacts == {"method": None}
    result = Experiment.execute(prepared)
    assert not result.failures and all(len(run.log.steps) == 2 for run in result.runs)
    assert result.runs[0].log.metadata["selected_decision_steps"] == 5
    assert prepared.metadata["method"][0].decision_steps == 5


def test_unsupported_environment_and_unselected_clean_config_fail_before_world_creation():
    config, _ = configuration(environment={"world": {"weather": "HardRainNoon"}})
    with pytest.raises(ValueError, match="world"):
        Experiment.prepare(config)
    assert not config.simulator.worlds and not config.attacks["method"].builders
    with pytest.raises(ValueError, match="requires an attack"):
        Experiment.prepare(replace(config, attacks={}, treatments=(), environment={}))


def test_reused_factory_objects_produce_failed_results_instead_of_reusing_state():
    config, _ = configuration()
    driver = Driver(0)
    config = replace(config, avstack=DrivingAdapter(lambda seed: driver))
    result = Experiment.execute(Experiment.prepare(config))
    assert result.runs[0].log.status == "completed"
    assert result.runs[1].log.status == "failed" and "reused" in result.runs[1].error
    assert driver.calls == 2 and all(world.closed for world in config.simulator.worlds)


def test_treatments_detach_start_payloads_and_use_experiment_seed_without_mutating_hooks(monkeypatch):
    observed_draws = []
    initialize = World._initialize_world

    def check_start(self, source, initialization, environment):
        assert source.payload["scenario"] == "s"
        initialize(self, source, initialization, environment)

    def change_world(value, context):
        context.simulator_runtime._start.payload["scenario"] = "changed in this run"
        observed_draws.append(context.rng.integers(1000))
        return value

    monkeypatch.setattr(World, "_initialize_world", check_start)
    hooks = HookSet([Hook("simulator.setup", change_world)], seed=99)
    config, _ = configuration(seeds=(7,), hooks=hooks, treatments=(
        TreatmentConfiguration("clean"), TreatmentConfiguration("first", "method"),
        TreatmentConfiguration("second", "method"),
    ))
    result = Experiment.execute(Experiment.prepare(config))
    assert not result.failures and len(result.runs) == 3
    assert hooks.seed == 99
    assert observed_draws == [np.random.default_rng(7).integers(1000)] * 2


def test_attacks_select_and_prepare_independently_with_layered_treatment_environment():
    class Independent(Method):
        def __init__(self, scenario_id, start, brake, environment):
            super().__init__()
            self.scenario_id, self.start, self.brake = scenario_id, start, brake
            self.environment = environment

        def build_metadata(self, view):
            self.builders.append(view.scenario_id)
            if view.scenario_id != self.scenario_id:
                return ()
            return (AttackSegment(view.scenario_id, view.frames[self.start],
                                  view.frames[self.start + 2]),)

        def run_offline(self, context):
            self.offline_calls += 1
            assert len(context.metadata) == 1
            assert context.metadata[0].scenario_id == self.scenario_id
            return {"brake": self.brake}

    first = Independent("first", 0, 0.25, {"world": {"weather": {"precipitation": 40}}})
    second = Independent("second", 2, 0.75, {"world": {"weather": {"precipitation": 80}}})
    noise = {"type": "GaussianNoise", "options": {"severity": 0.0}, "camera": "front"}
    config, _ = configuration(names=("first", "second"))
    config.simulator.supported_world_settings = frozenset({"weather"})
    config = replace(config, attacks={"one": first, "two": second},
                     environment={"world": {"weather": {"cloudiness": 10}}, "sensors": [noise]},
                     treatments=(TreatmentConfiguration("clean"),
                                 TreatmentConfiguration("one_run", "one", {
                                     "world": {"weather": {"cloudiness": 90}}, "sensors": [],
                                 }),
                                 TreatmentConfiguration("two_run", "two")))
    prepared = Experiment.prepare(config)
    assert first.builders == second.builders == ["first", "second"]
    assert first.offline_calls == second.offline_calls == 1
    assert prepared.artifacts == {"one": {"brake": 0.25}, "two": {"brake": 0.75}}
    result = Experiment.execute(prepared)
    assert not result.failures
    assert set(result.groups) == {("one", 0, 0), ("two", 0, 0)}
    assert set(result.groups[("one", 0, 0)]) == {"clean", "one_run"}
    assert set(result.groups[("two", 0, 0)]) == {"clean", "two_run"}
    for group, runs in result.groups.items():
        attack_name = group[0]
        attacked = runs["one_run" if attack_name == "one" else "two_run"]
        expected_start, expected_brake = (0, 0.25) if attack_name == "one" else (2, 0.75)
        assert attacked.steps[0].physical_pre.frame == expected_start
        assert runs["clean"].steps[0].physical_pre.frame == expected_start
        assert [step.executed_command.brake for step in attacked.steps] == [expected_brake] * 2
        assert runs["clean"].metadata["environment"] == {
            "world": {"weather": {"cloudiness": 10}}, "sensors": [noise],
        }
        expected_environment = {
            "world": {"weather": {"cloudiness": 90, "precipitation": 40}}, "sensors": [],
        } if attack_name == "one" else {
            "world": {"weather": {"cloudiness": 10, "precipitation": 80}}, "sensors": [noise],
        }
        assert attacked.metadata["environment"] == expected_environment
    assert [world.environment for world in config.simulator.worlds] == [
        {"weather": {"cloudiness": 10}}, {"weather": {"cloudiness": 90, "precipitation": 40}},
        {"weather": {"cloudiness": 10}}, {"weather": {"cloudiness": 10, "precipitation": 80}},
    ]


@pytest.mark.parametrize("failure", ["exception", "wrong_type"])
def test_attack_construction_failure_is_saved_while_clean_and_other_segments_continue(tmp_path, failure):
    class Failing(Method):
        def create_runtime(self, segment, artifacts):
            if segment.scenario_id == "first":
                if failure == "exception":
                    raise RuntimeError("Unable to create attack")
                return None
            return super().create_runtime(segment, artifacts)

    config, _ = configuration(Failing(), names=("first", "second"), output=tmp_path)
    prepared = Experiment.prepare(config)
    result = Experiment.execute(prepared)
    first, second = result.groups[("method", 0, 0)], result.groups[("method", 1, 0)]
    assert first["method"].status == "failed" and first["clean"].status == "completed"
    assert all(log.status == "completed" for log in second.values())
    assert len(result.failures) == 1
    expected = "Unable to create attack" if failure == "exception" else "AttackRuntime"
    assert expected in result.failures[0].error
    saved = RunLog.read(tmp_path / "runs/method/segment-00000/seed-0/method")
    assert saved.status == "failed" and saved.steps == []
    assert any(expected in str(event.data) for event in saved.events)
    assert all(world.closed for world in config.simulator.worlds)
    strict_prepared = replace(prepared, configuration=replace(config, strict=True, output=None))
    with pytest.raises((TypeError, RuntimeError), match=expected):
        Experiment.execute(strict_prepared)


class RecipeData(Data):
    """A CARLA-shaped dataset: actor evidence is generated for each execution seed."""

    def scenarios(self):
        return iter(ScenarioDescriptor(name, "CarlaDataset") for name in self.timelines)

    def frame_data(self, scenario_id, frame):
        from avsectester.datasets import Unavailable
        self.checked_frame(scenario_id, frame)
        return Unavailable("recipe")

    def scenario_start(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return ScenarioStart("CarlaDataset", scenario_id, frame, {"recipe": "fixed"})


class RecipeWorld(World):
    def __init__(self, simulator, seed):
        super().__init__(simulator)
        self.seed = seed

    def generate_frame_data(self, start, frames, *, world=None, deployment=None):
        from avsectester.closed_loop.frame_data import FrameDataSequence
        self.simulator.generations.append((self.seed, deployment.ticks, world))
        actors = {"a": ActorState(np.eye(4))}
        if self.seed not in self.simulator.missing_host_seeds:
            actors["b"] = ActorState(np.eye(4))
        return FrameDataSequence(tuple(FrameData(frame, None, actors) for frame in frames),
                                 {"seed": self.seed}, {"deployment_ticks": deployment.ticks})

    def _initialize_world(self, start, initialization, environment):
        super()._initialize_world(start, initialization, environment)
        assert initialization.resolved_scenario == {"seed": self.seed}
        assert initialization.expected.frame == start.frame
        self.simulator.initializations.append((self.seed, start.frame.position,
                                              self._deployment.ticks, environment))

    def _check_initialization(self, initialization):
        pass

    def _hold_tick(self):
        # Deployment ticks run before the first selected capture and are not decisions.
        pass


class RecipeFactory(Factory):
    dataset_kinds = ("CarlaDataset",)

    def __init__(self, missing_host_seeds=()):
        super().__init__()
        self.missing_host_seeds = missing_host_seeds
        self.generations, self.initializations = [], []

    def create_runtime(self, seed=None):
        world = RecipeWorld(self, seed)
        self.worlds.append(world)
        return world


class TwoSegments(Method):
    def build_metadata(self, view):
        first = super().build_metadata(view)[0]
        return first, replace(first, start=view.frames[3], stop=view.frames[5])

    def create_runtime(self, segment, artifacts):
        runtime = super().create_runtime(segment, artifacts)
        runtime.setup_ticks = 1
        return runtime


def recipe_configuration(*, missing_host_seeds=(), strict=False, output=None):
    config, _ = configuration(TwoSegments(roles=True), seeds=(0, 1))
    return replace(config, dataset=RecipeData(), simulator=RecipeFactory(missing_host_seeds),
                   strict=strict, output=output)


def test_carla_execution_reuses_frame_cache_across_segments_and_shares_deployment_window():
    config = recipe_configuration()
    prepared = Experiment.prepare(config)
    assert len(config.simulator.generations) == 1
    result = Experiment.execute(prepared)
    assert not result.failures and len(result.runs) == 8
    # One preparation sequence; one execution sequence per seed, shared by both segments.
    assert [(seed, ticks) for seed, ticks, _ in config.simulator.generations] == [(0, 0), (0, 1), (1, 1)]
    assert len(config.simulator.initializations) == 8
    assert {ticks for _, _, ticks, _ in config.simulator.initializations} == {1}
    assert all(world.closed for world in config.simulator.worlds)
    assert prepared.configuration.attacks["method"].offline_calls == 1
    assert prepared.configuration.attacks["method"].builders == ["s"]
    for group in result.groups.values():
        assert group["clean"].steps[0].physical_pre == group["method"].steps[0].physical_pre


def test_missing_role_in_an_execution_seed_fails_without_reselection_and_is_saved(tmp_path):
    config = recipe_configuration(missing_host_seeds=(1,), output=tmp_path)
    prepared = Experiment.prepare(config)
    result = Experiment.execute(prepared)
    assert len(result.failures) == 4
    assert all(run.seed == 1 and "seed 1" in run.error and "b" in run.error for run in result.failures)
    assert len(config.simulator.initializations) == 4  # only seed 0 reaches driving initialization
    assert all(seed == 0 for seed, *_ in config.simulator.initializations)
    assert config.attacks["method"].builders == ["s"]
    assert config.attacks["method"].offline_calls == 1
    for run in result.failures:
        reopened = RunLog.read(run.log.path)
        assert reopened.status == "failed" and not reopened.steps
        assert any("seed 1" in event.data.get("message", "") for event in reopened.events)
    with pytest.raises(ValueError, match="seed 1.*b"):
        Experiment.execute(replace(prepared, configuration=replace(config, strict=True, output=None)))



def test_hidden_constructor_parameters_do_not_block_preparation_or_execution():
    import warnings

    class Hidden(Method):
        def __init__(self, texture_path):
            super().__init__()
            self.assets = texture_path

    attack = Hidden("texture.png")
    with warnings.catch_warnings(record=True) as emitted:
        config, _ = configuration(attack)
        prepared = Experiment.prepare(config)
        result = Experiment.execute(prepared)
    assert not emitted
    assert not result.failures and len(result.runs) == 2
    assert attack.builders == ["s"] and attack.offline_calls == 1
    assert not hasattr(attack, "texture_path")
