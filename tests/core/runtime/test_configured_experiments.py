"""Configured treatments use the public preparation and execution interfaces end to end."""

from collections import Counter
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace

import pytest

from avsectester.closed_loop.avstack import AVStack
from avsectester.experiment import run_configured
from avsectester.closed_loop.datatypes import Control, StateEstimate
from avsectester.preparation import Intervention
from avsectester.closed_loop.hooks import Hook, Plugin, Runtime
from avsectester.closed_loop.simulators.nurec import NuRecSimulatorRuntime, NuRecSimulator, StubRenderer


@pytest.fixture
def configured_methods(monkeypatch):
    module = ModuleType("avsec_test_configured_methods")
    methods, handlers, worlds, stacks, localizers, noises = [], [], [], [], [], []
    created = Counter()

    class World(NuRecSimulatorRuntime):
        def __init__(self, config):
            super().__init__(NuRecSimulator(config, renderer=StubRenderer(height=2, width=3)))
            self.prepares = self.resets = self.closes = 0
            self.world_conditions = {}

        def configure_conditions(self, world):
            self.world_conditions = dict(world)

        def prepare_clean_attack_pair(self):
            self.prepares += 1

        def _initialize_world(self, source, initialization, environment):
            self.resets += 1
            return super()._initialize_world(source, initialization, environment)

        def close(self):
            self.closes += 1
            super().close()

    class Stack(AVStack):
        def __init__(self):
            self.samples = []

        def __call__(self, observation):
            self.samples.append(int(observation.sensor_data["camera_front"][0, 0, 0]))
            return Control(throttle=0.1)

    class Handler(Plugin):
        def __init__(self, kind):
            self.kind = kind
            handlers.append(self)

        def reset(self, context):
            self.calls = 0

        def transform(self, command, context):
            self.calls += 1
            throttle = (
                command.throttle * 2
                if self.kind == "attack"
                else max(0, command.throttle - 0.05)
            )
            command = replace(command, throttle=throttle)
            context.record_output(self.kind, {"throttle": throttle, "calls": self.calls})
            return command

        def close(self):
            self.closed = True

    class Method(Intervention):
        def __init__(self, kind):
            self.kind, self.prepares = kind, 0
            methods.append(self)

        def prepare(self, scenario):
            # Preparation must precede formal backend creation, without an extra preview.
            assert not worlds
            assert scenario.match is None
            assert not {"runtime", "interventions", "conditions", "stack"} & scenario.config.keys()
            self.prepares += 1
            return {"kind": self.kind}

        def hooks(self, artifacts):
            assert artifacts == {"kind": self.kind}
            return [Hook("command", Handler(self.kind))]

    class Localizer(Plugin):
        def __init__(self):
            localizers.append(self)
            self.calls, self.resets = 0, 0

        def reset(self, context):
            self.calls = 0
            self.resets += 1

        def transform(self, observation, context):
            self.calls += 1
            context.record_output("localizer", self.calls)
            return StateEstimate(observation.vehicle_state, observation.ego_speed)

        def close(self):
            self.closed = True

    class Noise(Plugin):
        def __init__(self):
            noises.append(self)

        def reset(self, context):
            self.values = []

        def transform(self, observation, context):
            value = int(context.rng.integers(1, 255))
            self.values.append(value)
            observation.sensor_data["camera_front"][0, 0, 0] = value
            context.record_output("noise", value)
            return observation

        def close(self):
            self.closed = True

    def make_raw_hook():
        created["raw"] += 1

        def raw(command, context):
            context.rng.random(500)
            command = replace(command, throttle=0.4)
            context.record_output("raw", command.throttle)
            return command

        return raw

    def make_stack():
        stack = Stack()
        stacks.append(stack)
        return stack

    def make_backend(config):
        backend = World(config)
        worlds.append(backend)
        return backend

    module.Method, module.Localizer, module.Noise = Method, Localizer, Noise
    module.make_raw_hook, module.make_stack = make_raw_hook, make_stack
    monkeypatch.setitem(sys.modules, module.__name__, module)
    path = module.__name__ + ":"
    config = {
        "backend": {"type": "NuRecSimulator"}, "dt": 0.1,
        "stack": {"factory": path + "make_stack"},
        "runtime": {"hooks": [{"stage": "command", "factory": path + "make_raw_hook"}]},
        "interventions": [
            {
                "factory": path + "Method", "options": {"kind": kind},
                "kind": kind, "name": kind,
            }
            for kind in ("attack", "defense")
        ],
    }
    return SimpleNamespace(
        config=config, make_backend=make_backend, methods=methods, handlers=handlers,
        worlds=worlds, stacks=stacks, created=created, localizers=localizers,
        noises=noises, path=path,
    )


@pytest.mark.parametrize("treatment,kinds,throttle", [
    ("clean", [], 0.1), ("attack", ["attack"], 0.8),
    ("defense", ["defense"], 0.05), ("attack-defense", ["attack", "defense"], 0.75),
])
def test_single_configured_treatment_prepares_only_enabled_methods(
    configured_methods, treatment, kinds, throttle,
):
    env = configured_methods
    results = run_configured(
        env.config, frames=2, mode="single", treatment=treatment,
        backend_factory=env.make_backend, provenance={"source": "user-config"},
    )
    assert list(results) == [treatment]
    trace = results[treatment]
    assert [record.throttle for record in trace.records] == pytest.approx([throttle, throttle])
    assert [method.kind for method in env.methods] == kinds
    assert all(method.prepares == 1 for method in env.methods)
    assert env.created["raw"] == int("attack" in kinds)
    expected_order = (["raw"] if "attack" in kinds else []) + kinds
    assert [output.name for output in trace.outputs] == expected_order * 2
    assert all(handler.calls == 2 and handler.closed for handler in env.handlers)
    assert len(env.worlds) == len(env.stacks) == 1
    backend = env.worlds[0]
    assert backend.prepares == 0 and backend.resets == backend.closes == 1
    assert trace.provenance["source"] == "user-config"
    assert [item["kind"] for item in trace.provenance["interventions"]] == kinds


def test_combined_treatment_preserves_user_order_instead_of_grouping_by_kind(configured_methods):
    env = configured_methods
    env.config["interventions"].reverse()
    trace = run_configured(
        env.config, frames=1, mode="single", treatment="attack-defense",
        backend_factory=env.make_backend,
    )["attack-defense"]
    assert trace.records[0].throttle == pytest.approx(0.7)
    assert [output.name for output in trace.outputs] == ["raw", "defense", "attack"]


@pytest.mark.parametrize("treatment", ["clean", "defense"])
def test_disabled_attack_factories_and_raw_hooks_are_not_imported(configured_methods, treatment):
    env = configured_methods
    env.config["interventions"][0]["factory"] = "missing_disabled_attack_module:Method"
    env.config["runtime"]["hooks"][0]["factory"] = "missing_disabled_attack_module:hook"
    result = run_configured(
        env.config, frames=1, mode="single", treatment=treatment,
        backend_factory=env.make_backend,
    )[treatment]
    assert len(result.records) == 1
    assert all(method.kind == "defense" for method in env.methods)
    assert env.created["raw"] == 0


@pytest.mark.parametrize("mode,treatment,labels", [
    ("single", "clean", ["clean"]),
    ("single", "attack", ["attack"]),
    ("pair", "clean", ["clean", "baseline"]),
    ("pair", "attack-defense", ["clean", "attack-defense"]),
], ids=["single-clean", "single-attack", "repeated-clean", "clean-vs-combined"])
def test_configured_localizers_and_conditions_are_fresh_for_each_executed_treatment(
    configured_methods, mode, treatment, labels,
):
    env = configured_methods
    env.config["runtime"]["localizer"] = {"factory": env.path + "Localizer"}
    env.config["conditions"] = {
        "world": {"weather": "ClearNoon"}, "seed": 91,
        "sensors": [{"factory": env.path + "Noise"}],
    }
    results = run_configured(
        env.config, frames=2, mode=mode, treatment=treatment,
        backend_factory=env.make_backend,
    )
    assert list(results) == labels
    assert len(env.localizers) == len(labels)
    assert all(
        localizer.calls == 2 and localizer.resets == 1 and localizer.closed
        for localizer in env.localizers
    )
    assert len(env.noises) == len(labels)
    assert all(noise.closed and len(noise.values) == 2 for noise in env.noises)
    assert all(noise.values == env.noises[0].values for noise in env.noises)
    assert all(stack.samples == env.stacks[0].samples for stack in env.stacks)
    assert len(env.worlds) == 1
    backend = env.worlds[0]
    assert backend.world_conditions == {"weather": "ClearNoon"}
    assert backend.prepares == int(mode == "pair")
    assert backend.resets == len(labels) and backend.closes == 1
    for trace in results.values():
        assert [v.value for v in trace.outputs if v.name == "localizer"] == [1, 2]
        assert trace.provenance["conditions"]["world"] == {"weather": "ClearNoon"}


def test_explicit_treated_runtime_replaces_configured_interventions(configured_methods):
    env = configured_methods
    env.config["interventions"][0]["factory"] = "missing_replaced_attack_module:Method"
    env.config["interventions"][1]["factory"] = "missing_replaced_defense_module:Method"
    env.config["runtime"]["hooks"][0]["factory"] = "missing_replaced_hook_module:hook"

    def brake(command, context):
        context.record_output("explicit", True)
        return Control(brake=1)

    results = run_configured(
        env.config, frames=1, mode="pair", treatment="attack-defense",
        backend_factory=env.make_backend,
        treated_runtime=Runtime([Hook("command", brake)], seed=17),
    )
    assert results["clean"].records[0].throttle == 0.1
    assert results["attack-defense"].records[0].brake == 1
    assert [v.name for v in results["attack-defense"].outputs] == ["explicit"]
    assert env.methods == [] and env.created["raw"] == 0


def test_configured_modes_are_used_unless_explicitly_overridden(configured_methods):
    env = configured_methods
    env.config.update(mode="single", treatment="clean")
    traces = run_configured(env.config, frames=1, backend_factory=env.make_backend)
    assert list(traces) == ["clean"]
    assert traces["clean"].records[0].throttle == 0.1
    assert env.methods == []
    traces = run_configured(env.config, frames=1, mode="pair", treatment="attack",
                            backend_factory=env.make_backend,
                            # Preparation is independently exercised by the earlier tests.
                            treated_runtime=Runtime([Hook("command", lambda value, context: value)]))
    assert list(traces) == ["clean", "attack"]


def test_repeated_clean_uses_the_explicit_baseline_in_both_runs(configured_methods):
    env = configured_methods
    runtime = Runtime([Hook("command", lambda value, context: replace(value, brake=0.5))], seed=17)
    traces = run_configured(env.config, frames=1, mode="pair", treatment="clean",
                            clean_runtime=runtime, backend_factory=env.make_backend)
    assert list(traces) == ["clean", "baseline"]
    assert all(trace.records[0].brake == 0.5 and trace.provenance["runtime_seed"] == 17
               for trace in traces.values())


@pytest.mark.parametrize("treatment", ["attack", "defense", "attack-defense"])
def test_missing_selected_methods_are_configuration_errors_before_backend_creation(
    configured_methods, treatment,
):
    from avsectester.experiment import ConfigurationError

    env = configured_methods
    env.config.pop("interventions")
    env.config.pop("runtime")
    with pytest.raises(ConfigurationError, match="No enabled intervention"):
        run_configured(env.config, frames=1, mode="single", treatment=treatment,
                       backend_factory=env.make_backend)
    assert env.worlds == []
    env.config["defenses"] = [{"factory": "unused:method"}]
    with pytest.raises(ConfigurationError, match="top-level defenses"):
        run_configured(env.config, frames=1, backend_factory=env.make_backend)
    assert env.worlds == []
