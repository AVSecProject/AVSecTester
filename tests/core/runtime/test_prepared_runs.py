"""Prepared scenarios retain bindings and conditions across named experiment treatments."""

from copy import deepcopy

import pytest

from avsectester.backend import AVStack
from avsectester.conditions import ScenarioConditions, SensorCondition
from avsectester.plane import Control
from avsectester.runner import run_prepared
from avsectester.runtime import Hook, Plugin, Runtime
from avsectester.scenarios.requirement import ScenarioMatch
from avsectester.scenarios.scene import EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import ScenarioInstance
from avsectester.simulators.nurec import NuRecBackend, StubRenderer


class MemoryWorld(NuRecBackend):
    def __init__(self):
        super().__init__({"dt": 0.1}, renderer=StubRenderer(height=2, width=3))
        self.prepares = self.resets = self.closes = 0

    def prepare_clean_attack_pair(self):
        self.prepares += 1

    def reset(self):
        self.resets += 1
        return super().reset()

    def close(self):
        self.closes += 1
        super().close()


class ReadingStack(AVStack):
    def __init__(self):
        self.samples = []
        self.resets = 0

    def reset(self, observation):
        self.resets += 1

    def __call__(self, observation):
        self.samples.append((
            observation.ego_speed,
            int(observation.sensor_data["camera_front"][0, 0, 0]),
        ))
        return Control()


def selected_match():
    actor = ObjectGT("selected-host", "vehicle", (10, 0, 0), extent=(4, 2, 1.5))
    scene = SceneGT(0, 0.0, EgoState(), {}, [actor], source={"clip": "prepared-clip"})
    return ScenarioMatch(scene, actor, None, bindings={"attacker": (actor,)})


@pytest.mark.parametrize("names", [
    ("clean",), ("clean", "attacked"), ("clean", "attacked", "defended"),
], ids=["single", "pair", "three-treatments"])
def test_prepared_treatments_share_origin_bindings_and_independent_condition_state(names):
    backend = MemoryWorld()
    stacks, condition_handlers, cases_seen = [], [], []

    class SensorNoise(Plugin):
        def reset(self, context):
            self.history = []

        def transform(self, observation, context):
            self.history.append(int(context.rng.integers(1, 255)))
            observation.sensor_data["camera_front"][0, 0, 0] = self.history[-1]
            observation.ego_speed = len(self.history)
            return observation

        def close(self):
            self.closed = True

    def make_condition():
        handler = SensorNoise()
        condition_handlers.append(handler)
        return handler

    def make_stack():
        stack = ReadingStack()
        stacks.append(stack)
        return stack

    def observe_command(command, context):
        cases_seen.append(context.case)
        context.record_output("binding", context.case.binding_ids)
        return command

    def consume_attack_randomness(observation, context):
        context.rng.random(1000)
        return observation

    match = selected_match()
    scenario = ScenarioInstance(
        lambda: backend, match,
        provenance={"clip": "prepared-clip", "selection": {"camera": "front"}},
        conditions=ScenarioConditions(sensors=(SensorCondition(make_condition),), seed=91),
    )
    runtimes = {
        name: Runtime([
            *([Hook("observation", consume_attack_randomness)] if index else []),
            Hook("command", observe_command),
        ], seed=10 + index)
        for index, name in enumerate(names)
    }
    results = run_prepared(scenario, make_stack, frames=3, runtimes=runtimes)

    assert tuple(results) == names
    assert backend.prepares == int(len(names) > 1)
    assert backend.resets == len(names) and backend.closes == 1
    assert len({id(stack) for stack in stacks}) == len(names)
    assert all(stack.resets == 1 and len(stack.samples) == 3 for stack in stacks)
    assert all(stack.samples == stacks[0].samples for stack in stacks)
    assert [sample[0] for sample in stacks[0].samples] == [1, 2, 3]
    assert len({id(handler) for handler in condition_handlers}) == len(names)
    assert all(handler.closed and len(handler.history) == 3 for handler in condition_handlers)
    assert all(handler.history == condition_handlers[0].history for handler in condition_handlers)
    assert all(case is match for case in cases_seen)
    for name, trace in results.items():
        assert trace.provenance["treatment"] == name
        assert trace.provenance["clip"] == "prepared-clip"
        assert trace.provenance["conditions"] == scenario.conditions.provenance()
        assert [event.value for event in trace.outputs] == [{"attacker": ("selected-host",)}] * 3
        assert [(event.step, event.frame) for event in trace.outputs] == [(0, 0), (1, 1), (2, 2)]

    scenario.provenance["selection"]["camera"] = "changed-after-run"
    match.bindings.clear()
    assert all(trace.provenance["selection"]["camera"] == "front" for trace in results.values())
    assert all(trace.outputs[0].value == {"attacker": ("selected-host",)} for trace in results.values())


def test_prepared_pass_through_runs_once_without_inventing_a_selection_match():
    backend = MemoryWorld()
    scenario = ScenarioInstance(lambda: backend)
    cases = []

    def record(command, context):
        cases.append(context.case)
        return command

    result = run_prepared(
        scenario, ReadingStack, frames=1,
        runtimes={"single": Runtime([Hook("command", record)])},
    )
    assert cases == [None]
    assert backend.prepares == 0 and backend.resets == backend.closes == 1
    assert len(result["single"].records) == 1


def test_prepared_runtime_factory_receives_backend_and_preserves_an_explicit_case():
    backend = MemoryWorld()
    scenario = ScenarioInstance(lambda: backend, selected_match())
    explicit = deepcopy(scenario.match)
    explicit.target.track_id = "explicit-host"
    seen = []

    def build_runtime(prepared_backend):
        assert prepared_backend is backend
        assert backend.prepares == 1 and backend.resets == 0

        def record(command, context):
            seen.append(context.case)
            return command

        return Runtime([Hook("command", record)], case=explicit)

    results = run_prepared(
        scenario, ReadingStack, frames=1,
        runtimes={"clean": None, "explicit": build_runtime},
    )
    assert set(results) == {"clean", "explicit"}
    assert seen == [explicit]
    assert backend.prepares == 1 and backend.resets == 2 and backend.closes == 1


def test_unsupported_world_conditions_fail_before_reset_and_close_the_backend():
    backend = MemoryWorld()
    scenario = ScenarioInstance(
        lambda: backend,
        conditions=ScenarioConditions(world={"weather": "HardRainNoon"}),
    )
    with pytest.raises(ValueError, match="does not support world conditions"):
        run_prepared(scenario, ReadingStack, frames=1)
    assert backend.prepares == backend.resets == 0
    assert backend.closes == 1


def test_failure_in_later_treatment_closes_plugins_and_world_with_fresh_stack_history():
    backend = MemoryWorld()
    scenario = ScenarioInstance(lambda: backend, selected_match())
    stacks = []
    failure = RuntimeError("adaptive attack failed")

    class FailingAttack(Plugin):
        def reset(self, context):
            self.started = True

        def transform(self, observation, context):
            if context.step == 1:
                raise failure
            return observation

        def close(self):
            self.closed = True

    attack = FailingAttack()

    def make_stack():
        stack = ReadingStack()
        stacks.append(stack)
        return stack

    with pytest.raises(RuntimeError, match="adaptive attack failed") as caught:
        run_prepared(
            scenario, make_stack, frames=2,
            runtimes={"clean": None, "attacked": Runtime([Hook("observation", attack)])},
        )
    assert caught.value is failure and attack.started and attack.closed
    assert backend.prepares == 1 and backend.resets == 2 and backend.closes == 1
    assert len(stacks) == 2 and stacks[0] is not stacks[1]
    assert [len(stack.samples) for stack in stacks] == [2, 1]
    assert backend._runtime is None and all(stack._runtime is None for stack in stacks)


def test_prepared_runner_keeps_custom_recording_results():
    backend = MemoryWorld()
    scenario = ScenarioInstance(lambda: backend)

    def record(backend, stack, frames, *, runtime):
        from avsectester.evaluation.component_log import run_logged

        return run_logged(backend, stack, frames, runtime=runtime)

    results = run_prepared(scenario, ReadingStack, frames=2, run_fn=record)
    trace, components = results["clean"]
    assert len(trace.records) == 2 and not components.steps
    assert trace.provenance["treatment"] == "clean"
    assert trace.provenance["conditions"] == scenario.conditions.provenance()
    assert backend.prepares == 0 and backend.resets == backend.closes == 1
