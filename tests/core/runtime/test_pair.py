"""Paired origin preparation, independent stack history and failure cleanup."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from avsectester.backend import AVStack, run
from avsectester.interventions import ObservationTransform, configured_runtime_pair
from avsectester.plane import Control, StateEstimate
from avsectester.runner import run_pair
from avsectester.runtime import Hook, Plugin, Runtime, UnsupportedStageError
from avsectester.simulators.nurec import NuRecBackend, StubRenderer


def test_pair_uses_prepared_origin_fresh_stacks_and_separate_runtime_sessions():
    backend = NuRecBackend({"ego0": {"x": 7, "speed": 2}}, renderer=StubRenderer(height=2, width=2))
    backend.prepare_clean_attack_pair = Mock()
    backend.close = Mock(wraps=backend.close)
    histories = []

    class Stack(AVStack):
        def __init__(self):
            self.history = []
            histories.append(self.history)

        def __call__(self, observation):
            self.history.append((observation.vehicle_state.x, observation.ego_speed))
            return Control()

    spoof = lambda observation: replace(observation, ego_speed=10)
    clean, attacked = run_pair(backend, Stack, 2, attacked_runtime=Runtime([
        Hook("observation", ObservationTransform(spoof)),
    ]))
    backend.prepare_clean_attack_pair.assert_called_once()
    backend.close.assert_called_once()
    assert histories == [[(7, 2), (7.1, 2)], [(7, 10), (7.1, 10)]]
    assert [record.speed for record in clean.records] == [record.speed for record in attacked.records]


@pytest.mark.parametrize("fail_run", [1, 2])
def test_pair_releases_backend_if_either_run_fails(fail_run):
    backend = NuRecBackend({}, renderer=StubRenderer(height=2, width=2))
    backend.close = Mock(side_effect=RuntimeError("cleanup failed"))
    calls = 0

    def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_run:
            raise RuntimeError("failed run")
        return run(*args, **kwargs)

    class Stack(AVStack):
        def __call__(self, observation):
            return Control()

    with pytest.raises(RuntimeError, match="failed run") as failure:
        run_pair(backend, Stack, 1, run_fn=execute)
    backend.close.assert_called_once()
    assert "cleanup failed" in failure.value.__notes__[0]


@pytest.mark.parametrize("treatment", ["clean", "attacked"])
@pytest.mark.parametrize("from_factory", [False, True])
def test_pair_rejects_unsupported_stages_before_any_run(treatment, from_factory):
    backend = NuRecBackend({}, renderer=StubRenderer(height=2, width=2))
    backend.prepare_clean_attack_pair = Mock()
    backend.reset = Mock(wraps=backend.reset)
    backend.close = Mock()
    backend.setup_ticks = 2
    handler = Mock(spec=Plugin)
    invalid = Runtime([Hook("percepton.post", handler)])

    class Stack(AVStack):
        def __call__(self, observation):
            return Control()

    def factory(prepared_backend):
        assert prepared_backend is backend
        backend.prepare_clean_attack_pair.assert_called_once()
        return invalid

    with pytest.raises(UnsupportedStageError, match="percepton.post"):
        run_pair(backend, Stack, 1, **{
            f"{treatment}_runtime": factory if from_factory else invalid,
        })
    assert backend.prepare_clean_attack_pair.call_count == int(from_factory)
    backend.reset.assert_not_called()
    handler.reset.assert_not_called()
    handler.transform.assert_not_called()
    handler.close.assert_not_called()
    backend.close.assert_called_once()
    assert backend.setup_ticks == 2


def test_pair_validates_all_stage_sources_without_reusing_stack_or_handler_history():
    backend = NuRecBackend({}, renderer=StubRenderer(height=2, width=2))
    backend.prepare_clean_attack_pair = Mock()
    constructed, resets, events = [], [], []

    class Stack(AVStack):
        @property
        def supported_stages(self):
            return frozenset({"policy.post"})

        def __init__(self):
            constructed.append(self)

        def __call__(self, observation):
            return self._emit("policy.post", Control())

    class Handler(Plugin):
        def reset(self, context):
            self.seen = []
            resets.append(context.stack)

        def transform(self, value, context):
            self.seen.append(context.stage)
            return value

    handler = Handler()
    runtime = Runtime([
        Hook(stage, handler)
        for stage in ("world.setup", "policy.post", "localization.pre", "command")
    ], localizer=lambda observation, context: StateEstimate(
        observation.vehicle_state, observation.ego_speed,
    ))

    def execute(backend, stack, frames, *, runtime):
        result = run(backend, stack, frames, runtime=runtime)
        events.append(tuple(handler.seen))
        return result

    run_pair(backend, Stack, 1, clean_runtime=runtime, attacked_runtime=runtime, run_fn=execute)
    assert len(constructed) == 2 and constructed[0] is not constructed[1]
    assert resets == constructed
    assert events == [("world.setup", "localization.pre", "policy.post", "command")] * 2
    backend.prepare_clean_attack_pair.assert_called_once()


@pytest.mark.parametrize("baseline", ["default", "yaml", "python"])
def test_configured_pair_respects_explicit_baseline_and_python_runtime(baseline):
    localizer = lambda observation, context: StateEstimate()
    attacked = Runtime([Hook("command", lambda value, context: value)],
                       localizer=localizer, case=object(), seed=7)
    scene = {"runtime": {"seed": 99}}
    clean_override = None
    if baseline == "yaml":
        scene["clean_runtime"] = {"seed": 11}
    elif baseline == "python":
        scene["clean_runtime"] = {"seed": 11}
        clean_override = Runtime(seed=13)
    clean, actual_attacked = configured_runtime_pair(
        scene, attacked_runtime=attacked, clean_runtime=clean_override,
    )
    assert actual_attacked is attacked
    assert clean.hooks == ()
    if baseline == "default":
        assert clean.seed == 7 and clean.localizer is localizer and clean.case is attacked.case
    elif baseline == "yaml":
        assert clean.seed == 11 and clean.localizer is None
    else:
        assert clean is clean_override
