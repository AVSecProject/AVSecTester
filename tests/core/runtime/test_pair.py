"""Paired origin preparation, independent stack history and failure cleanup."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from avsectester.backend import AVStack, run
from avsectester.interventions import ObservationTransform
from avsectester.plane import Control
from avsectester.runner import run_pair
from avsectester.runtime import Hook, Runtime
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
