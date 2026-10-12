"""Assemble CARLA and the modular stack for single or paired runtime execution."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from avsectester.backend import AVStack
from avsectester.evaluation.component_log import run_logged
from avsectester.interventions import configured_runtime, configured_runtime_pair
from avsectester.plane import Trace
from avsectester.runner import run_pair
from avsectester.runtime import Runtime
from avsectester.simulators.carla import CarlaBackend
from avsectester.stacks.modular import ModularAVStack

__all__ = ["run_scenario", "run_scenario_pair"]


def _stack(scenario: dict) -> ModularAVStack:
    stack = ModularAVStack(scenario["ego"]["pipeline"])
    stack.instrument(stages=("perception",))
    return stack


def _logged_run(backend, stack, frames, *, runtime) -> Trace:
    trace, components = run_logged(backend, stack, frames, runtime=runtime)
    trace.replay_scenario = deepcopy(backend.replay_scenario)
    for record, step in zip(trace.records, components.steps):
        record.n_detections = step.count("perception")
    return trace


def run_scenario(
    scenario: dict,
    attacks: list[dict] | None = None,
    frames: int = 40,
    settle_iters: int = 100,
    patches: list[dict] | None = None,
    *,
    runtime: Runtime | None = None,
) -> Trace:
    """Run once. Interventions are explicit, even when the scene contains convenience fields.

    ``attacks`` and ``patches`` are signature adapters to runtime hooks, not separate execution
    paths. Omitting them and ``runtime`` runs clean. Use ``run_scenario_pair`` to execute a
    configured YAML comparison with shared initial conditions and fresh stacks.
    """
    scenario = deepcopy(scenario)
    adapted = configured_runtime({"attacks": attacks or [], "patches": patches or []})
    runtime = (runtime or Runtime()).with_hooks(adapted.hooks, prepend=True)
    backend = CarlaBackend(scenario, settle_iters=settle_iters)
    try:
        return _logged_run(backend, _stack(scenario), frames, runtime=runtime)
    finally:
        backend.close()


def run_scenario_pair(
    scenario: dict,
    frames: int = 40,
    *,
    clean_runtime: Runtime | None = None,
    attacked_runtime: Runtime | None = None,
    settle_iters: int = 100,
    stack_factory: Callable[[], AVStack] | None = None,
) -> tuple[Trace, Trace]:
    """Run a configured CARLA pair through the common paired runner.

    The attacked runtime defaults to ``runtime``, ``attacks`` and ``patches`` in the scene.
    The default clean runtime retains the configured localizer and seed, without attack hooks.
    An explicit ``clean_runtime`` YAML mapping can define a different baseline treatment.
    Explicit Python runtime arguments replace the corresponding YAML treatment configuration.
    """
    scenario = deepcopy(scenario)
    clean_runtime, attacked_runtime = configured_runtime_pair(
        scenario, clean_runtime=clean_runtime, attacked_runtime=attacked_runtime,
    )
    backend = CarlaBackend(scenario, settle_iters=settle_iters)
    return run_pair(
        backend, stack_factory or (lambda: _stack(scenario)), frames,
        clean_runtime=clean_runtime, attacked_runtime=attacked_runtime, run_fn=_logged_run,
    )
