"""Assemble CARLA and the modular stack for single or paired runtime execution."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from avsectester.backend import AVStack, run
from avsectester.experiment import ExecutionMode, TreatmentMode, configured_stack, run_configured
from avsectester.plane import Trace
from avsectester.runtime import Runtime
from avsectester.simulators.carla import CarlaBackend
from avsectester.stacks.modular import ModularAVStack

__all__ = ["run_scenario", "run_scenario_pair"]


def _stack(scenario: dict) -> AVStack:
    if scenario.get("stack") is not None:
        return configured_stack(scenario)
    stack = ModularAVStack(scenario["ego"]["pipeline"])
    stack.instrument(stages=("perception",))
    return stack


def _logged_run(backend, stack, frames, *, runtime) -> Trace:
    trace = run(backend, stack, frames, runtime=runtime)
    trace.replay_scenario = deepcopy(backend.replay_scenario)
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
    # Signature adapters still enable only explicitly supplied interventions for one run.
    scenario["attacks"], scenario["patches"] = attacks or [], patches or []
    scenario.pop("interventions", None)
    scenario.pop("runtime", None)
    scenario.pop("clean_runtime", None)
    from avsectester.scenarios.setup import scenario_from_config

    instance = scenario_from_config(
        scenario,
        backend_factory=None if scenario.get("selection") is not None else
        lambda config: CarlaBackend(config, settle_iters=settle_iters),
    )
    if runtime is not None:
        # Prepare the convenience methods and prepend them to the caller's Runtime.
        from avsectester.interventions import prepare_from_config
        runtime = prepare_from_config(scenario, instance).runtime(runtime, runtime.case)
    treatment = TreatmentMode.ATTACK if runtime is not None or attacks or patches else TreatmentMode.CLEAN
    results = run_configured(
        scenario, frames, mode=ExecutionMode.SINGLE,
        treatment=treatment, treated_runtime=runtime,
        backend_factory=lambda config: CarlaBackend(config, settle_iters=settle_iters),
        stack_factory=lambda: _stack(scenario), run_fn=_logged_run,
        scenario=instance,
    )
    return results[treatment.value]


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
    results = run_configured(
        scenario, frames, mode=ExecutionMode.PAIR, treatment=TreatmentMode.ATTACK,
        clean_runtime=clean_runtime, treated_runtime=attacked_runtime,
        backend_factory=None if scenario.get("selection") is not None else
        lambda config: CarlaBackend(config, settle_iters=settle_iters),
        stack_factory=stack_factory or (lambda: _stack(scenario)), run_fn=_logged_run,
    )
    return results["clean"], results["attack"]
