"""Paired execution using the same single-run loop and selected initial origin."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from typing import Any, TYPE_CHECKING, TypeVar, overload

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import WorldBackend
from avsectester.backend import run
from avsectester.closed_loop.hooks import HookSet, Runtime
from avsectester.closed_loop.datatypes import Trace

Result = TypeVar("Result")
RuntimeFactory = HookSet | Callable[[WorldBackend], HookSet] | None

if TYPE_CHECKING:
    from avsectester.scenarios.source import ScenarioInstance


def _driving_trace(result: Any) -> Trace | None:
    if isinstance(result, Trace):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], Trace):
        return result[0]
    return None


@overload
def run_pair(
    backend: WorldBackend, stack_factory: Callable[[], AVStack], frames: int, *,
    clean_runtime: RuntimeFactory = None, attacked_runtime: RuntimeFactory = None,
    run_fn: None = None, close_backend: bool = True,
) -> tuple[Trace, Trace]: ...


@overload
def run_pair(
    backend: WorldBackend, stack_factory: Callable[[], AVStack], frames: int, *,
    clean_runtime: RuntimeFactory = None, attacked_runtime: RuntimeFactory = None,
    run_fn: Callable[..., Result], close_backend: bool = True,
) -> tuple[Result, Result]: ...


def run_pair(
    backend: WorldBackend,
    stack_factory: Callable[[], AVStack],
    frames: int,
    *,
    clean_runtime: RuntimeFactory = None,
    attacked_runtime: RuntimeFactory = None,
    run_fn: Callable[..., Any] | None = None,
    close_backend: bool = True,
) -> tuple[Any, Any]:
    """Run a comparison from one prepared origin, with a fresh stack per run.

    Runtime factories receive the backend before reset. Resolve live actors in world hooks,
    not in those factories. Each runtime gets its own session and lifecycle. Initial deployment
    ticks are matched across runs. The caller owns the backend when ``close_backend=False``.
    Both runtimes are checked against the first stack before clean execution. Stack factories
    must provide the same capabilities for both runs. Each run also checks its actual stack.
    No metric is computed here. ``run_fn`` may use the common recorder or component logger.
    """
    results = _execute_runs(
        backend, stack_factory, frames, (clean_runtime, attacked_runtime),
        run_fn=run_fn, close_backend=close_backend, prepare_origin=True,
    )
    return results[0], results[1]


def _execute_runs(
    backend: WorldBackend,
    stack_factory: Callable[[], AVStack],
    frames: int,
    configured_runtimes: tuple[RuntimeFactory, ...],
    *,
    run_fn: Callable[..., Any] | None,
    close_backend: bool,
    prepare_origin: bool,
) -> list[Any]:
    """Execute single and comparison runs with one validation and cleanup contract."""
    previous_ticks = backend.setup_ticks
    failure = None
    try:
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
            raise ValueError("frames must be a nonnegative integer")
        stack = stack_factory()
        for configured in configured_runtimes:
            if isinstance(configured, HookSet):
                configured.validate_stages(backend, stack)
            elif configured is None:
                Runtime().validate_stages(backend, stack)
            elif configured is not None and not callable(configured):
                raise TypeError("Expected Runtime or a runtime factory")
        if prepare_origin:
            backend.prepare_clean_attack_pair()
        runtimes = []
        for configured in configured_runtimes:
            runtime = configured(backend) if callable(configured) else configured
            if runtime is None:
                runtime = Runtime()
            if not isinstance(runtime, HookSet):
                raise TypeError("A runtime factory must return Runtime")
            runtime.validate_stages(backend, stack)
            runtimes.append(runtime)
        backend.setup_ticks = max(previous_ticks, *(runtime.setup_ticks for runtime in runtimes))
        results: list[Any] = []
        for index, runtime in enumerate(runtimes):
            if index:
                stack = stack_factory()
            result = (run_fn or run)(backend, stack, frames, runtime=runtime)
            trace = _driving_trace(result)
            if trace is not None:
                if trace.replay_scenario is None:
                    trace.replay_scenario = deepcopy(getattr(backend, "replay_scenario", None))
                trace.provenance.update({
                    "runtime_seed": runtime.seed,
                    "runtime_stages": [hook.stage for hook in runtime.hooks],
                })
            results.append(result)
        return results
    except BaseException as error:
        failure = error
        raise
    finally:
        backend.setup_ticks = previous_ticks
        if close_backend:
            try:
                backend.close()
            except Exception as error:
                if failure is None:
                    raise
                failure.add_note(f"Backend cleanup also failed: {error!r}")


def run_prepared(
    scenario: ScenarioInstance,
    stack_factory: Callable[[], AVStack],
    frames: int,
    *,
    runtimes: Mapping[str, RuntimeFactory] | None = None,
    run_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run named treatments of one ScenarioInstance and return results by treatment name.

    One treatment runs once. Multiple treatments replay the same prepared origin with fresh
    stacks. Conditions are applied to every treatment with fresh sensor handlers. Existing
    explicit runtime cases are preserved, otherwise the scenario's optional match is supplied.
    ScenarioInstance.make_backend applies world conditions. This function owns backend teardown.
    """
    return _run_prepared(scenario, stack_factory, frames, runtimes=runtimes, run_fn=run_fn)


def _run_prepared(
    scenario: ScenarioInstance,
    stack_factory: Callable[[], AVStack],
    frames: int,
    *,
    runtimes: Mapping[str, RuntimeFactory] | None = None,
    run_fn: Callable[..., Any] | None = None,
    backend_ready: Callable[[WorldBackend], bool] | None = None,
) -> dict[str, Any]:
    """Shared execution with optional legacy initialization before origin preparation.

    The initializer sees a backend with world conditions configured. False skips execution
    and closes the backend. Keeping this adapter internal preserves the public preparation API.
    """
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise ValueError("frames must be a nonnegative integer")
    configured = dict(runtimes) if runtimes is not None else {"clean": None}
    if not configured or any(not isinstance(name, str) or not name for name in configured):
        raise ValueError("Provide at least one named treatment")

    def adapt(runtime: HookSet | None) -> HookSet:
        runtime = runtime or Runtime()
        if not isinstance(runtime, HookSet):
            raise TypeError("A runtime factory must return Runtime")
        if runtime.case is None:
            runtime = copy(runtime)
            runtime.case = scenario.match
        return scenario.conditions.apply_runtime(runtime)

    effective: list[RuntimeFactory] = []

    def adapt_factory(factory: Callable[[WorldBackend], HookSet]) -> Callable[[WorldBackend], HookSet]:
        def build(backend: WorldBackend) -> HookSet:
            return adapt(factory(backend))
        return build

    for runtime in configured.values():
        if runtime is None or isinstance(runtime, HookSet):
            effective.append(adapt(runtime))
        elif callable(runtime):
            effective.append(adapt_factory(runtime))
        else:
            raise TypeError("Expected Runtime or a runtime factory")
    backend = scenario.make_backend()
    if backend_ready is not None:
        try:
            proceed = backend_ready(backend)
        except BaseException as error:
            try:
                backend.close()
            except Exception as cleanup:
                error.add_note(f"Backend cleanup also failed: {cleanup!r}")
            raise
        if not proceed:
            backend.close()
            return {}
    results = _execute_runs(
        backend, stack_factory, frames, tuple(effective), run_fn=run_fn,
        close_backend=True, prepare_origin=len(configured) > 1,
    )
    named = dict(zip(configured, results))
    for name, result in named.items():
        trace = _driving_trace(result)
        if trace is not None:
            trace.provenance = {
                **trace.provenance, **deepcopy(scenario.provenance), "treatment": name,
                "conditions": scenario.conditions.provenance(),
            }
    return named
