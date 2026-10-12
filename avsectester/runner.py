"""Paired execution using the same single-run loop and selected initial origin."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

from avsectester.backend import AVStack, WorldBackend, run
from avsectester.runtime import Runtime
from avsectester.plane import Trace

Result = TypeVar("Result")
RuntimeFactory = Runtime | Callable[[WorldBackend], Runtime] | None


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
    No metric is computed here. ``run_fn`` may use the common recorder or component logger.
    """
    previous_ticks = backend.setup_ticks
    failure = None
    try:
        if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
            raise ValueError("frames must be a nonnegative integer")
        backend.prepare_clean_attack_pair()
        runtimes = []
        for configured in (clean_runtime, attacked_runtime):
            runtime = configured(backend) if callable(configured) else configured
            if runtime is None:
                runtime = Runtime()
            if not isinstance(runtime, Runtime):
                raise TypeError("A runtime factory must return Runtime")
            runtimes.append(runtime)
        backend.setup_ticks = max(previous_ticks, *(runtime.setup_ticks for runtime in runtimes))
        results: list[Any] = []
        for runtime in runtimes:
            results.append((run_fn or run)(backend, stack_factory(), frames, runtime=runtime))
        return results[0], results[1]
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
