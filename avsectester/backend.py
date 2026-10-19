"""Legacy driving-loop entry point, removed in phase 4.

``run`` drives one legacy run through :class:`~avsectester.closed_loop.runner.ClosedLoopRunner`.
The legacy experiment path and :mod:`avsectester.evaluation` still call it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avsectester.closed_loop.avstack import AVStack
    from avsectester.closed_loop.datatypes import Control, Observation, Trace
    from avsectester.closed_loop.hooks import HookSet
    from avsectester.closed_loop.simulator import WorldBackend

__all__ = ["run"]


def run(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    perturb: Callable[[Observation], Observation] | None = None,
    on_step: Callable[[int, Observation, Control], None] | None = None,
    *,
    runtime: HookSet | None = None,
) -> Trace:
    """Drive ``stack`` in ``backend`` for ``frames`` steps and return the driving :class:`Trace`.

    ``runtime`` configures ordered stage handlers and their per-run lifecycle. ``perturb`` is
    an optional one-argument observation transform, applied before runtime observation handlers.
    Both act on model-visible data independently of the backend's physical state. The Trace
    records true backend time and speed and the final command applied to the world.

    ``on_step(i, seen, control)`` receives independent copies after command handlers and before
    actuation. The caller retains ownership of the backend and is responsible for closing it.
    """
    from avsectester.closed_loop.runner import run_trace

    return run_trace(backend, stack, frames, perturb, on_step, hooks=runtime)
