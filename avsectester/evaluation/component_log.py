"""Component-level logging — gather per-frame, per-layer stack outputs for in-system attack analysis.

The stack already knows how to expose its component outputs (``ModularAVStack.instrument`` attaches an
avstack post-hook per stage and ``component_log()`` returns ``{stage: output}``). This module just
*gathers* those per-frame snapshots across a run and *processes* them. :func:`run_logged` is the
instrumented twin of :func:`avsectester.backend.run` — same loop (via its ``on_step`` hook), but it also
collects the stack's ``component_log()`` each frame into a :class:`ComponentTrace`.

The analysis is deliberately thin and reuses avstack: per-stage **counts** come from ``len()`` of the raw
output (no new schema), and per-stage **performance vs ground truth** reuses
``avstack.metrics.get_instantaneous_metrics`` (TP/FP/FN by nearest-neighbour assignment). The headline
in-system signal is the **clean-vs-attacked** per-stage diff — which layer the attack first changes, and
how it ripples forward — which needs no ground truth.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from avsectester.backend import run as _run_loop

if TYPE_CHECKING:
    from avsectester.backend import AVStack, WorldBackend
    from avsectester.plane import Observation, Trace
    from avsectester.runtime import Runtime


@runtime_checkable
class InstrumentedStack(Protocol):
    """The **common component-logging interface** — the one thing a stack adds to be loggable.

    ``component_log() -> {component_name: free_output} | None`` returns the latest frame's per-component
    outputs, or None when the stack is not instrumented. It is deliberately a plain dict of the stack's
    *already-produced* ("free") outputs — no bespoke schema — so it fits either stack shape:

    * **modular** (``ModularAVStack``): one key per pipeline stage, ``{"perception": detections,
      "tracking": tracks, "planning": plan, "control": control}`` (captured via avstack post-hooks).
    * **end-to-end** (``AlpamayoAVStack``): ``{"policy": prediction, "action": control}`` from what the
      policy already returns (candidate trajectories + the emitted command). Sparse, but the same contract.

    :func:`run_logged` collects whatever keys a stack emits. :class:`ComponentTrace` processes them
    uniformly (counts/degradation per key). Keeping the interface alive for the E2E stack now means the
    logger and report need no change when the E2E stack later exposes more.
    """

    def component_log(self) -> dict[str, Any] | None:
        ...


def _count(output: Any) -> int:
    """Size of a stage output: ``len`` for a detections/tracks container, else 1 if present, 0 if None."""
    if output is None:
        return 0
    try:
        return len(output)
    except TypeError:
        return 1


@dataclass
class StepLog:
    """One frame's per-stage stack outputs (raw avstack objects), keyed by stage name."""

    frame: int
    stages: dict[str, Any] = field(default_factory=dict)

    def count(self, stage: str) -> int:
        return _count(self.stages.get(stage))


@dataclass
class ComponentTrace:
    """Per-frame component outputs across a run, with thin per-stage processing."""

    steps: list[StepLog] = field(default_factory=list)

    @property
    def stage_names(self) -> list[str]:
        return list(self.steps[0].stages) if self.steps else []

    def counts(self, stage: str) -> list[int]:
        """Per-frame output size of ``stage`` (e.g. n_detections, n_tracks)."""
        return [s.count(stage) for s in self.steps]

    def degradation(self, other: ComponentTrace, stage: str) -> list[int]:
        """Per-frame count drop ``self[stage] - other[stage]`` — the in-system effect of the difference
        between two runs (e.g. ``clean.degradation(attacked, "perception")`` is how many detections the
        attack removed each frame)."""
        return [a - b for a, b in zip(self.counts(stage), other.counts(stage))]

    def performance(self, stage: str, truths: list[Any], assign_radius: float = 4.0) -> list[Any]:
        """Per-frame detection/tracking performance of ``stage`` vs ground truth, reusing avstack's
        ``get_instantaneous_metrics`` (returns its per-frame metrics object. Needs ``truths[i]`` — the GT
        objects for frame ``i`` — in a matching reference frame)."""
        from avstack.metrics import get_instantaneous_metrics

        out = []
        for i, step in enumerate(self.steps):
            objs = step.stages.get(stage)
            data = getattr(objs, "data", objs) or []
            out.append(get_instantaneous_metrics(tracks=list(data), truths=truths[i],
                                                  assign_radius=assign_radius))
        return out


def run_logged(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    perturb: Callable[[Observation], Observation] | None = None,
    *,
    runtime: Runtime | None = None,
) -> tuple[Trace, ComponentTrace]:
    """Drive the scenario like :func:`avsectester.backend.run`, additionally collecting the stack's
    ``component_log()`` each frame. Returns the driving :class:`Trace` and the :class:`ComponentTrace`
    (empty when the stack is not instrumented, so the call site is uniform)."""
    component = ComponentTrace()
    instrumented = isinstance(stack, InstrumentedStack)

    def _capture(i: int, seen: Observation, control: Any) -> None:
        if instrumented:
            # Tracks, plans and their reference frames are mutable across steps. Copy the
            # complete output graph now, while preserving shared references within this step.
            component.steps.append(StepLog(frame=i, stages=deepcopy(stack.component_log() or {})))

    trace = _run_loop(backend, stack, frames, perturb=perturb, on_step=_capture, runtime=runtime)
    return trace, component
