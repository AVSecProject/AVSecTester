"""Compose a CARLA backend and modular AV stack for a clean or attacked run."""

from __future__ import annotations

from copy import deepcopy

from avsectester.evaluation.component_log import run_logged
from avsectester.plane import Trace
from avsectester.simulators.carla import CarlaBackend
from avsectester.stacks.modular import ModularAVStack

__all__ = ["run_scenario"]


def run_scenario(
    scenario: dict,
    attacks: list[dict] | None = None,
    frames: int = 40,
    settle_iters: int = 100,
    patches: list[dict] | None = None,
) -> Trace:
    """Run the CARLA + modular demo through the generic interface and return the driving Trace.

    Assembles a :class:`~avsectester.simulators.carla.CarlaBackend` (world + ego + traffic) and a
    :class:`~avsectester.stacks.modular.ModularAVStack` (the AV box), attaches any modular ``attacks``
    as hooks on the stack's pipeline, and drives them with :func:`run_logged`. ``patches``
    are world-level physical-patch attacks applied by the backend at reset (pass them only for the
    attacked run). ``replay_scenario`` (actual spawn transforms) is carried on the returned Trace for a
    paired run; strict-spawn replay is always used.
    """
    scenario = deepcopy(scenario)
    backend = CarlaBackend(scenario, settle_iters=settle_iters, patches=patches)
    stack = ModularAVStack(scenario["ego"]["pipeline"])
    for atk in attacks or []:
        stack.attach(atk["stage"], atk["hook"])
    stack.instrument(stages=("perception",))  # per-frame detection telemetry via the common log interface
    try:
        trace, components = run_logged(backend, stack, frames)
        trace.replay_scenario = backend.replay_scenario
        for record, step in zip(trace.records, components.steps):
            record.n_detections = step.count("perception")
    finally:
        backend.close()
    return trace
