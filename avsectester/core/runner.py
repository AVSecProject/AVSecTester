"""Runner — drive an Environment through a System, and run paired experiments.

The runtime loop is identical for dataset replay and simulation: the system processes each
frame (attacks fire at its seams), the resulting control is fed back to the environment, and
the environment yields the next frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .environment import Environment
from .metric import Metric
from .system import System
from .trace import Trace


def run(env: Environment, system: System, run_id: str = "run") -> Trace:
    """Drive ``env`` through ``system`` to completion, collecting a Trace."""
    trace = Trace(run_id=run_id)
    frame = env.reset()
    done = False
    try:
        while not done:
            out = system.process(frame)
            trace.add(out.record)
            frame, done = env.step(out.control)
    finally:
        system.close()
        env.close()
    return trace


@dataclass
class Result:
    clean: Trace
    attacked: Trace
    metrics: dict[str, Any]
    defended: Trace | None = None
    defended_metrics: dict[str, Any] | None = None


def run_experiment(
    make_env: Callable[[], Environment],
    make_system: Callable[[], System],
    metric: Metric,
    attack: Any = None,
    defense: Any = None,
) -> Result:
    """Run clean, attacked, and (if a defense is given) defended passes, and score them.

    ``make_env`` / ``make_system`` are factories so each pass gets a fresh env + system.
    """
    clean = run(make_env(), make_system(), "clean")

    sys_a = make_system()
    if attack is not None:
        attack.reset()
        sys_a.attach(attack)
    attacked = run(make_env(), sys_a, "attacked")
    res = Result(clean, attacked, metric.compute(clean, attacked))

    if attack is not None and defense is not None:
        sys_d = make_system()
        attack.reset()
        sys_d.attach(attack)
        defense.reset()
        sys_d.attach(defense)
        res.defended = run(make_env(), sys_d, "defended")
        res.defended_metrics = metric.compute(clean, res.defended)
    return res
