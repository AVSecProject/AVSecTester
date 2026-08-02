"""Execution traces and clean-vs-attacked diffing (PLAN.md Phase 4).

A ``Trace`` is the per-frame record of every instrumented component's I/O for one run.
Two paired runs (attack off/on, same seed+scenario) are diffed to find the first
meaningful divergence, which seeds the Attack Escalation DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentIO:
    frame: int
    stage: str
    component: str
    inputs: Any = None
    outputs: Any = None
    aux: dict[str, Any] = field(default_factory=dict)  # confidence, timing, etc.


@dataclass
class Trace:
    run_id: str
    records: list[ComponentIO] = field(default_factory=list)

    def add(self, io: ComponentIO) -> None:
        self.records.append(io)

    def by_frame(self, frame: int) -> list[ComponentIO]:
        return [r for r in self.records if r.frame == frame]


def diff_traces(clean: Trace, attacked: Trace, **kwargs: Any) -> ComponentIO | None:
    """Return the earliest attacked record that meaningfully diverges from clean.

    TODO(phase4): per-stage comparators (bbox IoU / count deltas for perception, track-id
    churn for tracking, trajectory L2 for planning, control deltas) with tolerances.
    """
    raise NotImplementedError("phase 4: implement per-stage divergence detection")
