"""Execution traces and clean-vs-attacked diffing (PLAN.md Phase 4).

A ``Trace`` is the per-frame record of every instrumented stage's I/O for one run.
Backends emit a standardized per-frame record dict; ``build_trace`` lifts a stream of
those into a ``Trace`` of per-stage ``ComponentIO``. Two paired runs (attack off/on, same
scenario) are diffed to find the first meaningful divergence, which seeds the Escalation DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.escalation import Stage
from ..core.interfaces import MonitorBase


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

    def index(self) -> dict[tuple[int, str], ComponentIO]:
        return {(r.frame, r.stage): r for r in self.records}

    def series(self, stage: str, key: str) -> list:
        return [
            r.outputs[key]
            for r in self.records
            if r.stage == stage and isinstance(r.outputs, dict) and key in r.outputs
        ]


# stage order for "earliest divergence" ranking
_STAGE_ORDER = {s.value: i for i, s in enumerate(Stage)}


def record_to_ios(rec: dict) -> list[tuple[str, str, dict]]:
    """Map a backend per-frame record to (stage, component, outputs) tuples."""
    return [
        (Stage.ATTACK_SURFACE.value, "perception_input", {"count": rec["n_input"]}),
        (Stage.PERCEPTION.value, "detector", {"count": rec["n_detections"]}),
        (Stage.TRACKING.value, "tracker", {"count": rec["n_tracks"]}),
        (
            Stage.CONTROL.value,
            "controller",
            {
                "braking": rec["braking"],
                "throttle": rec["throttle"],
                "brake": rec["brake"],
                "hazard_dist": rec["hazard_dist"],
            },
        ),
        (Stage.CONSEQUENCE.value, "ego", {"speed": rec["ego_speed"]}),
    ]


class TraceMonitor(MonitorBase):
    """Collects per-stage I/O into a ``Trace``.

    Usable two ways: ``ingest_record(rec)`` for the standardized backend record stream
    (what the engine uses), or the generic ``begin_frame`` + ``observe`` hook contract for
    future in-pipeline (avstack-hook) instrumentation.
    """

    def __init__(self, run_id: str = "run") -> None:
        self.trace = Trace(run_id)
        self._frame = 0

    def begin_frame(self, frame: int) -> None:
        self._frame = frame

    def observe(self, stage: str, component: str, data: Any) -> None:
        self.trace.add(ComponentIO(self._frame, stage, component, outputs=data))

    def ingest_record(self, rec: dict) -> None:
        for stage, component, outputs in record_to_ios(rec):
            self.trace.add(ComponentIO(rec["frame"], stage, component, outputs=outputs))


def build_trace(records: list[dict], run_id: str = "run") -> Trace:
    """Lift a stream of backend per-frame records into a Trace."""
    mon = TraceMonitor(run_id)
    for rec in records:
        mon.ingest_record(rec)
    return mon.trace


def _diverges(stage: str, clean: ComponentIO | None, attacked: ComponentIO, speed_tol: float) -> bool:
    ao = attacked.outputs if isinstance(attacked.outputs, dict) else {}
    co = clean.outputs if (clean and isinstance(clean.outputs, dict)) else {}
    if stage == Stage.CONSEQUENCE.value:
        return abs(ao.get("speed", 0.0) - co.get("speed", 0.0)) > speed_tol
    if stage == Stage.CONTROL.value:
        return bool(ao.get("braking")) != bool(co.get("braking"))
    if "count" in ao:
        return ao.get("count", 0) != co.get("count", 0)
    return False


def diff_traces(clean: Trace, attacked: Trace, speed_tol: float = 1.0) -> ComponentIO | None:
    """Return the earliest attacked record that meaningfully diverges from clean."""
    ci = clean.index()
    ordered = sorted(
        attacked.records, key=lambda r: (r.frame, _STAGE_ORDER.get(r.stage, 99))
    )
    for ar in ordered:
        if _diverges(ar.stage, ci.get((ar.frame, ar.stage)), ar, speed_tol):
            return ar
    return None
