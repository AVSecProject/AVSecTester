"""Escalation metric: quantify how an attack propagates to a driving consequence.

Consumes paired clean/attacked traces (see ``avsectester.monitors``) and produces both a
scalar metric dict (activation / propagation / persistence / safety) and the populated
attack-escalation DAG — the framework's central abstraction (PROJECT.md 10, Thrust 1).
"""

from __future__ import annotations

from typing import Any

from ..config import METRICS
from ..core.escalation import EscalationDAG, EscalationEdge, EscalationNode, Stage
from ..core.interfaces import MetricBase
from ..monitors.trace import Trace

# stages we compare, in propagation order, with a human-readable error description
_STAGE_DESCR = {
    Stage.ATTACK_SURFACE: ("perception_input", "injected object(s) at the perception input"),
    Stage.PERCEPTION: ("detector", "phantom detection(s) produced"),
    Stage.TRACKING: ("tracker", "phantom track(s) confirmed"),
    Stage.CONTROL: ("controller", "unwarranted brake command"),
    Stage.CONSEQUENCE: ("ego", "ego speed diverges (unsafe slow/stop)"),
}


@METRICS.register_module()
class EscalationMetric(MetricBase):
    name = "escalation"

    def __init__(self, speed_tol: float = 1.0, stop_speed: float = 0.5, warmup: int = 15) -> None:
        self.speed_tol = speed_tol
        self.stop_speed = stop_speed
        self.warmup = warmup  # ignore startup-from-rest frames for min-speed metrics

    def _first_divergences(self, clean: Trace, attacked: Trace) -> dict[Stage, dict]:
        ci = clean.index()
        first: dict[Stage, dict] = {}
        for ar in sorted(attacked.records, key=lambda r: r.frame):
            try:
                stage = Stage(ar.stage)
            except ValueError:
                continue
            if stage in first:
                continue
            cr = ci.get((ar.frame, ar.stage))
            ao = ar.outputs if isinstance(ar.outputs, dict) else {}
            co = cr.outputs if (cr and isinstance(cr.outputs, dict)) else {}
            diverged = False
            evidence: dict[str, Any] = {"frame": ar.frame}
            if stage == Stage.CONSEQUENCE:
                delta = (co.get("speed", 0.0)) - (ao.get("speed", 0.0))
                diverged = abs(delta) > self.speed_tol
                evidence.update(speed_attacked=ao.get("speed"), speed_clean=co.get("speed"),
                                speed_drop=round(delta, 2))
            elif stage == Stage.CONTROL:
                diverged = bool(ao.get("braking")) != bool(co.get("braking"))
                evidence.update(braking=ao.get("braking"), hazard_dist=ao.get("hazard_dist"))
            elif "count" in ao:
                extra = ao.get("count", 0) - co.get("count", 0)
                diverged = extra != 0
                evidence.update(clean_count=co.get("count", 0), attacked_count=ao.get("count", 0),
                                delta=extra)
            if diverged:
                first[stage] = evidence
        return first

    def compute(self, clean: Trace, attacked: Trace, **kwargs: Any) -> dict[str, Any]:
        first = self._first_divergences(clean, attacked)
        ordered = [s for s in Stage if s in first]

        # -- build the escalation DAG (chain in propagation order) --
        dag = EscalationDAG()
        prev_id = None
        for s in ordered:
            component, descr = _STAGE_DESCR.get(s, (s.value, ""))
            node = EscalationNode(
                id=f"{s.value}@{first[s]['frame']}", stage=s, component=component,
                description=descr, evidence=first[s],
            )
            dag.add_node(node)
            if prev_id is not None:
                dag.add_edge(EscalationEdge(prev_id, node.id, condition="propagated"))
            prev_id = node.id

        # -- scalar metrics --
        def _speeds_after(trace):
            return [
                r.outputs["speed"]
                for r in trace.records
                if r.stage == Stage.CONSEQUENCE.value
                and isinstance(r.outputs, dict)
                and "speed" in r.outputs
                and r.frame >= self.warmup
            ]

        speeds_att = _speeds_after(attacked)
        speeds_cln = _speeds_after(clean)
        # final speeds come from the full series (last frame)
        all_att = attacked.series(Stage.CONSEQUENCE.value, "speed")
        brake_att = sum(1 for b in attacked.series(Stage.CONTROL.value, "braking") if b)
        brake_cln = sum(1 for b in clean.series(Stage.CONTROL.value, "braking") if b)

        # persistence: frames where attacked has more confirmed tracks than clean
        ci = clean.index()
        persistence = 0
        for ar in attacked.records:
            if ar.stage == Stage.TRACKING.value:
                cr = ci.get((ar.frame, ar.stage))
                c = cr.outputs.get("count", 0) if cr else 0
                if ar.outputs.get("count", 0) > c:
                    persistence += 1

        activated = Stage.TRACKING in first or Stage.PERCEPTION in first
        reached_consequence = Stage.CONSEQUENCE in first
        min_att = min(speeds_att) if speeds_att else None
        min_cln = min(speeds_cln) if speeds_cln else None
        stopped = min_att is not None and min_att < self.stop_speed
        activation_frame = (
            first.get(Stage.TRACKING, first.get(Stage.PERCEPTION, {})).get("frame")
        )

        metrics = {
            "activated": activated,
            "activation_frame": activation_frame,
            "stages_reached": [s.value for s in ordered],
            "propagation_depth": len(ordered),
            "reached_consequence": reached_consequence,
            "persistence_frames": persistence,
            "brake_frames_attacked": brake_att,
            "brake_frames_clean": brake_cln,
            "min_speed_attacked": round(min_att, 2) if min_att is not None else None,
            "min_speed_clean": round(min_cln, 2) if min_cln is not None else None,
            "final_speed_attacked": round(all_att[-1], 2) if all_att else None,
            "stopped": stopped,
            "speed_suppression": (
                round(min_cln - min_att, 2) if (min_att is not None and min_cln is not None) else None
            ),
            # top-line verdict: attack activated AND caused an unsafe driving consequence
            "escalated": bool(activated and reached_consequence and stopped and brake_cln == 0),
        }
        return {"metrics": metrics, "dag": dag}
