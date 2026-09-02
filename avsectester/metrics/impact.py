"""ImpactMetric — did the attack change the driving outcome (vs the clean run)?

A minimal, DAG-free comparison of two traces: whether the attacked run brakes/stops when the
clean run does not. (Richer escalation/attribution analysis is intentionally left out — to be
added later per the project owner's design.)
"""

from __future__ import annotations

from typing import Any

from ..config import METRICS
from ..core.metric import Metric
from ..core.trace import Trace


@METRICS.register_module()
class ImpactMetric(Metric):
    def __init__(self, warmup: int = 15, stop_speed: float = 0.5) -> None:
        self.warmup = warmup
        self.stop_speed = stop_speed

    def _min_speed(self, trace: Trace) -> float | None:
        speeds = [s for s in trace.series("ego_speed")[self.warmup:] if s is not None]
        return min(speeds) if speeds else None

    def compute(self, clean: Trace, attacked: Trace) -> dict[str, Any]:
        brake_cln = clean.count("braking")
        brake_att = attacked.count("braking")
        min_cln = self._min_speed(clean)
        min_att = self._min_speed(attacked)
        final_att = attacked.series("ego_speed")[-1] if len(attacked) else None
        stopped = min_att is not None and min_att <= self.stop_speed
        return {
            "brake_frames_clean": brake_cln,
            "brake_frames_attacked": brake_att,
            "min_speed_clean": round(min_cln, 2) if min_cln is not None else None,
            "min_speed_attacked": round(min_att, 2) if min_att is not None else None,
            "final_speed_attacked": round(final_att, 2) if final_att is not None else None,
            "stopped": stopped,
            "speed_suppression": (
                round(min_cln - min_att, 2) if (min_cln is not None and min_att is not None) else None
            ),
            # top-line verdict: the attack induced braking + a stop the clean run never had
            "impacted": bool(brake_att > brake_cln and stopped and brake_cln == 0),
        }
