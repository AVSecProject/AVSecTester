"""Baseline mock defense: confidence gate on the perception input (PLAN.md Phase 5).

Drops perception-input objects whose detection ``score`` is below a threshold — the
simplest realistic sanitizer. It attaches at the exact same perception-input seam as the
attack (avstack pre-hook shape ``apply(data, ego_state=...) -> data``), so the engine can
run an attacked-and-defended pass to measure mitigation.

Baseline only: a real defense would use raw-sensor corroboration / temporal consistency /
physical plausibility rather than a single confidence value. See DefenseBase.
"""

from __future__ import annotations

from typing import Any

from ..config import DEFENSES
from ..core.interfaces import DefenseBase, DefenseOutcome


@DEFENSES.register_module()
class ScoreGateDefense(DefenseBase):
    category = "input_sanitize"
    # Gates the passthrough detector's object-level input by confidence. Only meaningful on a
    # ground-truth stack (objects carry a score at the input); the neural-stack counterpart
    # gates detections at perception_out (added with the perception-output defenses).
    seams = ("perception_input",)

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def apply(self, data: Any, ego_state: Any = None, ctx: Any = None, **kwargs: Any) -> Any:
        from avstack.datastructs import DataContainer

        kept, dropped = [], []
        for o in data:
            (kept if getattr(o, "score", 1.0) >= self.threshold else dropped).append(o)
        self.record_outcome(
            ctx,
            DefenseOutcome(
                seam=self.current_seam(ctx) or "perception_input",
                frame=getattr(ctx, "frame", 0),
                kept=len(kept),
                dropped=[getattr(o, "ID", None) for o in dropped],
                reason=f"score < {self.threshold}",
            ),
        )
        return DataContainer(data.frame, data.timestamp, kept, getattr(data, "source_identifier", "def"))
