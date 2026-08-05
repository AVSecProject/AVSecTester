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
from ..core.interfaces import DefenseBase


@DEFENSES.register_module()
class ScoreGateDefense(DefenseBase):
    seam = "perception_input"  # gates the detector input (same surface as the attack)

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def apply(self, data: Any, ego_state: Any = None, **kwargs: Any) -> Any:
        kept = [o for o in data if getattr(o, "score", 1.0) >= self.threshold]
        from avstack.datastructs import DataContainer

        return DataContainer(data.frame, data.timestamp, kept, getattr(data, "source_identifier", "def"))
