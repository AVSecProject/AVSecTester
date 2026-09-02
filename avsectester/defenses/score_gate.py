"""Baseline defense: confidence gate on the perception input.

Drops object-level inputs whose ``score`` is below a threshold — the simplest realistic
sanitizer. It attaches at the same ``perception_input`` seam as the object-level attacks, so
the runner can run an attacked-and-defended pass to measure mitigation. Baseline only: a real
defense would use raw-sensor corroboration / temporal consistency / physical plausibility.
"""

from __future__ import annotations

from typing import Any

from ..config import DEFENSES
from ..core.attack import Defense
from ..core.context import Context
from ..core.seam import Seam


@DEFENSES.register_module()
class ScoreGateDefense(Defense):
    seams = (Seam.PERCEPTION_INPUT,)

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def apply(self, payload: Any, ctx: Context) -> Any:
        from avstack.datastructs import DataContainer

        kept = [o for o in payload if getattr(o, "score", 1.0) >= self.threshold]
        return DataContainer(
            payload.frame, payload.timestamp, kept, getattr(payload, "source_identifier", "def")
        )
