"""Context — the runtime state handed to an attack/defense at a seam."""

from __future__ import annotations

from dataclasses import dataclass

from .frame import Frame
from .seam import Seam


@dataclass
class Context:
    frame: Frame   # the current frame (ego, sensors, ground_truth, meta)
    seam: Seam     # which seam is firing right now
