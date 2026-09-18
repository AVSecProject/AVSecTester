"""Data sources feeding the attack. EoT (Expectation-over-Transformation) lives here, not in PGD.

``FrameSource`` holds pre-captured (background, target, placement) samples and, on each ``sample``,
draws a random subset and freshly jitters the per-sample photometric context (brightness/contrast) —
so a patch optimized over it survives varied appearance. One frame with no jitter degrades to a plain
single-image attack.
"""

from __future__ import annotations

import random

from .interface import AdvSample, DataSource


class FrameSource(DataSource):
    """EoT over a fixed set of captured frames + random photometric jitter per draw."""

    def __init__(self, samples: list[AdvSample], brightness=(0.8, 1.2), contrast=(0.8, 1.2), seed=0):
        if not samples:
            raise ValueError("FrameSource needs at least one AdvSample")
        self.samples = samples
        self.brightness = brightness
        self.contrast = contrast
        self._rng = random.Random(seed)

    def sample(self, batch: int) -> list[AdvSample]:
        out = []
        for _ in range(batch):
            base = self._rng.choice(self.samples)
            ctx = dict(base.context)
            ctx["brightness"] = self._rng.uniform(*self.brightness)
            ctx["contrast"] = self._rng.uniform(*self.contrast)
            out.append(AdvSample(x=base.x, target=base.target, context=ctx))
        return out
