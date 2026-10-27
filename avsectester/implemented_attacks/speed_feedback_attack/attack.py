"""A feedback attack that changes the ego speed seen by the driving system."""
from __future__ import annotations

import math

from avsectester.attack.attack import Attack
from avsectester.attack.segments import validate_segment_options
from .metabuilder import build_metadata
from .online import SpeedFeedbackAttackRuntime


class SpeedFeedbackAttack(Attack):
    """Adapt a nonnegative model-visible speed bias from actual physical speed."""

    def __init__(self, *, decision_steps: int, target_speed: float, gain: float = 0.1,
                 max_bias: float = 5.0, initial_bias: float = 0.0,
                 max_segments: int | None = None, keep_tail: bool = False):
        validate_segment_options(decision_steps, max_segments, keep_tail)
        self.decision_steps, self.max_segments, self.keep_tail = decision_steps, max_segments, keep_tail
        self.target_speed, self.gain = float(target_speed), float(gain)
        self.max_bias, self.initial_bias = float(max_bias), float(initial_bias)
        values = (self.target_speed, self.gain, self.max_bias, self.initial_bias)
        if (not all(math.isfinite(value) for value in values)
                or min(values) < 0 or self.initial_bias > self.max_bias):
            raise ValueError("Speed-feedback parameters must be finite, nonnegative and ordered")

    def build_metadata(self, scenario):
        return build_metadata(self, scenario)

    def create_runtime(self, segment, artifacts):
        return SpeedFeedbackAttackRuntime(self)


__all__ = [
    'SpeedFeedbackAttack',
]
