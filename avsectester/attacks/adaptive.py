"""Small feedback-driven attack illustrating the preparation and runtime interfaces."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from avsectester.closed_loop.datatypes import Observation
from avsectester.preparation import Intervention
from avsectester.closed_loop.hooks import Hook, Plugin, RunContext, StageContext


class SpeedFeedbackPlugin(Plugin):
    """Update a reported-speed bias from physical-speed feedback after each driving step.

    This is an interface example rather than a GNSS model or a claim of attack effectiveness.
    The physical state is read only. Only the model-visible speed measurement is changed.
    """

    def __init__(
        self, target_speed: float, gain: float = 0.1, max_bias: float = 5.0,
        initial_bias: float = 0.0,
    ) -> None:
        values = (target_speed, gain, max_bias, initial_bias)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Speed-feedback parameters must be finite")
        if target_speed < 0 or gain < 0 or max_bias < 0 or not 0 <= initial_bias <= max_bias:
            raise ValueError("Speeds, gain and bias bounds must be nonnegative and ordered")
        self.target_speed, self.gain = float(target_speed), float(gain)
        self.max_bias, self.initial_bias = float(max_bias), float(initial_bias)
        self.bias = self.initial_bias
        self.feedback: list[float] = []

    def reset(self, context: RunContext) -> None:
        self.bias = self.initial_bias
        self.feedback = []

    def transform(self, value: Observation, context: StageContext) -> Observation:
        return replace(value, ego_speed=value.ego_speed + self.bias)

    def after_step(self, context: StageContext) -> None:
        speed = context.ground_truth.ego_speed
        self.feedback.append(speed)
        self.bias = max(0.0, min(self.max_bias, self.bias + self.gain * (
            self.target_speed - speed
        )))


class SpeedFeedbackAttack(Intervention):
    """Configure a feedback attack and instantiate independent handlers for each run."""

    def __init__(
        self, target_speed: float, gain: float = 0.1, max_bias: float = 5.0,
        initial_bias: float = 0.0,
    ) -> None:
        self.options = dict(target_speed=target_speed, gain=gain, max_bias=max_bias,
                            initial_bias=initial_bias)
        SpeedFeedbackPlugin(**self.options)

    def prepare(self, scenario) -> dict[str, float]:
        return dict(self.options)

    def hooks(self, artifacts: Any) -> tuple[Hook, ...]:
        return (Hook("observation", SpeedFeedbackPlugin(**artifacts)),)
