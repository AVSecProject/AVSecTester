"""All speed-feedback state belongs to one runtime."""
from dataclasses import replace

from avsectester.attack.runtime import AttackRuntime, AVStackAttackAction, ResultSubscription


class SpeedFeedbackAttackRuntime(AttackRuntime):
    """Keep the physical-speed controller and model-visible bias for one run."""

    subscriptions = (ResultSubscription("physical_state"),)

    def __init__(self, attack):
        self.target_speed, self.gain = attack.target_speed, attack.gain
        self.max_bias, self.bias = attack.max_bias, attack.initial_bias

    def step(self, context):
        if context.physical_state is None:
            raise ValueError("Speed feedback requires physical_state")
        if context.step:
            speed = context.physical_state.ego_speed
            self.bias = max(0.0, min(self.max_bias, self.bias + self.gain * (self.target_speed - speed)))
        context.record_output("speed_bias", self.bias)
        return (AVStackAttackAction("observation", self.modify),)

    def modify(self, observation, context):
        return replace(observation, ego_speed=observation.ego_speed + self.bias)
