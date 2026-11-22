"""Apply the relocation at its configured decision."""

from avsectester.attack import AttackRuntime, OnlineContext
from avsectester.attack.actor_actions import SetActorState


class ActorRelocationRuntime(AttackRuntime):
    def __init__(self, decision: int, action: SetActorState) -> None:
        self.decision = decision
        self.action = action

    def step(self, context: OnlineContext) -> tuple[SetActorState, ...]:
        return (self.action,) if context.step == self.decision else ()
