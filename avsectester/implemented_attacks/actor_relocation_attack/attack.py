"""Select the scene origin and configure an online actor relocation."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from avsectester.attack import Attack, AttackRuntime, AttackSegment
from avsectester.attack.actor_actions import SetActorState
from avsectester.data_types import ActorState
from avsectester.geometry.poses import rotation_matrix, validated_vector
from avsectester.validation import require_nonnegative_integer

from .online import ActorRelocationRuntime

if TYPE_CHECKING:
    from avsectester.execution.services import ScenarioView


class ActorRelocationAttack(Attack):
    """Move a non-ego actor instantly at the given zero-based decision.

    The destination is in framework world coordinates, in meters and degrees.
    The attack uses the scene's full decision interval from its origin and sets
    the actor's velocity to zero. It requires no offline artifacts.
    """

    def __init__(self, actor_id: str, x: float, y: float, z: float,
                 yaw_deg: float, decision: int) -> None:
        require_nonnegative_integer(decision, "decision")
        pose = np.eye(4)
        pose[:3, 3] = validated_vector((x, y, z), "destination")
        pose[:3, :3] = rotation_matrix((0., 0., yaw_deg))
        self._action = SetActorState(actor_id, ActorState(pose, (0., 0., 0.)))
        if actor_id == "ego":
            raise ValueError("ActorRelocationAttack requires a non-ego actor")
        self.actor_id = actor_id
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.yaw_deg = float(yaw_deg)
        self.decision = decision

    def build_metadata(self, scenario: "ScenarioView") -> Sequence[AttackSegment]:
        frames = scenario.frames
        with scenario.services.session(scenario.seed) as session:
            if not session.can_start_at(0):
                scenario.record_reason("Actor relocation requires the scene origin")
                return ()
        if len(frames) <= self.decision + 1:
            scenario.record_reason("The relocation decision is outside the scene")
            return ()
        return (AttackSegment(scenario.scenario_id, frames[0], frames[-1]),)

    def validate_segment(self, segment: AttackSegment) -> None:
        if segment.start.position != 0 or self.decision >= segment.decision_steps:
            raise ValueError("The segment must start at the origin and include the relocation decision")

    def create_runtime(self, segment: AttackSegment, artifacts: Any) -> AttackRuntime:
        self.validate_segment(segment)
        return ActorRelocationRuntime(self.decision, self._action)


__all__ = ['ActorRelocationAttack']
