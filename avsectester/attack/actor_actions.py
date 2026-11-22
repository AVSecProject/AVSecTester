"""World actions that operate on actors already in the simulator."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from avsectester.attack.runtime import WorldAttackAction
from avsectester.data_types import ActorState
from avsectester.geometry.poses import validated_transform, validated_vector

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.base import SimulatorRuntime


@dataclass(frozen=True)
class SetActorState(WorldAttackAction):
    """Set the world pose and velocity of an existing non-ego actor."""

    actor_id: str
    state: ActorState

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or not self.actor_id.strip():
            raise ValueError("actor_id must be a nonempty string")
        if not isinstance(self.state, ActorState):
            raise TypeError("state must be an ActorState")
        if self.state.velocity is None:
            raise ValueError("SetActorState requires a world velocity")
        x, y, z = (float(value) for value in validated_vector(self.state.velocity, "velocity"))
        object.__setattr__(self, "state", replace(
            self.state, pose=validated_transform(self.state.pose), velocity=(x, y, z)))

    def apply(self, simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None:
        """Apply the actor state using the simulator's native motion interface."""
        simulator_runtime.set_actor_state(self.actor_id, self.state)

    def receipt_details(self) -> Mapping[str, Any]:
        """Return the actor ID, world-frame pose and velocity that were requested."""
        return {"actor_id": self.actor_id, "pose": self.state.pose.copy(),
                "velocity": self.state.velocity}


__all__ = ['SetActorState']
