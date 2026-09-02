"""Object-removal attack (LiDAR-spoofing vector): erase a real obstacle.

A false-negative method — drops a genuine obstacle from the object-level detector input, so
the ego fails to perceive it and does not slow or evade. Target: an explicit ``target_id``, or
(default) the nearest real obstacle ahead of the ego within the braking corridor.
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.attack import Attack
from ...core.context import Context
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel
from .vector import LidarSpoofingVector


@ATTACKS.register_module()
class ObjectRemovalAttack(Attack):
    seams = LidarSpoofingVector.seams   # ("perception_input",)

    def __init__(
        self,
        target_id: int | None = None,
        corridor: float = 3.0,
        max_range: float = 40.0,
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.vector = LidarSpoofingVector()
        self.target_id = target_id
        self.corridor = corridor
        self.max_range = max_range
        self._removed_id: int | None = None
        self.threat_model = threat_model or ThreatModel(
            goal="Hide a real obstacle so the ego fails to slow or evade.",
            knowledge=Knowledge.BLACKBOX,
            access=[AccessLevel.SENSOR, AccessLevel.PHYSICAL_ENVIRONMENT],
            target="ego LiDAR perception",
            capabilities=["remove_lidar_points"],
            success_criteria="Target obstacle absent from perception -> ego does not brake.",
        )

    def reset(self) -> None:
        self._removed_id = None

    def apply(self, payload: Any, ctx: Context) -> Any:
        if self.target_id is not None:
            target_id = self.target_id
        else:
            ego = ctx.frame.ego
            if ego is None:
                return payload
            target = self.vector.select_forward_target(
                payload, ego, corridor=self.corridor, max_range=self.max_range
            )
            if target is None:
                return payload
            target_id = getattr(target, "ID", None)
        self._removed_id = target_id
        return self.vector.remove_objects(payload, lambda o: getattr(o, "ID", None) == target_id)
