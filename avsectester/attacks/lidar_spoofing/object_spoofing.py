"""Object-spoofing attack (LiDAR-spoofing vector): inject a phantom obstacle.

A false-positive method — appends a phantom ``ObjectState`` at the passthrough detector's
object-level input, so it propagates detection -> track -> an unsafe stop. The world-anchored
phantom is a stationary obstacle the ego then approaches.
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.attack import Attack
from ...core.context import Context
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel
from .vector import LidarSpoofingVector


@ATTACKS.register_module()
class ObjectSpoofingAttack(Attack):
    seams = LidarSpoofingVector.seams   # ("perception_input",)

    def __init__(
        self,
        target_xyz: list[float] | None = None,
        n_points: int = 200,
        score: float = 1.0,
        obj_type: str = "car",
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.vector = LidarSpoofingVector()
        self.target_xyz = target_xyz or [12.0, 0.0, 0.0]  # [forward, left, up] in ego frame
        self.n_points = n_points
        self.score = score
        self.obj_type = obj_type
        self._phantom_world = None
        self._phantom_id = 90001
        self.threat_model = threat_model or ThreatModel(
            goal="Create a phantom obstacle to induce unsafe braking.",
            knowledge=Knowledge.BLACKBOX,
            access=[AccessLevel.SENSOR, AccessLevel.PHYSICAL_ENVIRONMENT],
            target="ego LiDAR perception",
            capabilities=["inject_lidar_points"],
            constraints=["points_within_sensor_fov", "n_points<=1000"],
            success_criteria="Phantom object detected -> ego decelerates hard.",
        )

    def reset(self) -> None:
        self._phantom_world = None

    def apply(self, payload: Any, ctx: Context) -> Any:
        ego = ctx.frame.ego
        if ego is None:
            return payload
        if self._phantom_world is None:
            self._phantom_world = self.vector.anchor_world(ego, self.target_xyz)
        phantom = self.vector.make_phantom_object(
            payload.timestamp, self._phantom_world, ego,
            obj_type=self.obj_type, score=self.score, oid=self._phantom_id,
        )
        return self.vector.add_object(payload, phantom)
