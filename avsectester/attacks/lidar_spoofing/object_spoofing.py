"""Object-spoofing attack (LiDAR-spoofing vector): inject a phantom obstacle.

A **false-positive** method — it adds an obstacle that is not there, so it propagates
detection -> track -> (via the forward-collision reflex) an unsafe stop. On a ground-truth
passthrough stack it appends a phantom ``ObjectState`` at the object-level input; the
higher-fidelity raw-point realization is carried by the shared vector's ``raw_lidar``
binding (optimization track).
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.interfaces import AttackBase
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel
from .vector import LidarSpoofingVector


@ATTACKS.register_module()
class ObjectSpoofingAttack(AttackBase):
    category = "lidar_spoofing"
    bindings = LidarSpoofingVector.bindings  # shared with every LiDAR-spoofing method

    def __init__(
        self,
        target_xyz: list[float] | None = None,
        n_points: int = 200,
        score: float = 1.0,  # detector confidence of the phantom (defenses may gate on this)
        obj_type: str = "car",
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.vector = LidarSpoofingVector()
        self.target_xyz = target_xyz or [12.0, 0.0, 0.0]  # [forward, left, up] in ego frame
        self.n_points = n_points
        self.score = score
        self.obj_type = obj_type
        self._phantom_world = None  # cached absolute position (fixed obstacle in the world)
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

    def validate(self, spec) -> None:
        self.vector.check_budget(self.n_points)

    def reset(self) -> None:
        """Forget the cached phantom position (call between runs)."""
        self._phantom_world = None

    def apply(self, data: Any, ego_state: Any = None, ctx: Any = None, **kwargs: Any) -> Any:
        seam = self.bound_seam or "perception_input"
        if seam == "raw_lidar":
            return self.vector.add_points(data, self.target_xyz, self.n_points)
        # object-level (perception_input)
        if ego_state is None:
            return data
        if self._phantom_world is None:
            self._phantom_world = self.vector.anchor_world(ego_state, self.target_xyz)
        phantom = self.vector.make_phantom_object(
            data.timestamp, self._phantom_world, ego_state,
            obj_type=self.obj_type, score=self.score, oid=self._phantom_id,
        )
        return self.vector.add_object(data, phantom)


# Backward-compatible name (configs/scripts referenced ``LidarSpoofAttack``).
LidarSpoofAttack = ObjectSpoofingAttack
ATTACKS.register_module(name="LidarSpoofAttack", module=ObjectSpoofingAttack)
