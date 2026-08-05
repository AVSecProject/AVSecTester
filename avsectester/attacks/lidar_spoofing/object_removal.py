"""Object-removal attack (LiDAR-spoofing vector): erase a real obstacle.

A **false-negative** method — the mirror image of object spoofing. Rather than adding a
phantom, it deletes a genuine obstacle from the ego's view (points removed / occluded), so
the ego fails to perceive it and does not slow or evade. On a ground-truth passthrough stack
it drops the target ``ObjectState`` from the object-level input; the raw-point erasure
realization is carried by the shared vector's ``raw_lidar`` binding (optimization track).

Target selection: an explicit ``target_id``, or (default) the nearest real obstacle ahead
of the ego within the braking corridor — the one whose removal is most safety-relevant.
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.interfaces import AttackBase
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel
from .vector import LidarSpoofingVector


@ATTACKS.register_module()
class ObjectRemovalAttack(AttackBase):
    category = "lidar_spoofing"
    bindings = LidarSpoofingVector.bindings  # shared with every LiDAR-spoofing method

    def __init__(
        self,
        target_id: int | None = None,
        corridor: float = 3.0,   # lateral half-width of the "ahead of ego" region (m)
        max_range: float = 40.0,  # only remove obstacles within this forward distance (m)
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.vector = LidarSpoofingVector()
        self.target_id = target_id
        self.corridor = corridor
        self.max_range = max_range
        self._removed_id: int | None = None  # which object was hidden (telemetry)
        self.threat_model = threat_model or ThreatModel(
            goal="Hide a real obstacle so the ego fails to slow or evade.",
            knowledge=Knowledge.BLACKBOX,
            access=[AccessLevel.SENSOR, AccessLevel.PHYSICAL_ENVIRONMENT],
            target="ego LiDAR perception",
            capabilities=["remove_lidar_points"],
            constraints=["target_within_sensor_fov"],
            success_criteria="Target obstacle absent from perception -> ego does not brake.",
        )

    def validate(self, spec) -> None:
        if self.corridor <= 0 or self.max_range <= 0:
            raise ValueError("corridor and max_range must be positive")

    def reset(self) -> None:
        self._removed_id = None

    def apply(self, data: Any, ego_state: Any = None, ctx: Any = None, **kwargs: Any) -> Any:
        seam = self.bound_seam or "perception_input"
        if seam == "raw_lidar":
            return self.vector.remove_points(data, self.target_id)
        # object-level (perception_input)
        if self.target_id is not None:
            target_id = self.target_id
        else:
            if ego_state is None:
                return data
            target = self.vector.select_forward_target(
                data, ego_state, corridor=self.corridor, max_range=self.max_range
            )
            if target is None:
                return data
            target_id = getattr(target, "ID", None)
        self._removed_id = target_id
        return self.vector.remove_objects(data, lambda o: getattr(o, "ID", None) == target_id)
