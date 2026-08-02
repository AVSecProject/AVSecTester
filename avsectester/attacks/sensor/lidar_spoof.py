"""LiDAR spoofing / point-cloud manipulation attack (PLAN.md Phase 3).

Reference example of the attack contract. Injected as a **pre-hook** on the perception
module (or the sensor boundary): it receives avstack ``LidarData`` and returns manipulated
``LidarData`` (inject a phantom cluster / remove points from a region / shift points).

TODO(phase3):
  - Real point-injection geometry in the sensor reference frame (use avstack refchoc).
  - Respect threat-model constraints (max #points, spatial region, LOS feasibility).
  - Optimization mode (attack *generation*) vs. fixed replay (attack *evaluation*).
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.interfaces import AttackBase
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel


@ATTACKS.register_module()
class LidarSpoofAttack(AttackBase):
    def __init__(
        self,
        target_xyz: list[float] | None = None,
        n_points: int = 200,
        mode: str = "inject",  # inject | remove | shift
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.target_xyz = target_xyz or [10.0, 0.0, 0.0]
        self.n_points = n_points
        self.mode = mode
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
        # TODO(phase3): check n_points against declared budget, target within FOV, etc.
        if self.n_points > 1000:
            raise ValueError("n_points exceeds declared threat-model budget (<=1000)")

    def apply(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        # TODO(phase3): manipulate the avstack LidarData point cloud in-place / copy.
        raise NotImplementedError("phase 3: implement point-cloud manipulation")
