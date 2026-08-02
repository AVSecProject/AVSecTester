"""LiDAR spoofing attack: inject a phantom obstacle (PLAN.md Phase 3).

Attaches as a **pre-hook on the perception input** (avstack pre-hook shape
``apply(data, ego_state=...) -> data``). It fabricates a phantom vehicle a fixed
distance ahead of the ego and adds it to the perception input so it propagates
detection → track → (via the backend's forward-collision reflex) an unsafe stop —
the attack-escalation path in miniature.

Seam note: against the ground-truth ``Passthrough3DObjectDetector`` the injection is
at the object level (append a phantom ``ObjectState``). The identical hook moves to the
raw ``LidarData`` boundary — spraying ``n_points`` into the sensor FOV — once a neural
detector replaces the passthrough; ``n_points`` is carried for that seam and budgeted here.

TODO(phase3):
  - Raw-LiDAR point-injection variant (spray n_points in the sensor frame via refchoc).
  - remove / shift modes; LOS-feasibility + FOV constraint checks.
  - Optimization mode (attack *generation*) vs. fixed replay (attack *evaluation*).
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.interfaces import AttackBase
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel

# phantom vehicle box [h, w, l] in metres
_PHANTOM_EXTENT = [1.6, 1.8, 4.0]


@ATTACKS.register_module()
class LidarSpoofAttack(AttackBase):
    def __init__(
        self,
        target_xyz: list[float] | None = None,
        n_points: int = 200,
        mode: str = "inject",  # inject | remove | shift
        score: float = 1.0,  # detector confidence of the phantom (defenses may gate on this)
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.target_xyz = target_xyz or [12.0, 0.0, 0.0]  # [forward, left, up] in ego frame
        self.n_points = n_points
        self.mode = mode
        self.score = score
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
        if self.n_points > 1000:
            raise ValueError("n_points exceeds declared threat-model budget (<=1000)")
        if self.mode != "inject":
            raise NotImplementedError(f"mode {self.mode!r} not implemented yet (inject only)")

    def reset(self) -> None:
        """Forget the cached phantom position (call between runs)."""
        self._phantom_world = None

    def apply(self, data: Any, ego_state: Any = None, **kwargs: Any) -> Any:
        """Append a phantom ``ObjectState`` a fixed distance ahead of the ego."""
        if self.mode != "inject" or ego_state is None:
            return data

        import numpy as np
        from avstack.environment.objects import ObjectState
        from avstack.geometry import (
            Acceleration,
            AngularVelocity,
            Attitude,
            Box3D,
            GlobalOrigin3D,
            Position,
            Velocity,
        )
        from avstack.geometry import transformations as tforms

        # Fix the phantom in the world the first time it fires, using the ego-relative
        # offset at that moment — a stationary obstacle the ego then approaches.
        if self._phantom_world is None:
            rot = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, "+z")
            self._phantom_world = ego_state.position.x + rot @ np.asarray(
                self.target_xyz, dtype=float
            )

        pos = Position(self._phantom_world, GlobalOrigin3D)
        att = Attitude(ego_state.attitude.q, GlobalOrigin3D)
        box = Box3D(pos, att, _PHANTOM_EXTENT, where_is_t="bottom")
        phantom = ObjectState("car", ID=self._phantom_id, score=self.score)
        phantom.set(
            data.timestamp,
            pos,
            box,
            Velocity(np.zeros(3), GlobalOrigin3D),
            Acceleration(np.zeros(3), GlobalOrigin3D),
            att,
            AngularVelocity(np.quaternion(1), GlobalOrigin3D),
        )
        data.append(phantom)
        return data
