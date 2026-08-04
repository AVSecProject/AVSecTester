"""Detection-level phantom injection (perception-output seam).

Appends a fabricated ``BoxDetection`` to a detector's output, so a phantom obstacle
propagates detection -> track -> (via the backend's forward-collision reflex) an unsafe
stop, without needing to fool the neural network from raw points.

Threat model: this abstracts an adversary who can make the perception stage emit a false
positive -- e.g. a successful upstream sensor spoof, or a compromised perception module.
It is the reliable detection-level counterpart to a raw-LiDAR point-injection attack
(which must actually fool the detector and is handled by the optimization track).

Seam: attaches as a **post-hook on the detector** (``perception_out``); the hook adapter
calls ``apply(detections, ego_state=..., ctx=...) -> detections``.
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.interfaces import AttackBase
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel

# phantom vehicle box [h, w, l] in metres
_PHANTOM_EXTENT = [1.6, 1.8, 4.0]


@ATTACKS.register_module()
class PhantomDetectionAttack(AttackBase):
    def __init__(
        self,
        target_xyz: list[float] | None = None,
        obj_type: str = "Car",
        score: float = 0.9,
        extent: list[float] | None = None,
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.target_xyz = target_xyz or [12.0, 0.0, -1.5]  # [forward, left, up] in detection frame
        self.obj_type = obj_type
        self.score = score
        self.extent = extent or _PHANTOM_EXTENT
        self._phantom_id = 90002
        self.threat_model = threat_model or ThreatModel(
            goal="Emit a phantom obstacle detection to induce unsafe braking.",
            knowledge=Knowledge.GRAYBOX,
            access=[AccessLevel.SENSOR, AccessLevel.SOFTWARE],
            target="ego 3D object detector output",
            capabilities=["inject_false_detection"],
            constraints=["obj_type_in_detector_whitelist"],
            success_criteria="Phantom detection confirmed as a track -> ego decelerates hard.",
        )

    def validate(self, spec) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("score must be in [0, 1]")

    def reset(self) -> None:
        """Stateless; provided for a uniform attack lifecycle."""

    def _reference(self, detections: Any, ego_state: Any):
        """Pick the frame to place the phantom in: the detections' own frame if present."""
        if len(detections) > 0:
            return detections[0].box.reference
        if ego_state is not None:
            return ego_state.reference
        from avstack.geometry import GlobalOrigin3D

        return GlobalOrigin3D

    def apply(self, data: Any, ego_state: Any = None, ctx: Any = None, **kwargs: Any) -> Any:
        """Append a phantom ``BoxDetection`` ``target_xyz`` ahead of the ego."""
        import numpy as np
        from avstack.geometry import Attitude, Box3D, Position
        from avstack.modules.perception.detections import BoxDetection

        ref = self._reference(data, ego_state)
        pos = Position(np.asarray(self.target_xyz, dtype=float), ref)
        att = Attitude(np.quaternion(1), ref)  # aligned with the detection frame
        box = Box3D(pos, att, self.extent, where_is_t="bottom")
        phantom = BoxDetection(
            data=box,
            noise=np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) ** 2,
            source_identifier=data.source_identifier,
            reference=ref,
            obj_type=self.obj_type,
            score=self.score,
        )
        phantom.ID = self._phantom_id
        data.append(phantom)
        return data
