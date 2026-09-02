"""Detection-injection attack (detection-manipulation vector): inject a phantom detection.

A false-positive method — appends a fabricated ``BoxDetection`` at the detector's output, so a
phantom obstacle propagates detection -> track -> (via the forward-collision reflex) an unsafe
stop, without fooling the network from raw points. No offline preparation needed.
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.attack import Attack
from ...core.context import Context
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel
from .vector import DetectionManipulationVector


@ATTACKS.register_module()
class PhantomDetectionAttack(Attack):
    seams = DetectionManipulationVector.seams   # ("perception_out",)

    def __init__(
        self,
        target_xyz: list[float] | None = None,
        obj_type: str = "Car",
        score: float = 0.9,
        extent: list[float] | None = None,
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.vector = DetectionManipulationVector()
        self.target_xyz = target_xyz or [12.0, 0.0, -1.5]  # [forward, left, up] in detection frame
        self.obj_type = obj_type
        self.score = score
        self.extent = extent
        self._phantom_id = 90002
        self.threat_model = threat_model or ThreatModel(
            goal="Emit a phantom obstacle detection to induce unsafe braking.",
            knowledge=Knowledge.GRAYBOX,
            access=[AccessLevel.SENSOR, AccessLevel.SOFTWARE],
            target="ego 3D object detector output",
            capabilities=["inject_false_detection"],
            success_criteria="Phantom detection confirmed as a track -> ego decelerates hard.",
        )

    def apply(self, payload: Any, ctx: Context) -> Any:
        return self.vector.add_detection(
            payload, self.target_xyz, ego_state=ctx.frame.ego,
            obj_type=self.obj_type, score=self.score, extent=self.extent, oid=self._phantom_id,
        )
