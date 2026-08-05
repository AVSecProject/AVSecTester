"""Detection-removal attack (detection-manipulation vector): suppress a real detection.

A **false-negative** method — the mirror image of detection injection. It drops a genuine
detection from the detector's output (an adversary suppressing a true positive), so the
obstacle never becomes a track and the ego does not slow or evade. The detection-level
counterpart to LiDAR object removal.

Target selection: an explicit ``target_id``, or (default) the nearest real detection ahead
of the ego within the braking corridor. Injected phantoms are never targeted.
"""

from __future__ import annotations

from typing import Any

from ...config import ATTACKS
from ...core.interfaces import AttackBase
from ...core.threat_model import AccessLevel, Knowledge, ThreatModel
from .vector import DetectionManipulationVector


@ATTACKS.register_module()
class DetectionRemovalAttack(AttackBase):
    category = "detection_manipulation"
    bindings = DetectionManipulationVector.bindings  # shared with every detection method

    def __init__(
        self,
        target_id: int | None = None,
        corridor: float = 3.0,
        max_range: float = 40.0,
        threat_model: ThreatModel | None = None,
    ) -> None:
        self.vector = DetectionManipulationVector()
        self.target_id = target_id
        self.corridor = corridor
        self.max_range = max_range
        self._removed_id: int | None = None
        self.threat_model = threat_model or ThreatModel(
            goal="Suppress a real obstacle detection so the ego fails to slow or evade.",
            knowledge=Knowledge.GRAYBOX,
            access=[AccessLevel.SENSOR, AccessLevel.SOFTWARE],
            target="ego 3D object detector output",
            capabilities=["drop_true_detection"],
            constraints=["target_within_detector_range"],
            success_criteria="Target detection absent -> no track -> ego does not brake.",
        )

    def validate(self, spec) -> None:
        if self.corridor <= 0 or self.max_range <= 0:
            raise ValueError("corridor and max_range must be positive")

    def reset(self) -> None:
        self._removed_id = None

    def apply(self, data: Any, ego_state: Any = None, ctx: Any = None, **kwargs: Any) -> Any:
        if self.target_id is not None:
            target_id = self.target_id
        else:
            target = self.vector.select_forward_detection(
                data, corridor=self.corridor, max_range=self.max_range
            )
            if target is None:
                return data
            target_id = getattr(target, "ID", None)
        self._removed_id = target_id
        return self.vector.drop_detections(data, lambda d: getattr(d, "ID", None) == target_id)
