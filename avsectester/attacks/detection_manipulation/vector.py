"""The detection-manipulation vector: add/drop detections at the detector output.

The perception-output counterpart to the LiDAR-spoofing vector. Instead of manipulating the
raw sensor, it manipulates the detector's *output* — abstracting an adversary who can make
the perception stage emit a false positive (a successful upstream spoof, or a compromised
perception module) or suppress a true positive. Both detection-injection and
detection-removal methods compose this vector.

It binds at ``perception_out`` (an avstack post-hook on the detector) and needs no special
capability: it works on any detector, ground-truth passthrough or neural, so it is the
reliable detection-level counterpart to the raw-point attacks the LiDAR vector aspires to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...core.binding import BindingSpec
from ..vector import AttackVector

# phantom vehicle box [h, w, l] in metres
PHANTOM_EXTENT = [1.6, 1.8, 4.0]
# IDs reserved for injected phantoms (so removal never targets an injected detection)
PHANTOM_IDS = (90001, 90002)


class DetectionManipulationVector(AttackVector):
    # Detection-level manipulation works on any detector output -> no capability requirement,
    # only that a perception_out seam exists.
    bindings = (BindingSpec("perception_out", payload="detections", fidelity=1),)

    # -- frame selection ------------------------------------------------------
    @staticmethod
    def reference(detections: Any, ego_state: Any = None) -> Any:
        """The frame to place/interpret a detection in: the detections' own frame if present."""
        if len(detections) > 0:
            return detections[0].box.reference
        if ego_state is not None:
            return ego_state.reference
        from avstack.geometry import GlobalOrigin3D

        return GlobalOrigin3D

    @staticmethod
    def forward_lateral(det: Any) -> tuple[float, float]:
        """(forward, lateral) metres of a detection in its own (ego/sensor) frame."""
        p = det.position.x
        return float(p[0]), float(p[1])

    # -- primitives -----------------------------------------------------------
    @staticmethod
    def add_detection(
        detections: Any, target_xyz: list[float], *, ego_state: Any = None,
        obj_type: str = "Car", score: float = 0.9, extent: list[float] | None = None,
        oid: int = 90002,
    ) -> Any:
        """Append a fabricated ``BoxDetection`` ``target_xyz`` ahead of the ego (in place)."""
        import numpy as np
        from avstack.geometry import Attitude, Box3D, Position
        from avstack.modules.perception.detections import BoxDetection

        ref = DetectionManipulationVector.reference(detections, ego_state)
        pos = Position(np.asarray(target_xyz, dtype=float), ref)
        att = Attitude(np.quaternion(1), ref)  # aligned with the detection frame
        box = Box3D(pos, att, extent or PHANTOM_EXTENT, where_is_t="bottom")
        phantom = BoxDetection(
            data=box,
            noise=np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) ** 2,
            source_identifier=detections.source_identifier,
            reference=ref,
            obj_type=obj_type,
            score=score,
        )
        phantom.ID = oid
        detections.append(phantom)
        return detections

    @staticmethod
    def drop_detections(detections: Any, predicate: Callable[[Any], bool]) -> Any:
        """Return a new container with detections matching ``predicate`` removed."""
        from avstack.datastructs import DataContainer

        kept = [d for d in detections if not predicate(d)]
        return DataContainer(
            detections.frame, detections.timestamp, kept,
            getattr(detections, "source_identifier", "atk"),
        )

    def select_forward_detection(
        self, detections: Any, corridor: float = 3.0, max_range: float = 40.0,
        exclude_ids: tuple[int, ...] = PHANTOM_IDS,
    ) -> Any:
        """Nearest real detection ahead of the ego within ``corridor`` lateral / ``max_range``."""
        best, best_fwd = None, float("inf")
        for d in detections:
            if getattr(d, "ID", None) in exclude_ids:
                continue
            fwd, lat = self.forward_lateral(d)
            if 0.0 < fwd < min(best_fwd, max_range) and abs(lat) <= corridor:
                best, best_fwd = d, fwd
        return best
