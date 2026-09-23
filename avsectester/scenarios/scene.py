"""Canonical ground-truth scene state — what a scenario requirement is evaluated against.

A *provider adapter* maps its native ground truth into this schema so the same requirement predicate
(``requirement.py``) runs unchanged on either source:

  * CARLA: live actor 3-D boxes/poses + sensor calibration -> ``carla_scene_gt`` (phase 3);
  * a dataset (Alpamayo/NuRec clip labels): per-frame 3-D boxes + calibration -> a ``Dataset`` adapter.

Pure data — no simulator/dataset/torch imports — so the interface stays offline-importable and the
predicates are unit-testable. Coordinates are the **ego frame** (x forward, y left, z up), metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraCalib:
    """One camera's calibration. ``model`` is the opaque intrinsics (pinhole K, CARLA ``P``, or an
    f-theta spec) filled by the provider; ``cam_to_ego`` its extrinsic. Only ``name``/``width``/
    ``height`` are needed by the current constraints (they read 2-D boxes); the rest lets a provider
    project 3-D boxes to fill :attr:`ObjectGT.box2d` when the dataset gives only 3-D."""

    name: str
    width: int
    height: int
    model: Any = None
    cam_to_ego: Any = None


@dataclass
class ObjectGT:
    """A ground-truth object in the scene, in the ego frame.

    ``box2d`` maps ``camera_name -> (x1, y1, x2, y2)`` image-space boxes; a provider fills it from the
    dataset labels or by projecting the 3-D box. ``visibility`` is the unoccluded fraction in [0, 1]
    when known (else 1.0)."""

    track_id: str
    category: str  # "vehicle" | "pedestrian" | "cyclist" | ...
    center: tuple[float, float, float]  # (x fwd, y left, z up), metres, ego frame
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (length, width, height)
    yaw: float = 0.0  # heading relative to the ego (rad); 0 = same heading, pi = facing the ego
    box2d: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    visibility: float = 1.0

    @property
    def distance(self) -> float:
        """Ground-plane distance from the ego origin (metres)."""
        return math.hypot(self.center[0], self.center[1])

    @property
    def ahead(self) -> bool:
        """Is the object in front of the ego (positive x)?"""
        return self.center[0] > 0.0

    def image_area_frac(self, camera: str, calib: CameraCalib) -> float | None:
        """Fraction of the ``camera`` frame the object's 2-D box covers, or None if not projected."""
        box = self.box2d.get(camera)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(calib.width * calib.height)


@dataclass
class EgoState:
    """The ego's own state (what a requirement needs about the platform)."""

    speed: float = 0.0  # m/s
    pose: Any = None  # optional full pose, provider-specific


@dataclass
class SceneGT:
    """Ground truth for one frame: the ego, its cameras, and the objects — a requirement's input.

    ``source`` carries provenance so a match can be traced back to build the runnable scenario
    (e.g. ``{"backend": "nurec", "clip": "clipgt-01d5...", "frame": 42}`` or
    ``{"backend": "carla", "params": {...}}``)."""

    frame: int
    t: float
    ego: EgoState
    cameras: dict[str, CameraCalib]
    objects: list[ObjectGT]
    source: dict[str, Any] = field(default_factory=dict)

    def objects_of(self, category: str) -> list[ObjectGT]:
        return [o for o in self.objects if o.category == category]
