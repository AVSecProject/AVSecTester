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
    """Camera calibration in optical axes (x right, y down, z forward).

    ``model`` is a projection model, pinhole K, or provider lens specification.
    ``cam_to_ego`` maps optical coordinates into the scene's ego frame. Inserted geometry
    needs both fields. Existing annotation boxes can be queried using image dimensions alone.
    """

    name: str
    width: int
    height: int
    model: Any = None
    cam_to_ego: Any = None


@dataclass
class TargetGeometry:
    """A ground-truth object in the scene, in the ego frame.

    ``box2d`` maps ``camera_name -> (x1, y1, x2, y2)`` image-space boxes; a provider fills it from the
    dataset labels or by projecting the 3-D box. ``visibility`` is the visible fraction in [0, 1], including image truncation for rendered
    measurements. Dataset labels retain their documented semantics. Missing evidence is unknown."""

    track_id: str
    category: str  # "vehicle" | "pedestrian" | "cyclist" | ...
    center: tuple[float, float, float]  # (x fwd, y left, z up), metres, ego frame
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (length, width, height)
    yaw: float = 0.0  # heading relative to the ego (rad); 0 = same heading, pi = facing the ego
    box2d: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    visibility: Visibility | float | None = None
    pose: Any = None  # Optional ego-from-object 4x4 transform at the bounding-box centre.

    @property
    def distance(self) -> float:
        """Ground-plane distance from the ego origin (metres)."""
        return math.hypot(self.center[0], self.center[1])

    @property
    def ahead(self) -> bool:
        """Is the object in front of the ego (positive x)?"""
        return self.center[0] > 0.0

    def __post_init__(self):
        # Explicit legacy scalar values remain usable, but absence never means fully visible.
        if isinstance(self.visibility, (int, float)):
            self.visibility = Visibility(float(self.visibility), source="provided")

    def image_area_frac(self, camera: str, calib: CameraCalib) -> float | None:
        from .geometry import clipped_box_area

        box = self.box2d.get(camera)
        if box is None:
            return None
        return clipped_box_area(tuple(box), calib.width, calib.height) / (
            calib.width * calib.height
        )


@dataclass(frozen=True)
class Visibility:
    """Measured fraction or a coarse dataset representative. None camera means source-wide scope."""

    fraction: float
    source: str
    camera: str | None = None
    label: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError("visibility fraction must be in [0, 1]")


@dataclass
class ObjectGT(TargetGeometry):
    """An existing object identified by its track ID."""


@dataclass
class PlacementCandidate(TargetGeometry):
    """A proposed insertion, not an existing actor. Geometry is in the same ego frame as SceneGT.

    The attack or source supplies its projected footprint and optional visibility measurement.
    ``track_id`` is a stable placement ID, retained when the selected case is replayed.
    """


@dataclass
class EgoState:
    """Victim state. ``pose`` defines ego axes, while ``center`` locates its bounding box."""

    speed: float = 0.0  # m/s
    pose: Any = None  # Optional world-from-ego 4x4 transform, x forward, y left, z up.
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Bounding-box centre in ego axes.


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

    placements: list[PlacementCandidate] = field(default_factory=list)

    def objects_of(self, category: str) -> list[ObjectGT]:
        return [o for o in self.objects if o.category == category]
