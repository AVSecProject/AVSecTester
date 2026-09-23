"""The scenario-requirement DSL — an attack's preconditions as a composable predicate over ``SceneGT``.

An attack declares *what scene it needs* once, as a :class:`ScenarioRequirement` (a target selection +
a list of :class:`Constraint`). Its :meth:`ScenarioRequirement.match` is the predicate that both
providers in ``source.py`` consume: a :class:`DatasetFilter` uses it to *select* qualifying real-data
frames, a :class:`CarlaScenarioBuilder` to *validate* a scene it constructed. Defining the requirement
once is what lets one attack drive both real-data filtering and simulation building.

Constraints are Python objects (an embedded DSL): type-checked, composable, and directly executable; a
text/YAML front-end can be layered on later. Pure logic — offline-importable and unit-testable.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from avsectester.scenarios.scene import ObjectGT, SceneGT


@dataclass
class TargetSpec:
    """Which object the attack acts on: its ``category``, the ``camera`` it must be seen in, and a
    ``select`` rule to pick THE target when several candidates qualify."""

    category: str = "vehicle"
    camera: str = "front"
    select: str = "nearest_ahead"  # "nearest_ahead" | "largest" | "nearest"

    def candidates(self, scene: SceneGT) -> list[ObjectGT]:
        objs = [o for o in scene.objects_of(self.category) if self.camera in o.box2d]
        if self.select == "nearest_ahead":
            objs = [o for o in objs if o.ahead]
        return objs

    def pick(self, scene: SceneGT) -> ObjectGT | None:
        cands = self.candidates(scene)
        if not cands:
            return None
        if self.select == "largest":
            calib = scene.cameras.get(self.camera)
            return max(cands, key=lambda o: (o.image_area_frac(self.camera, calib) or 0.0)
                       if calib else 0.0)
        return min(cands, key=lambda o: o.distance)  # nearest / nearest_ahead


class Constraint(ABC):
    """A single precondition. ``holds(scene, target)`` — ``target`` is the selected object (may be
    ignored by ego-only constraints). Keep each one small and independently checkable."""

    @abstractmethod
    def holds(self, scene: SceneGT, target: ObjectGT | None) -> bool: ...


@dataclass
class InView(Constraint):
    """The target has a 2-D box in ``camera`` that overlaps the frame."""

    camera: str

    def holds(self, scene, target):
        calib = scene.cameras.get(self.camera)
        box = target.box2d.get(self.camera) if target else None
        if calib is None or box is None:
            return False
        x1, y1, x2, y2 = box
        return x2 > 0 and y2 > 0 and x1 < calib.width and y1 < calib.height and x2 > x1 and y2 > y1


@dataclass
class DistanceRange(Constraint):
    """Target ground-plane distance from the ego is within ``[min_m, max_m]``."""

    min_m: float
    max_m: float

    def holds(self, scene, target):
        return target is not None and self.min_m <= target.distance <= self.max_m


@dataclass
class ImageAreaFrac(Constraint):
    """Target's 2-D box covers a fraction of the ``camera`` frame within ``[min_frac, max_frac]`` —
    big enough to matter, not so big it is an implausible/near-full-frame false positive."""

    min_frac: float
    max_frac: float
    camera: str = "front"

    def holds(self, scene, target):
        calib = scene.cameras.get(self.camera)
        frac = target.image_area_frac(self.camera, calib) if (target and calib) else None
        return frac is not None and self.min_frac <= frac <= self.max_frac


@dataclass
class ViewpointRear(Constraint):
    """We see roughly the target's rear face — its heading is within ``max_deg`` of the ego's (so the
    back of the vehicle points toward us). Where a rear-surface patch attack needs the surface visible."""

    max_deg: float = 35.0

    def holds(self, scene, target):
        if target is None:
            return False
        # yaw ~ 0 means same heading as ego => we look at its rear; wrap to [-pi, pi]
        rel = math.atan2(math.sin(target.yaw), math.cos(target.yaw))
        return abs(math.degrees(rel)) <= self.max_deg


@dataclass
class MinVisibility(Constraint):
    """Target is at least ``min_vis`` unoccluded (so a patch on it is not itself hidden)."""

    min_vis: float = 0.7

    def holds(self, scene, target):
        return target is not None and target.visibility >= self.min_vis


@dataclass
class EgoMoving(Constraint):
    """The ego is driving at least ``min_speed`` — needed for attacks whose payload is a driving change
    (an already-stopped ego makes 'the attack stopped it' vacuous; cf. :mod:`avsectester.metric`)."""

    min_speed: float = 1.0

    def holds(self, scene, target):
        return scene.ego.speed >= self.min_speed


@dataclass
class ClearLaneAhead(Constraint):
    """No other object lies between the ego and the target within a lateral corridor — so the target is
    the relevant lead and nothing else occludes/pre-empts it."""

    corridor_half_width: float = 1.5

    def holds(self, scene, target):
        if target is None:
            return False
        for o in scene.objects:
            if o.track_id == target.track_id:
                continue
            if o.ahead and o.center[0] < target.center[0] and abs(o.center[1]) <= self.corridor_half_width:
                return False
        return True


@dataclass
class ScenarioMatch:
    """A scene that satisfied a requirement, plus the selected target + camera — the handle the eval
    harness needs to point the attack at the right object."""

    scene: SceneGT
    target: ObjectGT
    camera: str


@dataclass
class ScenarioRequirement:
    """An attack's preconditions: pick a target, then require all constraints to hold on it."""

    name: str
    target: TargetSpec
    constraints: list[Constraint] = field(default_factory=list)

    def match(self, scene: SceneGT) -> ScenarioMatch | None:
        """Return a :class:`ScenarioMatch` if this scene qualifies (target selectable + all constraints
        hold), else None. This is the shared predicate used by both scenario sources."""
        target = self.target.pick(scene)
        if target is None:
            return None
        if all(c.holds(scene, target) for c in self.constraints):
            return ScenarioMatch(scene=scene, target=target, camera=self.target.camera)
        return None
