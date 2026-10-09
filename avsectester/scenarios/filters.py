"""Composable initial-scene filters and three-valued results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Literal

from .context import FilterContext
from .geometry import rear_view_angle
from .scene import SceneGT


@dataclass(frozen=True)
class FilterResult:
    status: Literal["pass", "fail", "unknown"]
    reason: str = ""

    def __post_init__(self):
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError(f"Unknown filter status: {self.status}")

    @classmethod
    def from_bool(cls, passed: bool, reason: str) -> FilterResult:
        return cls("pass" if passed else "fail", reason)


def _all_results(results, empty_reason="No subjects available"):
    results = list(results)
    if not results:
        return FilterResult("unknown", empty_reason)
    statuses = [r.status for r in results]
    status = "fail" if "fail" in statuses else "unknown" if "unknown" in statuses else "pass"
    return FilterResult(status, ", ".join(r.reason for r in results if r.reason))


class Constraint(ABC):
    """Implement ``evaluate(context)`` for built-in or custom initial-scene filters."""

    @abstractmethod
    def evaluate(self, context: FilterContext) -> FilterResult: ...

    def holds(self, context: FilterContext | SceneGT, target=None) -> bool:
        """Boolean convenience adapter for single-target annotation queries."""
        if isinstance(context, SceneGT):
            bindings = {"attacker": (target,)} if target is not None else {}
            context = FilterContext(context, target=target, bindings=bindings)
        return self.evaluate(context).status == "pass"


@dataclass
class All(Constraint):
    filters: list[Constraint] = field(default_factory=list)

    def evaluate(self, context):
        if not self.filters:
            return FilterResult("pass", "No conditions")
        return _all_results(f.evaluate(context) for f in self.filters)


@dataclass
class Any(Constraint):
    filters: list[Constraint] = field(default_factory=list)

    def evaluate(self, context):
        results = [f.evaluate(context) for f in self.filters]
        statuses = [r.status for r in results]
        status = "pass" if "pass" in statuses else "unknown" if "unknown" in statuses else "fail"
        return FilterResult(status, ", ".join(r.reason for r in results if r.reason))


@dataclass
class Not(Constraint):
    filter: Constraint

    def evaluate(self, context):
        result = self.filter.evaluate(context)
        return FilterResult(
            {"pass": "fail", "fail": "pass", "unknown": "unknown"}[result.status], result.reason
        )


def _visual(context, subjects, check):
    targets = context.subjects(tuple(subjects))
    if subjects and len(targets) != len(subjects):
        return FilterResult("unknown", "A requested insertion subject is unavailable")
    return _all_results(check(target) for target in targets)


def _projection_unavailable(context, calib):
    if calib is None:
        return True
    return bool(context.insertions and (calib.model is None or calib.cam_to_ego is None))


@dataclass
class InView(Constraint):
    """Every selected insertion overlaps the victim camera, without asserting no occlusion."""

    camera: str
    subjects: tuple[str, ...] = ()

    def evaluate(self, context):
        calib = context.scene.cameras.get(self.camera)
        if _projection_unavailable(context, calib):
            return FilterResult("unknown", "Camera calibration unavailable")

        def check(target):
            if context.insertions:
                return FilterResult.from_bool(
                    context.in_view(target, self.camera),
                    f"{target.track_id}: opaque pixels overlap image",
                )
            frac = target.image_area_frac(self.camera, calib)
            return FilterResult.from_bool(
                frac is not None and frac > 0, f"{target.track_id}: projection overlaps image"
            )

        return _visual(context, self.subjects, check)


@dataclass
class DistanceRange(Constraint):
    """Every attacker-to-victim horizontal distance lies within inclusive metre bounds."""

    min_m: float
    max_m: float

    def __post_init__(self):
        if not 0 <= self.min_m <= self.max_m:
            raise ValueError("Distance bounds must satisfy 0 <= min_m <= max_m")

    def evaluate(self, context):
        def check(attacker):
            center = context.scene.ego.center
            distance = math.hypot(attacker.center[0] - center[0], attacker.center[1] - center[1])
            return FilterResult.from_bool(
                self.min_m <= distance <= self.max_m,
                f"Attacker {attacker.track_id}: distance {distance:.3f} m",
            )

        return _all_results(
            (check(attacker) for attacker in context.bindings.get("attacker", ())),
            "Attacker role unavailable",
        )


@dataclass
class ImageAreaFrac(Constraint):
    """Image-clipped insertion bounding-box fraction, not a silhouette measurement."""

    min_frac: float
    max_frac: float
    camera: str = "front"
    subjects: tuple[str, ...] = ()

    def __post_init__(self):
        if not 0 <= self.min_frac <= self.max_frac <= 1:
            raise ValueError("Image area bounds must be in [0, 1]")

    def evaluate(self, context):
        calib = context.scene.cameras.get(self.camera)
        if _projection_unavailable(context, calib):
            return FilterResult("unknown", "Camera calibration unavailable")

        def check(target):
            frac = target.image_area_frac(self.camera, calib)
            return FilterResult.from_bool(
                frac is not None and self.min_frac <= frac <= self.max_frac,
                f"{target.track_id}: clipped box area fraction {frac}",
            )

        return _visual(context, self.subjects, check)


@dataclass
class ViewpointRear(Constraint):
    """Horizontal rear-to-camera angle of actors in the named role."""

    max_deg: float = 35.0
    camera: str = "front"
    role: str = "attacker"

    def __post_init__(self):
        if not 0 <= self.max_deg <= 180:
            raise ValueError("Rear-view angle must be in [0, 180]")

    def evaluate(self, context):
        calib = context.scene.cameras.get(self.camera)
        if calib is None or calib.cam_to_ego is None:
            return FilterResult("unknown", "Camera-to-ego transform unavailable")
        position = [calib.cam_to_ego[i][3] for i in range(3)]

        def check(target):
            angle = rear_view_angle(target.center, target.yaw, position)
            if angle is None:
                return FilterResult(
                    "unknown", "Camera and target have coincident horizontal positions"
                )
            return FilterResult.from_bool(
                angle <= self.max_deg, f"{target.track_id}: rear-view angle {angle:.3f} degrees"
            )

        return _all_results(check(target) for target in context.bindings.get(self.role, ()))


@dataclass
class MinVisibility(Constraint):
    """Threshold each insertion's own visibility, including image truncation."""

    min_vis: float = 0.7
    camera: str = "front"
    subjects: tuple[str, ...] = ()

    def __post_init__(self):
        if not 0 <= self.min_vis <= 1:
            raise ValueError("Visibility threshold must be in [0, 1]")

    def evaluate(self, context):
        def check(target):
            visibility = context.visibility(target, self.camera)
            if visibility is None:
                return FilterResult("unknown", f"{target.track_id}: visibility unavailable")
            if visibility.camera is not None and visibility.camera != self.camera:
                return FilterResult("unknown", f"Visibility is for camera {visibility.camera}")
            return FilterResult.from_bool(
                visibility.fraction >= self.min_vis,
                f"{target.track_id}: visibility {visibility.fraction:g} ({visibility.source})",
            )

        return _visual(context, self.subjects, check)


@dataclass
class EgoMoving(Constraint):
    min_speed: float = 1.0

    def evaluate(self, context):
        return FilterResult.from_bool(
            context.scene.ego.speed >= self.min_speed, f"Ego speed {context.scene.ego.speed:g} m/s"
        )


@dataclass
class ClearLaneAhead(Constraint):
    """Existing centre-based corridor test. Neither lane geometry nor optical occlusion."""

    corridor_half_width: float = 1.5

    def evaluate(self, context):
        target = context.target
        if target is None:
            members = context.bindings.get("attacker", ())
            target = members[0] if members else None
        if target is None:
            return FilterResult("unknown", "No attacker bound for corridor test")
        for obj in context.scene.objects:
            if obj.track_id == target.track_id:
                continue
            if (
                obj.ahead
                and obj.center[0] < target.center[0]
                and abs(obj.center[1]) <= self.corridor_half_width
            ):
                return FilterResult("fail", f"Object {obj.track_id} occupies the forward corridor")
        return FilterResult("pass", "Forward corridor clear")
