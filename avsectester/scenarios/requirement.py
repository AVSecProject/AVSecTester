"""Composable filters and fixed-identity selection of initial experiment cases."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Any as TypingAny, Literal

from .context import FilterContext
from .geometry import rear_view_angle
from .scene import SceneGT, TargetGeometry


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


@dataclass
class TargetSpec:
    """Single-target shorthand for querying existing objects or provided placement geometry."""

    category: str = "vehicle"
    camera: str | None = "front"
    select: str = "nearest_ahead"
    kind: str = "object"

    def __post_init__(self):
        if self.select not in {"nearest_ahead", "nearest", "largest"}:
            raise ValueError(f"Unknown target selection: {self.select}")
        if self.kind not in {"object", "placement"}:
            raise ValueError(f"Unknown target kind: {self.kind}")
        if self.select == "largest" and self.camera is None:
            raise ValueError("largest selection requires a camera")

    def candidates(self, scene: SceneGT) -> list[TargetGeometry]:
        items = scene.objects if self.kind == "object" else scene.placements
        return [
            o
            for o in items
            if o.category == self.category and (self.select != "nearest_ahead" or o.ahead)
        ]

    def rank(self, scene, candidates):
        if self.select == "largest":
            calib = scene.cameras.get(self.camera)
            return sorted(
                [o for o in candidates if calib and self.camera in o.box2d],
                key=lambda o: -o.image_area_frac(self.camera, calib),
            )
        return sorted(candidates, key=lambda o: o.distance)

    def pick(self, scene, candidates=None):
        ordered = self.rank(scene, self.candidates(scene) if candidates is None else candidates)
        return ordered[0] if ordered else None


@dataclass
class RoleSpec:
    """Bind exact IDs or select ``count`` qualifying actors, without searching placements.

    Explicit ``ids`` are never replaced when absent or unsuitable. Selection order ranks
    complete role assignments only after every condition has been evaluated.
    """

    ids: tuple[str, ...] | None = None
    count: int = 1
    category: str = "vehicle"
    select: str = "nearest_ahead"
    camera: str | None = "front"

    def __post_init__(self):
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("Role count must be a positive integer")
        TargetSpec(self.category, self.camera, self.select)
        if self.ids is not None:
            self.ids = tuple(str(value) for value in self.ids)
            if len(set(self.ids)) != len(self.ids) or len(self.ids) != self.count:
                raise ValueError("Explicit role IDs must be distinct and match count")

    def choices(self, scene):
        if self.ids is not None:
            objects = {o.track_id: o for o in scene.objects}
            if all(i in objects for i in self.ids):
                yield tuple(objects[i] for i in self.ids)
            return
        spec = TargetSpec(self.category, self.camera, self.select)
        yield from combinations(spec.rank(scene, spec.candidates(scene)), self.count)


@dataclass(frozen=True)
class InitialWindow:
    """Consecutive prepared frames starting at the chosen experiment origin.

    Providers supply recorded or predefined states. This never invokes a driving policy.
    """

    frames: int = 1

    def __post_init__(self):
        if not isinstance(self.frames, int) or isinstance(self.frames, bool) or self.frames < 1:
            raise ValueError("InitialWindow.frames must be a positive integer")


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


@dataclass
class ScenarioMatch:
    scene: SceneGT
    target: TargetGeometry | None
    camera: str | None
    bindings: dict[str, tuple[TargetGeometry, ...]] = field(default_factory=dict)
    insertions: tuple[TypingAny, ...] = ()
    window: InitialWindow = field(default_factory=InitialWindow)

    @property
    def binding_ids(self):
        return {role: tuple(o.track_id for o in objects) for role, objects in self.bindings.items()}


@dataclass
class CandidateResult:
    target_id: str | None
    checks: list[tuple[str, FilterResult]]
    bindings: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return _all_results((r for _, r in self.checks)).status if self.checks else "pass"


@dataclass
class SelectionResult:
    status: str
    match: ScenarioMatch | None
    candidates: list[CandidateResult]
    reason: str = ""

    def summary(self) -> dict:
        """JSON-compatible selection evidence without images, SDK handles or native state."""
        from dataclasses import asdict

        return {
            "status": self.status,
            "reason": self.reason,
            "selected_target": self.match.target.track_id
            if self.match and self.match.target
            else None,
            "bindings": self.match.binding_ids if self.match else {},
            "insertions": [i.id for i in self.match.insertions] if self.match else [],
            "initial_frames": self.match.window.frames if self.match else None,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }


@dataclass
class ScenarioRequirement:
    """Select a fixed role assignment against every frame of an initial window.

    ``target`` is a single-object query shorthand. Rich insertion cases use ``roles``
    and user-specified ``insertions``. Only host identities are searched, never positions.
    """

    name: str
    target: TargetSpec | None = None
    constraints: list[Constraint] = field(default_factory=list)
    description: str = ""
    roles: dict[str, RoleSpec] = field(default_factory=dict)
    insertions: tuple[TypingAny, ...] = ()
    window: InitialWindow = field(default_factory=InitialWindow)
    max_bindings: int = 1000
    camera: str | None = None

    def __post_init__(self):
        if self.target is not None and self.roles:
            raise ValueError("Use target shorthand or named roles, not both")
        if "victim" in self.roles:
            raise ValueError("The victim is the scene ego, not a selectable role")
        if (
            not isinstance(self.max_bindings, int)
            or isinstance(self.max_bindings, bool)
            or self.max_bindings < 1
        ):
            raise ValueError("max_bindings must be a positive integer")
        self.insertions = tuple(self.insertions)
        if len({i.id for i in self.insertions}) != len(self.insertions):
            raise ValueError("Insertion IDs must be unique")

    def _camera(self, scene):
        if self.camera is not None:
            return self.camera
        if self.target is not None and self.target.camera is not None:
            return self.target.camera
        if "front" in scene.cameras:
            return "front"
        if len(scene.cameras) == 1:
            return next(iter(scene.cameras))
        return None

    def _assignments(self, context):
        if self.target is not None:
            for target in self.target.candidates(context.scene):
                bindings = dict(context.bindings)
                if self.target.kind == "object":
                    bindings["attacker"] = (target,)
                yield bindings, target
        elif self.roles:
            # Recursion keeps large role combinations lazy. itertools.product would
            # materialize every role's combination pool before its first result.
            roles = list(self.roles.items())

            def visit(index, bindings):
                if index == len(roles):
                    yield bindings, None
                    return
                name, spec = roles[index]
                for members in spec.choices(context.scene):
                    yield from visit(index + 1, {**bindings, name: members})

            yield from visit(0, dict(context.bindings))
        else:
            yield dict(context.bindings), context.target

    def _frames(self, context):
        if context.initial_contexts is not None:
            iterator = iter(context.initial_contexts())
            try:
                for _ in range(self.window.frames):
                    try:
                        yield next(iterator)
                    except StopIteration:
                        return
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        else:
            sequence = context.sequence or (context.scene,)
            for index, scene in enumerate(sequence[: self.window.frames]):
                yield context.derive(scene=scene) if index == 0 else context.at(scene, index)

    def evaluate(self, context: FilterContext | SceneGT) -> SelectionResult:
        if isinstance(context, SceneGT):
            context = FilterContext(context)
        insertions = tuple(self.insertions or context.insertions)
        if len({i.id for i in insertions}) != len(insertions):
            raise ValueError("Insertion IDs must be unique")
        context = context.derive(insertions=insertions)
        camera = self._camera(context.scene)
        assignments, checked = [], []
        exhausted = False
        for index, (bindings, target) in enumerate(self._assignments(context)):
            if index >= self.max_bindings:
                exhausted = True
                break
            assignments.append((bindings, target))
            checked.append(
                CandidateResult(
                    target.track_id if target else None,
                    [],
                    {
                        name: tuple(o.track_id for o in members)
                        for name, members in bindings.items()
                    },
                )
            )
        if not assignments:
            return SelectionResult(
                "fail", None, [], "Requested role IDs or actor counts unavailable"
            )

        previous_t, frame_count = None, 0
        for frame_index, prepared in enumerate(self._frames(context)):
            scene = prepared.scene
            if frame_index == 0 and (scene.frame, scene.t) != (
                context.scene.frame,
                context.scene.t,
            ):
                raise ValueError("Initial sequence must start at the selected scene")
            if previous_t is not None and scene.t <= previous_t:
                raise ValueError("Initial sequence timestamps must be strictly increasing")
            previous_t, frame_count = scene.t, frame_index + 1
            available = {o.track_id: o for o in [*scene.objects, *scene.placements]}
            # Evaluate every candidate against the same live frame before advancing the
            # provider. Native world APIs therefore agree with the normalized scene state.
            for (bindings, initial_target), record in zip(assignments, checked):
                current = {
                    name: tuple(available[i] for i in ids if i in available)
                    for name, ids in record.bindings.items()
                }
                target = available.get(initial_target.track_id) if initial_target else None
                missing = [
                    name for name, ids in record.bindings.items() if len(current[name]) != len(ids)
                ]
                if missing or (initial_target is not None and target is None):
                    record.checks.append(
                        (
                            f"frame[{frame_index}].bindings",
                            FilterResult("unknown", f"Bound identities unavailable: {missing}"),
                        )
                    )
                    continue
                frame = prepared.derive(
                    bindings=current, target=target, insertions=insertions, frame_index=frame_index
                )
                if insertions and scene.ego.pose is None:
                    record.checks.append(
                        (
                            f"frame[{frame_index}].geometry",
                            FilterResult(
                                "unknown", "Insertion geometry requires a world-from-ego pose"
                            ),
                        )
                    )
                    continue
                for insertion in insertions:
                    host = getattr(insertion.placement, "host", None)
                    if host in current and len(current[host]) > 1:
                        raise ValueError(
                            f"Host role {host!r} has multiple members, use an indexed alias"
                        )
                    if (
                        host is not None
                        and host not in available
                        and host not in frame.actor_aliases
                    ):
                        missing.append(host)
                if missing:
                    record.checks.append(
                        (
                            f"frame[{frame_index}].bindings",
                            FilterResult("unknown", f"Insertion hosts unavailable: {missing}"),
                        )
                    )
                    continue
                if insertions and not any(
                    isinstance(constraint, InView)
                    and constraint.camera == camera
                    and not constraint.subjects
                    for constraint in self.constraints
                ):
                    result = (
                        InView(camera).evaluate(frame)
                        if camera is not None
                        else FilterResult(
                            "unknown", "Select one victim camera for insertion visibility"
                        )
                    )
                    record.checks.append((f"frame[{frame_index}].InView[required]", result))
                for constraint in self.constraints:
                    result = constraint.evaluate(frame)
                    if not isinstance(result, FilterResult):
                        raise TypeError("Filters must return FilterResult")
                    label = type(constraint).__name__
                    if self.window.frames > 1:
                        label = f"frame[{frame_index}].{label}"
                    record.checks.append((label, result))
        if frame_count < self.window.frames:
            for record in checked:
                record.checks.append(
                    (
                        "InitialWindow",
                        FilterResult(
                            "unknown",
                            f"Only {frame_count} of {self.window.frames} prepared frames available",
                        ),
                    )
                )
        eligible = [
            assignment
            for assignment, result in zip(assignments, checked)
            if result.status == "pass"
        ]
        if eligible:
            bindings, target = eligible[0]
            if self.target:
                target = self.target.pick(context.scene, [target for _, target in eligible])
                if target is None:
                    return SelectionResult(
                        "unknown", None, checked, "Target ranking evidence unavailable"
                    )
                bindings = next(b for b, t in eligible if t is target)
            match = ScenarioMatch(context.scene, target, camera, bindings, insertions, self.window)
            return SelectionResult("pass", match, checked)
        status = "unknown" if exhausted or any(r.status == "unknown" for r in checked) else "fail"
        reason = "Role search budget exhausted" if exhausted else "No qualifying initial case"
        return SelectionResult(status, None, checked, reason)

    def match(self, context: FilterContext | SceneGT) -> ScenarioMatch | None:
        return self.evaluate(context).match
