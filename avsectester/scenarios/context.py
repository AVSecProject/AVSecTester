"""Selection-time access to normalized state and original provider resources.

Native objects are intentionally exposed without a method whitelist. A context is scoped to
candidate preparation, never retained as an active filter in the driving loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Any

from avsectester.insertion import Insertion
from avsectester.rendering.types import GeometryVisibilityEstimator, Visibility, VisibilityEvidence

from .scene import SceneGT, TargetGeometry


@dataclass
class FilterContext:
    """Inputs to one filter evaluation, with lazy insertion geometry and evidence.

    ``bindings`` maps role names to scene objects. ``native`` and ``metadata`` retain the
    provider's original objects, including fields absent from :class:`SceneGT`. A custom
    ``visibility_provider(context, insertion_id, camera)`` may acquire rendering evidence
    on demand. It returns ``Visibility`` or an estimator result with a ``visibility`` field.
    """

    scene: SceneGT
    bindings: dict[str, tuple[TargetGeometry, ...]] = field(default_factory=dict)
    insertions: tuple[Insertion, ...] = ()
    sequence: tuple[SceneGT, ...] = ()
    metadata: Any = None
    dataset: Any = None
    backend: Any = None
    renderer: Any = None
    native: Mapping[str, Any] = field(default_factory=dict)
    visibility_estimator: GeometryVisibilityEstimator | None = None
    visibility_provider: (
        Callable[[FilterContext, str, str], Visibility | VisibilityEvidence | None] | None
    ) = None
    frame_context: Callable[[SceneGT], FilterContext] | None = None
    initial_contexts: Callable[[], Iterator[FilterContext]] | None = None
    target: TargetGeometry | None = None
    frame_index: int = 0
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _resources: ExitStack = field(default_factory=ExitStack, repr=False, compare=False)

    def own(self, resource):
        """Register a resource's ``close()`` for this scope and return the resource.

        Fields such as ``renderer`` and ``native`` are borrowed unless registered here.
        Derived contexts share ownership. Closing any of them releases owned resources.
        """
        self._resources.callback(resource.close)
        return resource

    def close(self):
        """Release owned resources once, in reverse registration order."""
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def derive(self, **changes) -> FilterContext:
        """Create a context with a fresh geometry cache and shared native resources."""
        return replace(self, **changes)

    def at(self, scene: SceneGT, frame_index: int = 0) -> FilterContext:
        """Rebind the same identities in another frame of the prepared initial sequence."""
        available = {o.track_id: o for o in [*scene.objects, *scene.placements]}
        bindings = {
            role: tuple(available[o.track_id] for o in objects if o.track_id in available)
            for role, objects in self.bindings.items()
        }
        base = self.frame_context(scene) if self.frame_context else self
        if base._resources is not self._resources:
            self.own(base)
        return base.derive(
            scene=scene,
            bindings=bindings,
            insertions=self.insertions,
            sequence=self.sequence,
            frame_context=self.frame_context,
            frame_index=frame_index,
            target=available.get(self.target.track_id) if self.target else None,
        )

    @property
    def actor_aliases(self) -> dict[str, str]:
        """Single-member roles and indexed aliases such as ``hosts[0]``."""
        aliases = {}
        for role, members in self.bindings.items():
            if len(members) == 1:
                aliases[role] = members[0].track_id
            for index, member in enumerate(members):
                aliases[f"{role}[{index}]"] = member.track_id
        return aliases

    @property
    def actors(self) -> dict:
        """World poses keyed by native track ID and unambiguous single-member role alias."""
        if "actors" not in self._cache:
            import numpy as np

            from avsectester.insertion import ActorPose

            world_from_ego = self.world_from_ego
            actors = {}
            for obj in self.scene.objects:
                if obj.pose is None:
                    c, s = np.cos(obj.yaw), np.sin(obj.yaw)
                    local = np.eye(4)
                    local[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                    local[:3, 3] = obj.center
                else:
                    local = np.asarray(obj.pose, dtype=float)
                actors[obj.track_id] = ActorPose(world_from_ego @ local, obj.extent)
            for role, track_id in self.actor_aliases.items():
                if track_id in actors:
                    # An alias must not quietly shadow a different native actor.
                    if role in actors and role != track_id:
                        raise ValueError(f"Role {role!r} conflicts with an actor track ID")
                    actors[role] = actors[track_id]
            self._cache["actors"] = actors
        return self._cache["actors"]

    @property
    def world_from_ego(self):
        import numpy as np

        pose = self.scene.ego.pose
        if pose is None:
            raise ValueError("Insertion geometry requires the scene's world-from-ego pose")
        return np.asarray(pose, dtype=float)

    @property
    def victim(self):
        import numpy as np

        from avsectester.insertion import ActorPose

        local = np.eye(4)
        local[:3, 3] = self.scene.ego.center
        return ActorPose(self.world_from_ego @ local, self.scene.ego.extent)

    @property
    def resolved_insertions(self) -> dict:
        """Resolve user-specified placements without sampling or changing their positions."""
        if "resolved" not in self._cache:
            from avsectester.insertion import resolve_insertion

            self._cache["resolved"] = {
                insertion.id: resolve_insertion(insertion, self.actors, self.victim)
                for insertion in self.insertions
            }
        return self._cache["resolved"]

    def subjects(self, ids: tuple[str, ...] = ()) -> tuple[TargetGeometry, ...]:
        """Visual subjects are inserted objects when an insertion plan is present.

        Existing single-target requirements remain usable for dataset annotation queries.
        They never add a second host-visibility test to an insertion requirement.
        """
        if self.insertions:
            if "subjects" not in self._cache:
                from avsectester.insertion import ActorPose
                from .estimators import resolved_to_target

                self._cache["subjects"] = tuple(
                    resolved_to_target(item, ActorPose(self.world_from_ego), self.scene.cameras)
                    for item in self.resolved_insertions.values()
                )
            subjects = self._cache["subjects"]
        else:
            subjects = (self.target,) if self.target is not None else tuple(self.scene.placements)
        if ids:
            selected = {s.track_id: s for s in subjects}
            return tuple(selected[i] for i in ids if i in selected)
        return subjects

    def in_view(self, subject: TargetGeometry, camera: str) -> bool:
        """Test opaque insertion pixels against a calibrated image, independently of occlusion."""
        import numpy as np
        from avsectester.rendering.cameras import camera_from_calibration
        from avsectester.rendering.geometry import projected_silhouette

        key = ("in_view", subject.track_id, camera)
        if key not in self._cache:
            calib = self.scene.cameras[camera]
            mask = projected_silhouette(
                self.resolved_insertions[subject.track_id],
                camera_from_calibration(calib),
                np.linalg.inv(self.world_from_ego @ calib.cam_to_ego),
            )
            self._cache[key] = bool(mask.any())
        return self._cache[key]

    def visibility(self, subject: TargetGeometry, camera: str) -> Visibility | None:
        """Acquire each subject/camera measurement once within this candidate frame."""
        key = ("visibility", subject.track_id, camera)
        if key not in self._cache:
            evidence = subject.visibility
            if self.insertions and subject.track_id in self.resolved_insertions:
                if self.visibility_provider is not None:
                    evidence = self.visibility_provider(self, subject.track_id, camera)
                elif self.visibility_estimator is not None:
                    import numpy as np

                    from avsectester.rendering.cameras import camera_from_calibration

                    calib = self.scene.cameras.get(camera)
                    if calib is None or calib.cam_to_ego is None or calib.model is None:
                        return None
                    # Role aliases reference the same objects and must not duplicate occluders.
                    occluders = {o.track_id: self.actors[o.track_id] for o in self.scene.objects}
                    evidence = self.visibility_estimator.estimate(
                        self.resolved_insertions[subject.track_id],
                        camera_from_calibration(calib),
                        np.linalg.inv(self.world_from_ego @ calib.cam_to_ego),
                        occluders=occluders,
                        other_insertions=tuple(
                            item
                            for key, item in self.resolved_insertions.items()
                            if key != subject.track_id
                        ),
                        camera_name=camera,
                    )
                else:
                    evidence = None
            self._cache[key] = getattr(evidence, "visibility", evidence)
        return self._cache[key]
