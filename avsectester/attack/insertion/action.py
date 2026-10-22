"""Portable insertion actions: the simulator supplies geometry, the attack composites pixels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

from avsectester.attack.runtime import SensorAttackAction
from .assets import Insertion, resolve_insertion
from .compositor import PatchCompositor, render_resolved
from .visibility import CuboidVisibilityEstimator, DepthVisibilityEstimator


log = logging.getLogger(__name__)


@dataclass
class InsertionAttackAction(SensorAttackAction):
    """Render explicit planes with aligned depth when present, otherwise actor cuboids."""

    sensor: str
    insertion: Insertion
    compositor: PatchCompositor | None = None
    rendered: bool = field(default=False, init=False)
    reason: str = field(default="", init=False)
    occlusion_source: str = field(default="", init=False)
    ignored_occluders: tuple[str, ...] = field(default=(), init=False)

    def receipt_details(self):
        return {"rendered": self.rendered, "reason": self.reason,
                "occlusion_source": self.occlusion_source,
                "ignored_occluders": self.ignored_occluders}

    def resolve(self, view):
        return resolve_insertion(self.insertion, view.actors, view.ego)

    def evidence(self, view, resolved=None):
        if view.camera is None or view.cam_from_world is None:
            raise ValueError("InsertionAttackAction requires a calibrated camera SensorView")
        resolved = self.resolve(view) if resolved is None else resolved
        if view.depth is not None:
            return DepthVisibilityEstimator().estimate(
                resolved, view.camera, view.cam_from_world, scene_depth=view.depth,
                depth_convention="z", camera_name=view.sensor)
        return CuboidVisibilityEstimator().estimate(
            resolved, view.camera, view.cam_from_world, occluders=view.actors,
            camera_name=view.sensor, world_objects=tuple(view.objects.values()))

    def apply(self, data: Any, view):
        resolved = self.resolve(view)
        evidence = self.evidence(view, resolved)
        self.occlusion_source = "depth" if view.depth is not None else "actor_cuboids"
        self.ignored_occluders = evidence.ignored_occluders
        self.rendered = evidence.visibility is not None and bool(evidence.visible_mask.any())
        self.reason = evidence.reason if not self.rendered else ""
        if not self.rendered:
            self.reason = self.reason or "No visible insertion pixels"
            log.info("Insertion %s was not rendered: %s", self.insertion.id, self.reason)
            return data
        return render_resolved(data, view.camera, view.cam_from_world, resolved, evidence,
                               compositor=self.compositor)[0]
