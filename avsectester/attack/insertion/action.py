"""World-object and captured-image insertion actions for user attacks."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import logging

from avsectester.attack.runtime import SensorAttackAction, WorldAttackAction
from .assets import Insertion, resolve_insertion
from .compositor import PatchCompositor, render_resolved
from .visibility import CuboidVisibilityEstimator, DepthVisibilityEstimator


if TYPE_CHECKING:
    import numpy as np
    from avsectester.closed_loop.simulators.base import SimulatorRuntime
    from avsectester.rendering import Harmonizer, ObjectAsset


log = logging.getLogger(__name__)


@dataclass
class InsertObject(WorldAttackAction):
    """Insert arbitrary textured geometry into all cameras through the simulator.

    The object's origin is its bottom centre; ``world_from_object`` maps that
    right-handed object frame (+X front, +Y left, +Z up) into the world.
    Setup insertion is visible in the first capture; step insertion in the next.
    """

    object_id: str
    asset: ObjectAsset
    world_from_object: np.ndarray
    harmonizer: Harmonizer | None = None

    def __post_init__(self) -> None:
        from avsectester.rendering import PlacedObject

        placed = PlacedObject(self.object_id, self.asset, self.world_from_object, self.harmonizer)
        self.world_from_object = placed.world_from_object

    def apply(self, simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None:
        simulator_runtime.insert_object(self.object_id, self.asset, self.world_from_object,
                                        harmonizer=self.harmonizer)

        def remove_if_present() -> None:
            try:
                simulator_runtime.remove_object(self.object_id)
            except KeyError:
                # An explicit world action may have removed it before run cleanup.
                pass

        resources.callback(remove_if_present)

    def receipt_details(self) -> Mapping[str, Any]:
        return {
            "object_id": self.object_id, "asset_category": self.asset.category,
            "world_from_object": self.world_from_object.copy(),
        }


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
