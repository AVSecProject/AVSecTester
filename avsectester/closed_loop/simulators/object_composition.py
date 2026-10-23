"""Compose placed objects into native captures without owning simulator state or caches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from avsectester.data_types import Observation, read_camera_rgb, write_camera_rgb

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.base import SensorGeometry, SensorView
    from avsectester.rendering import CompositePlan, ObjectVisibility, PlacedObject


def compose_capture(
    observation: Observation,
    objects: Mapping[str, PlacedObject],
    camera_sensors: Iterable[str],
    sensor_geometry: Callable[[str], SensorGeometry],
    sensor_view: Callable[[str], SensorView],
    composite_plan: Callable[[SensorView, Sequence[PlacedObject]], CompositePlan],
) -> tuple[Observation, dict[str, SensorView], dict[str, dict[str, ObjectVisibility]]]:
    """Compose one capture and return its observation, camera views and object visibility.

    Projection screening uses lightweight camera geometry. A full native view, including
    optional scene depth, is requested only for cameras whose images contain an object, and
    ``composite_plan`` plans the objects into it. The simulator owns detachment, capture
    snapshots, successful caching and failed caching.
    """
    views: dict[str, SensorView] = {}
    visibility: dict[str, dict[str, ObjectVisibility]] = {identity: {} for identity in objects}
    if objects:
        import numpy as np

        from avsectester.geometry.projection import project_box
        from avsectester.rendering import ObjectComposition

        for sensor in sorted(camera_sensors):
            geometry = sensor_geometry(sensor)
            candidates = []
            for placed in objects.values():
                box_from_object = np.eye(4)
                box_from_object[2, 3] = placed.asset.extent[2] / 2
                if project_box(geometry.camera, geometry.cam_from_world,
                               placed.world_from_object @ box_from_object,
                               placed.asset.extent) is not None:
                    candidates.append(placed)
            if not candidates:
                continue
            view = sensor_view(sensor)
            captured = read_camera_rgb(observation, view.sensor)
            if captured is None:
                raise ValueError(f"Missing RGB camera {view.sensor!r}")
            plan = composite_plan(view, candidates)
            result = plan.apply(captured[1])
            observation = write_camera_rgb(observation, view.sensor, result.image)
            if view.depth is None:
                depth = None
            elif plan.occlusion_source == "scene_visibility":
                # Measured visibility plans no scene depth; visible surfaces replace the measured depth.
                depth = np.where(result.visible_mask, result.depth, view.depth)
            else:
                depth = result.depth
            views[view.sensor] = replace(
                view, data=result.image, depth=depth,
                objects={placed.object_id: placed for placed in candidates},
                composition=ObjectComposition(np.array(captured[1], copy=True), plan),
            )
            for placed in candidates:
                visibility[placed.object_id][view.sensor] = result.object_visibility[placed.object_id]
    return observation, views, visibility
