"""Measure native CARLA blueprints and match their dimensions to recorded actors."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from avsectester.geometry.poses import validated_transform, validated_vector

from .world import CarlaWorldSession


@dataclass(frozen=True)
class BlueprintGeometry:
    """Full box sizes and right-handed box center in meters, with optional wheel geometry.

    Rear axle and wheelbase use meters. The maximum steering angle uses degrees.
    """

    id: str
    dimensions: tuple[float, float, float]
    box_location: tuple[float, float, float]
    rear_axle_x: float | None = None
    wheelbase: float | None = None
    max_steer_angle: float | None = None

    def box_transform(self, world_from_actor: ArrayLike) -> np.ndarray:
        """Return the world transform of the box center for an actor at the given pose."""
        world_from_box = validated_transform(world_from_actor).copy()
        world_from_box[:3, 3] += world_from_box[:3, :3] @ np.asarray(self.box_location, dtype=float)
        return world_from_box


_GEOMETRY_CACHE: dict[tuple[str, str], tuple[BlueprintGeometry, ...]] = {}


def _geometry(actor: Any, identity: str) -> BlueprintGeometry:
    box = actor.bounding_box
    dimensions = (2. * box.extent.x, 2. * box.extent.y, 2. * box.extent.z)
    box_location = (box.location.x, -box.location.y, box.location.z)
    if not identity.startswith("vehicle."):
        return BlueprintGeometry(identity, dimensions, box_location)
    wheels = actor.get_physics_control().wheels
    if len(wheels) < 2:
        return BlueprintGeometry(identity, dimensions, box_location)
    actor_from_world = np.linalg.inv(np.asarray(actor.get_transform().get_matrix(), dtype=float))
    locations = np.array([(wheel.location.x, wheel.location.y, wheel.location.z) for wheel in wheels])
    wheel_x = locations @ actor_from_world[0, :3] + actor_from_world[0, 3]
    ordered = np.argsort(wheel_x)
    axle_size = 2 if len(wheels) >= 4 else 1
    rear_axle_x = float(wheel_x[ordered[:axle_size]].mean())
    front_indices = ordered[-axle_size:]
    wheelbase = float(wheel_x[front_indices].mean()) - rear_axle_x
    max_steer_angle = max(float(wheels[index].max_steer_angle) for index in front_indices)
    return BlueprintGeometry(identity, dimensions, box_location, rear_axle_x, wheelbase, max_steer_angle)


def probe_blueprints(session: CarlaWorldSession, pattern: str) -> tuple[BlueprintGeometry, ...]:
    """Measure spawnable blueprints, caching by server version and pattern."""
    import carla

    key = (session.client.get_server_version(), pattern)
    if key not in _GEOMETRY_CACHE:
        spawn_points = session.world.get_map().get_spawn_points()
        origin = spawn_points[0] if spawn_points else carla.Transform()
        location = origin.location
        elevated = carla.Transform(carla.Location(location.x, location.y, location.z + 2.),
                                   origin.rotation)
        geometries = []
        blueprints = session.world.get_blueprint_library().filter(pattern)
        for blueprint in sorted(blueprints, key=lambda item: item.id):
            actor = session.spawn(blueprint, elevated)
            if actor is None:
                continue
            try:
                session.tick()
                geometries.append(_geometry(actor, blueprint.id))
            finally:
                session.destroy(actor)
                session.tick()
        result = tuple(geometries)
        if result:
            _GEOMETRY_CACHE[key] = result
        return result
    return _GEOMETRY_CACHE[key]


def best_fit_blueprint(
    geometries: Sequence[BlueprintGeometry], dimensions: ArrayLike,
) -> BlueprintGeometry:
    """Choose the smallest squared error in half-dimensions, in input order on ties."""
    if not geometries:
        raise ValueError("No CARLA blueprint geometries are available")
    target = validated_vector(dimensions, "dimensions") * .5
    return min(geometries, key=lambda item: float(np.sum((np.asarray(item.dimensions) * .5 - target) ** 2)))


__all__ = ['BlueprintGeometry', 'best_fit_blueprint', 'probe_blueprints']
