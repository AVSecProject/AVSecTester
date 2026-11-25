"""Spawn and control a CARLA vehicle through a rear-axle rig reference."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from avsectester.geometry.poses import mirror_y_points, mirror_y_transform, validated_transform

from .conversions import carla_transform
from .vehicles import BlueprintGeometry
from .world import CarlaWorldSession


@dataclass(frozen=True)
class CarlaEgoState:
    """Rear-axle pose, velocity in m/s, yaw rate in rad/s and acceleration in m/s²."""

    world_from_rig: np.ndarray
    velocity: np.ndarray
    yaw_rate: float
    acceleration: np.ndarray


def _actor_from_rig(geometry: BlueprintGeometry) -> np.ndarray:
    if geometry.rear_axle_x is None:
        raise ValueError(f"CARLA ego blueprint {geometry.id!r} has no measured rear axle")
    transform = np.eye(4)
    transform[0, 3] = geometry.rear_axle_x
    return transform


def _world_vector(value: Any) -> np.ndarray:
    return mirror_y_points((value.x, value.y, value.z))


class CarlaEgoVehicle:
    """Control a native vehicle relative to its rear axle in right-handed CARLA coordinates."""

    def __init__(self, session: CarlaWorldSession, actor: Any, geometry: BlueprintGeometry) -> None:
        self.session, self.actor, self.geometry = session, actor, geometry
        self._actor_from_rig = _actor_from_rig(geometry)
        self._rig_from_actor = np.linalg.inv(self._actor_from_rig)
        self._physics_enabled = True
        self._spawn_lift_m = 0.

    @property
    def spawn_lift_m(self) -> float:
        """Vertical lift applied to the requested pose while finding a free spawn."""
        return self._spawn_lift_m

    @property
    def physics_enabled(self) -> bool:
        """Whether the vehicle is configured for native physics at its next tick."""
        return self._physics_enabled

    @classmethod
    def spawn(
        cls, session: CarlaWorldSession, geometry: BlueprintGeometry, world_from_rig: ArrayLike,
    ) -> CarlaEgoVehicle:
        """Try a rear-axle pose, then raise it in 0.05 m increments up to 1 m on collision."""
        world_from_actor = validated_transform(world_from_rig) @ np.linalg.inv(_actor_from_rig(geometry))
        blueprint = session.blueprint(geometry.id)
        for increment in range(21):
            pose = world_from_actor.copy()
            pose[2, 3] += .05 * increment
            actor = session.spawn(blueprint, carla_transform(pose))
            if actor is not None:
                ego = cls(session, actor, geometry)
                ego._spawn_lift_m = .05 * increment
                return ego
        raise RuntimeError(f"Could not spawn CARLA ego blueprint {geometry.id!r} within 1 m of the rig pose")

    def hold(self) -> None:
        """Queue full service brake and hand brake for settling."""
        import carla

        self.session.queue(carla.command.ApplyVehicleControl(
            self.actor.id, carla.VehicleControl(brake=1., hand_brake=True)))

    def release(self) -> None:
        """Queue release of the service and hand brakes."""
        self.apply_control(0., 0., 0.)

    def set_kinematic(self, world_from_rig: ArrayLike) -> None:
        """Queue a kinematic rig pose, switching off native physics when needed."""
        world_from_actor = validated_transform(world_from_rig) @ self._rig_from_actor
        if self._physics_enabled:
            self.session.place_kinematic(self.actor, world_from_actor)
            self._physics_enabled = False
        else:
            self.session.place(self.actor, world_from_actor)

    def enable_physics(self, velocity: ArrayLike) -> None:
        """Queue native physics and a target velocity in right-handed CARLA world axes."""
        import carla

        target = carla.Vector3D(*mirror_y_points(velocity))
        if not self._physics_enabled:
            self.session.queue(carla.command.SetSimulatePhysics(self.actor.id, True))
            self._physics_enabled = True
        self.session.queue(carla.command.ApplyTargetVelocity(self.actor.id, target))

    def apply_control(self, throttle: float, brake: float, steer: float) -> None:
        """Queue actuator commands with positive steering turning right."""
        import carla

        self.session.queue(carla.command.ApplyVehicleControl(
            self.actor.id, carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)))

    def box_transform(self, world_from_rig: ArrayLike) -> np.ndarray:
        """Return the world transform of the body box center for a rear-axle pose."""
        return self.geometry.box_transform(validated_transform(world_from_rig) @ self._rig_from_actor)

    def read(self) -> CarlaEgoState:
        """Read the rig pose and native velocity and acceleration after a world tick."""
        world_from_actor = mirror_y_transform(self.actor.get_transform().get_matrix())
        return CarlaEgoState(world_from_actor @ self._actor_from_rig,
                            _world_vector(self.actor.get_velocity()),
                            -math.radians(self.actor.get_angular_velocity().z),
                            _world_vector(self.actor.get_acceleration()))


__all__ = ['CarlaEgoState', 'CarlaEgoVehicle']
