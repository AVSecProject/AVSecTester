"""Replay prescribed CARLA motion until actors are released to driving or attack commands."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from avsectester.geometry.poses import mirror_y_points, mirror_y_transform
from avsectester.data_types import ActorState, Control

from .conversions import rotation_to_carla


def _place(actor, state):
    import carla

    pose = mirror_y_transform(np.asarray(state.pose, dtype=float))
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Prescribed CARLA pose must be a finite 4x4 matrix")
    pitch, yaw, roll = rotation_to_carla(pose[:3, :3])
    rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
    actor.set_transform(carla.Transform(carla.Location(*pose[:3, 3]), rotation))
    actor.set_target_velocity(carla.Vector3D(*mirror_y_points(state.velocity)))


def _trajectory_tick(actor, state, dt):
    """Apply the desired next state through a physics tick, retaining measured velocity.

    CARLA reports zero velocity for a teleported actor with physics disabled. Place the
    actor one integration interval before the requested position, then let native physics
    advance it with the prescribed velocity. References retain the actual resulting state.
    """
    import carla

    pose = np.asarray(state.pose, dtype=float).copy()
    pose[:3, 3] -= np.asarray(state.velocity) * dt
    actor.set_simulate_physics(True)
    # A prescribed 3-D path already specifies its vertical motion. Adding gravity here
    # would change its declared velocity by g*dt on every tick (even for a level path).
    actor.set_enable_gravity(False)
    actor.apply_control(carla.VehicleControl())
    actor.set_target_angular_velocity(carla.Vector3D())
    _place(actor, replace(state, pose=pose))


class RecipeReplay:
    """Prescribe background motion; release ego to driving commands after pre-roll."""

    def __init__(self, simulator_runtime, recipe):
        self.simulator_runtime, self.recipe = simulator_runtime, recipe
        self.index = 0
        self.pre_roll = True
        self.released: set[str] = set()  # actors an attack now controls

    def initialize(self):
        import carla
        for name, spec in self.recipe.motion.items():
            actor = self.simulator_runtime.native["actors"][name]
            actor.set_autopilot(False, self.simulator_runtime.client.traffic_manager_port)
            if spec.trajectory:
                actor.set_simulate_physics(False)
                _place(actor, spec.trajectory[0])
            else:
                actor.apply_control(carla.VehicleControl(brake=1.0))
        self.origins = None

    def activate(self):
        trajectories = {name: spec for name, spec in self.recipe.motion.items() if spec.trajectory}
        if trajectories:
            for name, spec in trajectories.items():
                _trajectory_tick(self.simulator_runtime.native["actors"][name], spec.trajectory[0], self.recipe.time_step)
            self.simulator_runtime.client.tick()
        self.origins = {}
        for name, actor in self.simulator_runtime.native["actors"].items():
            v = actor.get_velocity()
            self.origins[name] = ActorState(mirror_y_transform(np.asarray(actor.get_transform().get_matrix())),
                                           tuple(mirror_y_points((v.x, v.y, v.z))))
        return int(bool(trajectories))

    def hold_origin(self):
        for name, state in self.origins.items():
            if name not in self.released:
                _trajectory_tick(self.simulator_runtime.native["actors"][name], state, self.recipe.time_step)

    def finish_holding(self):
        for name, spec in self.recipe.motion.items():
            if not spec.trajectory:
                self.simulator_runtime.native["actors"][name].set_enable_gravity(True)

    def before_step(self):
        if self.index >= self.recipe.decision_steps:
            raise ValueError("CARLA step exceeds terminal recipe support")
        import carla

        for identity, motion in self.recipe.motion.items():
            if identity == "ego" and not self.pre_roll or identity in self.released:
                continue
            actor = self.simulator_runtime.native["actors"][identity]
            if motion.trajectory:
                _trajectory_tick(actor, motion.trajectory[self.index + 1], self.recipe.time_step)
            elif identity != "ego":
                control = motion.controls[self.index]
                actor.apply_control(carla.VehicleControl(
                    throttle=control.throttle, brake=control.brake, steer=control.steer,
                ))

    def after_step(self):
        self.index += 1

    def ego_command(self):
        motion = self.recipe.motion["ego"]
        return motion.controls[self.index] if motion.controls else Control()

    def release(self, identity):
        """Stop prescribing an actor's motion; physics and gravity act on it again."""
        self.released.add(identity)
        motion = self.recipe.motion.get(identity)
        if motion is not None and motion.trajectory:
            actor = self.simulator_runtime.native["actors"][identity]
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)

    def release_ego(self):
        self.pre_roll = False
        if self.recipe.motion["ego"].trajectory:
            self.simulator_runtime.ego.actor.set_simulate_physics(True)
            self.simulator_runtime.ego.actor.set_enable_gravity(True)
