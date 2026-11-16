"""Replay prescribed CARLA motion until actors are released to driving or attack commands."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from avsectester.geometry.poses import interpolate_transform, mirror_y_points, mirror_y_transform
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
    """Prescribe background motion. Release ego to driving commands after pre-roll."""

    def __init__(self, simulator_runtime, recipe):
        self.simulator_runtime, self.recipe = simulator_runtime, recipe
        self.index = 0
        self.tick_s = simulator_runtime.timing_capabilities.base_tick_us / 1e6
        self.ticks_per_decision = simulator_runtime.simulator.decision_ticks
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
                _trajectory_tick(self.simulator_runtime.native["actors"][name], spec.trajectory[0], self.tick_s)
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
                _trajectory_tick(self.simulator_runtime.native["actors"][name], state, self.tick_s)

    def finish_holding(self):
        for name, spec in self.recipe.motion.items():
            if not spec.trajectory:
                self.simulator_runtime.native["actors"][name].set_enable_gravity(True)

    def before_step(self):
        if self.index >= self.recipe.decision_steps * self.ticks_per_decision:
            raise ValueError("CARLA step exceeds terminal recipe support")
        import carla

        for identity, motion in self.recipe.motion.items():
            if identity == "ego" and not self.pre_roll or identity in self.released:
                continue
            actor = self.simulator_runtime.native["actors"][identity]
            if motion.trajectory:
                position, remainder = divmod(self.index + 1, self.ticks_per_decision)
                state = motion.trajectory[position]
                if remainder:
                    after = motion.trajectory[position + 1]
                    fraction = remainder / self.ticks_per_decision
                    state = replace(state,
                        pose=interpolate_transform(state.pose, after.pose, fraction),
                        velocity=tuple((1 - fraction) * np.asarray(state.velocity)
                                       + fraction * np.asarray(after.velocity)))
                _trajectory_tick(actor, state, self.tick_s)
            elif identity != "ego":
                control = motion.controls[self.index // self.ticks_per_decision]
                actor.apply_control(carla.VehicleControl(
                    throttle=control.throttle, brake=control.brake, steer=control.steer,
                ))

    def after_step(self):
        self.index += 1

    def ego_command(self):
        motion = self.recipe.motion["ego"]
        return motion.controls[self.index // self.ticks_per_decision] if motion.controls else Control()

    def release(self, identity):
        """Stop prescribing an actor's motion. Physics and gravity act on it again."""
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
