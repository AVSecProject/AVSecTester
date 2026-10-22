"""In-process ego motion from actuator or rig-frame trajectory commands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar, Protocol

from avsectester.data_types import Control, validate_command_modes


@dataclass
class EgoPose:
    """Ego world state we own and integrate (2-D kinematic dynamics)."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # rad
    speed: float = 0.0  # m/s
    t: float = 0.0  # s


class EgoDynamics(Protocol):
    """A custom ego-motion implementation, with an optional ``command_modes`` declaration."""

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
        """Advance the physical ego pose by ``dt`` seconds using the final command."""
        ...


@dataclass
class KinematicBicycle:
    """A simple, transparent ego dynamics model — the shared 'physical control' for NuRec worlds.

    Maps a :class:`~avsectester.data_types.Control` (throttle/steer/brake) to motion. Pure and in-process,
    so a step is trivially inspectable and deterministic.
    """

    wheelbase: float = 2.8
    max_accel: float = 3.0  # m/s^2 at full throttle
    max_brake: float = 8.0  # m/s^2 at full brake
    max_steer: float = 0.6  # rad at full steer
    command_modes: ClassVar[frozenset[str]] = frozenset({"actuator"})

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
        validate_command_modes(control, self.command_modes)
        accel = control.throttle * self.max_accel - control.brake * self.max_brake
        speed = max(0.0, pose.speed + accel * dt)
        steer = control.steer * self.max_steer
        yaw = pose.yaw + (speed / self.wheelbase) * math.tan(steer) * dt
        return EgoPose(
            x=pose.x + speed * math.cos(yaw) * dt,
            y=pose.y + speed * math.sin(yaw) * dt,
            yaw=yaw,
            speed=speed,
            t=pose.t + dt,
        )


@dataclass
class TrajectoryFollower:
    """Advance the ego along a planned trajectory (for trajectory-output stacks like Alpamayo).

    ``Control.trajectory`` is ``[((x,y,z),(w,x,y,z),t_us), ...]`` in the **rig frame** (x forward,
    y left) at absolute microsecond timestamps. Each step interpolates the rig-frame position at
    ``pose.t + dt`` and places the ego there in world coordinates — an open-loop follower (the ego
    realizes exactly what the policy planned). With no trajectory it coasts to a stop.
    """

    decel: float = 4.0  # m/s^2 when coasting with no plan
    command_modes: ClassVar[frozenset[str]] = frozenset({"trajectory"})

    def step(self, pose: EgoPose, control: Control, dt: float) -> EgoPose:
        validate_command_modes(control, self.command_modes)
        traj = control.trajectory or []
        if not traj:
            speed = max(0.0, pose.speed - self.decel * dt)
            return EgoPose(pose.x, pose.y, pose.yaw, speed, pose.t + dt)
        dx, dy = self._interp_rig_xy(traj, (pose.t + dt) * 1e6, pose.t * 1e6)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        dist = math.hypot(dx, dy)
        yaw = pose.yaw + math.atan2(dy, dx) if dist > 1e-3 else pose.yaw
        return EgoPose(
            x=pose.x + dx * c - dy * s,
            y=pose.y + dx * s + dy * c,
            yaw=yaw,
            speed=dist / dt,
            t=pose.t + dt,
        )

    @staticmethod
    def _interp_rig_xy(traj: list, target_us: float, origin_us: float) -> tuple[float, float]:
        pts = [(t_us, xyz[0], xyz[1]) for xyz, _quat, t_us in traj]
        if target_us <= pts[0][0]:
            # Rig-frame displacements start at the current ego position. A physics
            # substep before the first waypoint must cover only part of that segment.
            interval = pts[0][0] - origin_us
            fraction = (
                min(1.0, max(0.0, (target_us - origin_us) / interval))
                if interval > 0 else 1.0
            )
            return fraction * pts[0][1], fraction * pts[0][2]
        for (t0, x0, y0), (t1, x1, y1) in pairwise(pts):
            if target_us <= t1:
                a = (target_us - t0) / (t1 - t0) if t1 > t0 else 0.0
                return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
        return pts[-1][1], pts[-1][2]
