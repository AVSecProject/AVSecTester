"""In-process ego motion from actuator or rig-frame trajectory commands."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar, Protocol

from avsectester.data_types import Control, validate_command_modes
from avsectester.validation import require_positive_integer, require_positive_number


@dataclass
class PlanarEgoState:
    """Planar physical ego state in meters, radians and seconds.

    X/Y/yaw locate the ego origin in world coordinates. Speed and lateral_speed are
    longitudinal and lateral velocity in ego axes; yaw_rate is radians per second and
    acceleration is longitudinal meters per second squared.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # rad
    speed: float = 0.0  # m/s
    t: float = 0.0  # s
    lateral_speed: float = 0.0
    yaw_rate: float = 0.0
    acceleration: float = 0.0


class EgoDynamics(Protocol):
    """A custom ego-motion implementation, with an optional ``command_modes`` declaration."""

    def reset(self, state: PlanarEgoState) -> None:
        """Optionally initialize integrator state at the recorded start."""
        ...

    def step(self, pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState:
        """Advance the physical ego state by ``dt`` seconds using the final command."""
        ...


@dataclass
class KinematicBicycle:
    """Integrate actuator commands with a deterministic kinematic bicycle model.

    Maps a `avsectester.data_types.Control` (throttle/steer/brake) to motion. Pure and in-process,
    so a step is trivially inspectable and deterministic.
    """

    wheelbase: float = 2.8
    max_accel: float = 3.0  # m/s^2 at full throttle
    max_brake: float = 8.0  # m/s^2 at full brake
    max_steer: float = 0.6  # rad at full steer
    command_modes: ClassVar[frozenset[str]] = frozenset({"actuator"})
    'Control command modes this ego-motion model accepts.'

    def reset(self, state: PlanarEgoState) -> None:
        """No warmup is needed: this integrator carries all state in its argument."""

    def step(self, pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState:
        """Integrate actuator commands for one time interval using bicycle dynamics."""
        validate_command_modes(control, self.command_modes)
        require_positive_number(dt, "dt")
        accel = control.throttle * self.max_accel - control.brake * self.max_brake
        speed = max(0.0, pose.speed + accel * dt)
        steer = control.steer * self.max_steer
        yaw = pose.yaw + (speed / self.wheelbase) * math.tan(steer) * dt
        return PlanarEgoState(
            x=pose.x + speed * math.cos(yaw) * dt,
            y=pose.y + speed * math.sin(yaw) * dt,
            yaw=yaw,
            speed=speed,
            t=pose.t + dt,
            yaw_rate=(yaw - pose.yaw) / dt,
            acceleration=(speed - pose.speed) / dt,
        )


@dataclass
class TrajectoryFollower:
    """Realize a rig-frame planned trajectory by direct position interpolation.

    ``Control.trajectory`` is ``[((x,y,z),(w,x,y,z),t_us), ...]`` in the **rig frame** (x forward,
    y left) at absolute microsecond timestamps. Each step interpolates the rig-frame position at
    ``pose.t + dt`` and places the ego there in world coordinates — an open-loop follower (the ego
    realizes exactly what the policy planned). With no trajectory it coasts to a stop.
    """

    decel: float = 4.0  # m/s^2 when coasting with no plan
    command_modes: ClassVar[frozenset[str]] = frozenset({"trajectory"})
    'Control command modes this ego-motion model accepts.'

    def reset(self, state: PlanarEgoState) -> None:
        """No warmup is needed: the provided state anchors each planned trajectory."""

    def step(self, pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState:
        """Track the requested trajectory for one time interval."""
        validate_command_modes(control, self.command_modes)
        require_positive_number(dt, "dt")
        traj = control.trajectory or []
        if not traj:
            speed = max(0.0, pose.speed - self.decel * dt)
            return PlanarEgoState(pose.x, pose.y, pose.yaw, speed, pose.t + dt,
                                  acceleration=(speed - pose.speed) / dt)
        dx, dy = self._interp_rig_xy(traj, (pose.t + dt) * 1e6, pose.t * 1e6)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        dist = math.hypot(dx, dy)
        yaw = pose.yaw + math.atan2(dy, dx) if dist > 1e-3 else pose.yaw
        return PlanarEgoState(
            x=pose.x + dx * c - dy * s,
            y=pose.y + dx * s + dy * c,
            yaw=yaw,
            speed=dist / dt,
            t=pose.t + dt,
            yaw_rate=(yaw - pose.yaw) / dt,
            acceleration=(dist / dt - pose.speed) / dt,
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


@dataclass
class AlpaSimController:
    """AlpaSim's in-process controller and dynamic bicycle model.

    Defaults match AlpaSim affc2ea: linear MPC, a 20-step horizon and 0.1 s control
    period. Gains default to ControllerConfig's MPCGains; an explicit mapping overrides
    individual named gains. VehicleModel.Parameters are AlpaSim's Ford Fusion defaults,
    rather than values inferred from a USDZ. AlpaSim imports and controller construction
    happen on reset, so configuring this component requires no installed controller package.
    """

    mpc_implementation: str = "linear"
    n_horizon: int = 20
    dt_mpc: float = .1
    gains: Mapping[str, float | int] | None = None
    command_modes: ClassVar[frozenset[str]] = frozenset({"trajectory"})
    'Control command modes this ego-motion model accepts.'
    EPOCH_US: ClassVar[int] = 10_000_000
    'Timestamp offset in microseconds used by the trajectory interface.'

    def __post_init__(self) -> None:
        require_positive_integer(self.n_horizon, "n_horizon")
        require_positive_number(self.dt_mpc, "dt_mpc")
        if not isinstance(self.mpc_implementation, str) or not self.mpc_implementation:
            raise ValueError("mpc_implementation must be a nonempty string")
        if self.gains is not None:
            if not isinstance(self.gains, Mapping):
                raise TypeError("gains must be a mapping of MPCGains parameter names")
            self.gains = dict(self.gains)
        self._bindings: dict[str, Any] | None = None
        self._system: Any = None
        self._timestamp_us: int | None = None
        self._first_step = True

    @classmethod
    def _timestamp(cls, time: float) -> int:
        if not math.isfinite(time):
            raise ValueError("controller time must be finite")
        timestamp = round(time * 1e6) + cls.EPOCH_US
        if timestamp < 0:
            raise ValueError("controller time precedes the fixed timestamp offset")
        return timestamp

    def _state_message(self, state: PlanarEgoState) -> Any:
        assert self._bindings is not None
        make = self._bindings
        return make["StateAtTime"](
            timestamp_us=self._timestamp(state.t),
            pose=make["Pose"](
                vec=make["Vec3"](x=state.x, y=state.y, z=0.),
                quat=make["Quat"](w=math.cos(state.yaw / 2), z=math.sin(state.yaw / 2)),
            ),
            state=make["DynamicState"](
                linear_velocity=make["Vec3"](x=state.speed, y=state.lateral_speed, z=0.),
                angular_velocity=make["Vec3"](z=state.yaw_rate),
                linear_acceleration=make["Vec3"](x=state.acceleration),
            ),
        )

    def reset(self, state: PlanarEgoState) -> None:
        """Create a fresh controller at the supplied recorded physical state."""
        if self._bindings is None:
            from alpasim_controller.mpc_controller import ControllerConfig, MPCGains
            from alpasim_controller.system import create_system
            from alpasim_grpc.v0 import common_pb2, controller_pb2

            self._bindings = {
                name: getattr(common_pb2, name) for name in (
                    "StateAtTime", "Pose", "Vec3", "Quat", "DynamicState", "PoseAtTime", "Trajectory")
            }
            self._bindings.update(ControllerConfig=ControllerConfig, MPCGains=MPCGains,
                                  create_system=create_system,
                                  Request=controller_pb2.RunControllerAndVehicleModelRequest)
        make = self._bindings
        configuration = make["ControllerConfig"](
            mpc_implementation=self.mpc_implementation, n_horizon=self.n_horizon,
            dt_mpc=self.dt_mpc, gains=make["MPCGains"](**dict(self.gains or {})),
        )
        self._system = make["create_system"](
            log_file=os.devnull, initial_state=self._state_message(state),
            controller_config=configuration,
        )
        self._timestamp_us = self._timestamp(state.t)
        self._first_step = True

    def step(self, pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState:
        """Track the t0-led rig-frame plan and return the controller's final physical state."""
        import numpy as np

        validate_command_modes(control, self.command_modes)
        require_positive_number(dt, "dt")
        if self._system is None or self._bindings is None or self._timestamp_us is None:
            raise RuntimeError("Reset AlpaSimController before stepping")
        if self._timestamp(pose.t) != self._timestamp_us:
            raise ValueError("Controller state time must match its latest propagated state")
        trajectory = [] if control.trajectory is None else list(control.trajectory)
        if len(trajectory) < 2:
            raise ValueError("Controller trajectory must start at t0 and contain a future waypoint")
        origin, rotation, timestamp = trajectory[0]
        if (round(timestamp) != round(pose.t * 1e6)
                or not np.allclose(origin, (0., 0., 0.), atol=1e-6, rtol=0)
                or not np.allclose(np.abs(rotation), (1., 0., 0., 0.), atol=1e-6, rtol=0)):
            raise ValueError("Controller trajectory must start at t0 with zero position and unit rotation")
        timestamps = [round(time) + self.EPOCH_US for _xyz, _rotation, time in trajectory]
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("Controller trajectory timestamps must be strictly increasing")
        future = self._timestamp(pose.t + dt)
        if future <= self._timestamp_us:
            raise ValueError("dt must advance controller time by at least one microsecond")
        make = self._bindings
        waypoints = []
        for (position, quaternion, _time), time in zip(trajectory, timestamps):
            x, y, z = position
            w, qx, qy, qz = quaternion
            waypoints.append(make["PoseAtTime"](
                timestamp_us=time, pose=make["Pose"](
                    vec=make["Vec3"](x=x, y=y, z=z),
                    quat=make["Quat"](w=w, x=qx, y=qy, z=qz),
                ),
            ))
        response = self._system.run_controller_and_vehicle_model(make["Request"](
            state=self._state_message(pose),
            planned_trajectory_in_rig=make["Trajectory"](poses=waypoints),
            future_time_us=future, coerce_dynamic_state=self._first_step,
        ))
        final = response.states[-1]
        self._timestamp_us = final.timestamp_us
        self._first_step = False
        orientation = final.pose_local_to_rig.quat
        yaw = math.atan2(2 * (orientation.w * orientation.z + orientation.x * orientation.y),
                         1 - 2 * (orientation.y ** 2 + orientation.z ** 2))
        dynamic = final.dynamic_state
        return PlanarEgoState(
            x=final.pose_local_to_rig.vec.x, y=final.pose_local_to_rig.vec.y, yaw=yaw,
            speed=dynamic.linear_velocity.x,
            t=(final.timestamp_us - self.EPOCH_US) / 1e6,
            lateral_speed=dynamic.linear_velocity.y,
            yaw_rate=dynamic.angular_velocity.z,
            acceleration=dynamic.linear_acceleration.x,
        )


__all__ = [
    'PlanarEgoState',
    'EgoDynamics',
    'KinematicBicycle',
    'TrajectoryFollower',
    'AlpaSimController',
]
