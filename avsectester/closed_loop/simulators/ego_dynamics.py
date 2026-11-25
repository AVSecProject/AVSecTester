"""In-process ego motion from actuator or rig-frame trajectory commands."""

from __future__ import annotations

import math
import bisect
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields
from itertools import pairwise
from typing import Any, ClassVar, Protocol

from avsectester.validation import timestamp_to_microseconds
from avsectester.closed_loop.timing import seconds_to_microseconds
from avsectester.data_types import Control, validate_command_modes
from avsectester.validation import require_positive_integer, require_positive_number


@dataclass
class PlanarEgoState:
    """Planar physical ego state in meters, radians and seconds.

    X/Y/yaw locate the ego origin in world coordinates. Speed and lateral_speed are
    longitudinal and lateral velocity in ego axes. ``yaw_rate`` is radians per second and
    ``acceleration`` is longitudinal meters per second squared.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # rad
    speed: float = 0.0  # m/s
    t: float = 0.0  # s
    lateral_speed: float = 0.0
    yaw_rate: float = 0.0
    acceleration: float = 0.0


@dataclass(frozen=True)
class EgoMotion:
    """A completed motion interval with detached states at controller reporting times.

    ``at`` interpolates between neighboring reports without advancing dynamics or solving
    the controller again. Heading follows the shortest angular interval.
    """

    states: tuple[PlanarEgoState, ...]

    def __post_init__(self) -> None:
        if len(self.states) < 2 or any(not isinstance(state, PlanarEgoState) for state in self.states):
            raise ValueError("EgoMotion requires at least two PlanarEgoState reports")
        timestamps = [timestamp_to_microseconds(state.t) for state in self.states]
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("Motion report times must be strictly increasing")
        object.__setattr__(self, "states", deepcopy(tuple(self.states)))

    def at(self, time_us: int) -> PlanarEgoState:
        """Read a state at an integer time within this interval, as an independent copy."""
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("Motion sample time must be integer microseconds")
        times = [timestamp_to_microseconds(state.t) for state in self.states]
        if not times[0] <= time_us <= times[-1]:
            raise ValueError("Motion sample time lies outside the advanced interval")
        for timestamp, state in zip(times, self.states):
            if timestamp == time_us:
                return deepcopy(state)
        for left, right in pairwise(self.states):
            start, stop = timestamp_to_microseconds(left.t), timestamp_to_microseconds(right.t)
            if start < time_us < stop:
                fraction = (time_us - start) / (stop - start)
                values = {item.name: getattr(left, item.name) + fraction * (
                    getattr(right, item.name) - getattr(left, item.name)) for item in fields(left)}
                delta = math.atan2(math.sin(right.yaw - left.yaw), math.cos(right.yaw - left.yaw))
                values.update(yaw=left.yaw + fraction * delta, t=time_us / 1e6)
                return PlanarEgoState(**values)
        raise RuntimeError("Motion interval has no enclosing reports")


def _report_times(pose: PlanarEgoState, duration_us: int, requested: Sequence[int]) -> list[int]:
    require_positive_integer(duration_us, "duration_us")
    first = timestamp_to_microseconds(pose.t)
    times = []
    for time_us in requested:
        if (isinstance(time_us, bool) or not isinstance(time_us, int)
                or not first <= time_us <= first + duration_us):
            raise ValueError("Report times must be integer microseconds within the motion interval")
        times.append(time_us)
    return sorted({first, *times, first + duration_us})


def _step_motion(dynamics: EgoDynamics, pose: PlanarEgoState, control: Control, duration_us: int,
                 report_times_us: Sequence[int]) -> EgoMotion:
    """Evaluate a pure ``step`` from the interval start at each report time."""
    times = _report_times(pose, duration_us, report_times_us)
    first = times[0]
    states = [deepcopy(pose)]
    for time_us in times[1:]:
        state = dynamics.step(pose, control, (time_us - first) / 1e6)
        state.t = time_us / 1e6
        states.append(state)
    return EgoMotion(tuple(states))


def _advance_dynamics(dynamics: EgoDynamics, pose: PlanarEgoState, control: Control, duration_us: int,
                     report_times_us: Sequence[int] = ()) -> EgoMotion:
    """Advance one decision interval and report states at the requested integer times.

    Dynamics defining ``advance`` report their own states. Otherwise the pure ``step`` is
    evaluated from the interval start at each report time.
    """
    advance = getattr(dynamics, "advance", None)
    if advance is None:
        return _step_motion(dynamics, pose, control, duration_us, report_times_us)
    return advance(pose, control, duration_us, report_times_us)


class EgoDynamics(Protocol):
    """A custom ego-motion implementation, with an optional ``command_modes`` declaration.

    An implementation may also define ``advance(pose, control, duration_us, report_times_us)``
    returning `EgoMotion` when one controller interval reports intermediate states natively.
    Optional ``coerce_dynamic_state(state)`` aligns internal dynamics with the supplied physical
    state without resetting reference history.
    Simulators call it before each force-GT interval and at the handoff to driver control.
    """

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
        steer = -control.steer * self.max_steer
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
    _history: list[PlanarEgoState] = field(init=False, default_factory=list, repr=False)
    _oldest_reference_us: int | None = field(init=False, default=None, repr=False)

    def reset(self, state: PlanarEgoState) -> None:
        """Start the physical pose history used to anchor delayed trajectory references."""
        self._history = [deepcopy(state)]
        self._oldest_reference_us = timestamp_to_microseconds(state.t)

    def advance(self, pose: PlanarEgoState, control: Control, duration_us: int,
                report_times_us: Sequence[int] = ()) -> EgoMotion:
        """Report one interval and retain physical poses for subsequent delayed references."""
        start_us = timestamp_to_microseconds(pose.t)
        self._oldest_reference_us = (min(start_us, int(control.trajectory[0][2]))
                                     if control.trajectory else start_us)
        motion = _step_motion(self, pose, control, duration_us, report_times_us)
        history = [state for state in self._history
                   if timestamp_to_microseconds(state.t) < start_us] + list(motion.states)
        index = max(0, bisect.bisect_right(
            [timestamp_to_microseconds(state.t) for state in history], self._oldest_reference_us) - 1)
        self._history = history[index:]
        return motion

    def _reference_anchor(self, pose: PlanarEgoState, trajectory: Sequence[Any]) -> PlanarEgoState:
        timestamp = timestamp_to_microseconds(pose.t)
        history = [state for state in self._history if timestamp_to_microseconds(state.t) < timestamp]
        history.append(deepcopy(pose))
        reference_time = int(trajectory[0][2])
        if reference_time >= timestamp:
            return pose
        if reference_time < timestamp_to_microseconds(history[0].t):
            raise ValueError("Trajectory origin precedes the controller's physical pose history")
        if len(history) == 1:
            return history[0]
        return EgoMotion(tuple(history)).at(reference_time)

    def step(self, pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState:
        """Track the requested trajectory for one time interval."""
        validate_command_modes(control, self.command_modes)
        require_positive_number(dt, "dt")
        traj = control.trajectory or []
        target_us = timestamp_to_microseconds(pose.t) + seconds_to_microseconds(dt)
        if not traj:
            speed = max(0.0, pose.speed - self.decel * dt)
            return PlanarEgoState(pose.x, pose.y, pose.yaw, speed, target_us / 1e6,
                                  acceleration=(speed - pose.speed) / dt)
        anchor = self._reference_anchor(pose, traj)
        dx, dy = self._interp_rig_xy(traj, target_us, timestamp_to_microseconds(anchor.t))
        if anchor is not pose:
            cosine, sine = math.cos(anchor.yaw), math.sin(anchor.yaw)
            world_x = anchor.x + dx * cosine - dy * sine
            world_y = anchor.y + dx * sine + dy * cosine
            cosine, sine = math.cos(pose.yaw), math.sin(pose.yaw)
            dx, dy = (cosine * (world_x - pose.x) + sine * (world_y - pose.y),
                      -sine * (world_x - pose.x) + cosine * (world_y - pose.y))
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        dist = math.hypot(dx, dy)
        yaw = pose.yaw + math.atan2(dy, dx) if dist > 1e-3 else pose.yaw
        return PlanarEgoState(
            x=pose.x + dx * c - dy * s,
            y=pose.y + dx * s + dy * c,
            yaw=yaw,
            speed=dist / dt,
            t=target_us / 1e6,
            yaw_rate=(yaw - pose.yaw) / dt,
            acceleration=(dist / dt - pose.speed) / dt,
        )

    @staticmethod
    def _interp_rig_xy(traj: list, target_us: int, origin_us: int) -> tuple[float, float]:
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
    period. Gains default to ControllerConfig's MPCGains. An explicit mapping overrides
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
        timestamp = timestamp_to_microseconds(time) + cls.EPOCH_US
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

    def coerce_dynamic_state(self, state: PlanarEgoState) -> None:
        """Force the next native request's dynamics without discarding trajectory history."""
        self._first_step = True

    def step(self, pose: PlanarEgoState, control: Control, dt: float) -> PlanarEgoState:
        """Track the original rig-frame plan using the native controller's physical history."""
        return self._propagate(pose, control, dt, 0)[-1]

    def advance(self, pose: PlanarEgoState, control: Control, duration_us: int,
                report_times_us: Sequence[int] = ()) -> EgoMotion:
        """Run one controller interval and report poses without adding MPC solve times.

        Native pose reporting uses the configured regular reporting interval. The final
        control endpoint need not lie on that reporting grid. Zero extra reports uses the
        same native request as ``step``.
        """
        times = _report_times(pose, duration_us, report_times_us)
        interval = times[1] - times[0] if len(times) > 2 else 0
        reports = self._propagate(pose, control, duration_us / 1e6, interval)
        motion = EgoMotion((pose, *reports))
        states = []
        for time_us in times:
            state = motion.at(time_us)
            state.t = time_us / 1e6
            states.append(state)
        return EgoMotion(tuple(states))

    def _propagate(self, pose: PlanarEgoState, control: Control, dt: float,
                   reporting_interval_us: int) -> tuple[PlanarEgoState, ...]:

        validate_command_modes(control, self.command_modes)
        require_positive_number(dt, "dt")
        if self._system is None or self._bindings is None or self._timestamp_us is None:
            raise RuntimeError("Reset AlpaSimController before stepping")
        if self._timestamp(pose.t) != self._timestamp_us:
            raise ValueError("Controller state time must match its latest propagated state")
        trajectory = [] if control.trajectory is None else list(control.trajectory)
        if not trajectory:
            raise ValueError("Controller trajectory must contain at least one waypoint")
        timestamps = [round(time) + self.EPOCH_US for _xyz, _rotation, time in trajectory]
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("Controller trajectory timestamps must be strictly increasing")
        future = self._timestamp_us + seconds_to_microseconds(dt)
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
            pose_reporting_interval_us=reporting_interval_us,
        ))
        final = response.states[-1]
        self._timestamp_us = final.timestamp_us
        self._first_step = False
        result = []
        for report in response.states:
            orientation = report.pose_local_to_rig.quat
            yaw = math.atan2(2 * (orientation.w * orientation.z + orientation.x * orientation.y),
                             1 - 2 * (orientation.y ** 2 + orientation.z ** 2))
            dynamic = report.dynamic_state
            result.append(PlanarEgoState(
                x=report.pose_local_to_rig.vec.x, y=report.pose_local_to_rig.vec.y, yaw=yaw,
                speed=dynamic.linear_velocity.x,
                t=(report.timestamp_us - self.EPOCH_US) / 1e6,
                lateral_speed=dynamic.linear_velocity.y,
                yaw_rate=dynamic.angular_velocity.z,
                acceleration=dynamic.linear_acceleration.x,
            ))
        return tuple(result)


__all__ = [
    'PlanarEgoState',
    'EgoMotion',
    'EgoDynamics',
    'KinematicBicycle',
    'TrajectoryFollower',
    'AlpaSimController',
]
