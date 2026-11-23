"""Lifecycle of a driving system hosted beside a CARLA world."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from avsectester.closed_loop.ros_interception.declaration import InterceptionDeclaration
from avsectester.closed_loop.simulators.messages import MessageChannel


@dataclass(frozen=True)
class ExternalDrivingContext:
    """Running server, canonical-world goal and native tick rate for one session.

    ``ego_goal`` contains x and y in meters and yaw_deg in degrees. It uses the framework
    world frame. ``tick_hz`` is the native CARLA rate, which can exceed the decision rate.
    """

    host: str
    port: int
    map_name: str
    ego_goal: Mapping[str, float]
    tick_hz: float


class ExternalDrivingSession(Protocol):
    """One hosted driving system's live session.

    CARLA holds the ego while polling for readiness and calls ``check`` once after each
    native tick after handover. The session manages its asynchronous work.
    """

    @property
    def interception_channel(self) -> MessageChannel | None:
        """Return the message channel when this session intercepts topics."""
        return None

    def poll(self, ego_pose: np.ndarray, sim_time: float) -> bool:
        """Advance readiness without blocking. Raise if startup fails or times out.

        The ego pose is a framework-world transform and simulation time is in seconds.
        A poll after readiness supplies the current pose and time before engagement.
        """
        ...

    def engage(self) -> None:
        """Request control at the first capture without blocking."""
        ...

    def check(self) -> None:
        """Raise if engagement or the running system fails."""
        ...

    def telemetry(self) -> Mapping[str, Any]:
        """Return the latest driver state for a decision record."""
        ...

    def report(self) -> Mapping[str, Any]:
        """Return detached session evidence, available after stop."""
        ...

    def stop(self) -> None:
        """Release all session resources. Repeated calls are safe."""
        ...


class ExternalDrivingSystem(Protocol):
    """Reusable configuration for a hosted native ROS 2 driver."""

    @property
    def carla_publishes_tf(self) -> bool:
        """Whether CARLA publishes ROS transform frames for this driving system."""
        ...

    @property
    def identity(self) -> Mapping[str, Any]:
        """Settings that affect the prepared reference and run results."""
        ...

    @property
    def interception(self) -> InterceptionDeclaration | None:
        """Modified outputs, read-only observations and any pose projection for the relay."""
        return None

    def start(self, context: ExternalDrivingContext) -> ExternalDrivingSession:
        """Create a new session and begin startup without blocking CARLA ticks."""
        ...


__all__ = ["ExternalDrivingContext", "ExternalDrivingSession", "ExternalDrivingSystem"]
