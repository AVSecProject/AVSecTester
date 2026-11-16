"""The driving-system interface: an Observation in, a Control out.

``AVStack`` knows nothing about how the system works: a modular perception->tracking->planning
->control pipeline and a single end-to-end model both satisfy it. Implementations live in
`avsectester.closed_loop.avstacks`. This module imports nothing from avstack, avcarla or
carla. The implementations carry those dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import math
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from avsectester.data_types import Control, Detection, Observation, StateEstimate
from avsectester.closed_loop.hooks import _HookBindingMixin
from avsectester.closed_loop.timing import HistoryRequirement

if TYPE_CHECKING:
    from avsectester.closed_loop.hooks import HookDispatcher, HookSet
    from avsectester.closed_loop.simulators.base import SimulatorRuntime


def _planar_pose(state: Any) -> np.ndarray | None:
    try:
        pose = np.array([state.x, state.y, state.yaw], dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    return pose if np.isfinite(pose).all() else None


def _route_from_estimate(route: np.ndarray | None, source_state: Any, estimated_state: Any) -> np.ndarray | None:
    """Re-express ego-frame route points from the source pose in the estimated pose's frame."""
    if route is None or estimated_state is source_state:
        return route
    source, estimated = _planar_pose(source_state), _planar_pose(estimated_state)
    if source is None or estimated is None:
        raise ValueError("A route requires finite planar x, y and yaw before and after localization")
    if np.array_equal(source, estimated):
        return route
    points = np.array(route, dtype=float)

    def rotation(yaw: float) -> np.ndarray:
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.array([[cosine, -sine], [sine, cosine]])

    world = points[:, :2] @ rotation(float(source[2])).T + source[:2]
    points[:, :2] = (world - estimated[:2]) @ rotation(float(estimated[2]))
    return points


class ObservationPipeline:
    """Process driving-system inputs without changing output commands.

    Frame events pass through ``sensors.post``. Pose events pass through optional
    ``localization.pre``, the localizer and ``localization.post``. Decisions pass through
    ``observation``. The single-rate convenience path runs these stages in that order.
    When localization changes the planar ego pose, the route is re-expressed in the estimated
    ego frame with unchanged world positions, so model-visible navigation follows the estimate.

    The dispatcher manages the localizer's per-run lifecycle together with hook owners,
    so one object used at several stages is reset and closed exactly once. Resetting the
    driving model's history does not reset localization or process its first input again.
    """

    def __init__(self, localizer: Any = None) -> None:
        if localizer is not None and not callable(getattr(localizer, "transform", localizer)):
            raise TypeError("localizer must be callable or provide transform(value, context)")
        self.localizer = localizer

    def capture(self, observation: Observation, dispatcher: HookDispatcher) -> Observation:
        """Process one frame batch through sensors.post without running localization."""

        seen = dispatcher.context.simulator_runtime.copy_observation(observation)
        dispatcher.observation = seen
        seen = dispatcher.emit("sensors.post", seen)
        dispatcher._require(seen, Observation, "sensors.post")
        return seen

    def localize(self, observation: Observation, dispatcher: HookDispatcher) -> Observation:
        """Localize one pose timestamp, including timestamps without sensor frames."""
        from avsectester.closed_loop.hooks import _invoke

        seen = observation
        dispatcher.observation = seen
        if self.localizer is not None:
            seen = dispatcher.emit("localization.pre", seen)
            dispatcher._require(seen, Observation, "localization.pre")
            estimate = _invoke(self.localizer, seen, dispatcher.stage_context("localization"))
            dispatcher._require(estimate, StateEstimate, "localizer")
            # The localizer may reuse its state. Downstream handlers receive a copy.
            estimate = deepcopy(estimate)
        else:
            estimate = StateEstimate(seen.vehicle_state, seen.ego_speed)
        estimate = dispatcher.emit("localization.post", estimate)
        dispatcher._require(estimate, StateEstimate, "localization.post")
        seen = self.with_estimate(seen, estimate)
        dispatcher.observation = seen
        return seen

    @staticmethod
    def with_estimate(observation: Observation, estimate: StateEstimate) -> Observation:
        """Replace the model-visible pose and re-express navigation without running hooks."""
        if not isinstance(estimate, StateEstimate):
            raise TypeError("estimate must be StateEstimate")
        return replace(observation, vehicle_state=estimate.vehicle_state, ego_speed=estimate.ego_speed,
                       route=_route_from_estimate(observation.route, observation.vehicle_state,
                                                  estimate.vehicle_state))

    def decision(self, observation: Observation, dispatcher: HookDispatcher) -> Observation:
        """Apply decision input hooks after selecting delivered captures and history."""
        seen = dispatcher.emit("observation", observation)
        dispatcher._require(seen, Observation, "observation")
        dispatcher.observation = seen
        return seen

    def process(self, observation: Observation, dispatcher: HookDispatcher) -> Observation:
        """Process a capture followed immediately by one driving decision."""
        dispatcher.control = None
        return self.decision(self.localize(self.capture(observation, dispatcher), dispatcher), dispatcher)


class AVStack(_HookBindingMixin, ABC):
    """The driving system: an Observation in, a Control out.

    The interface knows nothing about the implementation: a modular perception->tracking->planning
    ->control pipeline and a single end-to-end model both satisfy it. Implementations adapt their
    own machinery (e.g. an avstack ``ModularDrivingPipeline``, or a camera end-to-end policy).
    """

    provides_detections: bool | None = None
    'Whether decisions expose detections compatible with detection metrics.'
    history: HistoryRequirement = HistoryRequirement()
    'Recorded input needed at the first decision, expressed as pose duration and sample count.'
    decision_time_step: ClassVar[float | None] = None
    'Decision interval in seconds. None leaves the interval unrestricted.'

    @classmethod
    def required_history(cls, options: Mapping[str, Any]) -> HistoryRequirement:
        """Recorded input needed at the first decision, resolved before model loading."""
        return cls.history

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "past_steps" in cls.__dict__:
            raise TypeError("AVStack.past_steps was replaced by history=HistoryRequirement. "
                            "Declare a duration or acquisition count, not a decision count")

    @property
    def observation_pipeline(self) -> ObservationPipeline:
        """Driving-system input processing, including its localizer lifecycle."""
        if "_observation_pipeline" not in self.__dict__:
            self._observation_pipeline = ObservationPipeline()
        return self._observation_pipeline

    @observation_pipeline.setter
    def observation_pipeline(self, value: ObservationPipeline) -> None:
        """Optional localization and observation pipeline applied before driving decisions."""
        if not isinstance(value, ObservationPipeline):
            raise TypeError("observation_pipeline must be ObservationPipeline")
        self._observation_pipeline = value

    def reset(self, observation: Observation) -> None:
        """Initialize from the first decision observation, including its recorded deliveries.

        ``history`` declares the recorded pose duration and acquisition count needed at reset.
        Implementations retain any history they need from subsequent incremental deliveries.
        """

    def ingest(self, observation: Observation) -> None:
        """Accept incremental input without inference when a force_gt decision skips driving.

        Normal decisions call ``__call__`` instead. Implementations with input caches retain
        their references here. Do not ingest and call the same observation twice.
        """

    @property
    def required_sensors(self) -> frozenset[str] | None:
        """Required delivered sensor IDs. None accepts the simulator's configured sensors."""
        return None

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Possible output command representations, or None for an undeclared driving system."""
        return None

    @property
    def models(self) -> Mapping[str, Any]:
        """Loaded native models available to user handlers, named by the driving system."""
        return {}

    def component_log(self) -> Mapping[str, Any] | None:
        """Already-produced component outputs for the current decision, or None.

        Recording must never run extra inference to obtain outputs. Implementations may
        enable their own instrumentation before the first call.
        """
        return None

    def detections(self) -> Sequence[Detection] | None:
        """Current decision's ego-frame detections. None means no detection output.

        An empty sequence means a detector ran and found no objects. Calling this method
        must not run inference or modify the driving system's state.
        """
        return None

    @property
    def detection_sensor(self) -> str | None:
        """Sensor whose latest delivered frame is associated with the model's detections.

        Matching uses its latest physical acquisition's truth. None uses decision-time truth.
        Decisions before a named sensor's first delivery are excluded. A run where it never
        delivers has unavailable detection evidence.
        """
        return None

    def close(self) -> None:
        """Release driving-system resources. The default implementation owns none."""

    @abstractmethod
    def __call__(self, observation: Observation) -> Control:
        """Advance driving inference once and return a command in ``command_modes``.

        Call after ``reset`` with the current model-visible observation. Keep per-run history
        in this instance. Do not modify or advance the simulator. Trajectories retain the
        generation-time rig frame and integer-microsecond timestamps for the controller.
        """


class CommandCompatibilityError(ValueError):
    """Declared AVStack commands cannot be consumed by the selected simulator."""


def validate_command_compatibility(
    simulator_runtime: SimulatorRuntime, avstack: AVStack, hooks: HookSet | None = None,
) -> None:
    """Check declared commands before world setup, allowing explicit hook conversion.

    A command-stage handler may translate the AVStack output into a different representation.
    In that case only the final command is checked, after all handlers have run.
    """
    accepted, produced = simulator_runtime.command_modes, avstack.command_modes
    if accepted is None or produced is None:
        return
    if hooks is not None and any(hook.stage == "command" for hook in hooks.hooks):
        return
    missing = produced - accepted
    if missing:
        raise CommandCompatibilityError(
            f"{type(avstack).__name__} produces {sorted(missing)} commands unsupported by "
            f"{type(simulator_runtime).__name__}. Select compatible dynamics or add a command conversion hook."
        )


__all__ = [
    'ObservationPipeline',
    'AVStack',
    'CommandCompatibilityError',
    'validate_command_compatibility',
]
