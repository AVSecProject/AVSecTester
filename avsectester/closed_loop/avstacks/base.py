"""The driving-system interface: an Observation in, a Control out.

``AVStack`` knows nothing about how the system works: a modular perception->tracking->planning
->control pipeline and a single end-to-end model both satisfy it. Implementations live in
`avsectester.closed_loop.avstacks`. This module imports nothing from avstack, avcarla or
carla; the implementations carry those dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

from avsectester.data_types import Control, Detection, Observation, StateEstimate
from avsectester.closed_loop.hooks import _StageAdapter

if TYPE_CHECKING:
    from avsectester.closed_loop.hooks import HookDispatcher, HookSet
    from avsectester.closed_loop.simulators.base import SimulatorRuntime


class ObservationPipeline:
    """Process driving-system inputs without changing output commands.

    Inputs pass through ``sensors.post``, optional ``localization.pre``, the localizer,
    ``localization.post`` and ``observation`` hooks in that order.

    The dispatcher manages the localizer's per-run lifecycle together with hook owners,
    so one object used at several stages is reset and closed exactly once. Resetting the
    driving model's history does not reset localization or process its first input again.
    """

    def __init__(self, localizer: Any = None) -> None:
        if localizer is not None and not callable(getattr(localizer, "transform", localizer)):
            raise TypeError("localizer must be callable or provide transform(value, context)")
        self.localizer = localizer

    def process(self, observation: Observation, dispatcher: HookDispatcher) -> Observation:
        """Apply localization and plugins, emitting their supported hook stages."""
        from avsectester.closed_loop.hooks import _invoke

        dispatcher.control = None
        seen = dispatcher.context.simulator_runtime.copy_observation(observation)
        dispatcher.observation = seen
        seen = dispatcher.emit("sensors.post", seen)
        dispatcher._require(seen, Observation, "sensors.post")
        if self.localizer is not None:
            seen = dispatcher.emit("localization.pre", seen)
            dispatcher._require(seen, Observation, "localization.pre")
            estimate = _invoke(self.localizer, seen, dispatcher.stage_context("localization"))
            dispatcher._require(estimate, StateEstimate, "localizer")
            # The localizer may reuse its own state; downstream handlers receive a copy.
            estimate = deepcopy(estimate)
        else:
            estimate = StateEstimate(seen.vehicle_state, seen.ego_speed)
        estimate = dispatcher.emit("localization.post", estimate)
        dispatcher._require(estimate, StateEstimate, "localization.post")
        seen = replace(seen, vehicle_state=estimate.vehicle_state, ego_speed=estimate.ego_speed)
        seen = dispatcher.emit("observation", seen)
        dispatcher._require(seen, Observation, "observation")
        dispatcher.observation = seen
        return seen


class AVStack(_StageAdapter, ABC):
    """The driving system: an Observation in, a Control out.

    The interface knows nothing about the implementation: a modular perception->tracking->planning
    ->control pipeline and a single end-to-end model both satisfy it. Implementations adapt their
    own machinery (e.g. an avstack ``ModularDrivingPipeline``, or a camera end-to-end policy).
    """

    provides_detections: bool | None = None
    'Whether decisions expose detections compatible with detection metrics.'
    past_steps: ClassVar[int] = 0
    'Recorded decision steps required before the selected start.'
    decision_time_step: ClassVar[float | None] = None
    'Decision interval in seconds; None leaves the interval unrestricted.'

    @property
    def observation_pipeline(self) -> ObservationPipeline:
        """Driving-system input processing, including its localizer lifecycle."""
        if "_observation_pipeline" not in self.__dict__:
            self._observation_pipeline = ObservationPipeline()
        return self._observation_pipeline

    @observation_pipeline.setter
    def observation_pipeline(self, value: ObservationPipeline) -> None:
        """Optional localization and observation transformation pipeline applied before driving decisions."""
        if not isinstance(value, ObservationPipeline):
            raise TypeError("observation_pipeline must be ObservationPipeline")
        self._observation_pipeline = value

    def reset(self, observation: Observation, past: Sequence[Observation] = ()) -> None:
        """Initialize from the first observation and recorded past, ordered oldest to newest.

        ``past_steps`` declares the required number of preceding decision observations.
        The default implementation requires none and performs no warmup.
        """

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Possible output command representations, or None for an undeclared adapter."""
        return None

    @property
    def models(self) -> Mapping[str, Any]:
        """Loaded native models available to user handlers. Adapters expose their own names."""
        return {}

    def component_log(self) -> Mapping[str, Any] | None:
        """Already-produced component outputs for the current decision, or None.

        Recording must never run extra inference to obtain outputs. Implementations may
        enable their own instrumentation before the first call.
        """
        return None

    def detections(self) -> Sequence[Detection] | None:
        """Current decision's ego-frame detections; None means no detection output.

        An empty sequence means a detector ran and found no objects. Calling this method
        must not run inference or modify the driving system's state.
        """
        return None

    def close(self) -> None:
        """Release driving-system resources. The default implementation owns none."""

    @abstractmethod
    def __call__(self, observation: Observation) -> Control:
        """Advance driving inference once and return a command in ``command_modes``.

        Call after ``reset`` with the current model-visible observation. Keep per-run history
        in this instance; do not modify or advance the simulator. Trajectories use the current
        rig frame and integer-microsecond timestamps.
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
