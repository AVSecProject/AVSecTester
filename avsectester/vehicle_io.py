"""Model-visible sensor processing and localization owned by the driving system."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from .plane import Observation, StateEstimate

if TYPE_CHECKING:
    from .runtime import HookDispatcher


class VehicleIO:
    """Apply sensor hooks, localization and observation hooks in their causal order.

    The dispatcher manages the localizer's per-run lifecycle together with hook owners,
    so one object used at several stages is reset and closed exactly once. Resetting the
    driving model's history does not reset localization or process its first input again.
    """

    def __init__(self, localizer: Any = None) -> None:
        if localizer is not None and not callable(getattr(localizer, "transform", localizer)):
            raise TypeError("localizer must be callable or provide transform(value, context)")
        self.localizer = localizer

    def process(self, observation: Observation, dispatcher: HookDispatcher) -> Observation:
        from .runtime import _invoke

        dispatcher.control = None
        seen = dispatcher.context.backend.copy_observation(observation)
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
