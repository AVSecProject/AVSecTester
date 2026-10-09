"""Ordered attack and defense callbacks with per-run lifecycle management.

Stages describe execution points, not attack families. A stage handler replaces its input
value. World stages receive ``None`` and use explicit native backend operations instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np

from avsectester.plane import Control, Observation, StateEstimate, WorldSnapshot

if TYPE_CHECKING:
    from avsectester.backend import AVStack, WorldBackend


@dataclass
class CallInputs:
    """Inputs to a component pre-hook, retaining positional and keyword arguments."""

    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.args, tuple) or not isinstance(self.kwargs, dict):
            raise TypeError("CallInputs requires tuple args and dict kwargs")


@dataclass
class RenderRequest:
    """A sensor render request. Its pose is independent of the backend's true pose."""

    pose: Any
    camera: str

    def __post_init__(self):
        if not isinstance(self.camera, str) or not self.camera:
            raise ValueError("RenderRequest camera must be a nonempty sensor name")


@dataclass(frozen=True)
class RunContext:
    """Borrowed run resources. Model, native SDK and metadata access is unrestricted."""

    backend: WorldBackend
    stack: AVStack
    case: Any
    rng: np.random.Generator

    @property
    def native(self) -> Mapping[str, Mapping[str, Any]]:
        return {"backend": self.backend.native, "stack": self.stack.native}


@dataclass(frozen=True)
class StageContext:
    """One callback invocation and an independent snapshot of its true world state.

    ``step`` counts driving decisions from zero, independently of simulator frame IDs.
    ``sensor`` identifies a render callback's output sensor. Native resources remain borrowed.
    """

    run: RunContext
    step: int
    stage: str
    ground_truth: WorldSnapshot
    sensor: str | None = None
    observation: Observation | None = None
    control: Control | None = None

    @property
    def backend(self):
        return self.run.backend

    @property
    def stack(self):
        return self.run.stack

    @property
    def case(self):
        return self.run.case

    @property
    def rng(self):
        return self.run.rng

    @property
    def native(self):
        return self.run.native

    @property
    def sim_time(self) -> float:
        return self.ground_truth.t


class Plugin(ABC):
    """Optional lifecycle base for a stateful attack, defense or localization adapter."""

    def reset(self, context: RunContext) -> None:
        """Clear per-run history. Actors may not exist until ``world.setup``."""

    @abstractmethod
    def transform(self, value: Any, context: StageContext) -> Any:
        """Return the value consumed by the next handler or downstream component."""

    def after_step(self, context: StageContext) -> None:
        """Receive feedback after the command has affected the world. Optional."""

    def close(self) -> None:
        """Release resources, including partially initialized resources after an error."""


@dataclass(frozen=True)
class Hook:
    """Bind a function or a lifecycle plugin to an adapter-supported execution stage."""

    stage: str
    handler: Any

    def __post_init__(self):
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("Hook stage must be a nonempty string")
        if not callable(getattr(self.handler, "transform", self.handler)):
            raise TypeError("Hook handler must be callable or provide transform(value, context)")


def _invoke(handler, value, context):
    return getattr(handler, "transform", handler)(value, context)


class Runtime:
    """A reusable execution configuration. Handler history resets for each run.

    Hooks execute in registration order at each stage. ``localizer`` consumes an Observation
    and returns StateEstimate. Without one, the backend's reported state is the initial estimate.
    A fixed seed initializes a fresh RNG for every run, without changing simulator/model seeds.
    """

    def __init__(
        self,
        hooks: Iterable[Hook] = (),
        *,
        localizer: Callable[[Observation, StageContext], StateEstimate] | Plugin | None = None,
        case: Any = None,
        seed: int | None = 0,
    ):
        self.hooks = tuple(hooks)
        if any(not isinstance(hook, Hook) for hook in self.hooks):
            raise TypeError("Runtime hooks must be Hook instances")
        if localizer is not None and not callable(getattr(localizer, "transform", localizer)):
            raise TypeError("localizer must be callable or provide transform(value, context)")
        self.localizer, self.case, self.seed = localizer, case, seed

    def session(self, backend: WorldBackend, stack: AVStack, perturb=None) -> RuntimeSession:
        """Create one execution session, validating capabilities before resetting the world."""
        return RuntimeSession(self, backend, stack, perturb)


class RuntimeSession:
    """One bound run. Custom adapters publish supported stages through ``emit``."""
    CORE_STAGES = frozenset({"sensors.post", "localization.post", "observation", "command"})

    def __init__(self, runtime, backend, stack, perturb):
        self.runtime = runtime
        self.context = RunContext(backend, stack, runtime.case, np.random.default_rng(runtime.seed))
        self.step = 0
        self.observation = None
        self.control = None
        self.perturb = perturb
        available = self.CORE_STAGES | backend.supported_stages | stack.supported_stages
        if runtime.localizer is not None:
            available |= {"localization.pre"}
        unknown = {hook.stage for hook in runtime.hooks} - available
        if unknown:
            raise ValueError(f"Unsupported runtime stages: {sorted(unknown)}. Available: {sorted(available)}")
        self.available = available
        self._handlers = {}
        for hook in runtime.hooks:
            self._handlers.setdefault(hook.stage, []).append(hook.handler)
        participants = [hook.handler for hook in runtime.hooks]
        if runtime.localizer is not None:
            participants.append(runtime.localizer)
        if perturb is not None:
            participants.append(perturb)
        # A single plugin can be bound to several stages but owns one lifecycle per run.
        self.participants = tuple({id(value): value for value in participants}.values())

    @contextmanager
    def activate(self):
        resources = ExitStack()
        failure = None
        try:
            resources.enter_context(self.context.backend.runtime_context(self))
            resources.enter_context(self.context.stack.runtime_context(self))
            for handler in self.participants:
                close = getattr(handler, "close", None)
                if close is not None:
                    resources.callback(close)
                reset = getattr(handler, "reset", None)
                if reset is not None:
                    reset(self.context)
            yield self
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                resources.close()
            except Exception as error:
                if failure is None:
                    raise
                failure.add_note(f"Runtime cleanup also failed: {error!r}")

    def stage_context(self, stage, sensor=None):
        truth = self.context.backend.ground_truth()
        return StageContext(
            self.context, self.step, stage, truth, sensor,
            self.observation, self.control if stage == "command" else deepcopy(self.control),
        )

    def emit(self, stage, value, sensor=None):
        if stage not in self.available:
            raise ValueError(f"Adapter emitted an unsupported stage: {stage!r}")
        # Context must describe the current input, including localization or legacy
        # perturbation replacements made before entering this stage.
        if isinstance(value, Observation):
            self.observation = value
        elif isinstance(value, Control):
            self.control = value
        for handler in self._handlers.get(stage, ()):
            value = _invoke(handler, value, self.stage_context(stage, sensor))
            expected = {
                "sensors.post": Observation,
                "localization.pre": Observation,
                "localization.post": StateEstimate,
                "observation": Observation,
                "command": Control,
                "render.pre": RenderRequest,
            }.get(stage)
            if expected is not None:
                self._require(value, expected, stage)
            elif stage in {"perception.pre", "tracking.pre", "planning.pre", "control.pre"}:
                self._require(value, CallInputs, stage)
            elif stage in {"world.setup", "world.step.pre", "world.step.post"}:
                if value is not None:
                    raise TypeError(f"{stage} handlers must return None and use native world operations")
            if isinstance(value, Observation):
                self.observation = value
            elif isinstance(value, Control):
                self.control = value
        return value

    def observe(self, observation):
        self.control = None  # The next decision has not produced a command yet.
        seen = self.context.backend.copy_observation(observation)
        self.observation = seen
        seen = self.emit("sensors.post", seen)
        self._require(seen, Observation, "sensors.post")
        if self.runtime.localizer is not None:
            seen = self.emit("localization.pre", seen)
            self._require(seen, Observation, "localization.pre")
            estimate = _invoke(self.runtime.localizer, seen, self.stage_context("localization"))
            self._require(estimate, StateEstimate, "localizer")
            # Estimators often return their own cached state. Downstream attacks
            # must not rewrite that history or a borrowed backend pose.
            estimate = deepcopy(estimate)
        else:
            estimate = StateEstimate(seen.vehicle_state, seen.ego_speed)
        self._require(estimate, StateEstimate, "localizer")
        estimate = self.emit("localization.post", estimate)
        self._require(estimate, StateEstimate, "localization.post")
        seen = replace(seen, vehicle_state=estimate.vehicle_state, ego_speed=estimate.ego_speed)
        if self.perturb is not None:
            seen = self.perturb(seen)
            self._require(seen, Observation, "perturb")
        seen = self.emit("observation", seen)
        self._require(seen, Observation, "observation")
        self.observation = seen
        return seen

    def command(self, control):
        self._require(control, Control, "stack")
        self.control = control
        control = self.emit("command", control)
        self._require(control, Control, "command")
        self.control = control
        return control

    def after_step(self, observation):
        # Feedback sees the next raw measurement, before input transforms for the next decision.
        self.observation = self.context.backend.copy_observation(observation)
        for handler in self.participants:
            callback = getattr(handler, "after_step", None)
            if callback is not None:
                # Feedback is observational. One plugin's bookkeeping must not
                # change another plugin's raw measurement or executed command.
                context = self.stage_context("step.end")
                callback(replace(
                    context,
                    observation=self.context.backend.copy_observation(self.observation),
                ))

    @staticmethod
    def _require(value, expected, stage):
        if not isinstance(value, expected):
            raise TypeError(f"{stage} must return {expected.__name__}, got {type(value).__name__}")
