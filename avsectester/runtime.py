"""Ordered attack and defense callbacks with per-run lifecycle management.

Stages describe execution points, not attack families. A stage handler replaces its input
value. World stages receive ``None`` and use explicit native backend operations instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np

from avsectester.plane import Control, InterventionOutput, Observation, StateEstimate, WorldSnapshot

if TYPE_CHECKING:
    from avsectester.backend import AVStack, WorldBackend


class UnsupportedStageError(ValueError):
    """A requested intervention stage is absent from the selected adapters."""


@dataclass
class CallInputs:
    """Inputs to a component pre-hook, retaining positional and keyword arguments."""

    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple) or not isinstance(self.kwargs, dict):
            raise TypeError("CallInputs requires tuple args and dict kwargs")


@dataclass
class RenderRequest:
    """A sensor render request. Its pose is independent of the backend's true pose."""

    pose: Any
    camera: str

    def __post_init__(self) -> None:
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
    _output_recorder: Callable[[StageContext, str, Any], None] | None = field(
        default=None, repr=False, compare=False,
    )

    def record_output(self, name: str, value: Any) -> None:
        """Save a named, independent output in this run's Trace with callback timing.

        Use this for attack decisions, defense alarms or estimator telemetry. Outputs
        supplement the transformed value and never replace the downstream input.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Output name must be a nonempty string")
        if self._output_recorder is None:
            raise RuntimeError("Outputs can only be recorded from an active runtime callback")
        self._output_recorder(self, name, value)

    @property
    def backend(self) -> WorldBackend:
        return self.run.backend

    @property
    def stack(self) -> AVStack:
        return self.run.stack

    @property
    def case(self) -> Any:
        return self.run.case

    @property
    def rng(self) -> np.random.Generator:
        return self.run.rng

    @property
    def native(self) -> Mapping[str, Mapping[str, Any]]:
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

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("Hook stage must be a nonempty string")
        if not callable(getattr(self.handler, "transform", self.handler)):
            raise TypeError("Hook handler must be callable or provide transform(value, context)")


def _invoke(handler: Any, value: Any, context: StageContext) -> Any:
    return getattr(handler, "transform", handler)(value, context)


class Runtime:
    """A reusable configuration that invokes handler lifecycle methods for each run.

    Hooks execute in registration order at each stage. ``localizer`` consumes an Observation
    and returns StateEstimate. Without one, the backend's reported state is the initial estimate.
    A fixed seed initializes a fresh RNG for every run, without changing simulator/model seeds.
    Stateful handlers must implement reset or be constructed afresh for each run.
    """

    def __init__(
        self,
        hooks: Iterable[Hook] = (),
        *,
        localizer: Callable[[Observation, StageContext], StateEstimate] | Plugin | None = None,
        case: Any = None,
        seed: int | None = 0,
    ) -> None:
        self.hooks = tuple(hooks)
        if any(not isinstance(hook, Hook) for hook in self.hooks):
            raise TypeError("Runtime hooks must be Hook instances")
        if localizer is not None and not callable(getattr(localizer, "transform", localizer)):
            raise TypeError("localizer must be callable or provide transform(value, context)")
        self.localizer, self.case, self.seed = localizer, case, seed

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None, *, case: Any = None) -> Runtime:
        """Load stage handlers from importable factories or native avstack configurations."""
        from .interventions import runtime_from_config

        return runtime_from_config(config, case=case)

    def with_hooks(self, hooks: Iterable[Hook], *, prepend: bool = False) -> Runtime:
        """Return a new configuration, preserving the original hook list and common settings."""
        extra = tuple(hooks)
        ordered = extra + self.hooks if prepend else self.hooks + extra
        return Runtime(ordered, localizer=self.localizer, case=self.case, seed=self.seed)

    def validate_stages(self, backend: WorldBackend, stack: AVStack) -> frozenset[str]:
        """Check adapter capabilities without binding a session or invoking handlers."""
        from .backend import validate_command_compatibility

        validate_command_compatibility(backend, stack, self)
        available = RuntimeSession.CORE_STAGES | backend.supported_stages | stack.supported_stages
        if self.localizer is not None:
            available |= {"localization.pre"}
        unknown = {hook.stage for hook in self.hooks} - available
        if unknown:
            raise UnsupportedStageError(
                f"Unsupported runtime stages: {sorted(unknown)}. Available: {sorted(available)}"
            )
        return available

    @property
    def setup_ticks(self) -> int:
        """Required deployment ticks, shared by both runs of a pair when supported."""
        handlers = {id(hook.handler): hook.handler for hook in self.hooks if hook.stage == "world.setup"}
        counts = [getattr(handler, "setup_ticks", 0) for handler in handlers.values()]
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
            raise ValueError("Plugin setup_ticks must be nonnegative integers")
        return sum(counts)

    def session(self, backend: WorldBackend, stack: AVStack, perturb: Callable[[Observation], Observation] | None = None) -> RuntimeSession:
        """Create one execution session, validating capabilities before resetting the world."""
        runtime = self
        if perturb is not None:
            from .interventions import ObservationTransform

            runtime = self.with_hooks([Hook("observation", ObservationTransform(perturb))], prepend=True)
        return RuntimeSession(runtime, backend, stack)


class RuntimeSession:
    """One bound run. Custom adapters publish supported stages through ``emit``."""
    CORE_STAGES = frozenset({"sensors.post", "localization.post", "observation", "command"})

    def __init__(self, runtime: Runtime, backend: WorldBackend, stack: AVStack) -> None:
        self.runtime = runtime
        self.context = RunContext(backend, stack, runtime.case, np.random.default_rng(runtime.seed))
        self.step = 0
        self.observation: Observation | None = None
        self.control: Control | None = None
        self.outputs: list[InterventionOutput] = []
        self._active = False
        self.available = runtime.validate_stages(backend, stack)
        self._handlers: dict[str, list[Any]] = {}
        for hook in runtime.hooks:
            self._handlers.setdefault(hook.stage, []).append(hook.handler)
        participants = [hook.handler for hook in runtime.hooks]
        if runtime.localizer is not None:
            participants.append(runtime.localizer)
        # A single plugin can be bound to several stages but owns one lifecycle per run.
        owners = [getattr(value, "lifecycle_owner", value) for value in participants]
        self.participants = tuple({id(value): value for value in owners}.values())

    @contextmanager
    def activate(self) -> Iterator[RuntimeSession]:
        if self._active:
            raise RuntimeError("Runtime session is already active")
        resources = ExitStack()
        failure = None
        try:
            backend = self.context.backend
            previous_ticks = backend.setup_ticks
            backend.setup_ticks = max(previous_ticks, self.runtime.setup_ticks)
            resources.callback(setattr, backend, "setup_ticks", previous_ticks)
            resources.enter_context(backend.runtime_context(self))
            resources.enter_context(self.context.stack.runtime_context(self))
            for handler in self.participants:
                close = getattr(handler, "close", None)
                if close is not None:
                    resources.callback(close)
                reset = getattr(handler, "reset", None)
                if reset is not None:
                    reset(self.context)
            self._active = True
            yield self
        except BaseException as error:
            failure = error
            raise
        finally:
            self._active = False
            try:
                resources.close()
            except Exception as error:
                if failure is None:
                    raise
                failure.add_note(f"Runtime cleanup also failed: {error!r}")

    def stage_context(self, stage: str, sensor: str | None = None) -> StageContext:
        truth = self.context.backend.ground_truth()
        return StageContext(
            self.context, self.step, stage, truth, sensor,
            self.observation, self.control if stage == "command" else deepcopy(self.control),
            self.record_output,
        )

    def record_output(self, context: StageContext, name: str, value: Any) -> None:
        if not self._active:
            raise RuntimeError("Outputs can only be recorded from an active runtime callback")
        self.outputs.append(InterventionOutput(
            name=name, value=deepcopy(value), step=context.step, stage=context.stage,
            frame=context.ground_truth.frame, t=context.sim_time, sensor=context.sensor,
        ))

    def emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        if stage not in self.available:
            raise ValueError(f"Adapter emitted an unsupported stage: {stage!r}")
        # Context describes the current input, including replacements by earlier handlers.
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

    def observe(self, observation: Observation) -> Observation:
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
        seen = self.emit("observation", seen)
        self._require(seen, Observation, "observation")
        self.observation = seen
        return seen

    def command(self, control: Control) -> Control:
        self._require(control, Control, "stack")
        self.control = control
        control = self.emit("command", control)
        self._require(control, Control, "command")
        self.context.backend.validate_command(control)
        self.control = control
        return control

    def after_step(self, observation: Observation) -> None:
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
    def _require(value: Any, expected: type, stage: str) -> None:
        if not isinstance(value, expected):
            raise TypeError(f"{stage} must return {expected.__name__}, got {type(value).__name__}")
