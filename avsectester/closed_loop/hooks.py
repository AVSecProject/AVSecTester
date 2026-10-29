"""Ordered attack and defense callbacks with per-run lifecycle management.

Stages describe execution points, not attack families. A stage handler replaces its input
value. Simulator stages receive ``None`` and use explicit native simulator operations instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Self, TYPE_CHECKING

import numpy as np

from avsectester.validation import require_nonnegative_integer
from avsectester.data_types import Control, Observation, RecordedOutput, StateEstimate, WorldSnapshot

if TYPE_CHECKING:
    from avsectester.closed_loop.avstacks.base import AVStack
    from avsectester.closed_loop.simulators.base import SimulatorRuntime
    from avsectester.closed_loop.avstacks.base import ObservationPipeline


class StageEmitter(Protocol):
    """The emitter a simulator runtime or driving system is bound to, normally the dispatcher."""

    def emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        """Run registered callbacks for a supported stage and return the resulting value."""
        ...


class _HookBindingMixin:
    """Shared stage binding for simulator runtimes and driving systems."""

    _dispatcher: StageEmitter | None = None

    @property
    def supported_stages(self) -> frozenset[str]:
        """Return the hook stages supported by this simulator runtime or driving system."""
        return frozenset()

    @property
    def native(self) -> Mapping[str, Any]:
        """Expose native components available to callbacks, keyed by component name."""
        return {}

    @contextmanager
    def bind_hooks(self, dispatcher: StageEmitter) -> Iterator[None]:
        """Bind one dispatcher for the scope. Reject an already active binding."""
        if getattr(self, "_dispatcher", None) is not None:
            raise RuntimeError(f"{type(self).__name__} is already bound to an active run")
        self._dispatcher = dispatcher
        try:
            yield
        finally:
            self._dispatcher = None

    def _emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        dispatcher = getattr(self, "_dispatcher", None)
        return dispatcher.emit(stage, value, sensor=sensor) if dispatcher is not None else value


class UnsupportedStageError(ValueError):
    """A requested hook stage is supported by neither the simulator runtime nor driving system."""


@dataclass
class CallInputs:
    """Inputs to a component pre-hook, retaining positional and keyword arguments."""

    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple) or not isinstance(self.kwargs, dict):
            raise TypeError("CallInputs requires tuple args and dict kwargs")


@dataclass
class CameraPoseRequest:
    """Camera name and rendering pose editable at a rendering simulator's ``render.pre`` stage.

    The requested pose is independent of the simulator's true physical pose.
    """

    pose: Any
    camera: str

    def __post_init__(self) -> None:
        if not isinstance(self.camera, str) or not self.camera:
            raise ValueError("CameraPoseRequest camera must be a nonempty sensor name")


@dataclass(frozen=True)
class RunContext:
    """Borrowed run resources. Model, native SDK and metadata access is unrestricted."""

    simulator_runtime: SimulatorRuntime
    avstack: AVStack
    rng: np.random.Generator

    @property
    def native(self) -> Mapping[str, Mapping[str, Any]]:
        """Access a native run resource of the simulator runtime or driving system by name."""
        return {"simulator_runtime": self.simulator_runtime.native, "avstack": self.avstack.native}


@dataclass(frozen=True)
class StageContext:
    """One callback invocation and an independent snapshot of the physical state at it.

    ``step`` counts driving decisions from zero, independently of simulator frame IDs.
    ``sensor`` identifies a render callback's output sensor. Native resources remain borrowed.
    """

    run: RunContext
    step: int
    stage: str
    physical_state: WorldSnapshot
    sensor: str | None = None
    observation: Observation | None = None
    control: Control | None = None
    _output_recorder: Callable[[StageContext, str, Any], None] | None = field(
        default=None, repr=False, compare=False,
    )

    def record_output(self, name: str, value: Any) -> None:
        """Save a named, independent output in this run's RunLog with callback timing.

        Use this for attack decisions, defense alarms or estimator telemetry. Outputs
        supplement the transformed value and never replace the downstream input.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Output name must be a nonempty string")
        if self._output_recorder is None:
            raise RuntimeError("Outputs can only be recorded from an active hook callback")
        self._output_recorder(self, name, value)

    @property
    def simulator_runtime(self) -> SimulatorRuntime:
        """Simulator runtime associated with the active run."""
        return self.run.simulator_runtime

    @property
    def avstack(self) -> AVStack:
        """Driving system associated with the active run."""
        return self.run.avstack

    @property
    def rng(self) -> np.random.Generator:
        """Random generator assigned to this callback invocation."""
        return self.run.rng

    @property
    def native(self) -> Mapping[str, Mapping[str, Any]]:
        """Access a native resource through the active run context."""
        return self.run.native

    @property
    def sim_time(self) -> float:
        """Physical simulation time at the active stage."""
        return self.physical_state.t


class Plugin(ABC):
    """Optional lifecycle base for a stateful attack, defense or driving-system localizer."""

    def reset(self, context: RunContext) -> None:
        """Clear per-run history. Actors may not exist until ``simulator.setup``."""

    @abstractmethod
    def transform(self, value: Any, context: StageContext) -> Any:
        """Return the value consumed by the next handler or downstream component."""

    def after_step(self, context: StageContext) -> None:
        """Receive feedback after the command has affected the world. Optional."""

    def close(self) -> None:
        """Release resources, including partially initialized resources after an error."""


@dataclass(frozen=True)
class Hook:
    """Bind a function or a lifecycle plugin to an execution stage."""

    stage: str
    handler: Any

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("Hook stage must be a nonempty string")
        if not callable(getattr(self.handler, "transform", self.handler)):
            raise TypeError("Hook handler must be callable or provide transform(value, context)")


def _invoke(handler: Any, value: Any, context: StageContext) -> Any:
    return getattr(handler, "transform", handler)(value, context)


class HookSet:
    """Reusable ordered hooks, independent of the driving system's ObservationPipeline.

    A fixed seed starts a fresh RNG for each run. Stateful handlers must implement
    ``reset`` or be constructed afresh.
    """

    def __init__(
        self,
        hooks: Iterable[Hook] = (),
        *,
        seed: int | None = 0,
    ) -> None:
        self.hooks = tuple(hooks)
        if any(not isinstance(hook, Hook) for hook in self.hooks):
            raise TypeError("HookSet hooks must be Hook instances")
        self.seed = seed

    def with_hooks(self, hooks: Iterable[Hook], *, prepend: bool = False) -> Self:
        """Copy this configuration with additional ordered hooks."""
        extra = tuple(hooks)
        if any(not isinstance(hook, Hook) for hook in extra):
            raise TypeError("HookSet hooks must be Hook instances")
        result = copy(self)
        result.hooks = extra + self.hooks if prepend else self.hooks + extra
        return result

    def _observation_pipeline(self, avstack: AVStack) -> ObservationPipeline:
        from avsectester.closed_loop.avstacks.base import ObservationPipeline

        result = avstack.observation_pipeline
        if not isinstance(result, ObservationPipeline):
            raise TypeError("avstack.observation_pipeline must be ObservationPipeline")
        return result

    def validate_stages(self, simulator_runtime: SimulatorRuntime, avstack: AVStack) -> frozenset[str]:
        """Check stage support without binding a dispatcher or invoking handlers."""
        from avsectester.closed_loop.avstacks.base import validate_command_compatibility

        validate_command_compatibility(simulator_runtime, avstack, self)
        available = HookDispatcher.CORE_STAGES | simulator_runtime.supported_stages | avstack.supported_stages
        if self._observation_pipeline(avstack).localizer is not None:
            available |= {"localization.pre"}
        unknown = {hook.stage for hook in self.hooks} - available
        if unknown:
            raise UnsupportedStageError(
                f"Unsupported hook stages: {sorted(unknown)}. Available: {sorted(available)}"
            )
        return available

    @property
    def setup_ticks(self) -> int:
        """Required deployment ticks, shared by every treatment in a comparison group."""
        handlers = {id(hook.handler): hook.handler for hook in self.hooks if hook.stage == "simulator.setup"}
        counts = [getattr(handler, "setup_ticks", 0) for handler in handlers.values()]
        for count in counts:
            require_nonnegative_integer(count, "Plugin setup_ticks")
        return sum(counts)

    def session(self, simulator_runtime: SimulatorRuntime, avstack: AVStack) -> HookDispatcher:
        """Create one execution session before resetting the world."""
        return HookDispatcher(self, simulator_runtime, avstack)


class HookDispatcher:
    """One bound run. Simulator runtimes and driving systems publish their stages through ``emit``."""
    CORE_STAGES = frozenset({"sensors.post", "localization.post", "observation", "command"})
    """Framework stages available independently of native simulator and driving-system stages."""

    def __init__(self, hook_set: HookSet, simulator_runtime: SimulatorRuntime, avstack: AVStack) -> None:
        self.hook_set = hook_set
        self.context = RunContext(simulator_runtime, avstack, np.random.default_rng(hook_set.seed))
        self.observation_pipeline = hook_set._observation_pipeline(avstack)
        self.step = 0
        self.observation: Observation | None = None
        self.control: Control | None = None
        self.outputs: list[RecordedOutput] = []
        self._active = False
        self.available = hook_set.validate_stages(simulator_runtime, avstack)
        self._handlers: dict[str, list[Any]] = {}
        for hook in hook_set.hooks:
            self._handlers.setdefault(hook.stage, []).append(hook.handler)
        participants = [hook.handler for hook in hook_set.hooks]
        if self.observation_pipeline.localizer is not None:
            participants.append(self.observation_pipeline.localizer)
        # A single plugin can be bound to several stages but owns one lifecycle per run.
        self.participants = tuple({id(value): value for value in participants}.values())

    @contextmanager
    def activate(self) -> Iterator[HookDispatcher]:
        """Bind the simulator runtime and driving system. Reset participants and close them on exit."""
        if self._active:
            raise RuntimeError("Hook dispatcher is already active")
        resources = ExitStack()
        failure = None
        try:
            simulator_runtime = self.context.simulator_runtime
            resources.enter_context(simulator_runtime.bind_hooks(self))
            resources.enter_context(self.context.avstack.bind_hooks(self))
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
                failure.add_note(f"Hook cleanup also failed: {error!r}")

    def stage_context(self, stage: str, sensor: str | None = None) -> StageContext:
        """Capture current physical state and model-visible values for a callback."""
        physical_state = self.context.simulator_runtime.state()
        return StageContext(
            self.context, self.step, stage, physical_state, sensor,
            self.observation, self.control if stage == "command" else deepcopy(self.control),
            self.record_output,
        )

    def record_output(self, context: StageContext, name: str, value: Any) -> None:
        """Snapshot a named output. Reject recording outside an active dispatcher scope."""
        if not self._active:
            raise RuntimeError("Outputs can only be recorded from an active hook callback")
        self.outputs.append(RecordedOutput(
            name=name, value=deepcopy(value), step=context.step, stage=context.stage,
            frame=context.physical_state.frame, t=context.sim_time, sensor=context.sensor,
        ))

    def drain_outputs(self) -> list[RecordedOutput]:
        """Transfer recorded outputs to a consumer without retaining the run's history.

        Direct session callers keep the existing accumulating ``outputs`` view until they
        explicitly drain it. Keep that list's identity for callers holding a reference.
        """
        result = list(self.outputs)
        self.outputs.clear()
        return result

    def emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        """Transform a value through the supported stage's ordered callbacks."""
        if stage not in self.available:
            raise ValueError(f"Emitted an unsupported stage: {stage!r}")
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
                "render.pre": CameraPoseRequest,
            }.get(stage)
            if expected is not None:
                self._require(value, expected, stage)
            elif stage in {"perception.pre", "tracking.pre", "planning.pre", "control.pre"}:
                self._require(value, CallInputs, stage)
            elif stage in {"simulator.setup", "simulator.step.pre", "simulator.step.post"}:
                if value is not None:
                    raise TypeError(f"{stage} handlers must return None and use native world operations")
            if isinstance(value, Observation):
                self.observation = value
            elif isinstance(value, Control):
                self.control = value
        return value

    def observe(self, observation: Observation) -> Observation:
        """Apply the driving system's ObservationPipeline with this run's ordered hooks."""
        return self.observation_pipeline.process(observation, self)

    def command(self, control: Control) -> Control:
        """Apply command hooks and validate the resulting command against the simulator."""
        self._require(control, Control, "avstack")
        self.control = control
        control = self.emit("command", control)
        self._require(control, Control, "command")
        self.context.simulator_runtime.validate_command(control)
        self.control = control
        return control

    def after_step(self, observation: Observation) -> None:
        """Notify participants with independent copies of the next raw measurement."""
        # Feedback sees the next raw measurement, before input transforms for the next decision.
        self.observation = self.context.simulator_runtime.copy_observation(observation)
        for handler in self.participants:
            callback = getattr(handler, "after_step", None)
            if callback is not None:
                # Feedback is observational. One plugin's bookkeeping must not
                # change another plugin's raw measurement or executed command.
                context = self.stage_context("step.end")
                callback(replace(
                    context,
                    observation=self.context.simulator_runtime.copy_observation(self.observation),
                ))

    @staticmethod
    def _require(value: Any, expected: type, stage: str) -> None:
        if not isinstance(value, expected):
            raise TypeError(f"{stage} must return {expected.__name__}, got {type(value).__name__}")


__all__ = [
    'StageEmitter',
    'UnsupportedStageError',
    'CallInputs',
    'CameraPoseRequest',
    'RunContext',
    'StageContext',
    'Plugin',
    'Hook',
    'HookSet',
    'HookDispatcher',
]
