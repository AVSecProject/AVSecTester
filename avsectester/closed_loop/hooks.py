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

from avsectester.closed_loop.datatypes import Control, Observation, RecordedOutput, StateEstimate, WorldSnapshot

if TYPE_CHECKING:
    from avsectester.closed_loop.avstack import AVStack
    from avsectester.closed_loop.simulator import SimulatorRuntime, WorldBackend
    from avsectester.closed_loop.vehicle_io import VehicleIO


class StageEmitter(Protocol):
    """What an adapter is bound to during a run: normally the run's hook dispatcher."""

    def emit(self, stage: str, value: Any, sensor: str | None = None) -> Any: ...


class _StageAdapter:
    """Shared stage binding for simulator and AVStack adapters."""

    _dispatcher: StageEmitter | None = None

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset()

    @property
    def native(self) -> Mapping[str, Any]:
        return {}

    @contextmanager
    def bind_hooks(self, dispatcher: StageEmitter) -> Iterator[None]:
        if getattr(self, "_dispatcher", None) is not None:
            raise RuntimeError("Adapter is already bound to an active run")
        self._dispatcher = dispatcher
        try:
            yield
        finally:
            self._dispatcher = None

    def _emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        dispatcher = getattr(self, "_dispatcher", None)
        return dispatcher.emit(stage, value, sensor=sensor) if dispatcher is not None else value


class UnsupportedStageError(ValueError):
    """A requested hook stage is absent from the selected adapters."""


@dataclass
class CallInputs:
    """Inputs to a component pre-hook, retaining positional and keyword arguments."""

    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple) or not isinstance(self.kwargs, dict):
            raise TypeError("CallInputs requires tuple args and dict kwargs")


@dataclass
class CameraRenderRequest:
    """A sensor render request. Its pose is independent of the simulator's true pose."""

    pose: Any
    camera: str

    def __post_init__(self) -> None:
        if not isinstance(self.camera, str) or not self.camera:
            raise ValueError("CameraRenderRequest camera must be a nonempty sensor name")


@dataclass(frozen=True)
class RunContext:
    """Borrowed run resources. Model, native SDK and metadata access is unrestricted."""

    simulator_runtime: SimulatorRuntime | WorldBackend
    avstack: AVStack
    case: Any
    rng: np.random.Generator

    @property
    def native(self) -> Mapping[str, Mapping[str, Any]]:
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
        """Save a named, independent output in this run's Trace with callback timing.

        Use this for attack decisions, defense alarms or estimator telemetry. Outputs
        supplement the transformed value and never replace the downstream input.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Output name must be a nonempty string")
        if self._output_recorder is None:
            raise RuntimeError("Outputs can only be recorded from an active hook callback")
        self._output_recorder(self, name, value)

    @property
    def simulator_runtime(self) -> SimulatorRuntime | WorldBackend:
        return self.run.simulator_runtime

    @property
    def avstack(self) -> AVStack:
        return self.run.avstack

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
        return self.physical_state.t


class Plugin(ABC):
    """Optional lifecycle base for a stateful attack, defense or localization adapter."""

    def reset(self, context: RunContext) -> None:
        """Clear per-run history. Actors may not exist until ``simulator.setup``."""

    @abstractmethod
    def transform(self, value: Any, context: StageContext) -> Any:
        """Return the value consumed by the next handler or downstream component."""

    def after_step(self, context: StageContext) -> None:
        """Receive feedback after the command has affected the world. Optional."""

    def close(self) -> None:
        """Release resources, including partially initialized resources after an error."""


class ObservationTransform:
    """Adapt a one-argument observation function without losing its lifecycle."""

    def __init__(self, transform: Callable[[Observation], Observation]) -> None:
        self.function = transform

    @property
    def lifecycle_owner(self) -> Any:
        return self.function

    def transform(self, observation: Observation, context: StageContext) -> Observation:
        return self.function(observation)


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


class HookSet:
    """Reusable ordered hooks, independent of the driving system's Vehicle I/O.

    A fixed seed starts a fresh RNG for each run. Stateful handlers must implement
    ``reset`` or be constructed afresh. ``Runtime`` retains the legacy localizer argument.
    """

    def __init__(
        self,
        hooks: Iterable[Hook] = (),
        *,
        case: Any = None,
        seed: int | None = 0,
    ) -> None:
        self.hooks = tuple(hooks)
        if any(not isinstance(hook, Hook) for hook in self.hooks):
            raise TypeError("HookSet hooks must be Hook instances")
        self.case, self.seed = case, seed

    def with_hooks(self, hooks: Iterable[Hook], *, prepend: bool = False) -> Self:
        """Copy this configuration while retaining compatibility-facade settings."""
        extra = tuple(hooks)
        if any(not isinstance(hook, Hook) for hook in extra):
            raise TypeError("HookSet hooks must be Hook instances")
        result = copy(self)
        result.hooks = extra + self.hooks if prepend else self.hooks + extra
        return result

    def _vehicle_io(self, avstack: AVStack) -> VehicleIO:
        from avsectester.closed_loop.vehicle_io import VehicleIO

        result = getattr(avstack, "vehicle_io", None)
        if result is None:
            return VehicleIO()
        if not isinstance(result, VehicleIO):
            raise TypeError("avstack.vehicle_io must be VehicleIO")
        return result

    def validate_stages(self, simulator_runtime: SimulatorRuntime | WorldBackend, avstack: AVStack) -> frozenset[str]:
        """Check adapter capabilities without binding a dispatcher or invoking handlers."""
        from avsectester.closed_loop.avstack import validate_command_compatibility

        validate_command_compatibility(simulator_runtime, avstack, self)
        available = HookDispatcher.CORE_STAGES | simulator_runtime.supported_stages | avstack.supported_stages
        if self._vehicle_io(avstack).localizer is not None:
            available |= {"localization.pre"}
        unknown = {hook.stage for hook in self.hooks} - available
        if unknown:
            raise UnsupportedStageError(
                f"Unsupported hook stages: {sorted(unknown)}. Available: {sorted(available)}"
            )
        return available

    @property
    def setup_ticks(self) -> int:
        """Required deployment ticks, shared by both runs of a pair when supported."""
        handlers = {id(hook.handler): hook.handler for hook in self.hooks if hook.stage == "simulator.setup"}
        counts = [getattr(handler, "setup_ticks", 0) for handler in handlers.values()]
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
            raise ValueError("Plugin setup_ticks must be nonnegative integers")
        return sum(counts)

    def session(self, simulator_runtime: SimulatorRuntime | WorldBackend, avstack: AVStack, perturb: Callable[[Observation], Observation] | None = None) -> HookDispatcher:
        """Create one execution session, validating capabilities before resetting the world."""
        hook_set = self
        if perturb is not None:
            hook_set = self.with_hooks([Hook("observation", ObservationTransform(perturb))], prepend=True)
        return HookDispatcher(hook_set, simulator_runtime, avstack)


class Runtime(HookSet):
    """Compatibility facade for ``Runtime(..., localizer=...)``.

    New driving adapters own ``VehicleIO``. A legacy localizer is held by a VehicleIO
    adapter for this run without mutating the caller's driving-system configuration.
    """

    def __init__(
        self,
        hooks: Iterable[Hook] = (),
        *,
        localizer: Callable[[Observation, StageContext], StateEstimate] | Plugin | None = None,
        case: Any = None,
        seed: int | None = 0,
    ) -> None:
        from avsectester.closed_loop.vehicle_io import VehicleIO

        super().__init__(hooks, case=case, seed=seed)
        self._legacy_vehicle_io = VehicleIO(localizer)

    @property
    def localizer(self) -> Any:
        return self._legacy_vehicle_io.localizer

    @localizer.setter
    def localizer(self, value: Any) -> None:
        from avsectester.closed_loop.vehicle_io import VehicleIO

        self._legacy_vehicle_io = VehicleIO(value)

    def _vehicle_io(self, avstack: AVStack) -> VehicleIO:
        driving_io = super()._vehicle_io(avstack)
        if self.localizer is None:
            return driving_io
        if driving_io.localizer is not None and driving_io.localizer is not self.localizer:
            raise ValueError("Specify a localizer on VehicleIO or legacy Runtime, not both")
        return self._legacy_vehicle_io

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None, *, case: Any = None) -> Runtime:
        """Load legacy stage handlers and localization configuration."""
        # The legacy Runtime facade still serves configured CLI and concrete attack callers.
        from avsectester.interventions import runtime_from_config

        return runtime_from_config(config, case=case)


class HookDispatcher:
    """One bound run. Custom adapters publish supported stages through ``emit``."""
    CORE_STAGES = frozenset({"sensors.post", "localization.post", "observation", "command"})

    def __init__(self, hook_set: HookSet, simulator_runtime: SimulatorRuntime | WorldBackend, avstack: AVStack) -> None:
        self.hook_set = hook_set
        self.context = RunContext(simulator_runtime, avstack, hook_set.case, np.random.default_rng(hook_set.seed))
        self.vehicle_io = hook_set._vehicle_io(avstack)
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
        if self.vehicle_io.localizer is not None:
            participants.append(self.vehicle_io.localizer)
        # A single plugin can be bound to several stages but owns one lifecycle per run.
        owners = [getattr(value, "lifecycle_owner", value) for value in participants]
        self.participants = tuple({id(value): value for value in owners}.values())

    @contextmanager
    def activate(self) -> Iterator[HookDispatcher]:
        if self._active:
            raise RuntimeError("Hook dispatcher is already active")
        resources = ExitStack()
        failure = None
        try:
            simulator_runtime = self.context.simulator_runtime
            previous_ticks = simulator_runtime.setup_ticks
            simulator_runtime.setup_ticks = max(previous_ticks, self.hook_set.setup_ticks)
            resources.callback(setattr, simulator_runtime, "setup_ticks", previous_ticks)
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
        from avsectester.closed_loop.simulator import SimulatorRuntime

        simulator_runtime = self.context.simulator_runtime
        physical_state = (simulator_runtime.state() if isinstance(simulator_runtime, SimulatorRuntime)
                          else simulator_runtime.ground_truth())  # legacy WorldBackend
        return StageContext(
            self.context, self.step, stage, physical_state, sensor,
            self.observation, self.control if stage == "command" else deepcopy(self.control),
            self.record_output,
        )

    def record_output(self, context: StageContext, name: str, value: Any) -> None:
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
                "render.pre": CameraRenderRequest,
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
        """Apply the driving system's Vehicle I/O with this run's ordered hooks."""
        return self.vehicle_io.process(observation, self)

    def command(self, control: Control) -> Control:
        self._require(control, Control, "avstack")
        self.control = control
        control = self.emit("command", control)
        self._require(control, Control, "command")
        self.context.simulator_runtime.validate_command(control)
        self.control = control
        return control

    def after_step(self, observation: Observation) -> None:
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
