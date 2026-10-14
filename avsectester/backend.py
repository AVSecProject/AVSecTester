"""The two interfaces the data plane connects — and the loop that drives them.

    WorldBackend : reset() / step(command) -> Observation      (CARLA, NuRec, custom worlds)
    AVStack      : (observation) -> Control                     (modular pipeline OR end-to-end model)

Both are deliberately ignorant of each other's internals. The backend owns world state and the
shared vehicle dynamics. The AV box owns the driving decision. Neither knows whether the other is
CARLA vs a neural-reconstruction world model, or a modular perception->planning->control pipeline
vs a single end-to-end model. Like :mod:`avsectester.plane`, this module imports nothing from
avstack/avcarla/carla — the backend and stack *implementations* carry those dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from .plane import Control, FrameRecord, Observation, Trace, WorldSnapshot

if TYPE_CHECKING:
    from .runtime import Runtime, RuntimeSession


class _RuntimeAdapter:
    """Shared stage binding for backend and stack adapters."""

    _runtime: RuntimeSession | None = None

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset()

    @property
    def native(self) -> Mapping[str, Any]:
        return {}

    @contextmanager
    def runtime_context(self, session: RuntimeSession) -> Iterator[None]:
        if getattr(self, "_runtime", None) is not None:
            raise RuntimeError("Adapter is already bound to an active run")
        self._runtime = session
        try:
            yield
        finally:
            self._runtime = None

    def _emit(self, stage: str, value: Any, sensor: str | None = None) -> Any:
        session = getattr(self, "_runtime", None)
        return session.emit(stage, value, sensor=sensor) if session is not None else value


class WorldBackend(_RuntimeAdapter, ABC):
    """A world that renders sensor data and applies shared physical control.

    Owns the world state and the ego dynamics. The AV stack never sees world state, only the
    :class:`~avsectester.plane.Observation`. ``step`` applies a :class:`~avsectester.plane.Control`
    through the shared physics, advances the world + traffic, renders the next sensors, and returns
    the next Observation. The causal invariant (so the loop serializes over gRPC): a control at
    frame *t* affects frame *t+1*, never the already-emitted sensors of frame *t*.
    """

    # Backend initialization steps required by world-deployment plugins. These do not
    # produce driving records and must be matched across comparison runs.
    setup_ticks: int = 0

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Command representations consumed by this backend, or None if undeclared.

        Built-in modes are ``actuator`` and ``trajectory``. A custom adapter can retain
        its existing command semantics by leaving the declaration unset.
        """
        return None

    def validate_command(self, command: Control) -> None:
        """Reject active command fields that this backend would silently ignore."""
        validate_command_modes(command, self.command_modes)

    def configure_conditions(self, world: Mapping[str, Any]) -> None:
        """Configure supported world conditions for subsequent resets.

        Custom backends override this method to validate and apply their own world
        parameters. Unimplemented conditions fail instead of silently doing nothing.
        """
        if world:
            raise ValueError(f"{type(self).__name__} does not support world conditions: {sorted(world)}")

    @abstractmethod
    def reset(self) -> Observation:
        """Build the world, spawn the ego + traffic, and return the first Observation."""

    @abstractmethod
    def step(self, control: Control) -> Observation:
        """Apply ``control`` via the shared dynamics, advance one frame, return the Observation."""

    def close(self) -> None:
        """Release simulator resources. Optional."""

    def copy_observation(self, observation: Observation) -> Observation:
        """Detach the complete mutable data graph before model-visible processing.

        Native sensor captures that cannot be copied require a backend override documenting
        which buffers are borrowed and how to replace them safely. Do not copy SDK handles.
        """
        return deepcopy(observation)

    def _remember_ground_truth(self, observation: Observation) -> None:
        self._truth = deepcopy(WorldSnapshot(
            observation.t, observation.frame, observation.vehicle_state, observation.ego_speed,
        ))

    def ground_truth(self) -> WorldSnapshot:
        """Return an independent true-state snapshot, never an attacked Observation.

        The default captures raw observations at reset/step for ideal-state custom backends.
        Backends returning estimated states must override this using their physical state.
        """
        truth = getattr(self, "_truth", None)
        if truth is None:
            raise RuntimeError("Backend has not produced a world state yet")
        return deepcopy(truth)

    def prepare_clean_attack_pair(self) -> None:
        """Prepare consecutive resets to replay one initial scene for clean and attack.

        Deterministic backends need no extra preparation. Backends with random scene choices or
        spawn fallback must resolve them once and replay the first run's actual initial settings.
        Subsequent world dynamics remain responsive to each run's controls.
        """


class AVStack(_RuntimeAdapter, ABC):
    """The AV 'box': an Observation in, a Control out.

    The interface knows nothing about how the box works — a modular perception->tracking->planning
    ->control pipeline and a single end-to-end model both satisfy it. Implementations adapt their
    own machinery (e.g. an avstack ``ModularDrivingPipeline``, or a camera end-to-end policy).
    """

    def reset(self, observation: Observation) -> None:
        """Optional warmup from the first Observation (e.g. seed a planner). Default no-op."""

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Possible output command representations, or None for an undeclared adapter."""
        return None

    @property
    def models(self) -> Mapping[str, Any]:
        """Loaded native models available to user handlers. Adapters expose their own names."""
        return {}

    @abstractmethod
    def __call__(self, observation: Observation) -> Control:
        """Decide the control command for this Observation."""


def validate_command_modes(command: Control, modes: frozenset[str] | None) -> None:
    """Validate the fields consumed by a backend or independently called dynamics model."""
    if not isinstance(command, Control):
        raise TypeError(f"Expected Control, got {type(command).__name__}")
    if modes is None:
        return
    if command.trajectory is not None and "trajectory" not in modes:
        raise ValueError("Backend does not support trajectory commands")
    active = [name for name in ("throttle", "brake", "steer") if getattr(command, name) != 0.0]
    if active and "actuator" not in modes:
        raise ValueError(f"Backend does not support actuator command fields: {active}")


class CommandCompatibilityError(ValueError):
    """Declared stack commands cannot be consumed by the selected backend."""


def validate_command_compatibility(
    backend: WorldBackend, stack: AVStack, runtime: Runtime | None = None,
) -> None:
    """Check declared commands before world setup, allowing explicit runtime conversion.

    A command-stage handler may translate the stack output into a different representation.
    In that case only the final command is checked, after all handlers have run.
    """
    accepted, produced = backend.command_modes, stack.command_modes
    if accepted is None or produced is None:
        return
    if runtime is not None and any(hook.stage == "command" for hook in runtime.hooks):
        return
    missing = produced - accepted
    if missing:
        raise CommandCompatibilityError(
            f"{type(stack).__name__} produces {sorted(missing)} commands unsupported by "
            f"{type(backend).__name__}. Select compatible dynamics or add a command conversion hook."
        )


def run(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    perturb: Callable[[Observation], Observation] | None = None,
    on_step: Callable[[int, Observation, Control], None] | None = None,
    *,
    runtime: Runtime | None = None,
) -> Trace:
    """Drive ``stack`` in ``backend`` for ``frames`` steps and return the driving :class:`Trace`.

    ``runtime`` configures ordered stage handlers and their per-run lifecycle. ``perturb`` is
    an optional one-argument observation transform, applied before runtime observation handlers.
    Both act on model-visible data independently of the backend's physical state. The Trace
    records true backend time and speed and the final command applied to the world.

    ``on_step(i, seen, control)`` receives independent copies after command handlers and before
    actuation. The caller retains ownership of the backend and is responsible for closing it.
    """
    from .runtime import Runtime

    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise ValueError("frames must be a nonnegative integer")
    session = (runtime or Runtime()).session(backend, stack, perturb)
    trace = Trace(outputs=session.outputs)
    with session.activate():
        obs = backend.reset()
        backend._remember_ground_truth(obs)
        # The initial input is transformed once and shared by reset and the first decision.
        seen = session.observe(obs) if frames else backend.copy_observation(obs)
        stack.reset(seen)
        for i in range(frames):
            session.step = i
            control = session.command(stack(seen))
            if on_step is not None:
                # Instrumentation cannot accidentally mutate the command or model input.
                on_step(i, backend.copy_observation(seen), deepcopy(control))
            obs = backend.step(control)
            backend._remember_ground_truth(obs)
            truth = backend.ground_truth()
            trace.records.append(FrameRecord(
                frame=i, t=truth.t, speed=truth.ego_speed,
                throttle=control.throttle, brake=control.brake, steer=control.steer,
            ))
            session.after_step(obs)
            if i + 1 < frames:
                session.step = i + 1
                seen = session.observe(obs)
    return trace
