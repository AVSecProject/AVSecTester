"""The simulator: a configuration (``Simulator``) and its live worlds (``SimulatorRuntime``).

``Simulator`` is what a user configures: settings and declarations, with no world state. Each
run or service operation creates one ``SimulatorRuntime`` through ``create_runtime``; the
runtime owns the world, clock, current capture and native connections until it is closed.
Concrete runtimes implement native initialization, one native advance and one native capture.
The runtime base owns the shared order: scenario-start reset, pre-roll, deployment window,
capture. It also owns the process-local reservation of the simulated world and its read-only
clock. Legacy ``WorldBackend`` implementations run through an internal adapter with the same
surface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from avsectester.closed_loop.hooks import _StageAdapter
from avsectester.datasets.base import ActorState, FrameData, FrameReference, ScenarioStart, nonnegative
from avsectester.closed_loop.datatypes import (
    Control, Observation, WorldSnapshot, validate_command_modes,
)

if TYPE_CHECKING:
    from avsectester.closed_loop.frame_data import FrameDataSequence
    import numpy as np

    from avsectester.geometry.poses import ActorPose
    from avsectester.geometry.cameras import Camera


class WorldBackend(_StageAdapter, ABC):
    """Legacy world interface and the base of ``Simulator``; removed as a public base in phase 4.

    A world that renders sensor data and applies shared physical control.

    Owns the world state and the ego dynamics. The AV stack never sees world state, only the
    :class:`~avsectester.closed_loop.datatypes.Observation`. ``step`` applies a :class:`~avsectester.closed_loop.datatypes.Control`
    through the shared physics, advances the world + traffic, renders the next sensors, and returns
    the next Observation. The causal invariant (so the loop serializes over gRPC): a control at
    frame *t* affects frame *t+1*, never the already-emitted sensors of frame *t*.
    """

    # Simulator initialization steps required by world-deployment plugins. These do not
    # produce driving records and must be matched across comparison runs.
    setup_ticks: int = 0

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Command representations consumed by this simulator, or None if undeclared.

        Built-in modes are ``actuator`` and ``trajectory``. A custom adapter can retain
        its existing command semantics by leaving the declaration unset.
        """
        return None

    def validate_command(self, command: Control) -> None:
        """Reject active command fields that this simulator would silently ignore."""
        validate_command_modes(command, self.command_modes)

    def configure_conditions(self, world: Mapping[str, Any]) -> None:
        """Configure supported world conditions for subsequent resets.

        Custom simulators override this method to validate and apply their own world
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

        Native sensor captures that cannot be copied require a simulator override documenting
        which buffers are borrowed and how to replace them safely. Do not copy SDK handles.
        """
        return deepcopy(observation)

    def _remember_ground_truth(self, observation: Observation) -> None:
        self._truth = deepcopy(WorldSnapshot(
            observation.t, observation.frame, observation.vehicle_state, observation.ego_speed,
        ))

    def ground_truth(self) -> WorldSnapshot:
        """Return an independent true-state snapshot, never an attacked Observation.

        The default captures raw observations at reset/step for ideal-state custom simulators.
        Simulators returning estimated states must override this using their physical state.
        """
        truth = getattr(self, "_truth", None)
        if truth is None:
            raise RuntimeError("Simulator has not produced a world state yet")
        return deepcopy(truth)

    def prepare_clean_attack_pair(self) -> None:
        """Prepare consecutive resets to replay one initial scene for clean and attack.

        Deterministic simulators need no extra preparation. Simulators with random scene choices or
        spawn fallback must resolve them once and replay the first run's actual initial settings.
        Subsequent world dynamics remain responsive to each run's controls.
        """


@dataclass(frozen=True)
class DeploymentPlan:
    """Shared tick budget for all treatments, including reference generation.

    Participants are the simulator.setup hooks bound by the runner, including attack runtimes
    whose setup returns world attack effects. They may consume ticks through setup_tick;
    the remainder is executed by the simulator. Neither path can exceed this budget.
    Selection metadata never enters the simulator.
    """

    ticks: int = 0

    def __post_init__(self) -> None:
        nonnegative(self.ticks, "deployment ticks")


@dataclass(frozen=True)
class ClockSnapshot:
    """Physical simulator time and frame, never a model-visible timestamp."""

    t: float
    frame: int


@dataclass(frozen=True)
class SensorView:
    """What a simulator provides for one sensor capture. It contains no insertion rendering.

    ``data`` is the captured camera image as RGB uint8 (H, W, 3). ``camera`` projects optical
    camera points to pixels and unprojects pixels to rays, hiding pinhole versus f-theta
    differences. ``cam_from_world`` maps world points to optical camera axes (x right, y down,
    z forward). ``actors`` holds world poses by stable actor ID and ``ego`` is the ego's cuboid.
    ``depth`` is optical-axis depth in metres per pixel, when the simulator measures it;
    without it, occlusion is estimated from actor cuboids. ``native`` gives simulator-specific
    access and gives up portability. World units are metres, with x forward, y left, z up.
    """

    sensor: str
    data: Any
    camera: Camera
    cam_from_world: np.ndarray
    actors: Mapping[str, ActorPose]
    ego: ActorPose
    time: float
    frame: int
    depth: np.ndarray | None = None
    native: Mapping[str, Any] = field(default_factory=dict)


class Simulator(ABC):
    """A configured simulator: settings and declarations, without world state.

    It can be reused. ``create_runtime`` creates one live world per run or service operation;
    the caller closes it. Declarations describe every runtime of this simulator.
    """

    dataset_kinds: tuple[str, ...] = ()
    can_render: bool = False
    supported_world_settings: frozenset[str] = frozenset()
    simulator_version = "1"
    # Native ticks that make simulator.setup effects visible in the first capture. A simulator
    # that renders on every tick (CARLA) needs one; one that renders at capture time needs none.
    capture_ticks = 0

    @abstractmethod
    def create_runtime(self, seed: int | None = None) -> SimulatorRuntime:
        """Create a new live world with this configuration; ``seed`` seeds its random choices."""

    @property
    def decision_time_step(self) -> float | None:
        return None

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Command representations consumed by this simulator, or None if undeclared."""
        return None

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset({"simulator.setup", "simulator.step.pre", "simulator.step.post"})

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"simulator": type(self).__name__, "version": self.simulator_version,
                "time_step": self.decision_time_step, "capture_ticks": self.capture_ticks}

    def validate_world_settings(self, settings: Mapping[str, Any]) -> None:
        """Check world-setting names and values without creating a live world.

        Simulator implementations extend this with checks that need no native connection.
        Actor existence and other live capabilities are checked during initialization.
        """
        if not isinstance(settings, Mapping):
            raise TypeError("Simulator world settings must be a mapping")
        unsupported = set(settings) - self.supported_world_settings
        if unsupported:
            raise ValueError(f"{type(self).__name__} does not support world settings "
                             f"{sorted(unsupported)}. Use explicitly configured sensors transforms instead")

    def validate_start(self, start: ScenarioStart | None) -> None:
        """Reject a scenario start this simulator cannot reset to, before any world exists."""
        if start is None:
            return
        if not isinstance(start, ScenarioStart):
            raise TypeError("Simulator reset requires ScenarioStart, never selection metadata")
        if start.kind not in self.dataset_kinds:
            raise ValueError(f"{type(self).__name__} requires {'/'.join(self.dataset_kinds)} ScenarioStart")
        dt = start.payload.get("time_step")
        if dt is not None and dt != self.decision_time_step:
            raise ValueError(f"Dataset time_step={dt} differs from simulator decision time step "
                             f"{self.decision_time_step}")


class SimulatorRuntime(_StageAdapter, ABC):
    """One live world of a ``Simulator``: world state, clock, actors, sensors and sensor views.

    A runtime is created by ``Simulator.create_runtime`` and closed by its creator, also as a
    context manager. ``reset`` starts a run and may be called again for a further run on the
    same world. Declarations come from ``simulator``.
    """

    # Simulator initialization steps required by legacy world-deployment plugins. They do not
    # produce driving records and must be matched across comparison runs.
    setup_ticks: int = 0
    # Owner of the process-local world reservation. Services set it so that the nested
    # operations of one session share their reservation.
    resource_owner: object | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if {"reset", "step"} & cls.__dict__.keys():
            raise TypeError("SimulatorRuntime subclasses implement native operations, not reset/step")

    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator
        self._capture: Observation | None = None
        self._world_initialized = False
        self._resetting = False
        self._stepping = False
        self._start: ScenarioStart | None = None
        self._deployment = DeploymentPlan()
        self._in_setup = False
        self._setup_ticks_used = 0
        self._exclusive = False
        # True while generate_frame_data replays a start: the runtime keeps following the
        # start's prescribed motion instead of handing the ego to a driving system.
        self._generating_frame_data = False
        self.initialization_report: dict[str, Any] = {}

    def __enter__(self) -> SimulatorRuntime:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # Declarations of the configured simulator.
    @property
    def dataset_kinds(self) -> tuple[str, ...]:
        return self.simulator.dataset_kinds

    @property
    def can_render(self) -> bool:
        return self.simulator.can_render

    @property
    def supported_world_settings(self) -> frozenset[str]:
        return self.simulator.supported_world_settings

    @property
    def capture_ticks(self) -> int:
        return self.simulator.capture_ticks

    @property
    def decision_time_step(self) -> float | None:
        return self.simulator.decision_time_step

    @property
    def command_modes(self) -> frozenset[str] | None:
        return self.simulator.command_modes

    @property
    def supported_stages(self) -> frozenset[str]:
        return self.simulator.supported_stages

    @property
    def identity(self) -> Mapping[str, Any]:
        return self.simulator.identity

    def validate_start(self, start: ScenarioStart | None) -> None:
        self.simulator.validate_start(start)

    def validate_command(self, command: Control) -> None:
        """Reject active command fields that this simulator would silently ignore."""
        validate_command_modes(command, self.command_modes)

    def configure_conditions(self, world: Mapping[str, Any]) -> None:
        """Legacy: configure world conditions for subsequent resets of this runtime."""
        if world:
            raise ValueError(f"{type(self).__name__} does not support world conditions: {sorted(world)}")

    def prepare_clean_attack_pair(self) -> None:
        """Legacy: make consecutive resets replay one initial scene for clean and attack."""

    def copy_observation(self, observation: Observation) -> Observation:
        """Detach the complete mutable data graph before model-visible processing.

        Native sensor captures that cannot be copied require a runtime override documenting
        which buffers are borrowed and how to replace them safely. Do not copy SDK handles.
        """
        return deepcopy(observation)

    def close(self) -> None:
        """Release the world and native connections. Optional."""

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Reserve this world for one operation; nested independent operations are rejected."""
        from avsectester.closed_loop.resources import reserve, simulator_key

        if self._exclusive:
            raise RuntimeError("Simulator already has an active operation")
        owner = self.resource_owner if self.resource_owner is not None else self
        with reserve(simulator_key(self), owner):
            self._exclusive = True
            try:
                yield
            finally:
                self._exclusive = False

    @property
    def clock(self) -> ClockSnapshot:
        state = self.state()
        return ClockSnapshot(t=state.t, frame=state.frame)

    def state(self) -> WorldSnapshot:
        """Read detached physical state without a capture, callback or simulator tick.

        Available after reset and during reset and step, so stage handlers can read it.
        """
        if not (self._world_initialized or self._resetting or self._stepping):
            raise RuntimeError("Reset the simulator before requesting its state")
        return self._state()

    def sensor_view(self, sensor: str) -> SensorView:
        """Return the view of the current capture of ``sensor``; it never advances time."""
        raise NotImplementedError(f"{type(self).__name__} does not provide sensor views")

    def control_actor(self, actor_id: str, control: Control) -> None:
        """Drive another actor with an actuator command from the next world step.

        The actor stops following its prescribed or traffic-manager motion for the rest of the
        run. Call it from a ``WorldAttackEffect``. The ego belongs to the driving system.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot control actors")

    def set_actor_state(self, actor_id: str, state: ActorState) -> None:
        """Move another actor to a canonical world state; visible from the next capture.

        The actor stops following its prescribed or traffic-manager motion for the rest of the
        run. Call it from a ``WorldAttackEffect``. The ego belongs to the driving system.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot set actor states")

    def reset(self, start: ScenarioStart | None = None, initialization: Any = None, *,
              deployment: DeploymentPlan | None = None,
              world: Mapping[str, Any] | None = None) -> None:
        """Start a run: reset to ``start`` (or the configured origin) and capture the first frame.

        The order is native initialization, pre-roll to the deployment window, simulator.setup
        and its deployment ticks, the capture ticks, then the first capture. Read it with
        ``observe``.
        """
        if start is None and initialization is not None:
            raise ValueError("An initialization plan requires ScenarioStart")
        self.validate_start(start)
        # The simulator owns only native world settings. Sensor environments belong to
        # the runner/services after capture; rejecting envelopes avoids silently losing them.
        world_environment = dict(world or {})
        self.simulator.validate_world_settings(world_environment)
        plan = deployment if deployment is not None else DeploymentPlan(self.setup_ticks)
        if plan.ticks < self.setup_ticks:
            raise ValueError("Shared deployment plan is smaller than participant setup_ticks")
        position = self._replay_position(start)
        # The deployment window ends at the selected capture: participant ticks plus the
        # simulator's capture ticks. It is carved out of the pre-roll, never appended to it.
        window = plan.ticks + self.capture_ticks
        if 0 < position < window:
            raise ValueError("Selected boundary is earlier than its deployment window")
        self._world_initialized = False
        self._capture = None
        self.initialization_report = {}
        self._start, self._deployment = start, plan
        self._resetting = True
        try:
            self._initialize_world(start, initialization, world_environment)
            for _ in range(max(0, position - window)):
                self._replay_tick()
            self._setup_ticks_used = 0
            self._in_setup = True
            try:
                self._emit("simulator.setup", None)
                while self._setup_ticks_used < plan.ticks:
                    before = self._setup_ticks_used
                    self.setup_tick()
                    if self._setup_ticks_used != before + 1:
                        raise RuntimeError("setup_tick must consume exactly one deployment tick")
            finally:
                self._in_setup = False
            for _ in range(self.capture_ticks):
                self._window_tick(position)
            self._before_selected_capture(position)
            self._capture = self._observe()
            self._world_initialized = True
            try:
                self._check_initialization(initialization)
            except BaseException:
                self._world_initialized = False
                self._capture = None
                raise
        finally:
            self._resetting = False

    def setup_tick(self) -> None:
        if not self._in_setup:
            raise RuntimeError("setup_tick is only available during simulator.setup")
        if self._setup_ticks_used >= self._deployment.ticks:
            raise RuntimeError("Deployment participant exceeded the shared tick budget")
        self._window_tick(self._replay_position(self._start))
        self._setup_ticks_used += 1

    def step(self, control: Control) -> None:
        """Apply ``control``, advance the world one decision and capture; read it with ``observe``."""
        self.validate_command(control)
        if not self._world_initialized:
            raise RuntimeError("Reset the simulator before stepping")
        self._capture = None
        self._world_initialized = False
        self._stepping = True
        try:
            self._emit("simulator.step.pre", None)
            self._advance_native(control)
            self._emit("simulator.step.post", None)
            self._capture = self._observe()
        finally:
            self._stepping = False
        self._world_initialized = True

    def observe(self) -> Observation:
        """Return an independent copy of the current capture; never advance or recapture."""
        if not self._world_initialized or self._capture is None:
            raise RuntimeError("Reset the simulator before observing")
        return self.copy_observation(self._capture)

    def _reset_scan(self, start: ScenarioStart, initialization: Any = None, *,
                    world: Mapping[str, Any] | None = None,
                    deployment: DeploymentPlan | None = None) -> None:
        """Initialize prescribed-motion scanning without releasing the ego to a driving model."""
        self._generating_frame_data = True
        try:
            self.reset(start, initialization, world=world, deployment=deployment)
        finally:
            self._generating_frame_data = False

    def _scan_forward(self, count: int) -> None:
        """Continue an effect-free service scan along prescribed motion, without stage hooks."""
        nonnegative(count, "scan steps")
        if not self._world_initialized or self._dispatcher is not None:
            raise RuntimeError("A sensor scan requires an initialized runtime without hooks")
        self._generating_frame_data = True
        try:
            for _ in range(count):
                self._capture = None
                self._replay_tick()
                self._capture = self._observe()
        except BaseException:
            self._world_initialized = False
            raise
        finally:
            self._generating_frame_data = False

    def _replay_position(self, start: ScenarioStart | None) -> int:
        return 0

    def _window_tick(self, position: int) -> None:
        """Advance one deployment-window tick: continue replay, or hold a p=0/plain origin."""
        if position:
            self._replay_tick()
        else:
            self._hold_tick()

    @abstractmethod
    def _initialize_world(self, start: ScenarioStart | None, initialization: Any,
                          environment: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def _advance_native(self, control: Control) -> None: ...

    @abstractmethod
    def _observe(self) -> Observation: ...

    @abstractmethod
    def _state(self) -> WorldSnapshot:
        """Return an independent snapshot of the physical state, never the model-visible data.

        It must work at every emitted stage, including ``simulator.setup`` during reset.
        """

    def _replay_tick(self) -> None:
        """Advance one frame along the start's prescribed motion, without hooks or a driver."""
        raise NotImplementedError("This simulator does not replay prescribed motion")

    def _hold_tick(self) -> None:
        raise ValueError("This simulator does not support physical deployment ticks")

    def _before_selected_capture(self, position: int) -> None:
        pass

    def _check_initialization(self, initialization: Any) -> None:
        if initialization is not None:
            raise ValueError("This simulator does not support replay initialization")

    def generate_frame_data(
        self, start: ScenarioStart, frames: Sequence[FrameReference], *,
        world: Mapping[str, Any] | None = None, deployment: DeploymentPlan | None = None,
    ) -> FrameDataSequence:
        """Record actual frame data by replaying ``start``'s prescribed motion over ``frames``.

        Used when the dataset has no frame data, such as a CARLA recipe. ``frames`` are the
        consecutive dataset frames beginning at ``start.frame``. ``world`` holds native world
        settings only, as for ``reset``. The runtime resets as for a run (same deployment plan
        and world settings), then advances one frame at a time along the
        prescribed motion and records each frame. No hook, attack or driving system runs.
        A runtime supports generation by implementing ``_replay_tick`` and
        ``_capture_frame_data``.
        """
        from avsectester.closed_loop.frame_data import FrameDataSequence

        if type(self)._capture_frame_data is SimulatorRuntime._capture_frame_data:
            raise NotImplementedError(f"{type(self.simulator).__name__} cannot generate frame data")
        frames = tuple(frames)
        if not frames or frames[0] != start.frame:
            raise ValueError("Frame data generation must begin at the start frame")
        if any(frame.scenario_id != start.scenario_id or frame.position != start.frame.position + i
               for i, frame in enumerate(frames)):
            raise ValueError("Frame data generation needs consecutive frames of the start's scenario")
        self._generating_frame_data = True
        try:
            self.reset(start, world=world, deployment=deployment)
            records = [self._capture_frame_data(frames[0])]
            for frame in frames[1:]:
                self._replay_tick()
                self._capture = self._observe()
                records.append(self._capture_frame_data(frame))
        finally:
            self._generating_frame_data = False
        if any(record.frame != frame for record, frame in zip(records, frames)):
            raise ValueError("Captured frame data must describe the requested frames")
        return FrameDataSequence(tuple(records), self._replayed_scenario(), {
            "simulator": dict(self.identity), "deployment_ticks": self._deployment.ticks,
            **self._frame_data_provenance(),
        })

    def _capture_frame_data(self, frame: FrameReference) -> FrameData:
        """Record the current capture and physical state as the frame data of ``frame``."""
        raise NotImplementedError(f"{type(self.simulator).__name__} cannot generate frame data")

    def _replayed_scenario(self) -> Mapping[str, Any]:
        """The resolved configuration that reproduces the generated frames, if any."""
        return {}

    def _frame_data_provenance(self) -> Mapping[str, Any]:
        """Simulator-specific provenance of generated frame data."""
        return {}


class LegacySimulatorAdapter:
    """Internal: give a legacy ``WorldBackend`` the simulator surface used by the runner.

    The adapter borrows the caller-owned world and caches the captures returned by its
    reset/step. It supports plain resets only. It is removed with ``WorldBackend`` in phase 4.
    """

    def __init__(self, backend: WorldBackend) -> None:
        if isinstance(backend, SimulatorRuntime):
            raise TypeError("Simulator runtimes do not need the legacy adapter")
        self.backend = backend
        self._capture: Observation | None = None
        self._ready = False
        self._resetting = False
        self._exclusive = False

    @property
    def setup_ticks(self) -> int:
        return self.backend.setup_ticks

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        from avsectester.closed_loop.resources import reserve, simulator_key

        if self._exclusive:
            raise RuntimeError("Simulator already has an active operation")
        with reserve(simulator_key(self.backend), self):
            self._exclusive = True
            try:
                yield
            finally:
                self._exclusive = False

    @property
    def clock(self) -> ClockSnapshot:
        state = self.state()
        return ClockSnapshot(t=state.t, frame=state.frame)

    def state(self) -> WorldSnapshot:
        if not self._ready and not self._resetting:
            raise RuntimeError("Reset the simulator before requesting its state")
        return self.backend.ground_truth()

    def sensor_view(self, sensor: str) -> SensorView:
        raise NotImplementedError(f"{type(self.backend).__name__} does not provide sensor views")

    def reset(self, start: Any = None, initialization: Any = None, *,
              deployment: DeploymentPlan | None = None, world: Any = None) -> None:
        if start is not None and not isinstance(start, ScenarioStart):
            raise TypeError("Simulator reset requires ScenarioStart, never selection metadata")
        if start is None and initialization is not None:
            raise ValueError("An initialization plan requires ScenarioStart")
        if start is not None or initialization is not None or deployment is not None or world:
            raise NotImplementedError("Legacy backend does not support ScenarioStart initialization")
        self._ready = False
        self._capture = None
        self._resetting = True
        try:
            self._remember(self.backend.reset())
            self._ready = True
        finally:
            self._resetting = False

    def observe(self) -> Observation:
        """Return an independent view of the current capture; never advance or recapture."""
        if not self._ready:
            raise RuntimeError("Reset the simulator before requesting an observation")
        assert self._capture is not None
        return self.backend.copy_observation(self._capture)

    def step(self, command: Control) -> None:
        """Advance the underlying world exactly once and retain the resulting capture."""
        if not self._ready:
            raise RuntimeError("Reset the simulator before stepping")
        self.backend.validate_command(command)
        # A failed advance must not leave a stale observation available as a successful result.
        self._ready = False
        self._capture = None
        self._remember(self.backend.step(command))
        self._ready = True

    def _remember(self, capture: Observation) -> None:
        if not isinstance(capture, Observation):
            raise TypeError("A legacy backend reset/step must return Observation")
        self.backend._remember_ground_truth(capture)
        self._capture = self.backend.copy_observation(capture)

    def close(self) -> None:
        """Release only this adapter's capture; the caller closes the borrowed backend."""
        self._capture = None
        self._ready = False
