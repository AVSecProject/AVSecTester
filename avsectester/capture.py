"""One capture lifecycle for source-aware simulators; native operations stay in backends."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .backend import WorldBackend
from .datasets.base import FrameRef, SourceData, nonnegative
from .plane import Control, Observation


@dataclass(frozen=True)
class DeploymentPlan:
    """Shared tick budget for all treatments, including reference materialization.

    Participants are the existing world.setup hooks bound by the runner. They may consume
    ticks through setup_tick; the remainder is executed by the backend. Neither path can
    exceed this budget. Selection metadata never enters the simulator.
    """

    ticks: int = 0

    def __post_init__(self) -> None:
        nonnegative(self.ticks, "deployment ticks")


class CaptureBackend(WorldBackend):
    source_kinds: tuple[str, ...] = ()
    can_render: bool = False
    decision_time_step: float | None = None
    simulator_version = "1"
    # Native ticks that make world.setup effects visible in the first capture. A simulator
    # that renders on every tick (CARLA) needs one; one that renders at capture time needs none.
    capture_ticks = 0

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if {"reset", "step"} & cls.__dict__.keys():
            raise TypeError("CaptureBackend subclasses implement native operations, not legacy reset/step")

    def __init__(self) -> None:
        self._capture: Observation | None = None
        self._world_initialized = False
        self._source: SourceData | None = None
        self._deployment = DeploymentPlan()
        self._in_setup = False
        self._setup_ticks_used = 0
        self.initialization_report: dict[str, Any] = {}

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset({"world.setup", "world.step.pre", "world.step.post"})

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"simulator": type(self).__name__, "version": self.simulator_version,
                "time_step": self.decision_time_step, "capture_ticks": self.capture_ticks}

    def validate_source(self, source: SourceData | None) -> None:
        if source is None:
            return
        if not isinstance(source, SourceData):
            raise TypeError("Simulator reset requires SourceData, never selection metadata")
        if source.kind not in self.source_kinds:
            raise ValueError(f"{type(self).__name__} requires {'/'.join(self.source_kinds)} SourceData")
        dt = source.payload.get("time_step")
        if dt is not None and dt != self.decision_time_step:
            raise ValueError(f"Dataset time_step={dt} differs from simulator decision time step "
                             f"{self.decision_time_step}")

    def reset_world(self, source: SourceData | None = None, initialization: Any = None, *,
                    deployment: DeploymentPlan | None = None,
                    environment: Mapping[str, Any] | None = None) -> None:
        self.validate_source(source)
        plan = deployment if deployment is not None else DeploymentPlan(self.setup_ticks)
        if plan.ticks < self.setup_ticks:
            raise ValueError("Shared deployment plan is smaller than participant setup_ticks")
        position = self._replay_position(source)
        # The deployment window ends at the selected capture: participant ticks plus the
        # simulator's capture ticks. It is carved out of the pre-roll, never appended to it.
        window = plan.ticks + self.capture_ticks
        if 0 < position < window:
            raise ValueError("Selected boundary is earlier than its deployment window")
        self._world_initialized = False
        self._capture = None
        self.initialization_report = {}
        self._source, self._deployment = source, plan
        self._initialize_world(source, initialization, environment or {})
        for _ in range(max(0, position - window)):
            self._replay_tick()
        self._setup_ticks_used = 0
        self._in_setup = True
        try:
            self._emit("world.setup", None)
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

    def setup_tick(self) -> None:
        if not self._in_setup:
            raise RuntimeError("setup_tick is only available during world.setup")
        if self._setup_ticks_used >= self._deployment.ticks:
            raise RuntimeError("Deployment participant exceeded the shared tick budget")
        self._window_tick(self._replay_position(self._source))
        self._setup_ticks_used += 1

    def reset(self) -> Observation:
        self.reset_world()
        return self.observe()

    def step(self, control: Control) -> Observation:
        self.advance(control)
        return self.observe()

    def advance(self, control: Control) -> None:
        self.validate_command(control)
        if not self._world_initialized:
            raise RuntimeError("Reset the simulator before advancing")
        self._capture = None
        self._world_initialized = False
        self._emit("world.step.pre", None)
        self._advance_native(control)
        self._emit("world.step.post", None)
        self._capture = self._observe()
        self._world_initialized = True

    def observe(self) -> Observation:
        if not self._world_initialized or self._capture is None:
            raise RuntimeError("Reset the simulator before observing")
        return self.copy_observation(self._capture)

    def _replay_position(self, source: SourceData | None) -> int:
        return 0

    def _window_tick(self, position: int) -> None:
        """Advance one deployment-window tick: continue replay, or hold a p=0/plain origin."""
        if position:
            self._replay_tick()
        else:
            self._hold_tick()

    def _initialize_world(self, source: SourceData | None, initialization: Any,
                          environment: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def _advance_native(self, control: Control) -> None:
        raise NotImplementedError

    def _observe(self) -> Observation:
        raise NotImplementedError

    def _replay_tick(self) -> None:
        raise NotImplementedError("This backend does not pre-roll")

    def _hold_tick(self) -> None:
        raise ValueError("This backend does not support physical deployment ticks")

    def _before_selected_capture(self, position: int) -> None:
        pass

    def _check_initialization(self, initialization: Any) -> None:
        if initialization is not None:
            raise ValueError("This backend does not support replay initialization")

    def materialize_reference(self, source: SourceData, frames: Sequence[FrameRef], **kwargs: Any) -> Any:
        raise NotImplementedError("This backend cannot materialize references")

    def render(self, observation: Observation, request: Any) -> Any:
        raise NotImplementedError("This backend does not render insertions")
