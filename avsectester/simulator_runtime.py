"""Clock/observation boundary over one simulator, with a legacy backend bridge.

Built-in backends implement ``reset_world``, ``observe`` and ``advance`` as their primary
operations. This adapter borrows that same world; it does not duplicate physics or own its
resources. Old custom backends are supported by caching the captures returned by reset/step.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .backend import WorldBackend
from .capture import CaptureBackend, DeploymentPlan
from .plane import Control, Observation, WorldSnapshot


@dataclass(frozen=True)
class ClockSnapshot:
    """Physical simulator time and frame, never a model-visible timestamp."""

    t: float
    frame: int


class SimulatorRuntime:
    """Expose separated operations while borrowing a caller-owned ``WorldBackend``.

    CaptureBackend implementations share the capture lifecycle. Other WorldBackend
    implementations use the compatibility bridge, including their reset/step overrides.
    """

    def __init__(self, backend: WorldBackend, *, resource_owner: object | None = None) -> None:
        self.backend = backend
        self._native = isinstance(backend, CaptureBackend)
        self._capture: Observation | None = None
        self._ready = False
        self._initializing = False
        self._resource_owner = resource_owner if resource_owner is not None else self
        self._active = False

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Reserve this world for a run; nested independent runs are never allowed."""
        from .resources import backend_key, reserve

        if self._active:
            raise RuntimeError("SimulatorRuntime already has an active operation")
        with reserve(backend_key(self.backend), self._resource_owner):
            self._active = True
            try:
                yield
            finally:
                self._active = False

    @property
    def clock(self) -> ClockSnapshot:
        state = self.state()
        return ClockSnapshot(t=state.t, frame=state.frame)

    def state(self) -> WorldSnapshot:
        """Read detached physical state without a capture, callback or simulator tick."""
        if not self._ready and not self._initializing:
            raise RuntimeError("Reset the simulator before requesting its state")
        return self.backend.ground_truth()

    def reset(self, source: Any = None, initialization: Any = None, *,
              deployment: DeploymentPlan | None = None, environment: Any = None) -> None:
        """Source-aware capture backends use one path; legacy worlds retain reset/step."""
        if source is not None:
            from .datasets.base import SourceData

            if not isinstance(source, SourceData):
                raise TypeError("Simulator reset requires SourceData, never selection metadata")
        elif initialization is not None:
            raise ValueError("An initialization plan requires SourceData")
        self._ready = False
        self._capture = None
        self._initializing = True
        try:
            if isinstance(self.backend, CaptureBackend):
                self.backend.reset_world(source, initialization, deployment=deployment,
                                         environment=environment)
            else:
                if source is not None or initialization is not None or deployment is not None or environment:
                    raise NotImplementedError("Legacy backend does not support SourceData initialization")
                self._remember(self.backend.reset())
            self._ready = True
        finally:
            self._initializing = False

    def observe(self) -> Observation:
        """Return an independent view of the current capture; never advance or recapture."""
        if not self._ready:
            raise RuntimeError("Reset the simulator before requesting an observation")
        if isinstance(self.backend, CaptureBackend):
            return self.backend.observe()
        assert self._capture is not None
        return self.backend.copy_observation(self._capture)

    def step(self, command: Control) -> None:
        """Advance the underlying world exactly once and retain the resulting capture."""
        if not self._ready:
            raise RuntimeError("Reset the simulator before stepping")
        self.backend.validate_command(command)
        # A failed advance must not leave a stale observation available as a successful result.
        self._ready = False
        if isinstance(self.backend, CaptureBackend):
            self.backend.advance(command)
        else:
            self._capture = None
            self._remember(self.backend.step(command))
        self._ready = True

    def _remember(self, capture: Observation) -> None:
        if not isinstance(capture, Observation):
            raise TypeError("A legacy backend reset/step must return Observation")
        self.backend._remember_ground_truth(capture)
        self._capture = self.backend.copy_observation(capture)

    def close(self) -> None:
        """Release only this bridge's capture; the caller closes the borrowed backend."""
        self._capture = None
        self._ready = False
