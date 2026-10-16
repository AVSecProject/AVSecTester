"""Clock/observation boundary over one simulator, with a legacy backend bridge.

Built-in backends implement ``reset_world``, ``observe`` and ``advance`` as their primary
operations. This adapter borrows that same world; it does not duplicate physics or own its
resources. Old custom backends are supported by caching the captures returned by reset/step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .backend import WorldBackend
from .plane import Control, Observation, WorldSnapshot


@dataclass(frozen=True)
class ClockSnapshot:
    """Physical simulator time and frame, never a model-visible timestamp."""

    t: float
    frame: int


class _SeparatedBackend(Protocol):
    """Operations checked by the legacy migration bridge before native dispatch."""

    def reset_world(self, *, before_first_capture: Callable[[], None] | None = None) -> None: ...

    def observe(self) -> Observation: ...

    def advance(self, command: Control) -> None: ...


def _owner(backend: WorldBackend, name: str) -> type | None:
    """Find the implementation owner, preserving legacy subclass overrides."""
    if name in vars(backend):
        return None
    return next((cls for cls in type(backend).__mro__ if name in vars(cls)), None)


def _migrated(backend: WorldBackend, old: str, new: str) -> bool:
    legacy_owner, primary_owner = _owner(backend, old), _owner(backend, new)
    if legacy_owner is None or primary_owner is None:
        return False
    hierarchy = type(backend).__mro__
    return (
        callable(getattr(backend, new, None))
        and hierarchy.index(primary_owner) <= hierarchy.index(legacy_owner)
    )


class SimulatorRuntime:
    """Expose separated operations while borrowing a caller-owned ``WorldBackend``.

    A legacy subclass that overrides reset/step without migrating their separated counterparts
    stays on the compatibility path. In particular, additional setup performed by an old reset
    override must not be silently skipped. Such adapters cannot promise a pre-capture callback.
    """

    def __init__(self, backend: WorldBackend) -> None:
        self.backend = backend
        self._native = all(
            _migrated(backend, old, new)
            for old, new in (("reset", "reset_world"), ("step", "advance"))
        ) and callable(getattr(backend, "observe", None))
        self._capture: Observation | None = None
        self._ready = False
        self._initializing = False

    @property
    def supports_before_first_capture(self) -> bool:
        return self._native

    @property
    def clock(self) -> ClockSnapshot:
        state = self.state()
        return ClockSnapshot(t=state.t, frame=state.frame)

    def state(self) -> WorldSnapshot:
        """Read detached physical state without a capture, callback or simulator tick."""
        if not self._ready and not self._initializing:
            raise RuntimeError("Reset the simulator before requesting its state")
        return self.backend.ground_truth()

    def reset(
        self,
        source: Any = None,
        initialization: Any = None,
        *,
        before_first_capture: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the existing configured world and capture its first observation.

        SourceData/initialization-plan translation belongs to the next migration phase; reject
        it explicitly here rather than silently ignoring it. A legacy backend cannot run a
        callback after reset and present it as a before-capture deployment.
        """
        if source is not None or initialization is not None:
            raise NotImplementedError("SourceData and initialization plans are not supported yet")
        if before_first_capture is not None and not self.supports_before_first_capture:
            raise NotImplementedError(
                "This legacy backend does not support a before-first-capture callback"
            )
        self._ready = False
        self._capture = None
        self._initializing = True
        try:
            if self._native:
                cast(_SeparatedBackend, self.backend).reset_world(
                    before_first_capture=before_first_capture,
                )
            else:
                capture = self.backend.reset()
                self._remember(capture)
            self._ready = True
        finally:
            self._initializing = False

    def observe(self) -> Observation:
        """Return an independent view of the current capture; never advance or recapture."""
        if not self._ready:
            raise RuntimeError("Reset the simulator before requesting an observation")
        if self._native:
            return cast(_SeparatedBackend, self.backend).observe()
        assert self._capture is not None
        return self.backend.copy_observation(self._capture)

    def step(self, command: Control) -> None:
        """Advance the underlying world exactly once and retain the resulting capture."""
        if not self._ready:
            raise RuntimeError("Reset the simulator before stepping")
        self.backend.validate_command(command)
        # A failed advance must not leave a stale observation available as a successful result.
        self._ready = False
        if self._native:
            cast(_SeparatedBackend, self.backend).advance(command)
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
