"""Abstract interfaces for the security-layer plugin types.

Concrete implementations live in the sibling packages (attacks/, defenses/, monitors/,
metrics/, backends/, search/) and register themselves with the corresponding registry.

Attacks/Defenses/Monitors are designed to be usable as **avstack pre/post hooks**: a hook
is any callable ``hook(*module_io) -> module_io``. Implementing ``__call__`` with that
contract lets an instance attach to an avstack module via its ``pre_hooks``/``post_hooks``
without any change to avstack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .experiment import ExperimentSpec
    from .threat_model import ThreatModel


class Backend(ABC):
    """Execution environment (CARLA closed-loop, offline dataset, replay).

    Wraps an avstack backend (avcarla / avapi) behind a uniform surface so one experiment
    spec runs across backends.
    """

    @abstractmethod
    def build(self, spec: ExperimentSpec) -> None:
        """Instantiate the AV system + scenario from the spec."""

    @abstractmethod
    def step(self) -> dict[str, Any]:
        """Advance one tick (closed-loop) or yield one frame (offline); return frame I/O."""

    @abstractmethod
    def run(self) -> Iterator[dict[str, Any]]:
        """Drive the scenario to completion, yielding per-frame records."""

    @abstractmethod
    def close(self) -> None:
        """Tear down actors/connections."""

    def add_perception_hook(self, hook: Any) -> None:
        """Attach an attack/defense/monitor at the perception-input seam.

        Hooks are applied in registration order each tick as ``hook(data, ego_state=...)
        -> data`` (avstack pre-hook shape). Backends that support live interception
        override this; the default refuses so misuse is loud.
        """
        raise NotImplementedError(f"{type(self).__name__} has no perception-hook seam")


class AttackBase(ABC):
    """An attack as a first-class entity (PROJECT.md 4.1).

    Subclasses declare the threat model they assume and implement ``apply`` as a hook that
    manipulates module I/O at the declared insertion point.
    """

    threat_model: ThreatModel

    @abstractmethod
    def validate(self, spec: ExperimentSpec) -> None:
        """Raise if the experiment would violate the declared threat model/constraints."""

    @abstractmethod
    def apply(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Perturb module I/O (hook contract). Return the (possibly) modified data."""

    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self.apply(data, *args, **kwargs)


class DefenseBase(ABC):
    """A defense/mitigation, also hook-shaped (PROJECT.md 6.5)."""

    @abstractmethod
    def apply(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Sanitize/monitor/mitigate. Return (possibly) corrected data; may flag detections."""

    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self.apply(data, *args, **kwargs)


class MonitorBase(ABC):
    """Runtime instrumentation hook: observe (never modify) module I/O (PROJECT.md 6.6)."""

    @abstractmethod
    def observe(self, stage: str, component: str, data: Any) -> None:
        """Record I/O for later trace diffing / escalation-DAG construction."""


class MetricBase(ABC):
    """A security metric over one or more (clean, attacked) traces (PROJECT.md 4.2)."""

    name: str

    @abstractmethod
    def compute(self, clean: Any, attacked: Any, **kwargs: Any) -> dict[str, Any]:
        """Return a dict of named metric values."""


class SearchStrategy(ABC):
    """Closed-loop vulnerability search (proposal Thrust 3)."""

    @abstractmethod
    def propose(self) -> list[dict[str, Any]]:
        """Propose the next batch of (scenario x attack) parameter points to evaluate."""

    @abstractmethod
    def observe(self, results: list[dict[str, Any]]) -> None:
        """Feed back evaluation results (fitness = activation/escalation signal)."""
