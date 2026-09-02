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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .capability import StackProfile
from .plugin import SecurityPlugin

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

    def profile(self) -> StackProfile:
        """Advertise which seams and capabilities this stack exposes.

        A plugin's declared seams are checked against this profile
        (:meth:`SecurityPlugin.check`). Concrete backends override it to reflect their
        configuration (e.g. ground-truth vs neural perception). The empty default supports
        nothing, so an unconfigured backend fails the compatibility check loudly.
        """
        return StackProfile()

    def attach(self, plugin: Any, seam: str = "perception_out") -> None:
        """Attach an attack/defense at a named ``seam`` of the pipeline.

        ``"perception_input"`` operates on the detector input (object/point level);
        ``"perception_out"`` operates on the detector output (detection level, via an
        avstack post-hook). Backends that support live interception override this.
        """
        raise NotImplementedError(f"{type(self).__name__} has no attach seam support")

    def add_perception_hook(self, hook: Any) -> None:
        """Legacy alias for the perception-input seam (``hook(data, ego_state=...) -> data``)."""
        self.attach(hook, seam="perception_input")


class AttackBase(SecurityPlugin):
    """An attack as a first-class entity (PROJECT.md 4.1).

    Subclasses declare the threat model they assume plus the :attr:`seams` they hook into
    (inherited from :class:`SecurityPlugin`) and implement ``apply`` as a hook that
    manipulates module I/O at the firing seam (dispatch on :meth:`current_seam`).
    """

    threat_model: ThreatModel

    def _kind(self) -> str:
        return "attack"

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        tm = getattr(self, "threat_model", None)
        if tm is not None:
            d["threat_model"] = {
                "goal": tm.goal,
                "knowledge": tm.knowledge.value,
                "access": [a.value for a in tm.access],
                "success_criteria": tm.success_criteria,
            }
        return d


@dataclass
class DefenseOutcome:
    """What a defense did on one tick — telemetry for scoring mitigation.

    Recorded into the run context (not returned) so a defense stays a plain avstack hook.
    ``dropped``/``flagged`` hold the IDs the defense removed / marked suspicious; ``kept``
    and ``reason`` describe the decision for the report.
    """

    seam: str
    frame: int = 0
    kept: int = 0
    dropped: list[Any] = field(default_factory=list)
    flagged: list[Any] = field(default_factory=list)
    reason: str = ""


class DefenseBase(SecurityPlugin):
    """A defense/mitigation, also hook-shaped (PROJECT.md 6.5).

    A defense records a :class:`DefenseOutcome` into the run context each tick via
    :meth:`record_outcome` so the report can score mitigation, while ``apply`` still returns
    the (sanitized) payload — keeping it a drop-in avstack hook.
    """

    category: str = "defense"

    def _kind(self) -> str:
        return "defense"

    def record_outcome(self, ctx: Any, outcome: DefenseOutcome) -> None:
        """Append ``outcome`` to ``ctx.defense_outcomes`` (duck-typed; no-op if no ctx)."""
        if ctx is None:
            return
        bucket = getattr(ctx, "defense_outcomes", None)
        if bucket is None:
            bucket = []
            try:
                ctx.defense_outcomes = bucket
            except AttributeError:  # ctx doesn't accept the attribute; skip telemetry
                return
        bucket.append(outcome)


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
