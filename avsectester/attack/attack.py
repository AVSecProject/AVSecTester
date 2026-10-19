"""The user's attack: the metabuilder, the offline phase and the online phase.

The framework calls the three parts in a fixed order:

1. ``build_metadata(scenario)`` selects the scenario's ``AttackSegment`` records. All selected
   segments together are the attack metadata. It is called once per scenario.
2. ``run_offline(context)`` turns the metadata and the assets into artifacts, for example by
   optimizing a patch. It is called once.
3. ``create_runtime(segment, artifacts)`` creates one ``AttackRuntime`` for each run.

The offline and online phases have defaults, so either may be trivial.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from avsectester.attack.runtime import AttackRuntime
from avsectester.datasets.base import FrameReference

if TYPE_CHECKING:
    from avsectester.closed_loop.services import ClosedLoopServices, ScenarioView


@dataclass(frozen=True)
class AttackSegment:
    """One selected attack instance: a scenario, its decision interval and the actors involved.

    Decisions run at positions ``[start, stop)``; ``stop`` is the terminal state. ``roles`` maps
    role names (such as ``attacker`` or ``hosts``) to stable actor IDs; ``placement`` is an
    optional insertion placement. Candidate rollouts use the same record.
    """

    scenario_id: str
    start: FrameReference
    stop: FrameReference
    roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    placement: Any = None

    def __post_init__(self) -> None:
        if self.start.scenario_id != self.scenario_id or self.stop.scenario_id != self.scenario_id:
            raise ValueError("Segment boundaries must belong to the segment's scenario")
        if self.stop.position <= self.start.position:
            raise ValueError("Segment stop must follow start and include the terminal state")
        roles = {}
        for role, identities in self.roles.items():
            if not isinstance(role, str) or not role or isinstance(identities, str):
                raise ValueError("Roles map nonempty names to sequences of stable actor IDs")
            roles[role] = tuple(identities)
            if any(not isinstance(i, str) or not i for i in identities):
                raise ValueError("Role actor IDs must be nonempty strings")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "placement", deepcopy(self.placement))

    @property
    def decision_steps(self) -> int:
        return self.stop.position - self.start.position


@dataclass(frozen=True)
class OfflineContext:
    """What ``run_offline`` receives: the closed-loop services and the attack metadata."""

    services: ClosedLoopServices
    metadata: tuple[AttackSegment, ...]


class Attack(ABC):
    """One attack, with its parameters and optional user-provided ``assets``.

    ``assets`` holds inputs such as the patch to insert. Attacks without assets, such as
    phantom injection or command perturbation, leave it ``None``.
    """

    assets: Any = None

    @abstractmethod
    def build_metadata(self, scenario: ScenarioView) -> Sequence[AttackSegment]:
        """Select this scenario's segments, using its data and the closed-loop services."""

    def run_offline(self, context: OfflineContext) -> Any:
        """Return the artifacts used by the online phase. By default, the assets unchanged."""
        return self.assets

    def create_runtime(self, segment: AttackSegment, artifacts: Any) -> AttackRuntime:
        """Create the runtime for one run. By default, a runtime without attack effects."""
        return AttackRuntime()
