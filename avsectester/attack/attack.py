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
import hashlib
import json
from inspect import Parameter, signature
import math
from numbers import Integral, Real
from pathlib import Path
import pickle
from types import MappingProxyType
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
    optional insertion placement. Candidate rollouts use the same record. Optional
    ``selection_score`` is larger for better candidates and ``selection_rank`` is zero-based.
    ``user_data`` stores attack-specific information without framework interpretation.
    """

    scenario_id: str
    start: FrameReference
    stop: FrameReference
    roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    placement: Any = None
    selection_score: float | None = None
    selection_rank: int | None = None
    user_data: Mapping[str, Any] = field(default_factory=dict)

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
            if any(not isinstance(i, str) or not i for i in roles[role]):
                raise ValueError("Role actor IDs must be nonempty strings")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "placement", deepcopy(self.placement))
        if self.selection_score is not None:
            if (isinstance(self.selection_score, bool) or not isinstance(self.selection_score, Real)
                    or not math.isfinite(self.selection_score)):
                raise ValueError("selection_score must be a finite number or None")
            object.__setattr__(self, "selection_score", float(self.selection_score))
        if self.selection_rank is not None:
            if (isinstance(self.selection_rank, bool) or not isinstance(self.selection_rank, Integral)
                    or self.selection_rank < 0):
                raise ValueError("selection_rank must be a nonnegative integer or None")
            object.__setattr__(self, "selection_rank", int(self.selection_rank))
        if not isinstance(self.user_data, Mapping):
            raise TypeError("user_data must be a mapping")
        object.__setattr__(self, "user_data", deepcopy(dict(self.user_data)))

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
    environment: Mapping[str, Any] = MappingProxyType({})

    @abstractmethod
    def build_metadata(self, scenario: ScenarioView) -> Sequence[AttackSegment]:
        """Select this scenario's segments, using its data and the closed-loop services."""

    def run_offline(self, context: OfflineContext) -> Any:
        """Return the artifacts used by the online phase. By default, the assets unchanged."""
        return self.assets

    def create_runtime(self, segment: AttackSegment, artifacts: Any) -> AttackRuntime:
        """Create the runtime for one run. By default, a runtime without attack effects."""
        return AttackRuntime()

    def configuration_parameters(self) -> Mapping[str, Any]:
        """Describe the constructor parameters used to validate saved preparations.

        The default reads attributes named by the constructor's explicit parameters, plus
        any non-None assets. Override when parameters are renamed, computed or hidden, returning
        data that identifies the configured inputs, never counters or live run state. The
        experiment requests this mapping only when describing saved or loaded preparations.
        An undescribable mapping never prevents configuration construction or an in-memory run.
        """
        parameters = {}
        for name, parameter in signature(type(self)).parameters.items():
            if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
                raise TypeError(f"{type(self).__name__}.configuration_parameters() must describe "
                                "variadic constructor parameters explicitly")
            if not hasattr(self, name):
                raise TypeError(f"{type(self).__name__}.configuration_parameters() must describe "
                                f"constructor parameter {name!r}: no matching attribute exists")
            parameters[name] = getattr(self, name)
        if self.assets is not None and "assets" not in parameters:
            parameters["assets"] = self.assets
        return parameters

    def save_artifacts(self, artifacts: Any, directory: str | Path) -> None:
        """Save the explicit ``artifacts`` value as pickle plus its content hash.

        Override together with ``load_artifacts`` for another storage format. The default
        format is intended for artifacts produced by trusted attack implementations.
        """
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(artifacts, protocol=pickle.HIGHEST_PROTOCOL)
        (destination / "artifacts.pkl").write_bytes(payload)
        (destination / "manifest.json").write_text(json.dumps({
            "format": "pickle", "sha256": hashlib.sha256(payload).hexdigest(),
        }) + "\n", encoding="utf-8")

    def load_artifacts(self, directory: str | Path) -> Any:
        """Return saved artifacts without changing this attack's configuration or state."""
        source = Path(directory)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "pickle":
            raise ValueError("Unsupported artifact format")
        payload = (source / "artifacts.pkl").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest.get("sha256"):
            raise ValueError("Artifact content hash does not match the saved manifest")
        return pickle.loads(payload)
