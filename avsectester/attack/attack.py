"""The user's attack: the metabuilder, the offline phase and the online phase.

The framework calls the three parts in a fixed order:

1. ``build_metadata(scenario)`` selects the scenario's ``AttackSegment`` records. All selected
   segments together are the attack metadata. It is called once per scenario.
2. ``run_offline(context)`` turns the metadata into artifacts, for example by
   optimizing a patch. It is called once.
3. ``create_runtime(segment, artifacts)`` creates one ``AttackRuntime`` for each run.

The offline and online phases have defaults, so either may be trivial.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from inspect import Parameter, signature
from pathlib import Path
import pickle
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from avsectester.attack.runtime import AttackRuntime
from avsectester.attack.offline_report import OfflineReport
from avsectester.datasets.base import FrameReference

if TYPE_CHECKING:
    from avsectester.execution.services import ClosedLoopServices, ScenarioView
    from avsectester.closed_loop.simulators.base import DeploymentPlan


@dataclass(frozen=True)
class AttackSegment:
    """One selected scenario and decision interval with attack-owned data.

    Decisions run at positions ``[start, stop)``; ``stop`` is the terminal state. Candidate
    rollouts use the same record. ``user_data`` stores all attack-specific information without
    framework interpretation. Persisted values must be JSON-like data or NumPy arrays.
    """

    scenario_id: str
    start: FrameReference
    stop: FrameReference
    user_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start.scenario_id != self.scenario_id or self.stop.scenario_id != self.scenario_id:
            raise ValueError("Segment boundaries must belong to the segment's scenario")
        if self.stop.position <= self.start.position:
            raise ValueError("Segment stop must follow start and include the terminal state")
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
    seed: int = 0
    environment: Mapping[str, Any] = field(default_factory=dict)
    deployment: DeploymentPlan | None = None
    validation_metadata: tuple[AttackSegment, ...] = ()
    report: OfflineReport = field(default_factory=lambda: OfflineReport(None))


class Attack(ABC):
    """One attack with its own parameters and optional offline preparation."""
    environment: Mapping[str, Any] = MappingProxyType({})

    def build_metadata(self, scenario: ScenarioView) -> Sequence[AttackSegment]:
        """Select segments, or leave selection to a shared metadata.load file."""
        raise NotImplementedError(
            f"{type(self).__name__} does not select segments; set metadata.load"
        )

    def validate_segment(self, segment: AttackSegment) -> None:
        """Accept shared metadata, or raise ValueError explaining why it is unsuitable."""

    def run_offline(self, context: OfflineContext) -> Any:
        """Return the artifacts used by the online phase, or None when no preparation is needed."""
        return None

    def deployment_ticks(self, segment: AttackSegment) -> int:
        """Declare native setup ticks before preparing artifacts; default has no world deployment."""
        return 0

    def create_runtime(self, segment: AttackSegment, artifacts: Any) -> AttackRuntime:
        """Create the runtime for one run. By default, a runtime without attack actions."""
        return AttackRuntime()

    def configuration_parameters(self) -> Mapping[str, Any]:
        """Describe the constructor parameters used to validate saved preparations.

        The default reads attributes named by the constructor's explicit parameters.
        Override when parameters are renamed, computed or hidden, returning
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
