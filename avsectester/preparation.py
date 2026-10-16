"""Prepare attack and defense artifacts separately from their per-run execution hooks."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from avsectester.runtime import Hook, HookSet, Runtime

if TYPE_CHECKING:
    from avsectester.scenarios.source import ScenarioInstance


class Intervention(ABC):
    """Preparation shared by attacks and defenses, including methods with no offline work.

    ``prepare`` receives the complete prepared scenario, whose ``match`` may be None.
    It may use independent model or rendering resources supplied by the caller. The backend
    for the formal driving run need not be live. ``hooks`` must create fresh per-run handlers
    or handlers that correctly reset their own state through the Plugin lifecycle.
    Reusable artifacts hold payloads and parameters. Keep per-run history in the handlers.
    """

    def prepare(self, scenario: ScenarioInstance) -> Any:
        """Create reusable artifacts. Methods without preparation may return None."""
        return None

    @abstractmethod
    def hooks(self, artifacts: Any) -> Sequence[Hook]:
        """Construct the ordered execution hooks for one run."""

    def export_artifacts(self, artifacts: Any) -> Any:
        """Return JSON data. Override for arrays, tensors or custom artifact objects.

        File paths may be stored as strings. Saving artifacts does not copy the files to
        which they refer. Methods that require another storage format own that conversion.
        """
        return artifacts

    def import_artifacts(self, data: Any) -> Any:
        """Restore exported data. Override alongside export_artifacts when needed."""
        return data


@dataclass(frozen=True)
class PreparedIntervention:
    """Reusable artifacts and their execution factory, with an explicit experiment role."""

    method: Intervention
    artifacts: Any
    kind: str = "attack"
    name: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    position: str = "after"

    def __post_init__(self) -> None:
        if self.kind not in {"attack", "defense"}:
            raise ValueError("Intervention kind must be 'attack' or 'defense'")
        if not callable(getattr(self.method, "hooks", None)):
            raise TypeError("Intervention methods must provide hooks(artifacts)")
        if not isinstance(self.name, str):
            raise TypeError("Intervention name must be a string")
        if not self.name:
            object.__setattr__(self, "name", type(self.method).__name__)
        if self.position not in {"before", "after"}:
            raise ValueError("Intervention position must be 'before' or 'after'")

    def hooks(self) -> tuple[Hook, ...]:
        """Construct handlers for this run without preparing the artifacts again."""
        hooks = tuple(self.method.hooks(self.artifacts))
        if any(not isinstance(hook, Hook) for hook in hooks):
            raise TypeError("Intervention hooks must be Hook instances")
        return hooks

    def save(self, path: str | Path) -> None:
        """Save metadata and explicitly exported artifacts as JSON, never executable objects."""
        export = getattr(self.method, "export_artifacts", lambda value: value)
        record = {
            "name": self.name,
            "kind": self.kind,
            "provenance": self.provenance,
            "position": self.position,
            "artifacts": export(self.artifacts),
        }
        # Validate the data before creating or replacing the output file.
        content = json.dumps(record, indent=2, allow_nan=False) + "\n"
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)

    @classmethod
    def load(cls, method: Intervention, path: str | Path) -> PreparedIntervention:
        """Load artifacts with the explicitly supplied method and its import function."""
        record = json.loads(Path(path).read_text())
        if not isinstance(record, dict) or set(record) != {
            "name", "kind", "provenance", "position", "artifacts",
        }:
            raise ValueError("Expected an intervention artifact record")
        if not isinstance(record["provenance"], dict):
            raise ValueError("Artifact provenance must be a mapping")
        restore = getattr(method, "import_artifacts", lambda value: value)
        return cls(method, restore(record["artifacts"]), record["kind"], record["name"],
                   record["provenance"], record["position"])


_UNSET = object()


def prepare_intervention(
    method: Intervention,
    scenario: ScenarioInstance,
    *,
    kind: str = "attack",
    name: str | None = None,
    artifacts: Any = _UNSET,
    position: str = "after",
) -> PreparedIntervention:
    """Prepare once or reuse supplied artifacts, including explicitly supplied None.

    Reusing the returned definition across scenarios or conditions is an explicit caller
    choice. The framework does not cache artifacts or infer where they remain suitable.
    """
    if kind not in {"attack", "defense"}:
        raise ValueError("Intervention kind must be 'attack' or 'defense'")
    if position not in {"before", "after"}:
        raise ValueError("Intervention position must be 'before' or 'after'")
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError("Intervention name must be a nonempty string")
    if not callable(getattr(method, "hooks", None)):
        raise TypeError("Intervention methods must provide hooks(artifacts)")
    if artifacts is _UNSET:
        prepare = getattr(method, "prepare", None)
        if not callable(prepare):
            raise TypeError("Intervention methods must provide prepare(scenario)")
        artifacts = prepare(scenario)
    return PreparedIntervention(
        method, artifacts, kind, name or type(method).__name__,
        {"scenario": deepcopy(scenario.provenance)}, position,
    )


@dataclass(frozen=True)
class Treatment:
    """Ordered prepared methods enabled for one experiment treatment."""

    interventions: tuple[PreparedIntervention, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "interventions", tuple(self.interventions))
        if any(not isinstance(item, PreparedIntervention) for item in self.interventions):
            raise TypeError("Treatment requires PreparedIntervention entries")

    def subset(self, kinds: Iterable[str]) -> Treatment:
        """Select attacks or defenses without preparing them again."""
        selected = frozenset(kinds)
        if selected - {"attack", "defense"}:
            raise ValueError("Treatment kinds must be 'attack' or 'defense'")
        return Treatment(tuple(item for item in self.interventions if item.kind in selected))

    def runtime(
        self, base: HookSet | None = None, case: Any = None, *,
        kinds: Iterable[str] | None = None,
    ) -> HookSet:
        """Create execution hooks afresh while retaining the base localizer and RNG seed."""
        treatment = self if kinds is None else self.subset(kinds)
        early, late = [], []
        for item in treatment.interventions:
            (early if item.position == "before" else late).extend(item.hooks())
        runtime = base or Runtime()
        result = runtime.with_hooks(early, prepend=True).with_hooks(late)
        if case is not None:
            result.case = case
        return result
