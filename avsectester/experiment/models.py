"""Prepared experiment data and per-treatment results, without live run resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from avsectester.attack import AttackSegment
from avsectester.runlog import RunLog

if TYPE_CHECKING:
    from .config import ExperimentConfiguration


@dataclass(frozen=True)
class SelectionDecision:
    """One metabuilder outcome for an attack and scenario, including any rejection or error."""

    attack_name: str
    scenario_id: str
    status: Literal["selected", "not_selected", "error"]
    reasons: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class PreparedExperiment:
    """Configured factories, selected metadata and artifacts, never a live simulator or session.

    The configuration describes how to create resources. Preparation services are closed before
    this object is returned. Frame caches belong to preparation, not the dataset.
    Saving does not pickle the configuration
    or its factories: loading takes an explicit configuration again.
    """

    configuration: ExperimentConfiguration
    metadata: Mapping[str, tuple[AttackSegment, ...]]
    artifacts: Mapping[str, Any]
    selections: tuple[SelectionDecision, ...] = ()

    def save(self, directory: str | Path) -> Path:
        from .persistence import save_prepared

        return save_prepared(self, directory)


    @classmethod
    def load(
        cls, config: ExperimentConfiguration | Mapping[str, Any] | str | Path,
        directory: str | Path,
    ) -> PreparedExperiment:
        """Load compatible metadata and artifacts without selecting or optimizing again."""
        from .config import load_configuration
        from .engine import validate_configuration_environment
        from .persistence import load_prepared

        configuration = load_configuration(config)
        validate_configuration_environment(configuration)
        return load_prepared(configuration, directory)


@dataclass(frozen=True)
class TreatmentResult:
    """The recorded outcome of one segment, seed and named treatment."""

    attack_name: str
    segment_index: int
    segment: AttackSegment
    treatment: str
    seed: int
    log: RunLog
    error: str | None = None

    @property
    def group(self) -> tuple[str, int, int]:
        return self.attack_name, self.segment_index, self.seed


@dataclass(frozen=True)
class ExperimentResult:
    """Completed and failed treatment runs, grouped with the preparation selection outcomes."""

    runs: tuple[TreatmentResult, ...] = ()
    selections: tuple[SelectionDecision, ...] = ()

    @property
    def groups(self) -> dict[tuple[str, int, int], dict[str, RunLog]]:
        result: dict[tuple[str, int, int], dict[str, RunLog]] = {}
        for run in self.runs:
            result.setdefault(run.group, {})[run.treatment] = run.log
        return result

    @property
    def failures(self) -> tuple[TreatmentResult, ...]:
        return tuple(run for run in self.runs if run.log.status != "completed")

    @property
    def selection_errors(self) -> tuple[SelectionDecision, ...]:
        return tuple(item for item in self.selections if item.status == "error")
