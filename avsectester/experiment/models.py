"""Preparation inputs, their validation, prepared metadata and per-treatment results.

These models describe experiments without owning live run resources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from avsectester.validation import require_nonnegative_integer
from avsectester.attack import AttackSegment
from avsectester.closed_loop.environment import merge_environments, normalize_environment
from avsectester.data_types import Unavailable
from avsectester.maps import ScenarioMap
from avsectester.runlog import RunLog
from avsectester.closed_loop.simulators.base import DeploymentPlan

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

    def as_dict(self) -> dict[str, Any]:
        """Return the shared plain-data representation used by preparation and reports."""
        return {"attack_name": self.attack_name, "scenario_id": self.scenario_id,
                "status": self.status, "reasons": list(self.reasons), "error": self.error}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SelectionDecision:
        """Restore a saved selection outcome."""
        return cls(value["attack_name"], value["scenario_id"], value["status"],
                   tuple(value["reasons"]), value.get("error"))


def _environment_record(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical data for one environment, independent of omitted optional entry keys."""
    from .configuration_summary import summary_value

    normalized = normalize_environment(environment)
    sensors = []
    for entry in normalized["sensors"]:
        record = {"type": entry["type"], "options": dict(entry["options"])}
        if entry.get("camera") is not None:
            record["camera"] = entry["camera"]
        sensors.append(record)
    return summary_value({"world": normalized["world"], "sensors": sensors})


def preparation_record(configuration: ExperimentConfiguration) -> dict[str, Any]:
    """The preparation inputs that saved metadata and artifacts depend on, as canonical data.

    Environment values must be data (as accepted by configuration summaries); anything else
    fails here, before selection or offline work starts.
    """
    return {
        "seed": configuration.runs.seeds[0],
        "split": None if configuration.split is None else {
            "train_segments": configuration.split.train_segments,
            "validation_segments": configuration.split.validation_segments,
            "seed": configuration.split.seed,
        },
        "offline_environments": {
            name: _environment_record(merge_environments(configuration.runs.environment,
                                                         attack.environment))
            for name, attack in configuration.attacks.items()
        },
    }


def validated_preparation(preparation: Any, attack_names: Any) -> Mapping[str, Any]:
    """Check a preparation record: the first seed and each attack's offline environment."""
    import json

    if not isinstance(preparation, Mapping) or set(preparation) != {"seed", "offline_environments", "split"}:
        raise ValueError("A preparation record must contain seed, offline_environments and split")
    seed = preparation["seed"]
    require_nonnegative_integer(seed, "preparation seed")
    split = preparation["split"]
    if split is not None:
        from .config import SplitConfiguration

        if not isinstance(split, Mapping) or set(split) != {"train_segments", "validation_segments", "seed"}:
            raise ValueError("Preparation split must contain train_segments, validation_segments and seed")
        SplitConfiguration(**split)
    environments = preparation["offline_environments"]
    if not isinstance(environments, Mapping) or set(environments) != set(attack_names):
        raise ValueError("A preparation record needs one offline environment for every attack")
    for name, environment in environments.items():
        if not isinstance(environment, Mapping) or set(environment) != {"world", "sensors"}:
            raise ValueError(f"Offline environment for {name!r} must contain world and sensors")
    try:
        json.dumps(dict(preparation), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"A preparation record must contain only JSON data: {error}") from None
    return preparation


def treatment_applies(treatment: Any, attack_name: str) -> bool:
    """Whether a treatment runs on this attack's selected metadata."""
    return treatment.attack_name is None or treatment.attack_name == attack_name


@dataclass(frozen=True)
class SegmentUnit:
    """One canonical segment shared by all attacks that selected exactly that segment."""

    index: int
    segment: AttackSegment
    attack_names: tuple[str, ...]
    part: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "scenario_id": self.segment.scenario_id,
                "start": self.segment.start.position, "stop": self.segment.stop.position,
                "attack_names": list(self.attack_names), "part": self.part}


def segment_key(segment: AttackSegment) -> str:
    """Canonical metadata identity, including attack-owned user_data."""
    import json
    from avsectester.serialization import DEFAULT_REGISTRY

    return json.dumps(DEFAULT_REGISTRY.encode(segment), sort_keys=True, allow_nan=False)


def build_units(configuration: ExperimentConfiguration,
                metadata: Mapping[str, tuple[AttackSegment, ...]], *,
                scenario_ids: Sequence[str] | None = None) -> tuple[SegmentUnit, ...]:
    """Build deterministic shared units and assign the configured random partition."""
    import numpy as np

    if scenario_ids is None:
        scenario_ids = tuple(scenario.id for scenario in configuration.dataset.scenarios())
    order = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    records: dict[str, tuple[AttackSegment, list[str], int]] = {}
    for attack_index, name in enumerate(configuration.attacks):
        for segment in metadata[name]:
            key = segment_key(segment)
            if key not in records:
                records[key] = (segment, [], attack_index)
            if name not in records[key][1]:
                records[key][1].append(name)
    ordered = sorted(records.values(), key=lambda item: (
        order[item[0].scenario_id], item[0].start.position, item[0].stop.position, item[2],
    ))
    parts: list[str | None] = [None] * len(ordered)
    split = configuration.split
    if split is not None:
        required = split.train_segments + split.validation_segments
        if required >= len(ordered):
            raise ValueError(f"train_segments + validation_segments must be less than the actual "
                             f"segment-unit count ({len(ordered)})")
        permutation = np.random.default_rng(split.seed).permutation(len(ordered))
        for rank, index in enumerate(permutation):
            parts[index] = ("train" if rank < split.train_segments else
                            "validation" if rank < required else "test")
    return tuple(SegmentUnit(index, segment, tuple(names), parts[index])
                 for index, (segment, names, _) in enumerate(ordered))


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
    preparation: Mapping[str, Any] = field(kw_only=True)
    units: tuple[SegmentUnit, ...] = field(default=(), kw_only=True)
    deployment: DeploymentPlan | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        validated_preparation(self.preparation, self.metadata)
        if not self.units and any(self.metadata.values()):
            object.__setattr__(self, "units", build_units(self.configuration, self.metadata))

    def save(self, directory: str | Path) -> Path:
        from .output_directory import save_prepared

        return save_prepared(self, directory)


    @classmethod
    def load(
        cls, config: ExperimentConfiguration | Mapping[str, Any] | str | Path,
        directory: str | Path,
    ) -> PreparedExperiment:
        """Load compatible metadata and artifacts without selecting or optimizing again."""
        from .config import load_configuration
        from .engine import validate_configuration_environment
        from .output_directory import load_prepared

        configuration = load_configuration(config)
        validate_configuration_environment(configuration)
        return load_prepared(configuration, directory)


@dataclass(frozen=True)
class TreatmentResult:
    """The recorded outcome of one segment, seed and named treatment."""

    attack_name: str | None
    segment_index: int
    segment: AttackSegment
    treatment: str
    seed: int
    log: RunLog
    error: str | None = None
    part: str | None = None

    @property
    def group(self) -> tuple[int, int]:
        return self.segment_index, self.seed


@dataclass(frozen=True)
class ExperimentResult:
    """Treatment runs, selection outcomes and saved maps for scenarios with selected segments.

    ``scenario_maps`` retains both semantic maps and explicit unavailable reasons. Loading and evaluation
    use this snapshot rather than accessing dataset map services.
    """

    runs: tuple[TreatmentResult, ...] = ()
    selections: tuple[SelectionDecision, ...] = ()
    notices: tuple[str, ...] = ()
    scenario_maps: Mapping[str, ScenarioMap | Unavailable] = field(default_factory=dict)

    @property
    def groups(self) -> dict[tuple[int, int], dict[str, RunLog]]:
        result: dict[tuple[int, int], dict[str, RunLog]] = {}
        for run in self.runs:
            result.setdefault(run.group, {})[run.treatment] = run.log
        return result

    @property
    def failures(self) -> tuple[TreatmentResult, ...]:
        return tuple(run for run in self.runs if run.log.status != "completed")

    @property
    def selection_errors(self) -> tuple[SelectionDecision, ...]:
        return tuple(item for item in self.selections if item.status == "error")

    @classmethod
    def load(cls, output: str | Path, configuration: ExperimentConfiguration) -> ExperimentResult:
        """Read results without loading artifacts, driving models or simulator runtimes.

        The preparation and configuration define the expected comparison groups. Missing
        run manifests become in-memory incomplete outcomes, so an interrupted/strict execution
        can still be evaluated without being mistaken for a fully completed experiment.
        Existing manifests must agree with both their directory and the prepared segment.
        """
        from .output_directory import load_result

        return load_result(output, configuration)
