"""Prepared experiment data and per-treatment results, without live run resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from avsectester.attack import AttackSegment
from avsectester.datasets.base import Unavailable
from avsectester.maps import ScenarioMap
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

    def as_dict(self) -> dict[str, Any]:
        """Return the shared plain-data representation used by preparation and reports."""
        return {"attack_name": self.attack_name, "scenario_id": self.scenario_id,
                "status": self.status, "reasons": list(self.reasons), "error": self.error}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SelectionDecision:
        """Restore a saved selection outcome."""
        return cls(value["attack_name"], value["scenario_id"], value["status"],
                   tuple(value["reasons"]), value.get("error"))


_SEGMENT_PREFIX = "segment-"
_SEED_PREFIX = "seed-"


def runs_directory(root: Path) -> Path:
    """Return the directory that holds every treatment run of an experiment output."""
    return Path(root) / "runs"


def run_directory(root: Path, attack_name: str, segment_index: int,
                  seed: int, treatment: str) -> Path:
    """Return the sole on-disk layout for an experiment's treatment runs."""
    return (runs_directory(root) / attack_name / f"{_SEGMENT_PREFIX}{segment_index:05d}"
            / f"{_SEED_PREFIX}{seed}" / treatment)


def saved_run_directories(root: Path) -> tuple[tuple[tuple[str, int, int, str], Path], ...]:
    """List saved runs (directories with a manifest) laid out by :func:`run_directory`.

    Each entry is ``((attack_name, segment_index, seed, treatment), directory)``. Directories
    that do not follow the layout are ignored.
    """
    found = []
    pattern = f"*/{_SEGMENT_PREFIX}*/{_SEED_PREFIX}*/*/manifest.json"
    for manifest in sorted(runs_directory(root).glob(pattern)):
        directory = manifest.parent
        seed_directory = directory.parent
        segment_directory = seed_directory.parent
        try:
            key = (segment_directory.parent.name,
                   int(segment_directory.name.removeprefix(_SEGMENT_PREFIX)),
                   int(seed_directory.name.removeprefix(_SEED_PREFIX)), directory.name)
        except ValueError:
            continue
        if run_directory(root, *key) == directory:
            found.append((key, directory))
    return tuple(found)


def validated_preparation(preparation: Any, attack_names: Any) -> Mapping[str, Any]:
    """Check a preparation record: the first seed and each attack's offline environment."""
    import json

    if not isinstance(preparation, Mapping) or set(preparation) != {"seed", "offline_environments"}:
        raise ValueError("A preparation record must contain seed and offline_environments")
    seed = preparation["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("A preparation seed must be a nonnegative integer")
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

    def __post_init__(self) -> None:
        validated_preparation(self.preparation, self.metadata)

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
    """Treatment runs, selection outcomes and saved maps for scenarios with selected segments.

    ``scenario_maps`` retains both semantic maps and explicit unavailable reasons. Loading and evaluation
    use this snapshot rather than accessing dataset map services.
    """

    runs: tuple[TreatmentResult, ...] = ()
    selections: tuple[SelectionDecision, ...] = ()
    notices: tuple[str, ...] = ()
    scenario_maps: Mapping[str, ScenarioMap | Unavailable] = field(default_factory=dict)

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

    @classmethod
    def load(cls, output: str | Path, configuration: ExperimentConfiguration) -> ExperimentResult:
        """Read results without loading artifacts, driving models or simulator runtimes.

        The preparation and configuration define the expected comparison groups. Missing
        run manifests become in-memory incomplete outcomes, so an interrupted/strict execution
        can still be evaluated without being mistaken for a fully completed experiment.
        Existing manifests must agree with both their directory and the prepared segment.
        """
        from .config import load_configuration
        from .persistence import load_prepared_metadata, load_scenario_maps

        root = Path(output)
        config = load_configuration(configuration)
        prepared = load_prepared_metadata(config, root)
        expected: dict[tuple[str, int, int, str], tuple[AttackSegment, Path]] = {}
        for attack_name, segments in prepared.metadata.items():
            for index, segment in enumerate(segments):
                for seed in config.runs.seeds:
                    for treatment_config in config.runs.treatments:
                        if not treatment_applies(treatment_config, attack_name):
                            continue
                        key = attack_name, index, seed, treatment_config.name
                        expected[key] = segment, run_directory(root, *key)
        destination = runs_directory(root)
        known_treatments = {item.name for item in config.runs.treatments}
        notices = tuple(
            f"Skipping saved run for removed treatment: {path}" if key[3] not in known_treatments
            else f"Skipping saved run that the saved preparation does not expect: {path}"
            for key, path in saved_run_directories(root) if key not in expected
        )
        found: dict[tuple[str, int, int, str], TreatmentResult] = {}
        for key, (segment, path) in expected.items():
            if not (path / "manifest.json").is_file():
                continue
            name, index, seed, treatment = key
            try:
                if not path.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f"Run manifest escapes the run output directory: {path}")
                log = RunLog.read(path)
                data = log.metadata
                identity = (data.get("attack_name"), data.get("segment_index"),
                            data.get("seed"), data.get("treatment"))
                if any(type(actual) is not type(wanted) or actual != wanted
                       for actual, wanted in zip(identity, key)):
                    raise ValueError(f"Run identity metadata and output directory disagree: {path}")
                if not _same_data(data.get("segment"), asdict(segment)):
                    raise ValueError(f"Run segment metadata differs from the saved preparation: {path}")
                errors = [event.data for event in log.events if event.kind == "failure"]
                error = None
                if errors:
                    error = f"{errors[-1].get('error_type', 'Error')}: {errors[-1].get('message', '')}"
                elif log.status != "completed":
                    error = f"Run ended with status {log.status}"
            except Exception as read_error:
                error = f"Run read failed: {type(read_error).__name__}: {read_error}"
                log = RunLog(metadata={
                    "attack_name": name, "segment_index": index, "seed": seed,
                    "treatment": treatment,
                }, status="failed", path=path)
            found[key] = TreatmentResult(name, index, segment, treatment, seed, log, error)
        runs: list[TreatmentResult] = []
        for (name, index, seed, treatment), (segment, path) in expected.items():
            key = name, index, seed, treatment
            if key in found:
                runs.append(found[key])
                continue
            message = "Expected run has no saved manifest; execution did not finish this treatment"
            missing = RunLog(metadata={
                "attack_name": name, "segment_index": index, "seed": seed,
                "treatment": treatment,
            }, status="incomplete", path=path)
            runs.append(TreatmentResult(name, index, segment, treatment, seed, missing, message))
        scenario_ids = tuple(dict.fromkeys(
            segment.scenario_id for segments in prepared.metadata.values() for segment in segments
        ))
        scenario_maps = load_scenario_maps(root, scenario_ids)
        return cls(tuple(runs), prepared.selections, notices, scenario_maps)


def _same_data(first: Any, second: Any) -> bool:
    """Compare recorded plain data, including independent arrays and list/tuple encoding."""
    import numpy as np

    if isinstance(first, np.generic):
        first = first.item()
    if isinstance(second, np.generic):
        second = second.item()
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(_same_data(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(_same_data(a, b) for a, b in zip(first, second))
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return bool(np.array_equal(first, second))
    return type(first) is type(second) and first == second
