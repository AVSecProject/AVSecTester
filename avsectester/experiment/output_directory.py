"""Define experiment output layout and read/write all files other than RunLog data.

Prepared metadata, artifacts, scenario maps and saved result discovery share this module.
RunLog reading remains delegated to the runlog package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from avsectester.attack import AttackSegment, OfflineReport
from avsectester.data_types import Unavailable
from avsectester.serialization import DEFAULT_REGISTRY
from avsectester.maps import ScenarioMap
from avsectester.runlog import RunLog
from avsectester.closed_loop.simulators.base import DeploymentPlan

from .models import (
    ExperimentResult, PreparedExperiment, SelectionDecision, TreatmentResult,
    SegmentUnit, build_units, preparation_record, validated_preparation,
)

if TYPE_CHECKING:
    from .config import ExperimentConfiguration


_SEGMENT_PREFIX = "segment-"
_SEED_PREFIX = "seed-"


def runs_directory(root: Path) -> Path:
    """Return the directory that holds every treatment run of an experiment output."""
    return Path(root) / "runs"


def run_directory(root: Path, segment_index: int, seed: int, treatment: str) -> Path:
    """Return one shared segment-unit/seed/treatment run directory."""
    return runs_directory(root) / f"{_SEGMENT_PREFIX}{segment_index:05d}" / f"{_SEED_PREFIX}{seed}" / treatment


def saved_run_directories(root: Path) -> tuple[tuple[tuple[int, int, str], Path], ...]:
    """List new-layout manifests as ((segment_index, seed, treatment), directory)."""
    found = []
    pattern = f"{_SEGMENT_PREFIX}*/{_SEED_PREFIX}*/*/manifest.json"
    for manifest in sorted(runs_directory(root).glob(pattern)):
        directory = manifest.parent
        try:
            key = (int(directory.parent.parent.name.removeprefix(_SEGMENT_PREFIX)),
                   int(directory.parent.name.removeprefix(_SEED_PREFIX)), directory.name)
        except ValueError:
            continue
        if run_directory(root, *key) == directory:
            found.append((key, directory))
    return tuple(found)


class ConfigurationMismatchError(ValueError):
    """Saved preparation and supplied execution configuration describe different inputs."""


def _validate_user_data(value: Any) -> None:
    """Keep persisted attack-owned data independent of framework record classes."""
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Saved user_data must contain finite numbers")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("user_data mappings require string keys")
        for item in value.values():
            _validate_user_data(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_user_data(item)
        return
    if isinstance(value, np.ndarray) and value.dtype.kind in "biufU":
        return
    raise TypeError("Saved user_data must contain only JSON-like data or NumPy arrays")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_metadata(prepared: PreparedExperiment, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, segments in prepared.metadata.items():
        destination = root / "attacks" / name
        destination.mkdir(parents=True, exist_ok=True)
        for segment in segments:
            _validate_user_data(segment.user_data)
        _write_json(destination / "metadata.json", {"segments": DEFAULT_REGISTRY.encode(tuple(segments))})


def _preparation_manifest(prepared: PreparedExperiment) -> dict[str, Any]:
    sha256 = None
    if prepared.configuration.metadata.load is not None:
        if (prepared.metadata_source is None
                or not isinstance(prepared.metadata_source.get("sha256"), str)):
            raise ValueError("Loaded metadata preparation requires its recorded source SHA256")
        sha256 = prepared.metadata_source["sha256"]
    return {"attacks": list(prepared.metadata),
            "configuration": prepared.configuration.describe_configuration(metadata_sha256=sha256),
            "preparation": prepared.preparation,
            "selections": [item.as_dict() for item in prepared.selections],
            "units": [unit.as_dict() for unit in prepared.units],
            "deployment_ticks": None if prepared.deployment is None else prepared.deployment.ticks,
            "metadata_source": prepared.metadata_source}


def save_preparation_progress(prepared: PreparedExperiment, directory: str | Path,
                              completed_attacks: Sequence[str] = ()) -> Path:
    """Checkpoint selection/partition data and the already published attack artifacts."""
    root = Path(directory)
    _save_metadata(prepared, root)
    manifest = _preparation_manifest(prepared)
    manifest["completed_attacks"] = list(completed_attacks)
    _write_json(root / "preparation-progress.json", manifest)
    return root


def save_completed_artifact(prepared: PreparedExperiment, name: str, directory: str | Path,
                            completed_attacks: Sequence[str]) -> Path:
    """Publish one completed offline artifact before advancing the preparation checkpoint."""
    root = Path(directory)
    prepared.configuration.attacks[name].save_artifacts(prepared.artifacts[name], root / "attacks" / name)
    prepared.offline_reports.get(name, OfflineReport(None)).save(root / "offline" / name)
    save_preparation_progress(prepared, root, completed_attacks)
    from .html_report import write_offline_html

    write_offline_html(root)
    return root


def save_prepared(prepared: PreparedExperiment, directory: str | Path) -> Path:
    """Publish a complete preparation only after every artifact has been saved."""
    root = Path(directory)
    _save_metadata(prepared, root)
    for name, attack in prepared.configuration.attacks.items():
        attack.save_artifacts(prepared.artifacts[name], root / "attacks" / name)
        prepared.offline_reports.get(name, OfflineReport(None)).save(root / "offline" / name)
    _write_json(root / "prepared.json", _preparation_manifest(prepared))
    from .html_report import write_offline_html

    write_offline_html(root)
    return root


def _scenario_map_path(directory: str | Path, scenario_id: str) -> Path:
    root = Path(directory) / "maps"
    path = root / f"{scenario_id}.json"
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Scenario map path escapes the output directory: {scenario_id!r}")
    return path


def save_scenario_maps(maps: Mapping[str, ScenarioMap | Unavailable],
                       directory: str | Path) -> None:
    """Persist each selected scenario's map or unavailable reason using metadata encoding."""
    for scenario_id, scenario_map in maps.items():
        if not isinstance(scenario_map, (ScenarioMap, Unavailable)):
            raise TypeError("Dataset.scenario_map must return ScenarioMap or Unavailable")
        destination = _scenario_map_path(directory, scenario_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(DEFAULT_REGISTRY.encode(scenario_map), indent=2, allow_nan=False) + "\n")


def load_scenario_maps(directory: str | Path,
                       scenario_ids: tuple[str, ...]) -> dict[str, ScenarioMap | Unavailable]:
    """Restore selected scenario maps without accessing dataset data."""
    result: dict[str, ScenarioMap | Unavailable] = {}
    for scenario_id in scenario_ids:
        path = _scenario_map_path(directory, scenario_id)
        if not path.is_file():
            result[scenario_id] = Unavailable("No saved scenario map is available")
            continue
        scenario_map = DEFAULT_REGISTRY.decode(json.loads(path.read_text()))
        if not isinstance(scenario_map, (ScenarioMap, Unavailable)):
            raise ValueError(f"Saved scenario map must contain ScenarioMap or Unavailable: {path}")
        result[scenario_id] = scenario_map
    return result


def _require_parameters(summary: Any) -> None:
    """Type-only records cannot establish parameter compatibility for loading."""
    if not isinstance(summary, Mapping) or not isinstance(summary.get("attacks"), Mapping):
        raise ValueError("Saved preparation has no valid configuration summary")
    unknown = [name for name, description in summary["attacks"].items()
               if not isinstance(description, Mapping)
               or description.get("parameters") is None
               or description.get("parameters_verified", True) is not True]
    if unknown:
        raise ValueError("Cannot validate preparation parameters for attacks: " + ", ".join(unknown)
                         + ". Override configuration_parameters() and save a validated preparation.")


@dataclass(frozen=True)
class PreparedMetadata:
    """Saved metadata for offline analysis; deliberately has no executable artifacts."""

    metadata: Mapping[str, tuple[AttackSegment, ...]]
    selections: tuple[SelectionDecision, ...]
    preparation: Mapping[str, Any]
    units: tuple[SegmentUnit, ...]
    deployment: DeploymentPlan | None
    metadata_source: Mapping[str, str] | None = None


def _canonical_text(value: Any) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True)


def _read_preparation(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    if "preparation" not in manifest:
        raise ValueError("Saved preparation has no preparation record; prepare it again")
    return validated_preparation(manifest["preparation"], manifest["attacks"])


def _check_preparation(configuration: ExperimentConfiguration,
                       saved: Mapping[str, Any]) -> None:
    expected = preparation_record(configuration)
    mismatches = []
    if _canonical_text(saved["seed"]) != _canonical_text(expected["seed"]):
        mismatches.append(f"preparation.seed: old={saved['seed']}, new={expected['seed']}")
    if _canonical_text(saved["split"]) != _canonical_text(expected["split"]):
        mismatches.append(f"preparation.split: old={saved['split']}, new={expected['split']}")
    for name, environment in expected["offline_environments"].items():
        stored = saved["offline_environments"][name]
        if _canonical_text(stored) != _canonical_text(environment):
            mismatches.append(
                f"preparation.offline_environments.{name}: "
                f"old={_canonical_text(stored)}, new={_canonical_text(environment)}"
            )
    if mismatches:
        raise ConfigurationMismatchError(
            "Saved preparation configuration differs from the supplied configuration: "
            + "; ".join(mismatches)
        )


def _load_metadata(configuration: ExperimentConfiguration, directory: str | Path,
                   *, validate_dataset: bool, manifest_name: str = "prepared.json") -> PreparedMetadata:
    root = Path(directory)
    manifest = json.loads((root / manifest_name).read_text())
    if set(manifest["attacks"]) != set(configuration.attacks):
        raise ConfigurationMismatchError("Saved attack names differ from the supplied configuration")
    preparation = _read_preparation(manifest)
    if validate_dataset:
        _check_preparation(configuration, preparation)
    elif _canonical_text(preparation["split"]) != _canonical_text(preparation_record(configuration)["split"]):
        raise ConfigurationMismatchError("Saved preparation.split differs from the supplied configuration")
    saved = manifest.get("configuration")
    _require_parameters(saved)
    has_saved_metadata = "metadata" in saved
    if has_saved_metadata != (configuration.metadata.load is not None):
        raise ConfigurationMismatchError("Saved metadata.load configuration differs from the supplied configuration")
    sha256 = None
    if has_saved_metadata:
        if validate_dataset:
            from .metadata_file import _read_document

            assert configuration.metadata.load is not None
            try:
                sha256 = _read_document(configuration.metadata.load).sha256
            except (OSError, TypeError, ValueError) as error:
                raise ConfigurationMismatchError(
                    f"Current metadata file cannot be validated against the saved preparation: {error}"
                ) from error
        else:
            description = saved["metadata"]
            if not isinstance(description, Mapping) or not isinstance(description.get("sha256"), str):
                raise ConfigurationMismatchError("Saved metadata SHA256 is unavailable")
            sha256 = description["sha256"]
    scenarios = None if validate_dataset else saved["dataset"]["scenarios"]
    expected = configuration.describe_configuration(scenarios=scenarios, metadata_sha256=sha256)
    _require_parameters(expected)
    mismatches = [key for key in expected if saved.get(key) != expected[key]]
    mismatches.extend(key for key in saved if key not in expected)
    if mismatches:
        raise ConfigurationMismatchError("Saved preparation configuration differs from the supplied configuration: "
                         + ", ".join(mismatches))
    metadata = {}
    for name in configuration.attacks:
        record = json.loads((root / "attacks" / name / "metadata.json").read_text())
        segments = DEFAULT_REGISTRY.decode(record["segments"])
        if not isinstance(segments, tuple) or any(not isinstance(s, AttackSegment) for s in segments):
            raise ValueError("Saved metadata must contain AttackSegment records")
        for segment in segments:
            _validate_user_data(segment.user_data)
        if validate_dataset:
            for segment in segments:
                configuration.dataset.checked_frame(segment.scenario_id, segment.start)
                configuration.dataset.checked_frame(segment.scenario_id, segment.stop)
        metadata[name] = segments
    decisions = tuple(SelectionDecision.from_dict(item) for item in manifest["selections"])
    if "units" not in manifest:
        raise ConfigurationMismatchError("Saved preparation uses an unsupported layout without segment units")
    try:
        units = build_units(configuration, metadata, scenario_ids=expected["dataset"]["scenarios"])
    except (ValueError, KeyError) as error:
        raise ConfigurationMismatchError(f"Saved preparation units cannot be reconstructed: {error}") from error
    if manifest["units"] != [unit.as_dict() for unit in units]:
        raise ConfigurationMismatchError("Saved preparation units or partition differ from the supplied configuration")
    ticks = manifest["deployment_ticks"]
    deployment = None if ticks is None else DeploymentPlan(ticks)
    return PreparedMetadata(metadata, decisions, preparation, units, deployment, manifest.get("metadata_source"))


def load_prepared_metadata(configuration: ExperimentConfiguration,
                           directory: str | Path) -> PreparedMetadata:
    """Read only preparation data for evaluation, without accessing dataset data or artifacts."""
    return _load_metadata(configuration, directory, validate_dataset=False)


def _load_offline_reports(names: Sequence[str], root: Path) -> dict[str, OfflineReport]:
    """Restore every attack report from its required offline output."""
    return {name: OfflineReport.load(root / "offline" / name) for name in names}


def load_prepared(configuration: ExperimentConfiguration,
                  directory: str | Path) -> PreparedExperiment:
    """Validate execution inputs and load all artifacts; never return a partially usable object."""
    saved = _load_metadata(configuration, directory, validate_dataset=True)
    assert saved.preparation is not None  # execution loading requires the record
    root = Path(directory)
    artifacts = {name: attack.load_artifacts(root / "attacks" / name)
                 for name, attack in configuration.attacks.items()}
    return PreparedExperiment(configuration, saved.metadata, artifacts, saved.selections,
                              preparation=saved.preparation, units=saved.units, deployment=saved.deployment,
                              offline_reports=_load_offline_reports(tuple(configuration.attacks), root),
                              metadata_source=saved.metadata_source)


def load_preparation_progress(configuration: ExperimentConfiguration,
                              directory: str | Path) -> tuple[PreparedExperiment, tuple[str, ...]]:
    """Validate an incomplete preparation, reusing selection and only completed artifacts."""
    root = Path(directory)
    manifest = json.loads((root / "preparation-progress.json").read_text())
    saved = _load_metadata(configuration, root, validate_dataset=True,
                           manifest_name="preparation-progress.json")
    completed = tuple(manifest["completed_attacks"])
    if len(set(completed)) != len(completed) or any(name not in configuration.attacks for name in completed):
        raise ValueError("Preparation progress has invalid completed attack names")
    artifacts = {name: configuration.attacks[name].load_artifacts(root / "attacks" / name)
                 for name in completed}
    return (PreparedExperiment(configuration, saved.metadata, artifacts, saved.selections,
                               preparation=saved.preparation, units=saved.units, deployment=saved.deployment,
                               offline_reports=_load_offline_reports(completed, root),
                              metadata_source=saved.metadata_source),
            completed)


def load_result(directory: str | Path, configuration: ExperimentConfiguration) -> ExperimentResult:
    """Read results without loading artifacts, driving models or simulator runtimes.

    The preparation and configuration define the expected comparison groups. Missing
    run manifests become in-memory incomplete outcomes, so an interrupted/strict execution
    can still be evaluated without being mistaken for a fully completed experiment.
    Existing manifests must agree with both their directory and the prepared segment.
    """
    from .config import load_configuration

    root = Path(directory)
    config = load_configuration(configuration)
    prepared = load_prepared_metadata(config, root)
    expected: dict[tuple[int, int, str], tuple[SegmentUnit, str | None, Path]] = {}
    for unit in prepared.units:
        if config.split is not None and unit.part not in config.split.execute:
            continue
        for seed in config.runs.seeds:
            for treatment_config in config.runs.treatments:
                name = treatment_config.attack_name
                if name is not None and name not in unit.attack_names:
                    continue
                key = unit.index, seed, treatment_config.name
                expected[key] = unit, name, run_directory(root, *key)
    destination = runs_directory(root)
    known_treatments = {item.name for item in config.runs.treatments}
    notices = tuple(
        f"Skipping saved run for removed treatment: {path}" if key[2] not in known_treatments
        else f"Skipping saved run that the saved preparation does not expect: {path}"
        for key, path in saved_run_directories(root) if key not in expected
    )
    runs: list[TreatmentResult] = []
    for (index, seed, treatment), (unit, name, path) in expected.items():
        segment = unit.segment
        description = {"attack_name": name, "segment_index": index, "seed": seed,
                       "treatment": treatment, "part": unit.part}
        if not (path / "manifest.json").is_file():
            message = "Expected run has no saved manifest; execution did not finish this treatment"
            log = RunLog(metadata=description, status="incomplete", path=path)
            runs.append(TreatmentResult(name, index, segment, treatment, seed, log, message, unit.part))
            continue
        try:
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"Run manifest escapes the run output directory: {path}")
            log = RunLog.read(path)
            data = log.metadata
            if any(type(data.get(key)) is not type(value) or data.get(key) != value
                   for key, value in description.items()):
                raise ValueError(f"Run identity metadata and output directory disagree: {path}")
            if not recorded_data_equal(data.get("segment"), asdict(segment)):
                raise ValueError(f"Run segment metadata differs from the saved preparation: {path}")
            errors = [event.data for event in log.events if event.kind == "failure"]
            error = None
            if errors:
                error = f"{errors[-1].get('error_type', 'Error')}: {errors[-1].get('message', '')}"
            elif log.status != "completed":
                error = f"Run ended with status {log.status}"
        except Exception as read_error:
            error = f"Run read failed: {type(read_error).__name__}: {read_error}"
            log = RunLog(metadata=description, status="failed", path=path)
        runs.append(TreatmentResult(name, index, segment, treatment, seed, log, error, unit.part))
    scenario_ids = tuple(dict.fromkeys(
        segment.scenario_id for segments in prepared.metadata.values() for segment in segments
    ))
    scenario_maps = load_scenario_maps(root, scenario_ids)
    return ExperimentResult(tuple(runs), prepared.selections, notices, scenario_maps)


def recorded_data_equal(first: Any, second: Any) -> bool:
    """Compare recorded plain data, including independent arrays and list/tuple encoding."""
    import numpy as np

    if isinstance(first, np.generic):
        first = first.item()
    if isinstance(second, np.generic):
        second = second.item()
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        return first.keys() == second.keys() and all(
            recorded_data_equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(recorded_data_equal(a, b) for a, b in zip(first, second))
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return bool(np.array_equal(first, second))
    return type(first) is type(second) and first == second
