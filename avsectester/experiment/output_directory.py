"""Define experiment output layout and read/write all files other than RunLog data.

Prepared metadata, artifacts, scenario maps and saved result discovery share this module.
RunLog reading remains delegated to the runlog package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from avsectester.attack import AttackSegment
from avsectester.data_types import Unavailable
from avsectester.serialization import DEFAULT_REGISTRY
from avsectester.maps import ScenarioMap
from avsectester.runlog import RunLog

from .models import (
    ExperimentResult, PreparedExperiment, SelectionDecision, TreatmentResult,
    preparation_record, treatment_applies, validated_preparation,
)

if TYPE_CHECKING:
    from .config import ExperimentConfiguration


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


def save_prepared(prepared: PreparedExperiment, directory: str | Path) -> Path:
    summary = prepared.configuration.configuration_summary
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, segments in prepared.metadata.items():
        destination = root / "attacks" / name
        destination.mkdir(parents=True, exist_ok=True)
        for segment in segments:
            _validate_user_data(segment.user_data)
        content = {"segments": DEFAULT_REGISTRY.encode(tuple(segments))}
        (destination / "metadata.json").write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
        attack = prepared.configuration.attacks[name]
        attack.save_artifacts(prepared.artifacts[name], destination)
    manifest = {
        "attacks": list(prepared.metadata),
        "configuration": summary,
        "preparation": prepared.preparation,
        "selections": [item.as_dict() for item in prepared.selections],
    }
    (root / "prepared.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False, sort_keys=True) + "\n")
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
    preparation: Mapping[str, Any] | None


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
                   *, validate_dataset: bool) -> PreparedMetadata:
    root = Path(directory)
    manifest = json.loads((root / "prepared.json").read_text())
    if set(manifest["attacks"]) != set(configuration.attacks):
        raise ConfigurationMismatchError("Saved attack names differ from the supplied configuration")
    preparation: Mapping[str, Any] | None
    if validate_dataset:
        preparation = _read_preparation(manifest)
        _check_preparation(configuration, preparation)
    else:
        # Evaluation reads results only; outputs saved before preparation records remain usable.
        preparation = (None if "preparation" not in manifest
                       else validated_preparation(manifest["preparation"], manifest["attacks"]))
    saved = manifest.get("configuration")
    _require_parameters(saved)
    scenarios = None if validate_dataset else saved["dataset"]["scenarios"]
    expected = configuration.describe_configuration(scenarios=scenarios)
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
    return PreparedMetadata(metadata, decisions, preparation)


def load_prepared_metadata(configuration: ExperimentConfiguration,
                           directory: str | Path) -> PreparedMetadata:
    """Read only preparation data for evaluation, without accessing dataset data or artifacts."""
    return _load_metadata(configuration, directory, validate_dataset=False)


def load_prepared(configuration: ExperimentConfiguration,
                  directory: str | Path) -> PreparedExperiment:
    """Validate execution inputs and load all artifacts; never return a partially usable object."""
    saved = _load_metadata(configuration, directory, validate_dataset=True)
    assert saved.preparation is not None  # execution loading requires the record
    root = Path(directory)
    artifacts = {name: attack.load_artifacts(root / "attacks" / name)
                 for name, attack in configuration.attacks.items()}
    return PreparedExperiment(configuration, saved.metadata, artifacts, saved.selections,
                              preparation=saved.preparation)


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
    return ExperimentResult(tuple(runs), prepared.selections, notices, scenario_maps)


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
