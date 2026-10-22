"""JSON attack metadata and explicit artifact storage for prepared experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from avsectester.attack import AttackSegment
from avsectester.closed_loop.environment import merge_environments, normalize_environment
from avsectester.datasets.base import Unavailable
from avsectester.maps import ScenarioMap

from .models import PreparedExperiment, SelectionDecision, validated_preparation

if TYPE_CHECKING:
    from .config import ExperimentConfiguration


class ConfigurationMismatchError(ValueError):
    """Saved preparation and supplied execution configuration describe different inputs."""


_RECORDS = {
    "avsectester.datasets.base": {"FrameReference", "ActorState", "Unavailable"},
    "avsectester.attack.attack": {"AttackSegment"},
    "avsectester.geometry.poses": {"ActorPose"},
    "avsectester.maps.scenario_map": {"ScenarioMap", "Lane", "TrafficControl", "StopLine"},
}


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


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Saved metadata must contain finite numbers")
        return value
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        if isinstance(value, AttackSegment):
            _validate_user_data(value.user_data)
        module, symbol = type(value).__module__, type(value).__qualname__
        name = f"{module}:{symbol}"
        if symbol not in _RECORDS.get(module, set()):
            raise TypeError(f"Unsupported metadata record: {name}; convert it to JSON data")
        return {"__type__": "record", "class": name,
                "fields": {f.name: _encode(getattr(value, f.name)) for f in fields(value) if f.init}}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Metadata mappings require string keys")
        return {"__type__": "mapping", "items": {key: _encode(item) for key, item in value.items()}}
    if isinstance(value, (tuple, list)):
        return {"__type__": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode(item) for item in value]}
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufU":
            raise TypeError("Metadata arrays require real numeric, boolean or Unicode values")
        return {"__type__": "array", "dtype": value.dtype.str,
                "shape": list(value.shape), "value": value.tolist()}
    if isinstance(value, np.generic):
        return _encode(value.item())
    raise TypeError(f"Metadata cannot contain {type(value).__name__}; provide JSON data")


def _decode(value: Any) -> Any:
    if not isinstance(value, dict) or "__type__" not in value:
        return value
    kind = value["__type__"]
    if kind == "path":
        return Path(value["value"])
    if kind == "mapping":
        return {key: _decode(item) for key, item in value["items"].items()}
    if kind in {"tuple", "list"}:
        items = [_decode(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "array":
        import numpy as np

        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biufU":
            raise ValueError("Metadata arrays require real numeric, boolean or Unicode values")
        shape = value["shape"]
        if not isinstance(shape, list) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape
        ):
            raise ValueError("Metadata array shape must contain nonnegative integers")
        return np.asarray(value["value"], dtype=dtype).reshape(shape)
    if kind == "record":
        # Only data records are restored. Never resolve an arbitrary configuration factory.
        module, name = value["class"].split(":", 1)
        if name not in _RECORDS.get(module, set()):
            raise ValueError(f"Unsupported metadata record: {value['class']}")
        record = getattr(importlib.import_module(module), name)
        decoded = record(**{key: _decode(item) for key, item in value["fields"].items()})
        if isinstance(decoded, AttackSegment):
            _validate_user_data(decoded.user_data)
        return decoded
    raise ValueError(f"Unknown metadata encoding: {kind}")


def save_prepared(prepared: PreparedExperiment, directory: str | Path) -> Path:
    summary = prepared.configuration.configuration_summary
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, segments in prepared.metadata.items():
        destination = root / "attacks" / name
        destination.mkdir(parents=True, exist_ok=True)
        content = {"segments": _encode(tuple(segments))}
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
        destination.write_text(json.dumps(_encode(scenario_map), indent=2, allow_nan=False) + "\n")


def load_scenario_maps(directory: str | Path,
                       scenario_ids: tuple[str, ...]) -> dict[str, ScenarioMap | Unavailable]:
    """Restore selected scenario maps without accessing dataset data."""
    result: dict[str, ScenarioMap | Unavailable] = {}
    for scenario_id in scenario_ids:
        path = _scenario_map_path(directory, scenario_id)
        if not path.is_file():
            result[scenario_id] = Unavailable("No saved scenario map is available")
            continue
        scenario_map = _decode(json.loads(path.read_text()))
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


def _environment_record(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical data for one environment, independent of omitted optional entry keys."""
    from .config import _summary_value

    normalized = normalize_environment(environment)
    sensors = []
    for entry in normalized["sensors"]:
        record = {"type": entry["type"], "options": dict(entry["options"])}
        if entry.get("camera") is not None:
            record["camera"] = entry["camera"]
        sensors.append(record)
    return _summary_value({"world": normalized["world"], "sensors": sensors})


def preparation_record(configuration: ExperimentConfiguration) -> dict[str, Any]:
    """The preparation inputs that saved metadata and artifacts depend on, as canonical data.

    Environment values must be data (as accepted by configuration summaries); anything else
    fails here, before selection or offline work starts.
    """
    return {
        "seed": configuration.runs.seeds[0],
        "offline_environments": {
            name: _environment_record(merge_environments(configuration.runs.environment,
                                                         attack.environment))
            for name, attack in configuration.attacks.items()
        },
    }


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
        segments = _decode(record["segments"])
        if not isinstance(segments, tuple) or any(not isinstance(s, AttackSegment) for s in segments):
            raise ValueError("Saved metadata must contain AttackSegment records")
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
