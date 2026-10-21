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

from .models import PreparedExperiment, SelectionDecision

if TYPE_CHECKING:
    from .config import ExperimentConfiguration


class ConfigurationMismatchError(ValueError):
    """Saved preparation and supplied execution configuration describe different inputs."""


_RECORDS = {
    "avsectester.datasets.base": {"FrameReference", "ActorState"},
    "avsectester.attack.attack": {"AttackSegment"},
    "avsectester.attack.placement": {"AttachedPlacement", "WorldPlacement", "Orientation"},
    "avsectester.geometry.poses": {"ActorPose"},
}


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
        return record(**{key: _decode(item) for key, item in value["fields"].items()})
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
        "selections": [item.as_dict() for item in prepared.selections],
    }
    (root / "prepared.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return root


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


def _load_metadata(configuration: ExperimentConfiguration, directory: str | Path,
                   *, validate_dataset: bool) -> PreparedMetadata:
    root = Path(directory)
    manifest = json.loads((root / "prepared.json").read_text())
    if set(manifest["attacks"]) != set(configuration.attacks):
        raise ConfigurationMismatchError("Saved attack names differ from the supplied configuration")
    saved = manifest.get("configuration")
    _require_parameters(saved)
    scenarios = None if validate_dataset else saved["dataset"]["scenarios"]
    expected = configuration.describe_configuration(scenarios=scenarios)
    _require_parameters(expected)
    if not isinstance(saved, dict) or saved != expected:
        mismatches = list(expected) if not isinstance(saved, dict) else [
            key for key in expected if saved.get(key) != expected[key]
        ]
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
    return PreparedMetadata(metadata, decisions)


def load_prepared_metadata(configuration: ExperimentConfiguration,
                           directory: str | Path) -> PreparedMetadata:
    """Read only preparation data for evaluation, without accessing dataset data or artifacts."""
    return _load_metadata(configuration, directory, validate_dataset=False)


def load_prepared(configuration: ExperimentConfiguration,
                  directory: str | Path) -> PreparedExperiment:
    """Validate execution inputs and load all artifacts; never return a partially usable object."""
    saved = _load_metadata(configuration, directory, validate_dataset=True)
    root = Path(directory)
    artifacts = {name: attack.load_artifacts(root / "attacks" / name)
                 for name, attack in configuration.attacks.items()}
    return PreparedExperiment(configuration, saved.metadata, artifacts, saved.selections)
