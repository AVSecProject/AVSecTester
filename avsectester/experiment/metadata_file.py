"""Share selected segments as a dataset-specific, versioned JSON metadata file."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from avsectester.attack import AttackSegment
from avsectester.serialization import DEFAULT_REGISTRY
from avsectester.type_paths import split_type_path

from .models import segment_key


@dataclass(frozen=True)
class _MetadataDocument:
    dataset_type: str
    segments: tuple[AttackSegment, ...]
    sha256: str


def _check_duplicates(segments: Sequence[AttackSegment]) -> None:
    seen: dict[str, int] = {}
    for index, segment in enumerate(segments):
        key = segment_key(segment)
        if key in seen:
            raise ValueError(f"Duplicate metadata segments at indices {seen[key]} and {index}")
        seen[key] = index


def _read_document(path: str | Path) -> _MetadataDocument:
    source = Path(path)
    payload = source.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("format") != "avsectester.metadata":
        raise ValueError("Metadata file format must be avsectester.metadata")
    version = document.get("version")
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported metadata file version; expected 1")
    dataset_type = document.get("dataset_type")
    if not isinstance(dataset_type, str):
        raise ValueError("Metadata dataset_type must be a module:Class path")
    try:
        split_type_path(dataset_type)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid metadata dataset_type: {error}") from error
    records = document.get("segments")
    if not isinstance(records, list):
        raise ValueError("Metadata segments must be a list")
    segments = []
    for index, record in enumerate(records):
        try:
            segment = DEFAULT_REGISTRY.decode(record)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Metadata segment {index} cannot be decoded: {error}") from error
        if not isinstance(segment, AttackSegment):
            raise ValueError(f"Metadata segment {index} must be an AttackSegment")
        segments.append(segment)
    _check_duplicates(segments)
    return _MetadataDocument(dataset_type, tuple(segments), hashlib.sha256(payload).hexdigest())


def save_metadata(segments: Sequence[AttackSegment], path: str | Path, *, dataset_type: str) -> None:
    """Save ordered segments without overwriting a different existing document.

    dataset_type is the dataset class's fully qualified module:Class path. Existing files
    with equal JSON content are retained byte for byte.
    """
    split_type_path(dataset_type)
    records = tuple(segments)
    for index, segment in enumerate(records):
        if not isinstance(segment, AttackSegment):
            raise TypeError(f"Metadata segment {index} must be an AttackSegment")
    _check_duplicates(records)
    document = {
        "format": "avsectester.metadata", "version": 1, "dataset_type": dataset_type,
        "segments": [DEFAULT_REGISTRY.encode(segment) for segment in records],
    }
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination = Path(path)
    if destination.exists():
        existing = _read_document(destination)
        saved = {
            "format": "avsectester.metadata", "version": 1, "dataset_type": existing.dataset_type,
            "segments": [DEFAULT_REGISTRY.encode(segment) for segment in existing.segments],
        }
        if json.dumps(saved, sort_keys=True, allow_nan=False) != json.dumps(
            document, sort_keys=True, allow_nan=False,
        ):
            raise ValueError(f"Metadata file already exists with different content: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another writer creating the same path.
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)


def load_metadata(path: str | Path) -> tuple[AttackSegment, ...]:
    """Load ordered segments after validating their file format and dataset type declaration.

    Experiment.prepare additionally checks the declared dataset type against its configured
    dataset and validates every segment against that dataset and its attacks.
    """
    return _read_document(path).segments

