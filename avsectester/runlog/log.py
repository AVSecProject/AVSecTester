"""Independent run snapshots and incremental Parquet persistence.

The in-memory API imports neither Arrow nor simulator/model packages. ``Recorder``
uses one snapshot path for memory and disk. Native adapter objects require registered
native types before persistence. Unknown objects are never pickled or stringified.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
import json
import os
from pathlib import Path
import sys
from threading import Condition, Thread
from time import monotonic
from typing import Any, TYPE_CHECKING
from weakref import ref

from avsectester.data_types import Detection, Observation, RosMessage, without_sensor_payloads
from avsectester.serialization import (
    DEFAULT_REGISTRY, SerializationRegistry, array_dtype_fields, decode_array,
)
from avsectester.structured_arrays import item_bytes

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.base import SensorGeometry
    from avsectester.rendering.compositing import ObjectVisibility

_TAG = "_avsectester_type"
_OBSERVATION_FIELDS = ("raw_observation", "model_observation")


def _record_payloads(value: Any, *, retain_sensor_data: bool = True) -> Any:
    observations = {}
    for name in _OBSERVATION_FIELDS:
        observation = getattr(value, name, None)
        if isinstance(observation, Observation):
            observations[name] = without_sensor_payloads(observation,
                keep_sensor_ids=observation.sensor_data if retain_sensor_data else ())
    return replace(value, **observations) if observations else value


def _without_encoded_sensor_data(value: Any) -> Any:
    """Return an encoded observation dataclass whose sensor data decodes to an empty mapping."""
    if (isinstance(value, dict) and value.get(_TAG) == "dataclass"
            and isinstance(value.get("fields"), dict) and "sensor_data" in value["fields"]):
        return {**value, "fields": {**value["fields"], "sensor_data": {}}}
    return value


@dataclass
class StepRecord:
    """One decision, with independently captured physical and model-visible data.

    ``detections`` holds canonical ego-frame boxes, or None if the driving system
    has no detection output. Recording never runs extra inference to obtain them.
    ``object_visibility`` maps world-object IDs and camera IDs to the visibility
    evidence for each camera's latest physical acquisition, before its sensor actions.
    Input hooks that remove a camera do not remove this physical evidence.
    Times belong to the captured values rather than a single timestamp for the entire decision.
    """

    step: int
    raw_observation: Any = None
    model_observation: Any = None
    physical_pre: Any = None
    physical_post: Any = None
    produced_command: Any = None
    executed_command: Any = None
    components: dict[str, Any] = field(default_factory=dict)
    detections: Sequence[Detection] | None = None
    sensor_geometry: Mapping[str, SensorGeometry] = field(default_factory=dict)
    object_visibility: Mapping[str, Mapping[str, ObjectVisibility]] = field(default_factory=dict)
    actions: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    executed_command_issue_time_us: int | None = None
    delivered_captures: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    'Capture IDs delivered to this decision. Driving systems record their own frame selection.'
    physical_captures: Mapping[str, int] = field(default_factory=dict)
    'Latest physically captured frame per sensor, independently of model input hooks.'
    detection_capture_id: int | None = None
    detection_sensor: str | None = None
    force_gt: bool = False
    'Whether this interval follows the configured recorded ground-truth motion.'

    def __post_init__(self) -> None:
        # A Parquet list column keeps variable detection counts in one typed schema.
        # General tuples elsewhere retain their existing explicit tuple encoding.
        if self.detections is not None:
            self.detections = list(self.detections)
            if any(not isinstance(item, Detection) for item in self.detections):
                raise TypeError("StepRecord.detections must contain Detection values")


@dataclass
class CaptureRecord:
    """One acquisition timestamp and its due sensors.

    ``model_observation`` holds the final per-frame input delivered to a decision, including
    localization and observation hooks. Its pose belongs to the frame end. It is None when
    no finalized decision input received the acquisition, including terminal feedback frames.
    """

    capture_id: int
    time_us: int
    sensor_ids: tuple[str, ...]
    raw_observation: Any = None
    model_observation: Any = None
    physical_state: Any = None
    sensor_geometry: Mapping[str, Any] = field(default_factory=dict)
    object_visibility: Mapping[str, Any] = field(default_factory=dict)
    frame_ranges_us: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    'Per-sensor half-open exposure or scan intervals. Instantaneous frames have equal endpoints.'


@dataclass
class EventRecord:
    """An initialization, termination or other non-step event."""

    kind: str
    step: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MessageRecord:
    """One message processed at a modifiable or observed ROS topic.

    ``arrival_time_us`` uses the simulation clock. ``stamp_ns`` is the ROS header timestamp,
    if available. Wall-clock timestamps in nanoseconds track relay reception
    (``relay_received_ns``), framework reception (``framework_received_ns``), processing
    start (``processing_started_ns``) and publication (``relay_sent_ns``). Observe-only
    topics have no publication timestamp. ``processing_s`` measures processing in seconds.

    ``modifiable`` marks forwarded topics, and ``modified`` marks changes to the message.
    Payload capture stores ``original`` and includes ``forwarded`` when the message changes.
    Metadata-only records leave both payloads empty.
    """

    topic: str
    type: str
    step: int
    stamp_ns: int | None
    arrival_time_us: int
    relay_received_ns: int
    framework_received_ns: int
    processing_started_ns: int
    processing_s: float
    relay_sent_ns: int | None
    modified: bool
    original: RosMessage | None = None
    forwarded: RosMessage | None = None
    modifiable: bool = True


_STREAM_RECORDS = {"steps": StepRecord, "events": EventRecord,
                   "captures": CaptureRecord, "messages": MessageRecord}
_MESSAGE_PAYLOADS = ("original", "forwarded")


@dataclass
class RunLog:
    """Captured records, optionally backed by an incrementally written directory."""

    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    status: str = "incomplete"
    path: Path | None = None
    captures: list[CaptureRecord] = field(default_factory=list)
    format_version: int = 2
    messages: list[MessageRecord] = field(default_factory=list)

    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        registry: SerializationRegistry = DEFAULT_REGISTRY,
        load_payloads: bool = True,
        load_sensor_data: bool = True,
        load_records: bool = True,
    ) -> RunLog:
        """Read completed parts, including those left by an interrupted writer.

        Only ``part-*.parquet`` files with a written footer are published. Pending
        ``*.open`` files are ignored. Native values decode to ordinary data, allowing
        analysis without simulator dependencies. User record types must be registered first.
        ``load_sensor_data=False`` reads raw and model observations with empty
        ``sensor_data`` and leaves their recorded payloads on disk. Acquisition times,
        poses and calibration remain available in ``deliveries`` without its payloads.
        It reads ROS message metadata without the original and forwarded values.
        ``load_records=False`` reads the manifest metadata, status and format version
        without reading any step, event, capture or message records. ``load_payloads`` also
        controls whether binary values referenced by the manifest metadata are loaded.
        """
        root = Path(path)
        manifest = _read_manifest(root)
        decoder = _Decoder(root, registry, load_payloads)
        log = cls(
            metadata=decoder.decode(manifest["metadata"]),
            status=manifest["status"],
            path=root,
            format_version=manifest.get("format_version", 1),
        )
        if not load_records:
            return log
        for row in read_table(root).to_pylist():
            if not load_sensor_data:
                row = {key: _without_encoded_sensor_data(value)
                       if key in _OBSERVATION_FIELDS else value for key, value in row.items()}
            log.steps.append(StepRecord(**{key: decoder.decode(value) for key, value in row.items()}))
        for row in read_table(root, stream="events").to_pylist():
            log.events.append(EventRecord(**{key: decoder.decode(value) for key, value in row.items()}))
        for row in read_table(root, stream="captures").to_pylist():
            if not load_sensor_data:
                row = {key: _without_encoded_sensor_data(value)
                       if key in _OBSERVATION_FIELDS else value for key, value in row.items()}
            log.captures.append(CaptureRecord(**{key: decoder.decode(value) for key, value in row.items()}))
        if (root / "messages").is_dir():
            columns = None if load_sensor_data else [f.name for f in fields(MessageRecord)
                                                       if f.name not in _MESSAGE_PAYLOADS]
            for row in read_table(root, columns=columns, stream="messages").to_pylist():
                log.messages.append(MessageRecord(**{key: decoder.decode(value) for key, value in row.items()}))
        return log


class Recorder:
    """Capture a run in memory and optionally stream it to Parquet.

    Record methods immediately snapshot metadata. The ``sensor_data`` values in raw and
    model observations are retained by reference and must not be modified afterwards.
    Call ``snapshot`` first when an independent copy is needed. ``retain_steps=False`` bounds
    disk-recorder memory by the open chunk and the write backlog.
    ``retain_sensor_data=False`` retains steps whose raw and model observations have
    empty ``sensor_data``. The written records keep their sensor data. Step delivery
    payloads reference their acquisition records rather than repeating their contents.
    ``record_messages=True`` enables ROS message records, including optional payloads.
    With ``retain_sensor_data=False``, message bodies are kept only in the written records.

    A directory must be empty. A background writer publishes completed chunks in order.
    Recording waits if two chunks are already queued. ``finish`` waits for publication
    before finalizing the manifest. ``abort`` discards pending chunks and leaves completed
    chunks readable. Neither method closes caller-owned simulator/model resources.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        registry: SerializationRegistry = DEFAULT_REGISTRY,
        chunk_steps: int = 128,
        chunk_bytes: int = 8 * 1024 * 1024,
        payload_shard_bytes: int = 64 * 1024 * 1024,
        retain_steps: bool = True,
        retain_sensor_data: bool = True,
        record_messages: bool = False,
    ) -> None:
        if chunk_steps <= 0 or chunk_bytes <= 0 or payload_shard_bytes <= 0:
            raise ValueError("Chunk limits must be positive")
        if not retain_steps and path is None:
            raise ValueError("retain_steps=False requires a persistent output path")
        if not retain_sensor_data and path is None:
            raise ValueError("retain_sensor_data=False requires a persistent output path")
        self.registry = registry
        self._writer: _Writer | None = None
        self._capture_ids: set[int] = set()
        self.log = RunLog(
            metadata=self.snapshot(metadata or {}), path=Path(path) if path is not None else None
        )
        self.retain_steps = retain_steps
        self.retain_sensor_data = retain_sensor_data
        self._record_messages = record_messages
        self._finished = False
        self._last_step = -1
        self._writer = (
            _Writer(self.log.path, self.registry, chunk_steps, chunk_bytes, payload_shard_bytes,
                    record_messages=record_messages)
            if self.log.path is not None else None
        )
        if self._writer is not None:
            try:
                self._writer.write_manifest(self.log)
            except BaseException as error:
                try:
                    self._writer.close()
                except BaseException as cleanup:
                    error.add_note(f"RunLog writer cleanup also failed: {cleanup!r}")
                raise

    def snapshot(self, value: Any) -> Any:
        """Take an independent snapshot, preserving native wrappers and mutable array semantics."""
        if self._writer is not None:
            self._writer.check_error()
        return self.registry.snapshot(value)

    def _record_snapshot(self, record: Any) -> Any:
        result = self.snapshot(_record_payloads(record, retain_sensor_data=False))
        for name in _OBSERVATION_FIELDS:
            source = getattr(record, name, None)
            if isinstance(source, Observation):
                getattr(result, name).sensor_data = dict(source.sensor_data)
        return result

    def record_step(self, record: StepRecord) -> StepRecord:
        """Append a decision record and queue a complete row group when due.

        Sensor payloads are retained by reference and must not be modified afterwards.
        Call ``snapshot`` first when an independent copy is needed.
        """
        self._check_open()
        if not isinstance(record, StepRecord):
            raise TypeError("record_step requires StepRecord")
        if (not isinstance(record.step, int) or isinstance(record.step, bool)
                or record.step <= self._last_step or record.step < 0):
            raise ValueError("Decision indices must be nonnegative and strictly increasing")
        snapshot = self._record_snapshot(record)
        if self._writer is not None:
            self._writer.append("steps", {f.name: getattr(snapshot, f.name) for f in fields(snapshot)})
        if self.retain_steps:
            self.log.steps.append(snapshot if self.retain_sensor_data else _record_payloads(snapshot, retain_sensor_data=False))
        self._last_step = record.step
        if self._writer is not None:
            self._writer.payloads.clear_bytes()
        return snapshot

    def record_event(self, kind: str, *, step: int | None = None, **data: Any) -> EventRecord:
        """Append a timestamped runtime event to the recording."""
        self._check_open()
        event = self.snapshot(EventRecord(kind=kind, step=step, data=data))
        if self._writer is not None:
            self._writer.append("events", {f.name: getattr(event, f.name) for f in fields(event)})
        if self.retain_steps:
            self.log.events.append(event)
        return event

    def record_capture(self, record: CaptureRecord) -> CaptureRecord:
        """Snapshot acquisition metadata and retain sensor payloads by reference.

        The caller must not modify those payloads afterwards. Call ``snapshot`` first
        when an independent copy is needed.
        """
        self._check_open()
        if not isinstance(record, CaptureRecord):
            raise TypeError("record_capture requires CaptureRecord")
        if record.capture_id in self._capture_ids:
            raise ValueError(f"Duplicate capture_id: {record.capture_id}")
        self._capture_ids.add(record.capture_id)
        snapshot = self._record_snapshot(record)
        if self._writer is not None:
            self._writer.append("captures", {f.name: getattr(snapshot, f.name) for f in fields(snapshot)})
        if self.retain_steps:
            self.log.captures.append(snapshot if self.retain_sensor_data else _record_payloads(snapshot, retain_sensor_data=False))
        return snapshot

    def record_message(self, record: MessageRecord) -> MessageRecord:
        """Snapshot and append one intercepted message to an enabled message stream."""
        self._check_open()
        if not self._record_messages:
            raise ValueError("Recording messages requires record_messages=True")
        if not isinstance(record, MessageRecord):
            raise TypeError("record_message requires MessageRecord")
        snapshot: MessageRecord = self.snapshot(record)
        if self._writer is not None:
            self._writer.append("messages", {f.name: getattr(snapshot, f.name) for f in fields(snapshot)})
        if self.retain_steps:
            self.log.messages.append(snapshot if self.retain_sensor_data else
                                     replace(snapshot, original=None, forwarded=None))
        return snapshot

    def flush(self) -> None:
        """Wait for publication of all recorded chunks without marking the run complete."""
        self._check_open()
        if self._writer is not None:
            self._writer.flush()

    def finish(self, status: str = "completed", **details: Any) -> RunLog:
        """Finish recording with a valid terminal status and persist its manifest."""
        if self._finished:
            return self.log
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("Final status must be completed, failed or cancelled")
        if self._writer is not None:
            # Retain the open chunk so the termination event joins it at the usual boundary.
            self._writer.wait()
            details["recording"] = {
                "backpressure_wait_s": self._writer.backpressure_wait_s,
                "backpressure_batches": self._writer.backpressure_batches,
            }
        self.record_event("termination", status=status, **details)
        if self._writer is not None:
            self._writer.flush()
        previous_status = self.log.status
        self.log.status = status
        if self._writer is not None:
            try:
                self._writer.write_manifest(self.log)
            except BaseException:
                self.log.status = previous_status
                raise
            self._writer.close()
        self._finished = True
        return self.log

    def abort(self) -> None:
        """Discard pending chunks and leave published records readable."""
        try:
            if not self._finished and self._writer is not None:
                self._writer.close()
        finally:
            self._finished = True

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError("Recorder is already closed")
        if self._writer is not None:
            self._writer.check_error()

    def __enter__(self) -> Recorder:
        self._check_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                self.finish()
            else:
                status = "cancelled" if issubclass(exc_type, (KeyboardInterrupt, SystemExit)) else "failed"
                self.finish(status, error_type=exc_type.__name__, error=str(exc))
        finally:
            if not self._finished:
                self.abort()


def read_table(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    stream: str = "steps",
) -> Any:
    """Read selected top-level columns without loading arrays or other columns.

    The returned Arrow table retains typed struct/list columns and payload
    references. Use ``RunLog.read`` to reconstruct core values and load payloads.
    Column schemas can gain fields across chunks. Incompatible types raise.
    """
    pa, pq = _arrow()
    root = Path(path)
    _read_manifest(root)
    if stream not in _STREAM_RECORDS:
        raise ValueError("stream must be steps, events, captures or messages")
    parts = sorted((root / stream).glob("part-*.parquet"))
    if not parts:
        record_type = _STREAM_RECORDS[stream]
        known = {f.name for f in fields(record_type)}
        wanted = list(columns) if columns is not None else [f.name for f in fields(record_type)]
        unknown = set(wanted) - known
        if unknown:
            raise ValueError(f"Unknown RunLog columns: {sorted(unknown)}")
        empty_schema = pa.schema([
            (name, pa.int64() if name == "step" else pa.string() if name == "kind" else pa.null())
            for name in wanted
        ])
        return pa.Table.from_batches([], schema=empty_schema)
    schemas = [pq.read_schema(part) for part in parts]
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    wanted = list(columns) if columns is not None else schema.names
    unknown = set(wanted) - set(schema.names)
    if unknown:
        raise ValueError(f"Unknown RunLog columns: {sorted(unknown)}")
    selected = pa.schema([schema.field(name) for name in wanted])
    tables = []
    for part, part_schema in zip(parts, schemas):
        available = [name for name in wanted if name in part_schema.names]
        table = pq.read_table(part, columns=available)
        for name in wanted:
            if name not in table.column_names:
                column = pa.nulls(table.num_rows, type=schema.field(name).type)
                table = table.append_column(name, column)
        table = table.select(wanted)
        if not table.schema.equals(selected):
            try:
                table = table.cast(selected)
            except pa.ArrowInvalid:
                # Arrow 21's nested nullable struct/list cast can produce invalid
                # child lengths even for valid arrays. Rebuild only this selected
                # chunk after validating its source. Never hide corrupt input.
                table.validate(full=True)
                table = pa.Table.from_pylist(table.to_pylist(), schema=selected)
        tables.append(table)
    return pa.concat_tables(tables)


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Parquet recording requires the project's pinned pyarrow dependency") from exc
    return pa, pq


def _read_manifest(root: Path) -> dict[str, Any]:
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version", 1) not in (1, 2):
        raise ValueError(f"Unsupported RunLog format version: {manifest['format_version']!r}")
    return manifest


class _Encoder:
    def __init__(self, registry: SerializationRegistry, payloads: _PayloadWriter) -> None:
        self.registry = registry
        self.payloads = payloads

    def dynamic(self, value: Any) -> dict[str, Any]:
        """Typed alternatives for a declared arbitrary-value field.

        Multiple named callback outputs share a list column even when one is
        a scalar TTC and another is a nested detection summary. Arrow has no
        Parquet union type, so store their alternatives as optional struct fields.
        Mapping entries and list members recurse through the same typed layout.
        """
        if value is None:
            return {_TAG: "dynamic", "kind": "null"}
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("RunLog mappings require string keys")
            return {_TAG: "dynamic", "kind": "mapping", "entries": [
                {"key": key, "value": self.dynamic(item)} for key, item in value.items()
            ]}
        if isinstance(value, (list, tuple)):
            return {_TAG: "dynamic", "kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [self.dynamic(item) for item in value]}
        encoded = self.encode(value)
        if isinstance(encoded, bool):
            kind = "boolean"
        elif isinstance(encoded, int):
            kind = "integer"
        elif isinstance(encoded, float):
            kind = "number"
        elif isinstance(encoded, str):
            kind = "string"
        else:
            kind = "encoded"
        return {_TAG: "dynamic", "kind": kind, kind: encoded}

    def encode(self, value: Any) -> Any:
        if isinstance(value, RosMessage):
            # Topic bodies have unrelated field types. Keep their shape records inside a
            # scalar so different ROS interfaces can share a chunk without a union schema.
            return {_TAG: "ros_message", "type": value.type,
                    "body": json.dumps(self.encode(value.fields), separators=(",", ":"))}
        encoded = self.registry.encode(
            value, encode_child=self.encode, encode_field=self._record_field,
            encode_leaf=self._payload_value,
        )
        if (isinstance(encoded, dict) and encoded.get(_TAG) == "dataclass"
                and encoded["fields"] == {}):
            # Parquet cannot store an empty struct. Null transports a record with no fields.
            return {**encoded, "fields": None}
        return encoded

    def _payload_value(self, value: Any) -> dict[str, Any]:
        if isinstance(value, bytes):
            return {_TAG: "bytes", **self.payloads.write(memoryview(value), value)}
        import numpy as np

        if not value.dtype.fields:
            array = np.ascontiguousarray(value)
            data = memoryview(array.reshape(-1)).cast("B")
        elif value.flags.c_contiguous:
            data = memoryview(value.reshape(-1).view(np.uint8))
        else:
            data = memoryview(item_bytes(value))
        return {
            _TAG: "array", **array_dtype_fields(value), "shape": list(value.shape),
            **self.payloads.write(data, value),
        }

    def _record_field(self, name: str, value: Any) -> Any:
        # Callback outputs deliberately permit heterogeneous values in one list column.
        return self.dynamic(value) if name == "value" else self.encode(value)


class _Decoder:
    def __init__(self, root: Path, registry: SerializationRegistry, load_payloads: bool) -> None:
        self.root = root
        self.registry = registry
        self.load_payloads = load_payloads

    def decode(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if not isinstance(value, dict):
            return value
        tag = value.get(_TAG)
        if tag in {"array", "bytes"}:
            if not self.load_payloads:
                return value
            data = self._payload(value)
            if tag == "bytes":
                return data
            return decode_array(data, value, "RunLog payload")
        if tag == "ros_message":
            return RosMessage(value["type"], self.decode(json.loads(value["body"])))
        if tag == "dynamic":
            kind = value["kind"]
            if kind == "null":
                return None
            if kind == "mapping":
                return {entry["key"]: self.decode(entry["value"]) for entry in value["entries"]}
            if kind in {"list", "tuple"}:
                items = [self.decode(item) for item in value["items"]]
                return tuple(items) if kind == "tuple" else items
            if kind not in {"boolean", "integer", "number", "string", "encoded"}:
                raise ValueError(f"Unknown RunLog dynamic value kind: {kind!r}")
            return self.decode(value[kind])
        if tag == "dataclass" and value["fields"] is None:
            value = {**value, "fields": {}}
        return self.registry.decode(value, decode_child=self.decode)

    def _payload(self, value: dict[str, Any]) -> bytes:
        pieces = []
        for chunk in value["chunks"]:
            path = (self.root / chunk["path"]).resolve()
            if not path.is_relative_to(self.root.resolve() / "payloads"):
                raise ValueError("RunLog payload path escapes its payload directory")
            offset, size = chunk["offset"], chunk["size"]
            if offset < 0 or size < 0:
                raise ValueError("RunLog payload has a negative extent")
            with path.open("rb") as handle:
                handle.seek(offset)
                piece = handle.read(size)
            if len(piece) != size:
                raise ValueError(f"Truncated RunLog payload: {chunk['path']}")
            pieces.append(piece)
        result = b"".join(pieces)
        if len(result) != value["nbytes"]:
            raise ValueError("RunLog payload length does not match its reference")
        return result


class _PayloadWriter:
    def __init__(self, root: Path, limit: int) -> None:
        self.root, self.limit = root, limit
        self.index = 0
        self._handle: Any = None
        self._size = 0
        self._open: Path | None = None
        self._sealed: list[Path] = []
        self._contents: dict[int, tuple[Any, dict[str, Any]]] = {}
        self._byte_contents: dict[int, tuple[Any, dict[str, Any]]] = {}

    def clear_bytes(self) -> None:
        self._byte_contents.clear()

    def write(self, data: memoryview, source: Any) -> dict[str, Any]:
        cache = self._byte_contents if isinstance(source, bytes) else self._contents
        previous = cache.get(id(source))
        if previous is not None:
            owner = previous[0] if isinstance(source, bytes) else previous[0]()
            if owner is source:
                return previous[1]
        chunks = []
        offset = 0
        while offset < len(data):
            if self._handle is None:
                self._open = self.root / "payloads" / f"part-{self.index:05d}.bin.open"
                self._handle = self._open.open("xb")
                self._size = 0
            length = min(self.limit - self._size, len(data) - offset)
            chunks.append({"path": f"payloads/part-{self.index:05d}.bin",
                           "offset": self._size, "size": length})
            self._handle.write(data[offset:offset + length])
            self._size += length
            offset += length
            if self._size == self.limit:
                self.seal_shard()
        reference = {"nbytes": len(data), "chunks": chunks}
        identity = id(source)
        owner = source if isinstance(source, bytes) else ref(source, lambda _ref: cache.pop(identity, None))
        cache[identity] = (owner, reference)
        return reference

    def seal_shard(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        assert self._open is not None
        self._sealed.append(self._open)
        self.index += 1
        self._open = None

    @property
    def has_sealed_shards(self) -> bool:
        return bool(self._sealed)

    def take_shards(self) -> list[Path]:
        shards, self._sealed = self._sealed, []
        return shards

    def abort(self) -> None:
        handle, self._handle = self._handle, None
        self._sealed.clear()
        if handle is not None:
            handle.close()
        # Open files are intentionally ignored by recovery, like a killed process.


@dataclass
class _WriteBatch:
    records: dict[str, list[dict[str, Any]]]
    shards: list[Path]
    manifest: dict[str, Any] | None = None


class _Writer:
    def __init__(self, root: Path, registry: SerializationRegistry, rows: int, size: int, shard: int,
                 *, record_messages: bool) -> None:
        self.pa, self.pq = _arrow()
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError(f"RunLog output directory must be empty: {root}")
        self.streams = ("steps", "events", "captures") + (("messages",) if record_messages else ())
        for name in (*self.streams, "payloads"):
            (root / name).mkdir()
        self.root = root
        self.rows, self.size = rows, size
        self.payloads = _PayloadWriter(root, shard)
        self.encoder = _Encoder(registry, self.payloads)
        self.buffers: dict[str, list[dict[str, Any]]] = {name: [] for name in self.streams}
        self.counts = dict.fromkeys(self.streams, 0)
        self.rows_written = dict.fromkeys(self.streams, 0)
        self.last_durable_step: int | None = None
        self.manifest: dict[str, Any] = {}
        self.schemas: dict[str, Any] = {}
        self.buffer_size = 0
        self.backpressure_wait_s = 0.
        self.backpressure_batches = 0
        self._condition = Condition()
        self._pending: deque[_WriteBatch] = deque()
        self._writing = False
        self._stopping = False
        self._error: BaseException | None = None
        self._thread = Thread(target=self._write_loop, name=f"avsectester-runlog-{root.name}",
                              daemon=True)
        self._thread.start()

    def append(self, stream: str, record: dict[str, Any]) -> None:
        encoded = {name: self.encoder.encode(value) for name, value in record.items()}
        self.buffers[stream].append(encoded)
        self.buffer_size += _memory_size(encoded)
        if len(self.buffers[stream]) >= self.rows or self.buffer_size >= self.size:
            self._submit()

    def flush(self) -> None:
        self._submit()
        self.wait()

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("RunLog background write failed") from self._error

    def wait(self) -> None:
        with self._condition:
            while self._pending or self._writing:
                self._condition.wait()
        try:
            self.check_error()
        except RuntimeError as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"RunLog writer cleanup also failed: {cleanup!r}")
            raise

    def _submit(self, manifest: dict[str, Any] | None = None) -> None:
        self.payloads.seal_shard()
        if not self.buffer_size and not self.payloads.has_sealed_shards and manifest is None:
            return
        buffers: dict[str, list[dict[str, Any]]] = {name: [] for name in self.streams}
        with self._condition:
            if len(self._pending) == 2:
                start = monotonic()
                while len(self._pending) == 2:
                    self._condition.wait()
                self.backpressure_wait_s += monotonic() - start
                self.backpressure_batches += 1
            self.check_error()
            # Keep the records and sealed shards together if the producer is interrupted waiting.
            self._pending.append(_WriteBatch(self.buffers, self.payloads.take_shards(), manifest))
            self.buffers = buffers
            self.buffer_size = 0
            self._condition.notify_all()

    def _write_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                batch = self._pending.popleft()
                self._writing = True
                self._condition.notify_all()
            try:
                self._write_batch(batch)
            except BaseException as error:
                with self._condition:
                    self._error = error
                    self._pending.clear()
                    self._writing = False
                    self._condition.notify_all()
                return
            del batch
            with self._condition:
                self._writing = False
                self._condition.notify_all()

    def _write_batch(self, batch: _WriteBatch) -> None:
        # A published record part may only reference published payload shards.
        for path in batch.shards:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(path, path.with_suffix(""))
        tables = {}
        for stream, records in batch.records.items():
            if records:
                try:
                    table = self.pa.Table.from_pylist(records)
                    previous = self.schemas.get(stream)
                    schema = table.schema if previous is None else self.pa.unify_schemas(
                        [previous, table.schema], promote_options="permissive"
                    )
                except (self.pa.ArrowInvalid, self.pa.ArrowTypeError) as exc:
                    raise TypeError("RunLog values need consistent typed registered records") from exc
                tables[stream] = table
                self.schemas[stream] = schema
        for stream, table in tables.items():
            part = self.root / stream / f"part-{self.counts[stream]:05d}.parquet"
            pending = part.with_suffix(".parquet.open")
            self.pq.write_table(table, pending, compression="zstd")
            with pending.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(pending, part)
            self.counts[stream] += 1
            self.rows_written[stream] += table.num_rows
            if stream == "steps":
                self.last_durable_step = table["step"][-1].as_py()
        if batch.manifest is not None:
            self.manifest = batch.manifest
        if tables or batch.manifest is not None:
            # This remains an incomplete checkpoint until finish publishes the
            # final status. Recovery also scans parts in case a process exits
            # after their rename but before this checkpoint is replaced.
            self._publish_manifest()

    def write_manifest(self, log: RunLog) -> None:
        self.wait()
        self._submit({"status": log.status, "metadata": self.encoder.encode(log.metadata),
                      "format_version": log.format_version})
        self.wait()

    def _publish_manifest(self) -> None:
        manifest = {
            **self.manifest,
            "step_parts": self.counts["steps"], "event_parts": self.counts["events"],
            "step_rows": self.rows_written["steps"], "event_rows": self.rows_written["events"],
            "last_durable_step": self.last_durable_step,
            "capture_parts": self.counts["captures"], "capture_rows": self.rows_written["captures"],
        }
        pending = self.root / "manifest.json.open"
        with pending.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, self.root / "manifest.json")

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending.clear()
            self._condition.notify_all()
        self._thread.join()
        self.payloads.abort()
        self.buffers = {name: [] for name in self.streams}
        self.buffer_size = 0


def _memory_size(value: Any) -> int:
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(sys.getsizeof(key) + _memory_size(item) for key, item in value.items())
    elif isinstance(value, list):
        size += sum(_memory_size(item) for item in value)
    return size


__all__ = [
    'StepRecord',
    'CaptureRecord',
    'EventRecord',
    'MessageRecord',
    'RunLog',
    'Recorder',
    'read_table',
]
