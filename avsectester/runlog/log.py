"""Independent run snapshots and incremental Parquet persistence.

The in-memory API imports neither Arrow nor simulator/model packages. ``Recorder``
uses one capture path for memory and disk; native adapter objects require registered
codecs before persistence. Unknown objects are never pickled or stringified.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

# Version 2 renamed the step ``actions`` field to ``effects`` (attack effect records).
SCHEMA_VERSION = 2
_TAG = "_avsectester_type"
_KEYS = "_avsectester_keys"
_DATATYPE_NAMES = {
    "Control", "Observation", "WorldSnapshot", "StateEstimate", "FrameRecord", "RecordedOutput",
}


@dataclass
class StepRecord:
    """One decision, with independently captured physical and model-visible data.

    ``telemetry`` holds already-produced compatibility measurements, including
    ``n_detections`` where available. Missing outputs stay missing; recording never
    runs a component to obtain them. Times belong to the captured values rather
    than a single ambiguous timestamp for the entire decision.
    """

    step: int
    raw_observation: Any = None
    model_observation: Any = None
    physical_pre: Any = None
    physical_post: Any = None
    produced_command: Any = None
    executed_command: Any = None
    components: dict[str, Any] = field(default_factory=dict)
    effects: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    """An initialization, termination or other non-step event."""

    kind: str
    step: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunLog:
    """Captured records, optionally backed by an incrementally written directory."""

    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    events: list[EventRecord] = field(default_factory=list)
    status: str = "incomplete"
    path: Path | None = None

    @classmethod
    def read(
        cls,
        path: str | Path,
        *,
        codecs: CodecRegistry | None = None,
        load_payloads: bool = True,
    ) -> RunLog:
        """Read completed parts, including those left by an interrupted writer.

        Only ``part-*.parquet`` files with a written footer are published. Pending
        ``*.open`` files are ignored. Unrecognized schema versions raise instead
        of being guessed. Without a native decoder, custom codec values remain
        ordinary decoded data, allowing analysis without simulator dependencies.
        """
        root = Path(path)
        manifest = _read_manifest(root)
        decoder = _Decoder(root, codecs or CodecRegistry(), load_payloads)
        log = cls(
            metadata=decoder.decode(manifest["metadata"]),
            status=manifest["status"],
            path=root,
        )
        for row in read_table(root).to_pylist():
            log.steps.append(StepRecord(**{key: decoder.decode(value) for key, value in row.items()}))
        for row in read_table(root, stream="events").to_pylist():
            log.events.append(EventRecord(**{key: decoder.decode(value) for key, value in row.items()}))
        return log


class CodecRegistry:
    """Explicit native-object conversion supplied by adapters.

    Encoders return plain data, supported core dataclasses or arrays. A decoder
    is optional: persisted component results can remain data for offline metrics.
    Codec names must remain stable across processes and releases. An optional
    ``snapshot(value, capture_child)`` handles SDK buffers that cannot be copied;
    it returns an independent native-compatible value and uses ``capture_child``
    for owned members. It must not capture its own input recursively.
    """

    def __init__(self) -> None:
        self._encoders: dict[type, tuple[str, Callable[[Any], Any]]] = {}
        self._decoders: dict[str, Callable[[Any], Any]] = {}
        self._snapshots: dict[type, Callable[[Any, Callable[[Any], Any]], Any]] = {}

    def register(
        self,
        native_type: type,
        encode: Callable[[Any], Any],
        *,
        name: str | None = None,
        decode: Callable[[Any], Any] | None = None,
        snapshot: Callable[[Any, Callable[[Any], Any]], Any] | None = None,
    ) -> CodecRegistry:
        label = name or f"{native_type.__module__}.{native_type.__qualname__}"
        if native_type in self._encoders or label in {item[0] for item in self._encoders.values()}:
            raise ValueError(f"Duplicate RunLog codec: {label}")
        self._encoders[native_type] = (label, encode)
        if decode is not None:
            self._decoders[label] = decode
        if snapshot is not None:
            self._snapshots[native_type] = snapshot
        return self

    def encoder(self, value: Any) -> tuple[str, Callable[[Any], Any]] | None:
        for native_type in type(value).__mro__:
            if native_type in self._encoders:
                return self._encoders[native_type]
        return None

    def snapshotter(self, value: Any) -> Callable[[Any, Callable[[Any], Any]], Any] | None:
        # Adapter registries may register already-loaded native types lazily.
        self.encoder(value)
        for native_type in type(value).__mro__:
            if native_type in self._snapshots:
                return self._snapshots[native_type]
        return None


class Recorder:
    """Capture a run in memory and optionally stream it to Parquet.

    ``record_step`` and ``record_event`` snapshot their inputs immediately.
    Call ``capture`` earlier when a subsequent hook might mutate a value before
    a complete step is available. ``retain_steps=False`` bounds disk-recorder
    memory by the open chunk instead of retaining the full run.

    A directory must be empty. ``finish`` publishes the last chunks before the
    final manifest. ``abort`` leaves completed chunks readable and discards the
    current chunks. Neither method closes caller-owned simulator/model resources.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        codecs: CodecRegistry | None = None,
        chunk_steps: int = 128,
        chunk_bytes: int = 8 * 1024 * 1024,
        payload_shard_bytes: int = 64 * 1024 * 1024,
        retain_steps: bool = True,
    ) -> None:
        if chunk_steps <= 0 or chunk_bytes <= 0 or payload_shard_bytes <= 0:
            raise ValueError("Chunk limits must be positive")
        if not retain_steps and path is None:
            raise ValueError("retain_steps=False requires a persistent output path")
        self.codecs = codecs or CodecRegistry()
        self.log = RunLog(
            metadata=self.capture(metadata or {}), path=Path(path) if path is not None else None
        )
        self.retain_steps = retain_steps
        self._finished = False
        self._last_step = -1
        self._writer = (
            _Writer(self.log.path, self.codecs, chunk_steps, chunk_bytes, payload_shard_bytes)
            if self.log.path is not None else None
        )
        if self._writer is not None:
            self._writer.write_manifest(self.log)

    def capture(self, value: Any) -> Any:
        """Copy mutable data now; tensors are detached without importing torch."""
        return _snapshot(value, {}, self.codecs)

    def record_step(self, record: StepRecord) -> StepRecord:
        self._check_open()
        if not isinstance(record, StepRecord):
            raise TypeError("record_step requires StepRecord")
        if (not isinstance(record.step, int) or isinstance(record.step, bool)
                or record.step <= self._last_step or record.step < 0):
            raise ValueError("Decision indices must be nonnegative and strictly increasing")
        snapshot = self.capture(record)
        if self._writer is not None:
            self._writer.append("steps", {f.name: getattr(snapshot, f.name) for f in fields(snapshot)})
        if self.retain_steps:
            self.log.steps.append(snapshot)
        self._last_step = record.step
        return snapshot

    def record_event(self, kind: str, *, step: int | None = None, **data: Any) -> EventRecord:
        self._check_open()
        event = self.capture(EventRecord(kind=kind, step=step, data=data))
        if self._writer is not None:
            self._writer.append("events", {f.name: getattr(event, f.name) for f in fields(event)})
        if self.retain_steps:
            self.log.events.append(event)
        return event

    def flush(self) -> None:
        """Publish the current chunks without marking the run complete."""
        self._check_open()
        if self._writer is not None:
            self._writer.flush()

    def finish(self, status: str = "completed", **details: Any) -> RunLog:
        if self._finished:
            return self.log
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("Final status must be completed, failed or cancelled")
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
        self._finished = True
        return self.log

    def abort(self) -> None:
        """Discard unpublished records and leave the persisted run incomplete."""
        if not self._finished and self._writer is not None:
            self._writer.abort()
        self._finished = True

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError("Recorder is already closed")

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.finish()
        else:
            try:
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
    Column schemas can gain fields across chunks; incompatible types raise.
    """
    pa, pq = _arrow()
    root = Path(path)
    _read_manifest(root)
    if stream not in {"steps", "events"}:
        raise ValueError("stream must be steps or events")
    parts = sorted((root / stream).glob("part-*.parquet"))
    if not parts:
        record_type = StepRecord if stream == "steps" else EventRecord
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
                table = table.append_column(name, pa.nulls(table.num_rows, type=schema.field(name).type))
        table = table.select(wanted)
        if not table.schema.equals(selected):
            try:
                table = table.cast(selected)
            except pa.ArrowInvalid:
                # Arrow 21's nested nullable struct/list cast can produce invalid
                # child lengths even for valid arrays. Rebuild only this selected
                # chunk after validating its source; never hide corrupt input.
                table.validate(full=True)
                table = pa.Table.from_pylist(table.to_pylist(), schema=selected)
        tables.append(table)
    return pa.concat_tables(tables)


def _snapshot(value: Any, memo: dict[int, Any], codecs: CodecRegistry) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    key = id(value)
    if key in memo:
        return memo[key]
    snapshot = codecs.snapshotter(value)
    if snapshot is not None:
        result = snapshot(value, lambda child: _snapshot(child, memo, codecs))
        if result is value:
            raise TypeError("A RunLog snapshot codec must return an independent snapshot")
        memo[key] = result
        return result
    if _is_tensor(value):
        result = value.detach().clone()
        memo[key] = result
        return result
    if type(value) is dict:
        result = {}
        memo[key] = result
        for k, v in value.items():
            result[deepcopy(k, memo)] = _snapshot(v, memo, codecs)
        return result
    if type(value) is list:
        result = []
        memo[key] = result
        result.extend(_snapshot(v, memo, codecs) for v in value)
        return result
    if type(value) is tuple:
        result = tuple(_snapshot(v, memo, codecs) for v in value)
        memo[key] = result
        return result
    if isinstance(value, dict):
        result = copy(value)
        memo[key] = result
        result.clear()
        for k, v in value.items():
            result[deepcopy(k, memo)] = _snapshot(v, memo, codecs)
        _snapshot_attributes(value, result, memo, codecs)
        return result
    if isinstance(value, list):
        result = copy(value)
        memo[key] = result
        result[:] = [_snapshot(v, memo, codecs) for v in value]
        _snapshot_attributes(value, result, memo, codecs)
        return result
    if isinstance(value, tuple):
        items = [_snapshot(v, memo, codecs) for v in value]
        if hasattr(value, "_fields"):
            result = getattr(type(value), "_make")(items)
        else:
            try:
                result = type(value)(items)
            except Exception:
                result = deepcopy(value, memo)
        memo[key] = result
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = object.__new__(type(value))
        memo[key] = result
        for f in fields(value):
            object.__setattr__(result, f.name, _snapshot(getattr(value, f.name), memo, codecs))
        return result
    return deepcopy(value, memo)


def _snapshot_attributes(
    source: Any, result: Any, memo: dict[int, Any], codecs: CodecRegistry,
) -> None:
    """Replace the attributes ``copy`` shared with ``source`` by independent snapshots."""
    state = getattr(source, "__dict__", None)
    if state:
        target = vars(result)
        for name, item in state.items():
            target[name] = _snapshot(item, memo, codecs)


def _is_tensor(value: Any) -> bool:
    return any(base.__module__.split(".")[0] == "torch" and base.__name__ == "Tensor"
               for base in type(value).__mro__)


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
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported RunLog schema version: {manifest.get('schema_version')!r}")
    return manifest


class _Encoder:
    def __init__(self, codecs: CodecRegistry, payloads: _PayloadWriter) -> None:
        self.codecs = codecs
        self.payloads = payloads

    def dynamic(self, value: Any) -> dict[str, Any]:
        """Typed alternatives for a declared arbitrary-value field.

        Multiple named intervention outputs share a list column even when one is
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
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, bytes):
            return {_TAG: "bytes", **self.payloads.write(memoryview(value))}
        codec = self.codecs.encoder(value)
        if codec is not None:
            name, encode = codec
            converted = encode(value)
            if converted is value:
                raise TypeError(f"RunLog codec {name} returned its input")
            return {_TAG: "codec", "name": name, "value": self.encode(converted)}
        if _is_tensor(value):
            value = value.detach().cpu().numpy()
        if any(base.__module__.split(".")[0] == "numpy" for base in type(value).__mro__):
            import numpy as np
            if isinstance(value, np.generic):
                return self.encode(value.item())
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject or value.dtype.fields:
                    raise TypeError("Object/structured arrays require an explicit RunLog codec")
                array = np.ascontiguousarray(value)
                return {
                    _TAG: "array", "dtype": value.dtype.str, "shape": list(value.shape),
                    **self.payloads.write(memoryview(array.reshape(-1)).cast("B")),
                }
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("RunLog mappings require string keys")
            if _TAG in value or _KEYS in value:
                raise ValueError("RunLog mappings contain a reserved serialization key")
            return {_KEYS: list(value), **{k: self.encode(v) for k, v in value.items()}}
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, tuple):
            return {_TAG: "tuple", "length": len(value),
                    **{f"item_{i}": self.encode(item) for i, item in enumerate(value)}}
        if type(value).__module__ == "avsectester.closed_loop.datatypes" and type(value).__name__ in _DATATYPE_NAMES:
            return {_TAG: type(value).__name__,
                    **{f.name: (
                        self.dynamic(getattr(value, f.name))
                        if type(value).__name__ == "RecordedOutput" and f.name == "value"
                        else self.encode(getattr(value, f.name))
                    ) for f in fields(value)}}
        if (type(value).__module__.startswith("avsectester.")
                and is_dataclass(value) and not isinstance(value, type)):
            # Framework records are explicitly data; external/native dataclasses
            # still need adapter codecs. Readers never import the encoded name.
            return {
                _TAG: "record", "name": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": {
                    _KEYS: [f.name for f in fields(value)],
                    **{f.name: (
                        self.dynamic(getattr(value, f.name)) if f.name == "value"
                        else self.encode(getattr(value, f.name))
                    ) for f in fields(value)},
                },
            }
        raise TypeError(
            f"No RunLog codec for {type(value).__module__}.{type(value).__qualname__}; "
            "register an adapter codec returning plain data"
        )


class _Decoder:
    def __init__(self, root: Path, codecs: CodecRegistry, load_payloads: bool) -> None:
        self.root = root
        self.codecs = codecs
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
            import numpy as np
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject or dtype.fields:
                raise ValueError("Unsafe array dtype in RunLog payload")
            return np.frombuffer(data, dtype=dtype).reshape(value["shape"]).copy()
        if tag == "tuple":
            return tuple(self.decode(value[f"item_{i}"]) for i in range(value["length"]))
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
        if tag == "codec":
            data = self.decode(value["value"])
            decode = self.codecs._decoders.get(value["name"])
            return decode(data) if decode is not None else data
        if tag == "record":
            return self.decode(value["value"])
        if tag in _DATATYPE_NAMES:
            from avsectester.closed_loop import datatypes
            record_type = getattr(datatypes, tag)
            return record_type(**{f.name: self.decode(value[f.name]) for f in fields(record_type)})
        if tag is not None:
            raise ValueError(f"Unknown RunLog value codec: {tag!r}")
        if _KEYS in value:
            return {key: self.decode(value[key]) for key in value[_KEYS]}
        return {key: self.decode(item) for key, item in value.items()}

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

    def write(self, data: memoryview) -> dict[str, Any]:
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
                self.publish()
        return {"nbytes": len(data), "chunks": chunks}

    def publish(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        assert self._open is not None
        os.replace(self._open, self._open.with_suffix(""))
        self.index += 1
        self._open = None

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        # Open files are intentionally ignored by recovery, like a killed process.


class _Writer:
    def __init__(self, root: Path, codecs: CodecRegistry, rows: int, size: int, shard: int) -> None:
        self.pa, self.pq = _arrow()
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError(f"RunLog output directory must be empty: {root}")
        for name in ("steps", "events", "payloads"):
            (root / name).mkdir()
        self.root = root
        self.rows, self.size = rows, size
        self.payloads = _PayloadWriter(root, shard)
        self.encoder = _Encoder(codecs, self.payloads)
        self.buffers: dict[str, list[dict[str, Any]]] = {"steps": [], "events": []}
        self.counts = {"steps": 0, "events": 0}
        self.rows_written = {"steps": 0, "events": 0}
        self.last_durable_step: int | None = None
        self.manifest: dict[str, Any] = {}
        self.schemas: dict[str, Any] = {}
        self.buffer_size = 0

    def append(self, stream: str, record: dict[str, Any]) -> None:
        encoded = {name: self.encoder.encode(value) for name, value in record.items()}
        self.buffers[stream].append(encoded)
        self.buffer_size += _memory_size(encoded)
        if len(self.buffers[stream]) >= self.rows or self.buffer_size >= self.size:
            self.flush()

    def flush(self) -> None:
        tables = {}
        for stream, records in self.buffers.items():
            if records:
                try:
                    table = self.pa.Table.from_pylist(records)
                    previous = self.schemas.get(stream)
                    schema = table.schema if previous is None else self.pa.unify_schemas(
                        [previous, table.schema], promote_options="permissive"
                    )
                except (self.pa.ArrowInvalid, self.pa.ArrowTypeError) as exc:
                    raise TypeError("RunLog values need consistent typed adapter codecs") from exc
                tables[stream] = table
                self.schemas[stream] = schema
        # Finalize all referenced payloads before publishing any record part.
        self.payloads.publish()
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
            self.buffers[stream].clear()
        self.buffer_size = 0
        if tables:
            # This remains an incomplete checkpoint until finish publishes the
            # final status. Recovery also scans parts in case a process exits
            # after their rename but before this checkpoint is replaced.
            self._publish_manifest()

    def write_manifest(self, log: RunLog) -> None:
        metadata = self.encoder.encode(log.metadata)
        self.payloads.publish()
        previous = self.manifest
        self.manifest = {
            "schema_version": SCHEMA_VERSION, "status": log.status, "metadata": metadata,
        }
        try:
            self._publish_manifest()
        except BaseException:
            self.manifest = previous
            raise

    def _publish_manifest(self) -> None:
        manifest = {
            **self.manifest,
            "step_parts": self.counts["steps"], "event_parts": self.counts["events"],
            "step_rows": self.rows_written["steps"], "event_rows": self.rows_written["events"],
            "last_durable_step": self.last_durable_step,
        }
        pending = self.root / "manifest.json.open"
        with pending.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, self.root / "manifest.json")

    def abort(self) -> None:
        self.payloads.abort()
        self.buffers = {"steps": [], "events": []}
        self.buffer_size = 0


def _memory_size(value: Any) -> int:
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(sys.getsizeof(key) + _memory_size(item) for key, item in value.items())
    elif isinstance(value, list):
        size += sum(_memory_size(item) for item in value)
    return size
