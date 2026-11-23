"""Process native driver messages independently of simulator ticks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from threading import Event
from time import monotonic, time_ns
from typing import Any

import numpy as np

from avsectester.closed_loop.hooks import HookDispatcher
from avsectester.closed_loop.simulators.messages import MessageChannel
from avsectester.closed_loop.simulators.base import SensorView, SimulatorRuntime
from avsectester.data_types import RosMessage
from avsectester.execution.attack_dispatch import AttackDispatcher
from avsectester.runlog import MessageRecord
from avsectester.serialization import DEFAULT_REGISTRY
from avsectester.structured_arrays import item_bytes


def _same_fields(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and item_bytes(a) == item_bytes(b)
    if isinstance(a, Mapping):
        return a.keys() == b.keys() and all(_same_fields(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_fields(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        # Compare bit patterns so NaN payloads and signed zeros count as changes.
        return np.float64(a).tobytes() == np.float64(b).tobytes()
    return bool(a == b)


class MessageProcessor:
    """Handle native driver messages on the channel's reader threads."""

    def __init__(self, channel: MessageChannel, simulator: SimulatorRuntime,
                 hooks: HookDispatcher, attacks: AttackDispatcher,
                 payload_topics: frozenset[str]) -> None:
        self.channel = channel
        self.simulator = simulator
        self.hooks = hooks
        self.attacks = attacks
        self.payload_topics = payload_topics
        self._step: int | None = None
        self._closed = False
        self._finished = Event()
        self._failure: Exception | None = None
        self._pending: dict[tuple[int, int], dict[str, Any]] = {}
        self._records: list[MessageRecord] = []
        self._post_run_observations: Counter[str] = Counter()
        self._reports: dict[int, dict[str, Any]] = {}
        self.statistics: dict[str, Any] | None = None

    def start(self, *, has_decisions: bool) -> None:
        """Start delivering messages and activate decision zero before issuing its actions."""
        self._step = 0 if has_decisions else None
        self.channel.start(self.handle)
        if has_decisions:
            self.channel.send({"kind": "origin"})

    def switch(self, step: int | None) -> None:
        """Set the active decision while the caller holds the shared callback lock."""
        stopping = step is None and self._step is not None
        self._step = step
        if stopping:
            self.channel.send({"kind": "stop"})

    def drain_records(self) -> list[MessageRecord]:
        """Detach records for the runner, which alone writes to the log."""
        with self.hooks.callback_lock:
            records, self._records = self._records, []
        return records

    def check(self) -> None:
        """Raise a processing failure before checking the transport."""
        if self._failure is not None:
            raise self._failure
        self.channel.check()

    def handle(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        """Process a frame and return the reply to write on its originating connection."""
        with self.hooks.callback_lock:
            if self._closed or self._failure is not None:
                return None
            try:
                return self._process(packet)
            except Exception as error:
                self._failure = error
                self._finished.set()
                return None

    def _process(self, packet: dict[str, Any]) -> dict[str, Any] | None:
        kind = packet["kind"]
        if kind in {"message", "observed"}:
            topic = packet["topic"]
            started_wall, started = time_ns(), monotonic()
            try:
                return self._message(packet, kind == "message", started_wall, started)
            except Exception as error:
                raise RuntimeError(f"{topic}: ROS message processing failed: {error}") from error
        if kind == "published":
            identity = packet["relay_index"], packet["id"]
            record = self._pending.pop(identity, None)
            if record is not None:
                self._records.append(MessageRecord(**record, relay_sent_ns=packet["sent_ns"]))
        elif kind == "statistics":
            relay = packet["relay_index"]
            if relay in self._reports:
                raise RuntimeError(f"ROS relay {relay} sent duplicate statistics")
            self._reports[relay] = packet
        else:
            raise RuntimeError(f"Unexpected ROS relay packet {kind!r}")
        if len(self._reports) == self.channel.relay_count and self._reports and not self._pending:
            self._merge_statistics()
            self._finished.set()
        return None

    def _merge_statistics(self) -> None:
        topics: dict[str, Any] = {}
        tf: dict[str, Any] = {}
        # A public companion topic may be both transparently forwarded and observed.
        # Its message stage is the observer's, so use that peer's count for the stage.
        for packet in sorted(self._reports.values(), key=lambda report: report.get("role") == "observation"):
            topics.update(packet["topics"])
            tf.update(packet["tf"])
        # One-way observations sent before the stop signal can reach us after the last decision.
        for topic, count in self._post_run_observations.items():
            topics[topic]["messages"] -= count
            topics[topic]["post_run_messages"] += count
        self.statistics = {"topics": topics, "tf": tf}

    def _message(self, packet: dict[str, Any], modifiable: bool,
                 started_wall: int, started: float) -> dict[str, Any] | None:
        topic = packet["topic"]
        spec = self.simulator.message_stages.get(topic)
        if spec is None or spec.modifiable != modifiable:
            raise ValueError(f"Unexpected message for {topic}")
        reply = {"kind": "reply", "id": packet["id"]} if modifiable else None
        if self._step is None:
            if reply is not None:
                return {**reply, "outside_run": True}
            self._post_run_observations[topic] += 1
            return None
        snapshot = self.simulator.message_snapshot
        if snapshot is None:
            raise RuntimeError("No completed simulator snapshot is available")
        time_us, state = snapshot
        value = RosMessage(packet["type"], packet["fields"])
        original = DEFAULT_REGISTRY.snapshot(value) if modifiable else value
        sensor = spec.sensor
        if modifiable and sensor is not None and state.ego is not None and self.attacks.attacks:
            world_from_sensor = state.ego.pose @ self.simulator.message_sensor_poses[sensor]
            view = SensorView(sensor, value, None, None, deepcopy(state.actors), deepcopy(state.ego),
                              state.t, state.frame, sensor_from_world=np.linalg.inv(world_from_sensor))
            value = self.attacks.apply_message_sensor_actions(topic, value, view, time_us)
        value = self.hooks.emit_message(topic, value, state, step=self._step, outputs=self.hooks.outputs)
        changed = modifiable and not _same_fields(original.fields, value.fields)
        forwarded = DEFAULT_REGISTRY.snapshot(value) if changed else None
        record: dict[str, Any] = {
            "topic": topic, "type": value.type, "step": self._step, "stamp_ns": packet.get("stamp_ns"),
            "arrival_time_us": time_us, "relay_received_ns": packet["received_ns"],
            "framework_received_ns": packet["read_ns"], "processing_started_ns": started_wall,
            "processing_s": monotonic() - started, "modified": changed, "modifiable": modifiable,
        }
        if topic in self.payload_topics:
            record["original"] = original
            if forwarded is not None:
                record["forwarded"] = forwarded
        if reply is None:
            self._records.append(MessageRecord(**record, relay_sent_ns=None))
            return None
        identity = packet["relay_index"], packet["id"]
        if identity in self._pending:
            raise ValueError(f"Duplicate message identity {identity}")
        self._pending[identity] = record
        if forwarded is not None:
            reply["fields"] = forwarded.fields
        return reply

    def finish(self) -> None:
        """Drain publication acknowledgments and retrieve relay totals."""
        self.check()
        self.channel.send({"kind": "finish"})
        finished = self._finished.wait(5.)
        self.check()
        if not finished:
            raise TimeoutError("ROS relay did not finish forwarding pending messages")

    def close(self) -> None:
        """Disable delivery before closing hooks or the runtime transport."""
        with self.hooks.callback_lock:
            self._closed = True
