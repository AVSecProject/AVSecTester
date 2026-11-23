"""Relay intercepted ROS messages between an isolated driver and its simulator."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import suppress
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py import set_message_fields
from rosidl_runtime_py.utilities import get_message

from .channel import RelayConnection
from .declaration import private_topic
from .projection import Pose, PoseDelta, ProjectionQueue
from .ros_messages import header_stamp_ns, publishable_fields, to_fields


_TF_TYPE = "tf2_msgs/msg/TFMessage"
# Only bounds a transform's wait when its pose topic stops publishing.
_PAIRING_TIMEOUT_S = 1.0
_GRAPH_INTERVAL_S = 0.1
_UNRESOLVED_NODE = "/_NODE_NAMESPACE_UNKNOWN_/_NODE_NAME_UNKNOWN_"


def _node_name(endpoint: Any) -> str:
    return "/" + "/".join(part.strip("/") for part in (endpoint.node_namespace, endpoint.node_name)
                          if part.strip("/"))


def _identity(endpoint: Any) -> tuple[str, str]:
    return _node_name(endpoint), endpoint.topic_type


def _qos_key(profile: Any) -> tuple[Any, ...]:
    fields = (profile.history, profile.depth, profile.reliability, profile.durability,
              profile.liveliness, profile.avoid_ros_namespace_conventions)
    return (*fields, profile.deadline.nanoseconds, profile.lifespan.nanoseconds,
            profile.liveliness_lease_duration.nanoseconds)


def _pose(position: Any, rotation: Any) -> Pose:
    return Pose((position.x, position.y, position.z), (rotation.x, rotation.y, rotation.z, rotation.w))


@dataclass
class _Pending:
    raw: bytes
    message: Any
    stamp_ns: int | None
    received_ns: int


@dataclass
class _Topic:
    name: str
    type_name: str
    publishers: tuple[str, ...]
    passthrough: bool = False
    observed: bool = False
    message_type: Any = None
    publisher: Any = None
    subscription: Any = None
    waiting: deque[int] = field(default_factory=deque)
    trailing: deque[bytes] = field(default_factory=deque)
    pre_origin: int = 0
    post_run: int = 0
    messages: int = 0
    modified: int = 0
    max_queue: int = 0
    latency_ns: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        values = self.latency_ns
        latency = {}
        if values:
            latency = {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
                       "max": max(values)}
        return {"pre_origin_messages": self.pre_origin, "post_run_messages": self.post_run,
                "messages": self.messages,
                "modified_messages": self.modified, "max_queue_length": self.max_queue,
                "latency_ns": latency}


class _TfRelay:
    def __init__(self, configuration: dict[str, Any]) -> None:
        # Keys follow InterceptionDeclaration.relay_configurations.
        self.pose_topic = configuration["pose_topic"]
        self.private_topic = configuration["private_topic"]
        self.parent, self.child = configuration["parent_frame"], configuration["child_frame"]
        self.expected = configuration["publisher"], _TF_TYPE
        self.message_type = get_message(_TF_TYPE)
        self.publisher: Any = None
        self.subscription: Any = None
        self.queue: ProjectionQueue[tuple[bytes, Any, tuple[int, ...]]] = ProjectionQueue(_PAIRING_TIMEOUT_S)
        self.seen = False
        self.counts = dict(pre_origin=0, post_run=0, forwarded=0, modified=0, paired=0, fallback=0)

    def discover(self, node: Node, callback: Any) -> str | None:
        endpoints = node.get_publishers_info_by_topic(self.private_topic)
        observed = [_identity(endpoint) for endpoint in endpoints]
        if observed != [self.expected]:
            return (f"{self.private_topic}: expected one producer {self.expected}, "
                    f"found {observed}")
        if self.publisher is None:
            profile = endpoints[0].qos_profile
            self.publisher = node.create_publisher(self.message_type, "/tf", profile)
            self.subscription = node.create_subscription(
                self.message_type, self.private_topic, callback, profile, raw=True)
        return None if self.seen else f"{self.private_topic}: no {self.parent}→{self.child} received"

    def receive(self, raw: bytes, *, origin: bool, stopped: bool) -> None:
        if stopped:
            # A trailing message cannot overtake one still waiting for its active pose.
            self.counts["post_run"] += 1
            self.queue.enqueue((raw, None, ()), (), time.monotonic())
            return
        message = deserialize_message(raw, self.message_type)
        indices = tuple(i for i, item in enumerate(message.transforms)
                        if item.header.frame_id == self.parent and item.child_frame_id == self.child)
        self.seen |= bool(indices)
        if not origin:
            self.publisher.publish(raw)
            self.counts["pre_origin"] += 1
            return
        stamps = tuple(header_stamp_ns(message.transforms[i]) or 0 for i in indices)
        self.queue.enqueue((raw, message, indices), stamps, time.monotonic())

    def pose(self, stamp_ns: int | None, original: Any, replacement: Any | None) -> None:
        if stamp_ns is None:
            raise ValueError("A projected pose requires a header stamp")
        delta = None
        if replacement is not None:
            if header_stamp_ns(replacement) != stamp_ns:
                raise ValueError("Projected pose replacements must preserve their header stamp")
            before, after = original.pose.pose, replacement.pose.pose
            delta = PoseDelta.between(_pose(before.position, before.orientation),
                                      _pose(after.position, after.orientation))
        self.queue.pose(stamp_ns, delta)

    def drain(self) -> None:
        for item in self.queue.drain(time.monotonic()):
            raw, message, indices = item.message
            if message is None:
                self.publisher.publish(raw)
                continue
            modified = False
            for index, (delta, paired) in zip(indices, item.pairs):
                self.counts["paired" if paired else "fallback"] += 1
                if delta is not None:
                    transform = message.transforms[index].transform
                    pose = delta.apply(_pose(transform.translation, transform.rotation))
                    transform.translation.x, transform.translation.y, transform.translation.z = pose.position
                    (transform.rotation.x, transform.rotation.y,
                     transform.rotation.z, transform.rotation.w) = pose.rotation
                    modified = True
            self.publisher.publish(message if modified else raw)
            self.counts["forwarded"] += 1
            self.counts["modified"] += modified


class _Relay(Node):
    def __init__(self, configuration: dict[str, Any], socket_path: Path) -> None:
        role = configuration.get("role")
        if role not in ("forwarding", "observation"):
            raise ValueError(f"Expected a forwarding or observation relay role, got {role!r}")
        super().__init__(f"avsectester_interception_{role}", enable_rosout=False,
                         start_parameter_services=False)
        self.role = role
        self.topics = [_Topic(topic["name"], topic["type"], tuple(topic["publishers"]),
                              topic["passthrough"]) for topic in configuration["topics"]]
        self.observations = [
            _Topic(name, "", (), observed=True) for name in configuration.get("observed_topics", ())
        ]
        self.observation_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.tf = _TfRelay(configuration["projection"]) if configuration["projection"] is not None else None
        self.origin = False
        self.ready = False
        self.stopped = False
        self.finishing = False
        self.finished = False
        self._id = 0
        self._pending: dict[int, tuple[_Topic, _Pending]] = {}
        self._replies: dict[int, dict[str, Any]] = {}
        self._deadline = time.monotonic() + configuration["startup_timeout_s"]
        self._stable = False
        self._graph_timer = self.create_timer(_GRAPH_INTERVAL_S, self.discover)
        self._tf_timer = (
            self.create_timer(_PAIRING_TIMEOUT_S, self._expire_tf, autostart=False)
            if self.tf is not None else None
        )
        self._tf_deadline: float | None = None
        incoming = self.create_guard_condition(self.frames)
        # The reader can wake the executor as soon as construction returns.
        self.connection = RelayConnection(socket_path, notify=incoming.trigger)

    def discover(self) -> None:
        reasons = [reason for topic in self.topics if (reason := self._discover_topic(topic))]
        if self.observations:
            types = dict(self.get_topic_names_and_types())
            reasons.extend(reason for topic in self.observations
                           if (reason := self._discover_observation(topic, types.get(topic.name, []))))
        if self.tf is not None:
            reason = self.tf.discover(self, self._receive_tf)
            if reason is not None:
                reasons.append(reason)
        if reasons:
            self._stable = False
            if time.monotonic() >= self._deadline:
                raise RuntimeError("ROS interception graph validation failed: " + "; ".join(reasons))
        elif self._stable:
            self.ready = True
            self._graph_timer.cancel()
            self.connection.send({"kind": "ready"})
        else:
            self._stable = True

    def _discover_topic(self, topic: _Topic) -> str | None:
        original = self.get_publishers_info_by_topic(topic.name)
        unexpected = [_identity(endpoint) for endpoint in original
                      if _node_name(endpoint) not in (self.get_fully_qualified_name(), _UNRESOLVED_NODE)]
        if unexpected:
            raise RuntimeError(f"{topic.name}: original topic still has non-relay publishers {unexpected}")
        private = private_topic(topic.name)
        endpoints = self.get_publishers_info_by_topic(private)
        observed = Counter(_identity(endpoint) for endpoint in endpoints)
        expected = Counter((node, topic.type_name) for node in topic.publishers)
        if observed != expected:
            # DDS can report a writer before the ROS graph identifies its node. A placeholder
            # cannot satisfy the expected set, but it also is not evidence of a different node.
            resolved = Counter(_identity(endpoint) for endpoint in endpoints
                               if _node_name(endpoint) != _UNRESOLVED_NODE)
            if resolved - expected:
                raise RuntimeError(f"{private}: expected producers {list(expected.elements())}, "
                                   f"found {list(observed.elements())}")
            return (f"{private}: expected producers {list(expected.elements())}, "
                    f"found {list(observed.elements())}")
        profiles = [_qos_key(endpoint.qos_profile) for endpoint in endpoints]
        if any(qos != profiles[0] for qos in profiles[1:]):
            raise RuntimeError(f"{private}: producer QoS differs: {list(zip(observed, profiles))}")
        if topic.publisher is None:
            profile = endpoints[0].qos_profile
            topic.message_type = get_message(topic.type_name)
            topic.publisher = self.create_publisher(topic.message_type, topic.name, profile)
            topic.subscription = self.create_subscription(
                topic.message_type, private, self._callback(topic), profile, raw=True)
        names = [_node_name(endpoint) for endpoint in original]
        return (None if names == [self.get_fully_qualified_name()]
                else f"{topic.name}: relay publisher not yet uniquely discovered: {names}")

    def _discover_observation(self, topic: _Topic, types: list[str]) -> str | None:
        if not types:
            return f"{topic.name}: observed topic has no type in the ROS graph"
        if len(types) != 1 or topic.type_name not in ("", types[0]):
            raise RuntimeError(
                f"{topic.name}: expected one stable type for observation, found {types}"
            )
        if topic.subscription is None:
            topic.type_name = types[0]
            topic.message_type = get_message(topic.type_name)
            topic.subscription = self.create_subscription(
                topic.message_type, topic.name, self._callback(topic), self.observation_qos, raw=True)
        return None

    def _receive_tf(self, raw: bytes) -> None:
        if self.tf is not None:
            try:
                self.tf.receive(
                    raw, origin=self.origin, stopped=self.stopped
                )
                self._drain_tf()
                self._send_statistics()
            except Exception as error:
                raise RuntimeError(f"{self.tf.private_topic}: {error}") from error

    def _callback(self, topic: _Topic) -> Any:
        def receive(raw: bytes) -> None:
            try:
                self._receive(topic, raw)
            except Exception as error:
                raise RuntimeError(f"{topic.name}: {error}") from error

        return receive

    def _receive(self, topic: _Topic, raw: bytes) -> None:
        if not self.origin or self.stopped or topic.passthrough:
            if self.stopped:
                topic.post_run += 1
            elif not self.origin:
                topic.pre_origin += 1
            else:
                topic.messages += 1
            if not topic.observed:
                if topic.waiting:
                    topic.trailing.append(raw)
                else:
                    topic.publisher.publish(raw)
            return
        received_ns = time.time_ns()
        message = deserialize_message(raw, topic.message_type)
        fields = to_fields(message, topic.type_name)
        stamp_ns = header_stamp_ns(message)
        frame = {"topic": topic.name, "type": topic.type_name, "fields": fields,
                 "received_ns": received_ns, "stamp_ns": stamp_ns}
        if topic.observed:
            topic.messages += 1
            self.connection.send({"kind": "observed", **frame})
            return
        identity = self._id
        self._id += 1
        self._pending[identity] = topic, _Pending(raw, message, stamp_ns, received_ns)
        topic.waiting.append(identity)
        topic.max_queue = max(topic.max_queue, len(topic.waiting))
        self.connection.send({"kind": "message", "id": identity, **frame})

    def _publish(self, topic: _Topic, identity: int) -> None:
        _, pending = self._pending.pop(identity)
        reply = self._replies.pop(identity)
        outside_run = reply.get("outside_run", False)
        replacement = None
        if not outside_run and "fields" in reply:
            replacement = topic.message_type()
            set_message_fields(replacement, publishable_fields(reply["fields"], topic.type_name))
        if self.tf is not None and self.tf.pose_topic == topic.name:
            self.tf.pose(pending.stamp_ns, pending.message, replacement)
        topic.publisher.publish(replacement if replacement is not None else pending.raw)
        sent_ns = time.time_ns()
        if outside_run:
            topic.post_run += 1
        else:
            topic.messages += 1
            topic.modified += replacement is not None
            topic.latency_ns.append(sent_ns - pending.received_ns)
        self.connection.send({"kind": "published", "id": identity, "sent_ns": sent_ns})

    def frames(self) -> None:
        for frame in self.connection.drain():
            kind = frame["kind"]
            if kind == "origin":
                self.origin = True
            elif kind == "stop":
                self.stopped = True
            elif kind == "finish":
                self.stopped = True
                self.finishing = True
            elif kind == "reply":
                identity = frame["id"]
                topic, _ = self._pending[identity]
                self._replies[identity] = frame
                try:
                    while topic.waiting and topic.waiting[0] in self._replies:
                        self._publish(topic, topic.waiting.popleft())
                    if not topic.waiting:
                        while topic.trailing:
                            topic.publisher.publish(topic.trailing.popleft())
                except Exception as error:
                    raise RuntimeError(f"{topic.name}: {error}") from error
            else:
                raise RuntimeError(f"Unknown ROS relay frame kind {kind!r}")
        self._drain_tf()
        self._send_statistics()

    def _expire_tf(self) -> None:
        assert self._tf_timer is not None
        self._tf_timer.cancel()
        self._tf_deadline = None
        self._drain_tf()
        self._send_statistics()

    def _drain_tf(self) -> None:
        if self.tf is None:
            return
        self.tf.drain()
        deadline = self.tf.queue.next_deadline
        if deadline == self._tf_deadline:
            return
        self._tf_deadline = deadline
        timer = self._tf_timer
        assert timer is not None
        if deadline is None:
            timer.cancel()
        else:
            timer.timer_period_ns = max(1, int((deadline - time.monotonic()) * 1e9))
            timer.reset()

    def _send_statistics(self) -> None:
        if (self.finishing and not self._pending and not self.finished
                and (self.tf is None or not self.tf.queue.waiting)):
            self.finished = True
            self.connection.send({"kind": "statistics", "role": self.role,
                                  "topics": {topic.name: topic.summary()
                                             for topic in (*self.topics, *self.observations)},
                                  "tf": dict(self.tf.counts) if self.tf is not None else {}})


def main() -> None:
    """Run the relay against its session socket and its configuration file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.configuration.read_text())
    rclpy.init()
    relay = None
    try:
        relay = _Relay(configuration, args.socket)
        rclpy.spin(relay)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    except Exception as error:
        if relay is not None:
            with suppress(Exception):
                relay.connection.send({"kind": "error", "reason": str(error)})
                relay.connection.flush(0.5)
        raise
    finally:
        if relay is not None:
            relay.connection.close()
            relay.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
