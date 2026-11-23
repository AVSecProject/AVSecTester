"""What a hosted ROS 2 driving system routes through the framework's relay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avsectester.closed_loop.simulators.messages import MessageStage, require_absolute_topic


PRIVATE_NAMESPACE = "/avsectester/intercepted"
"""Namespace of the private topics that redirected producers publish to."""

def private_topic(name: str) -> str:
    """Return the private ingress name for an absolute ROS topic name."""
    require_absolute_topic(name)
    return PRIVATE_NAMESPACE + name


@dataclass(frozen=True)
class InterceptedTopic:
    """One original topic whose producers publish on its private topic.

    ``publishers`` lists the expected producer node names on the private topic. ``sensor`` names
    the native rig sensor that produces the topic, if any. A passthrough topic is republished
    unchanged and is not exposed as a message stage.
    """

    name: str
    type: str
    publishers: tuple[str, ...]
    sensor: str | None = None
    passthrough: bool = False

    def __post_init__(self) -> None:
        private_topic(self.name)
        if not self.type or not self.publishers:
            raise ValueError(f"{self.name}: an intercepted topic requires a type and publishers")


@dataclass(frozen=True)
class PoseProjection:
    """Apply each pose topic change to one transform pair of a redirected TF publisher.

    The relay pairs a transform with the pose message that has the same header stamp. When a
    later pose arrives first, the transform uses the latest change.
    """

    pose_topic: str
    publisher: str
    parent_frame: str
    child_frame: str

    @property
    def private_topic(self) -> str:
        """Private topic carrying the redirected publisher's TF messages."""
        return private_topic("/tf")


@dataclass(frozen=True)
class InterceptionDeclaration:
    """Redirected and observed topics and an optional pose projection for a hosted driver.

    ``observed_sensors`` associates public observation topics with known native rig sensors.
    """

    topics: tuple[InterceptedTopic, ...]
    projection: PoseProjection | None = None
    observed_topics: tuple[str, ...] = ()
    observed_sensors: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        names = [topic.name for topic in self.topics]
        if len(set(names)) != len(names):
            raise ValueError("Intercepted topics must be unique")
        for name in self.observed_topics:
            require_absolute_topic(name)
        if len(set(self.observed_topics)) != len(self.observed_topics):
            raise ValueError("Observed topics must be unique")
        observed_sensor_topics = [topic for topic, _ in self.observed_sensors]
        if (len(set(observed_sensor_topics)) != len(observed_sensor_topics)
                or not set(observed_sensor_topics).issubset(self.observed_topics)):
            raise ValueError("Observed sensor associations must refer to distinct observed topics")
        modified = {topic.name for topic in self.topics if not topic.passthrough}
        overlap = sorted(modified.intersection(self.observed_topics))
        if overlap:
            raise ValueError(f"Topics cannot be both modified and observed: {', '.join(overlap)}")
        if self.projection is not None and self.projection.pose_topic not in modified:
            raise ValueError("A pose projection requires its pose topic as a modifiable message stage")

    @property
    def stages(self) -> dict[str, MessageStage]:
        """Message stages keyed by original topic, including read-only observations."""
        observed_sensors = dict(self.observed_sensors)
        stages = {name: MessageStage(sensor=observed_sensors.get(name), modifiable=False)
                  for name in self.observed_topics}
        stages.update({topic.name: MessageStage(topic.sensor)
                       for topic in self.topics if not topic.passthrough})
        return dict(sorted(stages.items()))

    def relay_configurations(self, *, startup_timeout_s: float) -> list[dict[str, Any]]:
        """Return one relay configuration per active role, forwarding before observation."""
        configurations: list[dict[str, Any]] = []
        projection = self.projection
        if self.topics:
            configurations.append({
                "role": "forwarding",
                "topics": [{"name": topic.name, "type": topic.type, "publishers": list(topic.publishers),
                            "passthrough": topic.passthrough} for topic in self.topics],
                "observed_topics": [],
                "projection": None if projection is None else {
                    "pose_topic": projection.pose_topic, "private_topic": projection.private_topic,
                    "publisher": projection.publisher, "parent_frame": projection.parent_frame,
                    "child_frame": projection.child_frame,
                },
                "startup_timeout_s": startup_timeout_s,
            })
        if self.observed_topics:
            configurations.append({
                "role": "observation",
                "topics": [],
                "observed_topics": list(self.observed_topics),
                "projection": None,
                "startup_timeout_s": startup_timeout_s,
            })
        return configurations


__all__ = [
    "PRIVATE_NAMESPACE",
    "InterceptedTopic",
    "InterceptionDeclaration",
    "PoseProjection",
    "private_topic",
]
