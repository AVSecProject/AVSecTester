"""Message stages and their transport contract for simulator integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re
from typing import Any


_ABSOLUTE_TOPIC = re.compile(r"/[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*")


def require_absolute_topic(name: str) -> None:
    """Require an absolute topic whose components start with a letter or underscore."""
    if not isinstance(name, str) or not _ABSOLUTE_TOPIC.fullmatch(name):
        raise ValueError(f"Expected an absolute ROS topic name: {name!r}")


@dataclass(frozen=True)
class MessageStage:
    """One message topic's callback capabilities.

    ``sensor`` identifies the native rig sensor when known. An observed stage exposes
    the received message to read-only callbacks.
    """

    sensor: str | None = None
    modifiable: bool = True


class MessageChannel(ABC):
    """Exchange framed messages with relays owned by the simulator.

    The framework broadcasts ``origin``, ``stop`` and ``finish`` to mark the start and end
    of message processing and to request terminal statistics. A relay sends ``ready`` after
    discovery or ``error`` on failure. It sends ``message`` to request a ``reply`` on the same
    connection and ``published`` to acknowledge that reply. The message ID matches both
    frames. An ``observed`` frame supplies a passive observation and expects no reply.
    Each relay sends one ``statistics`` frame after ``finish`` and its pending publications.
    Incoming frames carry ``read_ns``, the wall-clock time when the transport read the frame,
    and ``relay_index``, the zero-based identity of the connection for matching replies.
    """

    @abstractmethod
    def start(self, handler: Callable[[dict[str, Any]], Mapping[str, Any] | None]) -> None:
        """Deliver incoming message frames on the transport readers.

        Only relay readiness and errors may be received before this call. A returned frame
        is written directly to the originating relay. The handler owns processing failures.
        """

    @abstractmethod
    def send(self, frame: Mapping[str, Any]) -> None:
        """Queue a control frame without blocking on I/O.

        Every expected relay must have connected.
        """

    @abstractmethod
    def check(self) -> None:
        """Raise the first saved relay or transport failure."""

    @property
    @abstractmethod
    def relay_count(self) -> int:
        """Number of connected relays expected to report terminal statistics."""


__all__ = ["MessageChannel", "MessageStage", "require_absolute_topic"]
