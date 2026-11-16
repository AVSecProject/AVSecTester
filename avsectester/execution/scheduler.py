"""Deterministic event ordering for the closed-loop execution loop."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import heapq
from itertools import count
from typing import Any


@dataclass(order=True, frozen=True)
class _Event:
    time_us: int
    priority: int
    sequence: int
    value: Any = field(compare=False)


class EventQueue:
    """Advance simulation time only by removing ordered integer-time events."""

    def __init__(self, origin_us: int) -> None:
        self.time_us = origin_us
        self._sequence = count()
        self._events: list[_Event] = []

    def schedule(self, time_us: int, value: Any = None, *, priority: int = 10) -> None:
        if isinstance(time_us, bool) or not isinstance(time_us, int):
            raise ValueError("Event times must be integer microseconds")
        if time_us < self.time_us:
            raise ValueError("An event cannot precede the event currently being processed")
        heapq.heappush(self._events, _Event(time_us, priority, next(self._sequence), value))

    def __iter__(self) -> Iterator[_Event]:
        while self._events:
            event = heapq.heappop(self._events)
            self.time_us = event.time_us
            yield event



