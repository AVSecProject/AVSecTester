"""Environment — the common bridge over dataset replay and simulation.

An environment is a sequential source of :class:`Frame`\\ s. It is the *only* place the two
worlds differ: a dataset replays recorded frames and ignores the control fed back to it;
a simulator applies the control and advances its state to produce the next frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .frame import Frame


class Environment(ABC):
    @abstractmethod
    def reset(self) -> Frame:
        """Initialize and return the first frame."""

    @abstractmethod
    def step(self, control: Any = None) -> tuple[Frame, bool]:
        """Advance one tick; return ``(next_frame, done)``.

        Dataset replay ignores ``control`` (returns the next recorded frame); a simulator
        applies ``control`` and advances the world.
        """

    def close(self) -> None:
        """Tear down actors/connections (optional)."""
