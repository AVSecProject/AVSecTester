"""Metric — score a clean vs attacked (or defended) run."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .trace import Trace


class Metric(ABC):
    @abstractmethod
    def compute(self, clean: Trace, attacked: Trace) -> dict[str, Any]:
        """Return a JSON-serializable dict of named metric values."""
