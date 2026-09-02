"""Trace — the per-frame record of one run, consumed by metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trace:
    """An ordered list of per-frame record dicts (one per tick)."""

    records: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = "run"

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def series(self, key: str) -> list[Any]:
        return [r.get(key) for r in self.records]

    def count(self, key: str) -> int:
        """Number of records where ``record[key]`` is truthy."""
        return sum(1 for r in self.records if r.get(key))

    def __len__(self) -> int:
        return len(self.records)
