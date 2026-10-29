"""Preparation-owned caching of simulator-generated frame data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from avsectester.datasets.base import ScenarioStart
from avsectester.closed_loop.simulators import FrameDataSequence


def fingerprint(value: Any) -> str:
    def encode(item: Any) -> Any:
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Reference cache key cannot contain {type(item).__name__}")

    encoded = json.dumps(value, default=encode, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class FrameDataCache:
    """Cache simulator-generated frame sequences independently of the dataset.

    A failed generation is never published. The identity includes the complete recipe,
    simulator adapter/configuration/version and seed, supplied explicitly by preparation.
    """

    def __init__(self) -> None:
        self._sequences: dict[str, FrameDataSequence] = {}

    @staticmethod
    def key(start: ScenarioStart, simulator_identity: Mapping[str, Any], seed: int) -> str:
        """Hash start kind, scenario, payload, simulator identity and seed."""
        return fingerprint({"kind": start.kind, "scenario": start.scenario_id,
                            "payload": start.payload, "simulator": simulator_identity, "seed": seed})

    def get_or_generate(
        self, key: str, generate: Callable[[], FrameDataSequence],
    ) -> FrameDataSequence:
        """Return cached frame data or generate and retain it once for this key."""
        if key not in self._sequences:
            sequence = generate()
            if not sequence.frames or any(record.frame.position != i
                                          for i, record in enumerate(sequence.frames)):
                raise ValueError("Generated frame data must cover contiguous frame positions")
            self._sequences[key] = deepcopy(sequence)
        return deepcopy(self._sequences[key])

    def get(self, key: str) -> FrameDataSequence:
        """Return a copy of the cached sequence. Raise ``KeyError`` when the key is absent."""
        return deepcopy(self._sequences[key])


__all__ = [
    'FrameDataCache',
]
