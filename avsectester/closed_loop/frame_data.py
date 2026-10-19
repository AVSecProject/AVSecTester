"""Simulator-generated frame data, owned by experiment preparation, and selected-start replay inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np

from avsectester.datasets.base import FrameReference, FrameData, ScenarioStart


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


@dataclass(frozen=True)
class Initialization:
    """Source replay inputs only. No attack origin, score or placement enters the simulator."""

    expected: FrameData
    resolved_scenario: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameDataSequence:
    frames: tuple[FrameData, ...]
    resolved_scenario: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def at(self, frame: FrameReference) -> FrameData:
        if frame.position >= len(self.frames) or self.frames[frame.position].frame != frame:
            raise ValueError("Reference cache does not match the requested source frame")
        return deepcopy(self.frames[frame.position])


class FrameDataCache:
    """Owned by preparation; no mutation or hidden attachment to a Dataset.

    A failed generation is never published. The identity includes the complete recipe,
    simulator adapter/configuration/version and seed, supplied explicitly by preparation.
    """

    def __init__(self) -> None:
        self._sequences: dict[str, FrameDataSequence] = {}

    @staticmethod
    def key(start: ScenarioStart, simulator_identity: Mapping[str, Any], seed: int) -> str:
        return fingerprint({"schema": 1, "kind": start.kind, "scenario": start.scenario_id,
                            "payload": start.payload, "simulator": simulator_identity, "seed": seed})

    def get_or_generate(
        self, key: str, generate: Callable[[], FrameDataSequence],
    ) -> FrameDataSequence:
        if key not in self._sequences:
            sequence = generate()
            if not sequence.frames or any(record.frame.position != i
                                          for i, record in enumerate(sequence.frames)):
                raise ValueError("Generated frame data must cover contiguous frame positions")
            self._sequences[key] = deepcopy(sequence)
        return deepcopy(self._sequences[key])

    def get(self, key: str) -> FrameDataSequence:
        return deepcopy(self._sequences[key])
