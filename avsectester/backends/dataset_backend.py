"""Offline dataset backend (wraps avstack-api / avapi).

Phase 2 (PLAN.md). Component-level evaluation + open-loop replay over KITTI / nuScenes
via ``avapi``'s scene-manager / dataset classes. Useful for cheap, GPU-light attack
generation and component-level metrics before closed-loop search.

TODO(phase2): iterate frames, feed sensor data through an avstack perception pipeline,
apply attack/monitor hooks, yield per-frame records.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..config import BACKENDS
from ..core.interfaces import Backend


@BACKENDS.register_module()
class DatasetBackend(Backend):
    def __init__(self, dataset: str = "kitti", data_dir: str = "", split: str = "val") -> None:
        self.dataset = dataset
        self.data_dir = data_dir
        self.split = split
        self._manager = None  # avapi scene manager

    def build(self, spec) -> None:
        raise NotImplementedError("phase 2: open dataset via avapi")

    def step(self) -> dict[str, Any]:
        raise NotImplementedError("phase 2: next frame")

    def run(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError("phase 2")

    def close(self) -> None:
        self._manager = None
