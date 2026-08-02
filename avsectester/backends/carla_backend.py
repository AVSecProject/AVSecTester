"""Closed-loop CARLA backend (wraps lib-avstack-carla / avcarla).

Phase 2 (PLAN.md). Wraps ``avcarla.client.CarlaClient`` + ``avcarla.sensors`` + actors,
drives the world in **synchronous mode**, and runs an avstack pipeline as the ego AV.

TODO(phase2):
  - Build ego + NPCs + sensors from ScenarioSpec.
  - Register AVSecTester attack/defense/monitor hooks onto the pipeline modules.
  - Deterministic per-tick sensor alignment (match on frame id).
  - Emit per-frame records {sensor_in, module_io, ego_state, events}.

Coordinate-frame note: CARLA is left-handed; mmdet3d expects right-handed (KITTI/nuScenes).
Rely on avstack ``geometry/refchoc`` and add round-trip tests (PLAN.md risks).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ..config import BACKENDS
from ..core.interfaces import Backend


@BACKENDS.register_module()
class CarlaBackend(Backend):
    def __init__(self, connect_ip: str = "127.0.0.1", connect_port: int = 2000,
                 town: str = "Town10HD", fixed_delta_seconds: float = 0.05) -> None:
        self.connect_ip = connect_ip
        self.connect_port = connect_port
        self.town = town
        self.fixed_delta_seconds = fixed_delta_seconds
        self._client = None  # avcarla.CarlaClient

    def build(self, spec) -> None:
        raise NotImplementedError("phase 2: build ego/scenario via avcarla")

    def step(self) -> dict[str, Any]:
        raise NotImplementedError("phase 2: world.tick() + drain sensor queues")

    def run(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError("phase 2")

    def close(self) -> None:
        if self._client is not None:
            self._client = None
