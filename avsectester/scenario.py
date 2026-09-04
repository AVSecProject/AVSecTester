"""Closed-loop security scenario: a real avstack pipeline in real CARLA, attacked by avstack hooks.

A scenario is built **entirely from config through avstack/avcarla's own registries** — there is no
parallel environment/system machinery here:

* ``avcarla.CarlaClient``      owns the world, traffic manager, sync + ticking.
* ``avcarla.CarlaMobileActor`` is the ego: sensors (``CarlaLidar``) + an avstack
  ``ModularDrivingPipeline`` (neural perception -> tracking -> planning -> control) + control.
* ``avcarla.CarlaNpc``         is the background traffic.
* an **attack** is an ``avstack.HOOKS`` hook attached to a named pipeline stage (e.g.
  ``PhantomInjection`` on ``perception``) — the same mechanism avstack uses for any pre/post hook.

Running a scenario clean and then attacked, and diffing the driving record, is the whole security
test (see :mod:`avsectester.metric`). Everything below is real: no mock, no stubs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import avcarla  # noqa: F401  (CarlaClient / CarlaMobileActor / CarlaLidar / CarlaNpc)

# importing these registers the modules in avstack's MODELS/PIPELINE/HOOKS and avcarla's CARLA
import avstack.modules.control.vehicle
import avstack.modules.perception.object3d
import avstack.modules.pipeline
import avstack.modules.planning.vehicle
import avstack.modules.tracking.tracker3d  # noqa: F401  (BasicBoxTracker3D)
from avcarla.config import CARLA
from avstack.config import HOOKS

import avsectester.attacks  # noqa: F401  (registers PhantomInjection et al. in avstack HOOKS)


@dataclass
class FrameRecord:
    frame: int
    t: float
    n_detections: int
    speed: float
    throttle: float
    brake: float
    steer: float


@dataclass
class Trace:
    """The driving record of one run — the observable a security test scores."""

    records: list[FrameRecord] = field(default_factory=list)

    @property
    def final_speed(self) -> float:
        return self.records[-1].speed if self.records else 0.0

    @property
    def braking_frames(self) -> int:
        return sum(r.throttle == 0.0 and r.brake > 0.0 for r in self.records)

    @property
    def mean_detections(self) -> float:
        return sum(r.n_detections for r in self.records) / max(len(self.records), 1)


class _DetectionCounter:
    """A trivial avstack post-hook that records how many detections perception emitted."""

    def __init__(self) -> None:
        self.last = 0

    def __call__(self, detections: Any) -> tuple[Any]:
        self.last = len(detections)
        return (detections,)


def set_perception_gpu(scenario: dict, gpu: int | None) -> dict:
    """Override the perception stage's CUDA device in a scenario config, in place.

    Useful for host (non-Docker) runs: the scenario config targets ``gpu: 0`` for the container, but
    on a single host CARLA already renders on GPU 0, so point neural inference at a free GPU to avoid
    contention. No-op when ``gpu`` is None or the config has no such stage.
    """
    if gpu is not None:
        try:
            scenario["ego"]["pipeline"]["perception"]["gpu"] = gpu
        except (KeyError, TypeError):
            pass
    return scenario


def _build_npcs(spec: Any, client: Any) -> list:
    if not spec:
        return []
    if isinstance(spec, dict):  # compact form: {count, npc_type, spawn_start}
        start = spec.get("spawn_start", 1)
        spec = [
            {"type": "CarlaNpc", "spawn": start + i, "npc_type": spec.get("npc_type", "vehicle")}
            for i in range(spec["count"])
        ]
    return [CARLA.build(dict(n), default_args={"client": client}) for n in spec]


def run_scenario(
    scenario: dict,
    attacks: list[dict] | None = None,
    frames: int = 40,
    settle_iters: int = 100,
) -> Trace:
    """Build the avcarla closed loop from ``scenario`` config, optionally attach ``attacks`` (avstack
    hooks) to named pipeline stages, drive ``frames`` steps, and return the driving :class:`Trace`."""
    client = CARLA.build(dict(scenario["client"]))
    ego = CARLA.build(dict(scenario["ego"]), default_args={"client": client})
    npcs = _build_npcs(scenario.get("npcs"), client)

    # attacks are avstack hooks on pipeline stages; a detection counter rides behind them
    for atk in attacks or []:
        stage = getattr(ego.pipeline, atk["stage"])
        stage.register_post_hook(HOOKS.build(dict(atk["hook"])))
    counter = _DetectionCounter()
    ego.pipeline.perception.register_post_hook(counter)

    snap = client.world.get_snapshot()
    ego.initialize(snap.timestamp.elapsed_seconds, snap.frame)
    for npc in npcs:
        npc.initialize(snap.timestamp.elapsed_seconds, snap.frame)

    trace = Trace()
    try:
        for i in range(frames):
            client.tick()
            for _ in range(settle_iters):  # await this frame's async sensor delivery
                if not ego.sensor_data_manager.empty():
                    break
                time.sleep(0.005)
            snap = client.world.get_snapshot()
            ctrl = ego.tick(snap.timestamp.elapsed_seconds, snap.frame)
            state = ego.get_object_state()
            trace.records.append(
                FrameRecord(
                    frame=i,
                    t=snap.timestamp.elapsed_seconds,
                    n_detections=counter.last,
                    speed=float(state.velocity.norm()),
                    throttle=float(getattr(ctrl, "throttle", 0.0)),
                    brake=float(getattr(ctrl, "brake", 0.0)),
                    steer=float(getattr(ctrl, "steer", 0.0)),
                )
            )
    finally:
        ego.destroy()
        for npc in npcs:
            try:
                npc.destroy()
            except Exception:  # noqa: BLE001, S110 — best-effort teardown
                pass
    return trace
