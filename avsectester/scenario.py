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

import logging
import secrets
import time
from contextlib import ExitStack
from copy import deepcopy
from typing import Any

import avcarla  # noqa: F401  (CarlaClient / CarlaMobileActor / CarlaLidar / CarlaNpc)

# importing these registers the modules in avstack's MODELS/PIPELINE/HOOKS and avcarla's CARLA
import avstack.modules.control.vehicle
import avstack.modules.perception.object3d
import avstack.modules.pipeline
import avstack.modules.planning.vehicle
import avstack.modules.tracking.tracker3d  # noqa: F401  (BasicBoxTracker3D)
import numpy as np
from avcarla.config import CARLA
from avstack.config import HOOKS

import avsectester.attacks  # noqa: F401  (registers PhantomInjection et al. in avstack HOOKS)

# The driving record lives in the backend-agnostic data plane; re-exported here for compatibility.
from avsectester.plane import FrameRecord, Trace


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


def _npc_specs(spec: Any) -> list[dict]:
    if not spec:
        return []
    if isinstance(spec, dict):  # compact form: {count, npc_type, spawn_start}
        start = spec.get("spawn_start", 1)
        spec = [
            {"type": "CarlaNpc", "spawn": start + i, "npc_type": spec.get("npc_type", "vehicle")}
            for i in range(spec["count"])
        ]
    return list(spec)


def prepare_scenario(scenario: dict) -> dict:
    """Resolve random choices once for a pair of runs on a dedicated CARLA server.

    Explicit actor settings are preserved. The returned configuration contains no live
    simulator objects and can be reused after each world reset.
    """
    resolved = deepcopy(scenario)
    config = resolved["client"]
    if config.get("seed") is None:
        config["seed"] = secrets.randbelow(2**31)
    if config.get("traffic_manager_seed") is None:
        config["traffic_manager_seed"] = config["seed"]
    if config.get("reset_world") is False:
        raise ValueError("Paired experiments require reset_world")
    config["reset_world"] = True
    config.setdefault("strict_spawn", False)
    client = CARLA.build(deepcopy(config))
    try:
        config.update(client.scene_settings())
        rng = np.random.RandomState(config["seed"])
        blueprints = list(client.world.get_blueprint_library().filter("vehicle"))
        sorted_blueprints = sorted(blueprints, key=lambda bp: bp.id)
        actors = [resolved["ego"], *_npc_specs(resolved.get("npcs"))]
        # Reserve explicit indices before choosing any random spawn, including later NPCs.
        used = {actor["spawn"] for actor in actors if isinstance(actor.get("spawn"), int)}
        for actor in actors:
            field = "vehicle" if actor is resolved["ego"] else "npc_type"
            vehicle = actor.get(field, "random")
            if vehicle in ("random", "randint", "vehicle", "random-vehicle"):
                actor[field] = sorted_blueprints[int(rng.randint(len(sorted_blueprints)))].id
            elif isinstance(vehicle, int):
                actor[field] = blueprints[vehicle].id
            if actor.get("spawn") in ("random", "randint"):
                available = [i for i in range(len(client.spawn_points)) if i not in used]
                if not available:
                    raise ValueError("Not enough unoccupied spawn points for the scenario")
                actor["spawn"] = available[int(rng.randint(len(available)))]
                used.add(actor["spawn"])
            if isinstance(actor.get("destination"), str) and actor["destination"] in (
                "random",
                "randint",
            ):
                actor["destination"] = int(rng.randint(len(client.spawn_points)))
            for sensor in actor.get("sensors", []):
                # Stable source IDs also prevent IDs from drifting across repeated runs.
                if sensor.get("source_ID") is None:
                    sensor["source_ID"] = 0
                if sensor.get("type") == "CarlaLidar" and sensor.get("noise_seed") is None:
                    sensor["noise_seed"] = int(rng.randint(2**31))
        resolved["npcs"] = actors[1:]
    finally:
        client.close()
    return resolved


def _destroy_npc(npc):
    try:
        npc.destroy()
    except Exception:
        logging.getLogger(__name__).warning("Could not destroy NPC", exc_info=True)


def _spawn_config(spec: dict, actor: Any, vehicle_field: str) -> dict:
    """Capture the successful spawn before driving, in native CARLA world coordinates."""
    resolved = deepcopy(spec)
    transform = actor.spawn_transform
    resolved[vehicle_field] = actor.actor.type_id
    # Store the final transform, including spawn offsets and any retry displacement.
    # Replay bypasses those offsets rather than applying them a second time.
    resolved["spawn_transform"] = {
        "location": {axis: getattr(transform.location, axis) for axis in ("x", "y", "z")},
        "rotation": {axis: getattr(transform.rotation, axis) for axis in ("pitch", "yaw", "roll")},
    }
    return resolved


def run_scenario(
    scenario: dict,
    attacks: list[dict] | None = None,
    frames: int = 40,
    settle_iters: int = 100,
) -> Trace:
    """Build the avcarla closed loop from ``scenario`` config, optionally attach ``attacks`` (avstack
    hooks) to named pipeline stages, drive ``frames`` steps, and return the driving :class:`Trace`.

    The returned ``replay_scenario`` records actual spawn transforms for a subsequent run.
    Initial spawning allows relocation on failure unless ``client.strict_spawn`` is enabled;
    recorded transforms are always replayed strictly.
    """
    scenario = deepcopy(scenario)
    client_config = scenario["client"]
    client_config.setdefault("reset_world", True)
    client_config.setdefault("strict_spawn", False)
    replay = deepcopy(scenario)
    replay["client"]["strict_spawn"] = True
    replay["npcs"] = []
    trace = Trace(replay_scenario=replay)
    with ExitStack() as resources:
        client = CARLA.build(client_config)
        resources.callback(client.close)
        ego = CARLA.build(scenario["ego"], default_args={"client": client})
        resources.callback(ego.destroy)
        replay["ego"] = _spawn_config(scenario["ego"], ego, "vehicle")
        npcs = []
        for spec in _npc_specs(scenario.get("npcs")):
            npc = CARLA.build(dict(spec), default_args={"client": client})
            resources.callback(_destroy_npc, npc)
            npcs.append(npc)
            replay["npcs"].append(_spawn_config(spec, npc, "npc_type"))

        # Attacks are attached only to this run's freshly built pipeline.
        for atk in attacks or []:
            stage = getattr(ego.pipeline, atk["stage"])
            stage.register_post_hook(HOOKS.build(deepcopy(atk["hook"])))
        counter = _DetectionCounter()
        ego.pipeline.perception.register_post_hook(counter)

        snap = client.world.get_snapshot()
        ego.initialize(snap.timestamp.elapsed_seconds, snap.frame)
        for npc in npcs:
            npc.initialize(snap.timestamp.elapsed_seconds, snap.frame)

        for i in range(frames):
            client.tick()
            for _ in range(settle_iters):  # Allow asynchronous sensor delivery before driving.
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
    return trace
