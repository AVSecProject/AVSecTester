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
from avstack.config import HOOKS, PIPELINE

import avsectester.attacks.phantom  # noqa: F401  (registers PhantomInjection et al. in avstack HOOKS)
from avsectester.backend import AVStack, WorldBackend, run
from avsectester.plane import Control, Observation, Trace


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


def _destroy_actor(actor):
    """Best-effort teardown for a raw CARLA actor (patch prop or a synthesized lead vehicle)."""
    try:
        actor.destroy()
    except Exception:
        logging.getLogger(__name__).warning("Could not destroy CARLA actor", exc_info=True)


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


class CarlaBackend(WorldBackend):
    """A CARLA :class:`~avsectester.backend.WorldBackend`.

    Owns the client, the ego vehicle + its sensors, and the NPC traffic, and applies control through
    CARLA's shared physics. The ego's own avcarla pipeline is a no-op placeholder: driving is
    *external* (the :class:`AVStack`), so this backend only senses (``_observe``) and actuates
    (``step`` -> ``apply_control``). ``reset`` also captures a strict-spawn ``replay_scenario`` so a
    paired run reproduces the same scene.
    """

    def __init__(
        self, scenario: dict, settle_iters: int = 100, patches: list[dict] | None = None
    ) -> None:
        self.scenario = deepcopy(scenario)
        self.settle_iters = settle_iters
        # Physical-patch attacks are a WORLD-level seam applied at reset (attached to a target
        # vehicle), distinct from perturb(Observation) and the modular hooks. `patches` is passed
        # explicitly for the attacked run so a paired clean run stays clean; None means no patch.
        self.patch_specs = list(patches) if patches else []
        # An optional stationary lead vehicle spawned directly ahead of the ego, present in BOTH the
        # clean and attacked runs (scene content). A physical patch (attacked run) attaches to it, so
        # the only difference between the paired runs is the patch itself.
        self.lead_cfg = self.scenario.get("lead")
        self.lead = None
        self.client = None
        self.ego = None
        self.npcs: list = []
        self.replay_scenario: dict | None = None
        self._t0: float | None = None
        self._resources = ExitStack()

    def reset(self) -> Observation:
        scenario = self.scenario
        client_config = scenario["client"]
        client_config.setdefault("reset_world", True)
        client_config.setdefault("strict_spawn", False)
        replay = deepcopy(scenario)
        replay["client"]["strict_spawn"] = True
        replay["npcs"] = []

        # Register teardown as we build, so a failure mid-setup still cleans up (LIFO): npcs (best
        # effort) then ego (propagates) then client -- matching the paired-run cleanup contract.
        self.client = CARLA.build(client_config)
        self._resources.callback(self.client.close)
        # Driving is external: give the CARLA ego an empty pipeline so its own _tick never drives.
        ego_cfg = deepcopy(scenario["ego"])
        ego_cfg["pipeline"] = {"type": "SerialPipeline", "modules": []}
        self.ego = CARLA.build(ego_cfg, default_args={"client": self.client})
        self._resources.callback(self.ego.destroy)
        replay["ego"] = _spawn_config(scenario["ego"], self.ego, "vehicle")
        for spec in _npc_specs(scenario.get("npcs")):
            npc = CARLA.build(dict(spec), default_args={"client": self.client})
            self.npcs.append(npc)
            self._resources.callback(_destroy_npc, npc)
            replay["npcs"].append(_spawn_config(spec, npc, "npc_type"))
        self.replay_scenario = replay

        if self.lead_cfg is not None:  # scene content: a stationary lead ahead of the ego (both runs)
            self.lead = self._spawn_lead(self.lead_cfg)
        self._apply_patches()  # world-level physical patches (attacked run only)

        snap = self.client.world.get_snapshot()
        self.ego.initialize(snap.timestamp.elapsed_seconds, snap.frame)
        for npc in self.npcs:
            npc.initialize(snap.timestamp.elapsed_seconds, snap.frame)
        self.client.tick()  # produce the first sensor frame (ego stationary until first control)
        return self._observe()

    def step(self, control: Control) -> Observation:
        self.ego.apply_control(control)  # decision from the AVStack; CARLA physics does the rest
        self.client.tick()
        return self._observe()

    def _observe(self) -> Observation:
        for _ in range(self.settle_iters):  # await asynchronous sensor delivery for this frame
            if not self.ego.sensor_data_manager.empty():
                break
            time.sleep(0.005)
        # Refresh the ego body frame so the sensor->global reference chain is current this frame.
        pose = self.ego.get_pose()
        self.ego.reference.x = pose.position.x
        self.ego.reference.q = pose.attitude.q
        snap = self.client.world.get_snapshot()
        t = snap.timestamp.elapsed_seconds
        self._t0 = t if self._t0 is None else self._t0
        self.ego.timestamp = t
        sensor_data = self.ego.sensor_data_manager.pop()
        state = self.ego.get_object_state()
        return Observation(
            t=t,
            frame=snap.frame,
            sensor_data=sensor_data,
            calibration={},  # modular stack reads references off sensor_data; kept for the contract
            vehicle_state=state,
            ego_speed=float(state.velocity.norm()),
        )

    def _apply_patches(self) -> None:
        """Attach + paint each configured physical patch onto its target vehicle (attacked run)."""
        if not self.patch_specs:
            return
        from avsectester.attacks.physical_patch import build_patch

        world = self.client.world
        for spec in self.patch_specs:
            spec = dict(spec)
            target = self._resolve_patch_target(spec)
            spawned = build_patch(spec).apply(world, target)
            for actor in spawned:
                self._resources.callback(_destroy_actor, actor)

    def _resolve_patch_target(self, spec: dict) -> Any:
        """Resolve a patch's ``target``: the scene ``lead`` car, the ``ego``, or ``npc:<i>``."""
        target = spec.get("target", "lead")
        if target == "ego":
            return self.ego.actor
        if target.startswith("npc:"):
            return self.npcs[int(target.split(":", 1)[1])].actor
        if target == "lead":
            if self.lead is None:
                raise ValueError("patch target 'lead' needs a `lead:` section in the scenario")
            return self.lead
        raise ValueError(f"unknown patch target {target!r} (use 'lead', 'ego', or 'npc:<i>')")

    def _spawn_lead(self, cfg: dict) -> Any:
        """Spawn a stationary lead vehicle ``cfg['gap']`` metres directly ahead of the ego."""
        import carla

        world = self.client.world
        gap = float(cfg.get("gap", 9.0))
        vehicle = str(cfg.get("vehicle", "vehicle.tesla.model3"))
        # Use the recorded spawn transform (valid before the first world tick, unlike the live actor
        # transform which still reads the origin at this point).
        ego_tf = self.ego.spawn_transform
        fwd = ego_tf.get_forward_vector()
        loc = carla.Location(
            ego_tf.location.x + fwd.x * gap,
            ego_tf.location.y + fwd.y * gap,
            ego_tf.location.z + 0.3,
        )
        bp = world.get_blueprint_library().filter(vehicle)[0]
        lead = world.try_spawn_actor(bp, carla.Transform(loc, ego_tf.rotation))
        if lead is None:
            raise RuntimeError(f"could not spawn lead vehicle {gap} m ahead (spawn point blocked)")
        self._resources.callback(_destroy_actor, lead)
        world.tick()
        return lead

    def close(self) -> None:
        # Runs the registered teardown in LIFO order; ego-destroy failures propagate, npc/client
        # failures are swallowed (npc via _destroy_npc, client.close best-effort registration).
        self._resources.close()


class ModularAVStack(AVStack):
    """An :class:`~avsectester.backend.AVStack` backed by an avstack ``ModularDrivingPipeline``.

    Observation -> the pipeline (perception -> tracking -> planning -> control) -> a Control command.
    White-box modular attacks attach as avstack hooks on a stage via :meth:`attach` (e.g.
    ``PhantomInjection`` on ``perception``); universal sensor/world attacks instead use ``run``'s
    ``perturb`` seam. A detection counter (attached last, after any attack) records per-frame
    detection counts for telemetry.
    """

    def __init__(self, pipeline_cfg: dict) -> None:
        self.pipeline = PIPELINE.build(deepcopy(pipeline_cfg))
        self._counter: _DetectionCounter | None = None
        self.detection_counts: list[int] = []

    def attach(self, stage: str, hook_cfg: dict) -> None:
        getattr(self.pipeline, stage).register_post_hook(HOOKS.build(deepcopy(hook_cfg)))

    def attach_counter(self) -> None:
        """Register the detection counter last, so it counts any attack-injected detections too."""
        self._counter = _DetectionCounter()
        self.pipeline.perception.register_post_hook(self._counter)

    def __call__(self, observation: Observation) -> Control:
        ctrl = self.pipeline(observation.sensor_data, observation.vehicle_state)
        if self._counter is not None:
            self.detection_counts.append(self._counter.last)
        return Control(
            throttle=float(ctrl.throttle), steer=float(ctrl.steer), brake=float(ctrl.brake)
        )


def run_scenario(
    scenario: dict,
    attacks: list[dict] | None = None,
    frames: int = 40,
    settle_iters: int = 100,
    patches: list[dict] | None = None,
) -> Trace:
    """Run the CARLA + modular demo through the generic interface and return the driving Trace.

    Assembles a :class:`CarlaBackend` (world + ego + traffic) and a :class:`ModularAVStack` (the AV
    box), attaches any modular ``attacks`` as hooks on the stack's pipeline, and drives them with
    :func:`avsectester.backend.run`. ``patches`` are world-level physical-patch attacks applied by the
    backend at reset (pass them only for the attacked run). ``replay_scenario`` (actual spawn
    transforms) is carried on the returned Trace for a paired run; strict-spawn replay is always used.
    """
    scenario = deepcopy(scenario)
    backend = CarlaBackend(scenario, settle_iters=settle_iters, patches=patches)
    stack = ModularAVStack(scenario["ego"]["pipeline"])
    for atk in attacks or []:
        stack.attach(atk["stage"], atk["hook"])
    stack.attach_counter()
    try:
        trace = run(backend, stack, frames)
        trace.replay_scenario = backend.replay_scenario
        for record, n in zip(trace.records, stack.detection_counts):
            record.n_detections = n
    finally:
        backend.close()
    return trace
