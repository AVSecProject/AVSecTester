"""Everything CARLA: the :class:`CarlaBackend` WorldBackend + its view adapters for the viz pipeline.

Heavy imports (``avcarla`` / ``carla`` / the avstack CARLA registry) are lazy — done inside
``prepare_scenario`` and ``CarlaBackend.reset`` — so this module and the ``avsectester.simulators``
package import without a CARLA stack (the pure view adapters and config helpers stay offline-usable).

  * View adapters (:func:`camera_view`, :func:`lidar_bev`) feed the simulator-agnostic pipeline in
    :mod:`avsectester.simulators.viz`; they duck-type avstack ``ImageData`` / ``LidarData``.
  * :class:`CarlaBackend` owns the client, ego (+ sensors), traffic, an optional scene ``lead`` car,
    and applies physical-patch attacks at reset. Compose it with a stack via
    :func:`avsectester.scenario.run_scenario`.
"""

from __future__ import annotations

import logging
import secrets
import time
from contextlib import ExitStack
from copy import deepcopy
from typing import Any

from avsectester.backend import WorldBackend
from avsectester.plane import Control, Observation
from avsectester.simulators.viz import as_rgb


# ---------------------------------------------------------------------------------------------------
# View adapters (simulator-specific unwrapping for the generic viz pipeline)
# ---------------------------------------------------------------------------------------------------
def camera_view(observation: Observation, camera: str | None = None) -> Any:
    """Extract an RGB frame from a CARLA ``CarlaRgbCamera`` payload (avstack ``ImageData``)."""
    import numpy as np

    data = observation.sensor_data
    if not data:
        return None
    key = camera if camera in data else next(iter(data))
    frame = data[key]
    rgb = getattr(frame, "rgb_image", None)  # avstack ImageData -> channel-correct RGB
    if rgb is not None:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3].astype("uint8")
    return as_rgb(frame)  # already a raw ndarray


def lidar_bev(observation: Observation, size: int = 800, meters: float = 60.0) -> Any:
    """A top-down bird's-eye view of a CARLA ego lidar cloud (x forward = up, y left = left)."""
    import numpy as np

    data = next(iter(observation.sensor_data.values()), None)
    try:  # avstack LidarData wraps a CARLA measurement whose raw_data is float32 [x,y,z,intensity]
        raw = data.data.raw_data
        pts = np.frombuffer(bytes(raw), dtype=np.float32).reshape(-1, 4)[:, :3]
    except Exception:  # noqa: BLE001 - best-effort: any non-lidar/unparseable payload -> skip
        return None
    img = np.zeros((size, size, 3), dtype=np.uint8)
    scale = size / (2 * meters)
    u = (size / 2 - pts[:, 0] * scale).astype(int)  # forward -> up
    v = (size / 2 - pts[:, 1] * scale).astype(int)  # left -> left
    m = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    img[u[m], v[m]] = (0, 255, 0)
    img[size // 2 - 3 : size // 2 + 3, size // 2 - 3 : size // 2 + 3] = (255, 80, 80)  # ego
    return img


def _order_quad(pts: Any) -> Any:
    """Order 4 image points into (TL, TR, BR, BL) by pixel position — the compositor's quad order.

    Geometric ordering (not physical-corner order) so the warped patch stays upright and unmirrored
    regardless of the target's world orientation relative to the camera.
    """
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]  # x - y
    return np.stack([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


def lead_rear_quad(
    backend: Any, camera: str | None = None, width_frac: float = 0.8, height_frac: float = 0.55
) -> Any:
    """Return ``quad_of(observation) -> (4,2) px | None`` for the lead car's rear face (live actors).

    The backend-specific projection injected into :func:`avsectester.simulators.viz.composite_view`:
    reads the live CARLA camera + lead vehicle actors off ``backend`` each frame, builds the rear-face
    quad from the lead's bounding box (a centered panel scaled by ``width_frac`` / ``height_frac``),
    and projects it through the camera to pixels. Yields None when the face is behind the camera or
    off-frame, so the compositor leaves that frame clean. Camera intrinsics come from the sensor's own
    projection matrix ``P``; extrinsics from the live camera transform (CARLA UE axis convention).
    """
    import numpy as np

    from avsectester.attacks.patch_composite import carla_cam_coords, project_to_pixels

    def _quad_of(_observation: Observation) -> Any:
        import carla  # noqa: F401 - carla.Location used below

        lead = backend.lead
        if lead is None:  # also covers pre-reset: ego/lead not spawned yet
            return None
        sensors = backend.ego.sensors  # read lazily: actors exist only after backend.reset()
        sensor = sensors[camera] if camera in sensors else next(iter(sensors.values()))
        K = np.asarray(sensor.P)[:, :3]  # avcarla packs intrinsics in the 3x4 projection matrix
        h_img, w_img = sensor.imsize
        bb = lead.bounding_box
        xr = bb.location.x - bb.extent.x  # rear face plane (vehicle local, +x = forward)
        ey, ez = bb.extent.y * width_frac, bb.extent.z * height_frac
        cy, cz = bb.location.y, bb.location.z
        local = [(xr, cy - ey, cz + ez), (xr, cy + ey, cz + ez),
                 (xr, cy + ey, cz - ez), (xr, cy - ey, cz - ez)]
        tf = lead.get_transform()
        world = np.array([[(p := tf.transform(carla.Location(*c))).x, p.y, p.z] for c in local])
        inv = np.array(sensor.object.get_transform().get_inverse_matrix())
        cam_pts = carla_cam_coords(world, inv)
        if np.any(cam_pts[:, 2] <= 0.1):  # any corner behind the camera -> skip this frame
            return None
        px = project_to_pixels(cam_pts, K)
        if px[:, 0].max() < 0 or px[:, 0].min() > w_img or px[:, 1].max() < 0 or px[:, 1].min() > h_img:
            return None
        return _order_quad(px)

    return _quad_of


# ---------------------------------------------------------------------------------------------------
# Scenario config helpers (pure) + preparation (CARLA)
# ---------------------------------------------------------------------------------------------------
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
    import numpy as np
    from avcarla.config import CARLA

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


# ---------------------------------------------------------------------------------------------------
# The CARLA WorldBackend
# ---------------------------------------------------------------------------------------------------
class CarlaBackend(WorldBackend):
    """A CARLA :class:`~avsectester.backend.WorldBackend`.

    Owns the client, the ego vehicle + its sensors, and the NPC traffic, and applies control through
    CARLA's shared physics. The ego's own avcarla pipeline is a no-op placeholder: driving is
    *external* (the :class:`~avsectester.backend.AVStack`), so this backend only senses (``_observe``)
    and actuates (``step`` -> ``apply_control``). ``reset`` also captures a strict-spawn
    ``replay_scenario`` so a paired run reproduces the same scene, and applies any physical patches.
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
        from avcarla.config import CARLA

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
