"""Everything CARLA: :class:`CarlaSimulator`, its live :class:`CarlaSimulatorRuntime`, and view adapters.

Heavy imports (``avcarla`` / ``carla`` / the avstack CARLA registry) are lazy — done inside
``prepare_scenario`` and ``CarlaSimulatorRuntime.reset`` — so this module and the
``avsectester.closed_loop.simulators`` package import without a CARLA stack (the pure view
adapters and config helpers stay offline-usable).

  * View adapters (:func:`camera_view`, :func:`lidar_bev`) feed the simulator-agnostic pipeline in
    :mod:`avsectester.closed_loop.simulators.viz`. They duck-type avstack ``ImageData`` / ``LidarData``.
  * :class:`CarlaSimulatorRuntime` owns the client, ego (+ sensors), traffic, an optional scene
    ``lead`` car, and exposes simulator stages. ``PhysicalPatchEffect`` deploys panels at ``simulator.setup``.
    Compose the simulator with an AVStack via
    :func:`avsectester.carla_runner.run_scenario`.
"""

from __future__ import annotations

import logging
import secrets
import time
from contextlib import ExitStack
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from dataclasses import asdict, replace
import math
import numpy as np
from avsectester.datasets.base import ActorState, SensorDataReference, FrameData
from avsectester.datasets.carla import CarlaRecipe
from avsectester.closed_loop.frame_data import Initialization
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.simulators.viz import as_rgb


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


def carla_sensor_view(context, camera, observation, true_frame):
    """Build the sensor view of one CARLA camera capture from a same-frame selection context.

    ``context`` is a :class:`~avsectester.scenarios.context.FilterContext` for the current
    world frame, as made by ``carla_filter_context``. Depth is included when the context
    holds aligned depth for ``camera``.
    """
    from avsectester.perturbations.image import read_camera_rgb
    from avsectester.rendering.cameras import camera_from_calibration
    from avsectester.closed_loop.simulator import SensorView

    if context.scene.frame != true_frame:
        raise ValueError("Insertion geometry and the physical world must use the same frame")
    context = context.derive()  # derived geometry is cached; read the current scene
    source_key = context.native.get("camera_sources", {}).get(camera, camera)
    if source_key not in observation.sensor_data:
        raise ValueError(f"Missing insertion camera {camera!r} (sensor {source_key!r})")
    payload_frame = getattr(observation.sensor_data[source_key], "frame", None)
    frame_offset = context.native.get("camera_frame_offsets", {}).get(camera)
    if payload_frame is not None and frame_offset is not None:
        if payload_frame + frame_offset != true_frame:
            raise ValueError("Insertion camera payload and geometry must use the same frame")
    captured = read_camera_rgb(observation, source_key)
    if captured is None:
        raise ValueError(f"Missing insertion camera {camera!r}")
    calibration = context.scene.cameras[camera]
    depth = context.native.get("depth", {}).get(camera)
    return SensorView(
        source_key, captured[1], camera_from_calibration(calibration),
        np.linalg.inv(context.world_from_ego @ calibration.cam_to_ego), dict(context.actors),
        context.victim, context.scene.t, context.scene.frame,
        depth=None if depth is None else np.asarray(depth, dtype=float), native=context.native,
    )


def insertion_perturbation(simulator_runtime, insertions, *, bindings=None, camera="front",
                           compositor=None):
    """Legacy observation transform rendering explicit insertions into a CARLA camera image."""
    return _insertion_renderer(simulator_runtime, insertions, bindings=bindings, camera=camera,
                               compositor=compositor)


def _insertion_renderer(simulator_runtime, insertions, *, bindings=None, camera="front",
                        compositor=None):
    """Insert explicit, stable host-bound or world-fixed surfaces into model input.

    The runtime must provide ``selection_context()`` with current same-frame depth, as a
    runtime of ``CarlaSimulator(depth_camera=...)`` does.
    Geometry is updated each frame. Selection filters are never rerun here.
    Pass ``bindings=case.match.binding_ids`` to preserve selected role aliases across
    resets. Native logical host names such as ``lead`` need no additional mapping.
    """
    import numpy as np

    from avsectester.rendering.cameras import camera_from_calibration
    from avsectester.rendering.visibility import DepthVisibilityEstimator
    from avsectester.rendering.types import InsertionGeometry
    from avsectester.perturbations.image import read_camera_rgb, write_camera_rgb
    from avsectester.rendering.compositor import InsertionRenderer

    renderer = None
    current = None
    selected_ids = {}
    for role, ids in (bindings or {}).items():
        if isinstance(ids, str):
            raise TypeError("Each role binding must be a sequence of actor IDs, not one string")
        selected_ids[role] = tuple(ids)

    def geometry(observation):
        context = current
        calibration = context.scene.cameras[camera]
        return InsertionGeometry(context.actors, context.victim, np.linalg.inv(
            context.world_from_ego @ calibration.cam_to_ego
        ))

    def evidence_provider(observation, resolved, geometry_data):
        depth = current.native.get("depth", {}).get(camera)
        if depth is None:
            raise ValueError("CARLA insertion needs aligned depth for the selected camera")
        return {
            item.id: DepthVisibilityEstimator(tolerance_m=0.0002).estimate(
                item, renderer.camera, geometry_data.cam_from_world, scene_depth=depth,
                depth_convention="z", camera_name=camera,
                other_insertions=tuple(other for other in resolved if other.id != item.id),
            ) for item in resolved
        }

    def perturb(observation):
        nonlocal renderer, current
        current = simulator_runtime.selection_context()
        true_frame = simulator_runtime.state().frame
        if current.scene.frame != true_frame:
            raise ValueError("Insertion geometry and the physical world must use the same frame")
        if selected_ids:
            available = {actor.track_id: actor for actor in current.scene.objects}
            missing = {track_id for ids in selected_ids.values() for track_id in ids if track_id not in available}
            if missing:
                raise KeyError(f"Selected insertion actors unavailable: {sorted(missing)}")
            current = current.derive(bindings={
                role: tuple(available[track_id] for track_id in ids)
                for role, ids in selected_ids.items()
            })
        source_key = current.native.get("camera_sources", {}).get(camera, camera)
        if source_key not in observation.sensor_data:
            raise ValueError(f"Missing insertion camera {camera!r} (sensor {source_key!r})")
        payload_frame = getattr(observation.sensor_data[source_key], "frame", None)
        frame_offset = current.native.get("camera_frame_offsets", {}).get(camera)
        if payload_frame is not None and frame_offset is not None:
            if payload_frame + frame_offset != true_frame:
                raise ValueError("Insertion camera payload and geometry must use the same frame")
        got = read_camera_rgb(observation, source_key)
        if got is None:
            raise ValueError(f"Missing insertion camera {camera!r}")
        key, rgb = got
        if renderer is None:
            renderer = InsertionRenderer(
                insertions, camera_from_calibration(current.scene.cameras[camera]), geometry,
                compositor=compositor, evidence_provider=evidence_provider,
            )
        else:
            # A paired reset recreates sensors. Never retain a previous sensor's calibration.
            renderer.camera = camera_from_calibration(current.scene.cameras[camera])
        rendered = renderer(observation, rgb)
        perturb.evidence = renderer.evidence
        perturb.resolved = renderer.resolved
        return write_camera_rgb(observation, key, rendered)

    return perturb


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


def resolve_scenario_seed(scenario: dict) -> dict:
    """Copy a scene and supply a missing client seed without contacting CARLA."""
    resolved = deepcopy(scenario)
    if resolved["client"].get("seed") is None:
        resolved["client"]["seed"] = secrets.randbelow(2**31)
    return resolved


def prepare_scenario(scenario: dict) -> dict:
    """Resolve random choices once for a pair of runs on a dedicated CARLA server.

    Explicit actor settings are preserved. The returned configuration contains no live
    simulator objects and can be reused after each world reset.
    """
    import numpy as np
    from avcarla.config import CARLA

    resolved = resolve_scenario_seed(scenario)
    config = resolved["client"]
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
    """Best-effort teardown for a raw CARLA actor such as the synthesized lead vehicle."""
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


# A curated set of built-in ``carla.WeatherParameters`` presets for world-level robustness sweeps
# (pass one as the scenario ``weather`` key. See ``CarlaSimulator._apply_weather``). Any preset CARLA
# exposes also works — this is just the driving-relevant spread from clear to adverse.
CARLA_WEATHER_PRESETS = (
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon", "HardRainNoon",
    "ClearSunset", "CloudySunset", "WetSunset", "HardRainSunset",
)


# ---------------------------------------------------------------------------------------------------
# The CARLA WorldBackend
# ---------------------------------------------------------------------------------------------------
class SensorAlignmentError(RuntimeError):
    """A configured sensor has no capture for the current world frame."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = deepcopy(dict(report))
        super().__init__("Sensor capture is not aligned: " + "; ".join(report["failures"]))


_PINHOLE_ATTRIBUTES = {"lens_k": "0", "lens_kcube": "0", "lens_circle_multiplier": "0"}


def register_pinhole_camera():
    """Register the undistorted RGB sensor used with a depth camera, without importing CARLA eagerly.

    The class stays in this repository. The ordinary avcarla RGB sensor keeps its original
    behavior. Sensor factories reuse the registered subclass and its calibration.
    """
    from avcarla.config import CARLA
    from avcarla.sensors import CarlaRgbCamera

    existing = CARLA.module_dict.get("PinholeRgbCamera")
    if existing is not None:
        if existing.__module__ != __name__:
            raise ValueError("PinholeRgbCamera is already registered by another module")
        return existing

    @CARLA.register_module()
    class PinholeRgbCamera(CarlaRgbCamera):
        """An ideal pinhole RGB sensor for calibrated depth-based insertion."""

        def __init__(self, *args, do_spawn=True, **kwargs):
            super().__init__(*args, do_spawn=False, **kwargs)
            for name, value in _PINHOLE_ATTRIBUTES.items():
                self.bp.set_attribute(name, value)
            if do_spawn:
                self.spawn()

    return PinholeRgbCamera


def _use_pinhole_cameras(scenario):
    """Replace stock RGB cameras with the undistorted subclass; custom classes are kept."""
    for sensor in scenario.get("ego", {}).get("sensors", ()):
        if sensor.get("type") == "CarlaRgbCamera":
            sensor["type"] = "PinholeRgbCamera"
    return scenario


def _require_pinhole_sensor(sensor):
    """Reject camera geometry that the pinhole visibility provider cannot represent."""
    for name in _PINHOLE_ATTRIBUTES:
        value = float(sensor.attributes.get(name, -1 if name == "lens_k" else 0))
        if not np.isfinite(value) or value != 0:
            raise ValueError(
                f"Selection requires an undistorted camera ({name}=0), got {value}. "
                "Use PinholeRgbCamera or supply a custom calibrated visibility provider."
            )


def decode_carla_depth(image):
    """Decode CARLA's 24-bit depth image to forward camera depth in metres."""
    bgra = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    channels = bgra.astype(np.float64)
    return (channels[..., 2] + 256 * channels[..., 1] + 65536 * channels[..., 0]) * (
        1000.0 / 16777215
    )


class CarlaSimulator(Simulator):
    """CARLA configured by an avcarla scenario: client, ego and its sensors, NPCs and a lead.

    ``create_runtime`` connects a new live world. ``depth_camera`` names an ego RGB camera that
    gets an aligned depth camera, so sensor views of it carry measured depth; stock RGB cameras
    are then replaced by undistorted ``PinholeRgbCamera`` sensors matching the pinhole
    projection. Custom RGB classes are kept and must expose zero lens distortion.
    """

    dataset_kinds = ("CarlaDataset",)
    capture_ticks = 1  # CARLA renders on ticks; the first capture follows simulator.setup
    can_render = True

    def __init__(
        self, scenario: dict, settle_iters: int = 100, *, simulator_version="0.9.16",
        depth_camera: str | None = None,
    ) -> None:
        scenario = deepcopy(scenario)
        self.scenario = _use_pinhole_cameras(scenario) if depth_camera is not None else scenario
        self.settle_iters = settle_iters
        self.simulator_version = simulator_version
        self.depth_camera = depth_camera

    def create_runtime(self, seed: int | None = None) -> CarlaSimulatorRuntime:
        """A new live world; ``seed`` becomes the client seed when the scenario sets none."""
        return CarlaSimulatorRuntime(self, seed)

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"actuator"})

    @property
    def decision_time_step(self):
        return 1 / float(self.scenario["client"]["rate"])

    @property
    def identity(self):
        return {**super().identity, "scenario": deepcopy(self.scenario),
                "depth_camera": self.depth_camera, "start_sensor_setup_ticks": 1}

    def validate_start(self, start):
        super().validate_start(start)
        if start is not None:
            recipe = start.payload.get("recipe")
            if not isinstance(recipe, CarlaRecipe):
                raise TypeError("CARLA ScenarioStart requires an explicit CarlaRecipe")
            if not math.isclose(recipe.time_step, self.decision_time_step, abs_tol=1e-9):
                raise ValueError("Dataset time_step differs from simulator decision time step")
            if start.frame.position > recipe.decision_steps:
                raise ValueError("Selected start exceeds recipe support")


class CarlaSimulatorRuntime(SimulatorRuntime):
    """One live CARLA world of a :class:`CarlaSimulator`.

    Owns the client, the ego vehicle + its sensors, and the NPC traffic, and applies control through
    CARLA's shared physics. The ego's own avcarla pipeline is a no-op placeholder: driving is
    *external* (the :class:`~avsectester.closed_loop.avstack.AVStack`), so this runtime only senses (``_observe``)
    and actuates (``step`` -> ``apply_control``). ``reset`` also captures a strict-spawn
    ``replay_scenario`` so a paired run reproduces the same initial scene.
    """

    def __init__(self, simulator: CarlaSimulator, seed: int | None = None) -> None:
        super().__init__(simulator)
        self.scenario = deepcopy(simulator.scenario)
        if seed is not None and self.scenario["client"].get("seed") is None:
            self.scenario["client"]["seed"] = seed
        self.settle_iters = simulator.settle_iters
        self.depth_camera = simulator.depth_camera
        self.depth = None
        self.depth_frame = None
        self._depth_queue = None
        self._depth_observation = None
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
        self._pair_scenario: dict | None = None
        self._pair_replay: dict | None = None
        self._recipe = None
        self._recipe_replay = None

    @property
    def resource_key(self):
        client = self.scenario["client"]
        host = client.get("connect_ip", "127.0.0.1")
        if host in {"localhost", "127.0.0.1", "::1"}:
            host = "localhost"
        return ("carla", host, client.get("connect_port", 2000))

    def configure_conditions(self, world: Mapping[str, Any]) -> None:
        """Apply CARLA weather settings at every subsequent scene reset."""
        unknown = set(world) - {"weather"}
        if unknown:
            raise ValueError(f"Unsupported CARLA world conditions: {sorted(unknown)}")
        if "weather" in world:
            weather = deepcopy(world["weather"])
            self.scenario["weather"] = weather
            # A prepared or previously run pair must not retain the old weather.
            for prepared in (self._pair_scenario, self._pair_replay, self.replay_scenario):
                if prepared is not None:
                    prepared["weather"] = deepcopy(weather)

    @property
    def native(self) -> dict[str, Any]:
        """Expose live simulator resources for explicit world-level interventions."""
        actors = {f"npc:{i}": npc.actor for i, npc in enumerate(self.npcs)}
        if self.ego is not None:
            actors["ego"] = self.ego.actor
        if self.lead is not None:
            actors["lead"] = self.lead
        return {
            "client": self.client,
            "world": None if self.client is None else self.client.world,
            "ego": self.ego,
            "lead": self.lead,
            "npcs": tuple(self.npcs),
            "actors": actors,
        }

    def _state(self) -> WorldSnapshot:
        """Snapshot the latest completed tick, or the known spawn before the first actor tick."""
        if self.client is None or self.ego is None:
            raise RuntimeError("Reset the CARLA simulator before requesting its state")
        snapshot = self.client.world.get_snapshot()
        self.ego.timestamp = snapshot.timestamp.elapsed_seconds
        state = self.ego.get_object_state()
        find = getattr(snapshot, "find", None)
        if find is not None and find(self.ego.actor.id) is None:
            from avcarla.geometry import carla_transform_to_pose

            # A newly spawned actor is absent from CARLA's last-tick cache. Use the successful
            # spawn request rather than interpreting its default cached origin as its location.
            pose = carla_transform_to_pose(self.ego.spawn_transform)
            state.position.x = pose.position.x
            state.attitude.q = pose.attitude.q
        actors = {}
        if find is not None:
            # Actors spawned after the latest tick have no simulated state yet.
            others = {f"npc:{i}": npc.actor for i, npc in enumerate(self.npcs)}
            if self.lead is not None:
                others["lead"] = self.lead
            for name, actor in others.items():
                simulated = find(actor.id)
                if simulated is not None:
                    actors[name] = _actor_state(simulated, actor.type_id)
        return WorldSnapshot(
            snapshot.timestamp.elapsed_seconds, snapshot.frame,
            deepcopy(state), float(state.velocity.norm()), actors,
        )

    def control_actor(self, actor_id, control):
        import carla

        if control.trajectory is not None:
            raise ValueError("CARLA actors take actuator commands, not trajectories")
        actor = self._take_over(actor_id)
        actor.apply_control(carla.VehicleControl(
            throttle=control.throttle, brake=control.brake, steer=control.steer))

    def set_actor_state(self, actor_id, state):
        _place(self._take_over(actor_id), state)

    def _take_over(self, actor_id):
        """Stop an actor's prescribed or traffic-manager motion so an attack can move it."""
        if actor_id == "ego":
            raise ValueError("The ego is controlled by the driving system")
        actors = self.native["actors"]
        if actor_id not in actors:
            raise KeyError(f"Unknown CARLA actor {actor_id!r}; available: {sorted(actors)}")
        actor = actors[actor_id]
        if self._recipe_replay is not None:
            self._recipe_replay.release(actor_id)
        actor.set_autopilot(False, self.client.traffic_manager_port)
        return actor

    def copy_observation(self, observation: Observation) -> Observation:
        """Detach mutable payloads, calibration and reference graphs in one copy operation.

        Native CARLA sensor captures cannot be deep-copied and are borrowed. Their metadata
        properties are read-only, but SDK item assignment may modify capture points. Replace a
        capture with a private array or measurement to preserve the original input. The Python
        wrappers, calibration and reference graphs are copied. Capture edits do not move actors.
        """
        memo = {}
        for payload in observation.sensor_data.values():
            for candidate in (payload, getattr(payload, "data", None)):
                if (
                    candidate is not None
                    and type(candidate).__module__.startswith("carla.")
                    and hasattr(candidate, "raw_data")
                ):
                    memo[id(candidate)] = candidate
        return deepcopy(observation, memo)

    def prepare_clean_attack_pair(self) -> None:
        """Resolve scene choices once. The first reset may fall back to an available spawn."""
        self.close()
        # Keep the original config so preparing another pair can resolve random choices afresh.
        self._pair_scenario = prepare_scenario(self.scenario)
        self._pair_replay = None
        self.replay_scenario = None

    def _replay_position(self, start):
        return start.frame.position if start is not None else 0

    def _resolve_scenario(self, start, initialization, environment):
        if set(environment) - {"weather"}:
            raise ValueError("CARLA environment overrides support weather only")
        if start is not None:
            if initialization is None and start.frame.position:
                raise ValueError("A selected CARLA start requires materialized reference evidence")
            if initialization is not None:
                if not isinstance(initialization, Initialization) or initialization.expected.frame != start.frame:
                    raise ValueError("Initialization evidence does not match the selected start frame")
                scenario = deepcopy(initialization.resolved_scenario)
            else:
                scenario = deepcopy(start.payload["recipe"].scenario)
                if "seed" in self.scenario["client"]:
                    scenario["client"]["seed"] = self.scenario["client"]["seed"]
                # prepare_scenario connects to resolve spawns, so it must use the factory's
                # server, never the one written in the recipe.
                self._apply_connection(scenario)
                scenario = prepare_scenario(scenario)
            # Connection settings come from the factory; scenario recipes do not choose a server.
            self._apply_connection(scenario)
        else:
            scenario = deepcopy(self._pair_replay or self._pair_scenario) if self._pair_scenario is not None else deepcopy(self.scenario)
        scenario.update(deepcopy(dict(environment)))
        if self.depth_camera is not None:
            register_pinhole_camera()
            _use_pinhole_cameras(scenario)
        return scenario

    def _apply_connection(self, scenario):
        for key in ("connect_ip", "connect_port", "traffic_manager_port"):
            if key in self.scenario["client"]:
                scenario["client"][key] = self.scenario["client"][key]

    def _initialize_world(self, start, initialization, environment):
        from avcarla.config import CARLA

        scenario = self._resolve_scenario(start, initialization, environment)
        self.close()
        self._recipe = start.payload["recipe"] if start is not None else None
        # Plain resets always use the configured origin, never a previous start's resolved world.
        self._scenario_t0 = None
        self._trajectory_setup_ticks = 0
        client_config = scenario["client"]
        client_config.setdefault("reset_world", True)
        client_config.setdefault("strict_spawn", False)
        replay = deepcopy(scenario)
        replay["client"]["strict_spawn"] = True
        replay["npcs"] = []
        self.client = CARLA.build(client_config)
        self._resources.callback(self.client.close)
        ego_cfg = deepcopy(scenario["ego"])
        ego_cfg["pipeline"] = {"type": "SerialPipeline", "modules": []}
        self.ego = CARLA.build(ego_cfg, default_args={"client": self.client})
        self._resources.callback(self.ego.destroy)
        replay["ego"] = _spawn_config(scenario["ego"], self.ego, "vehicle")
        for spec in _npc_specs(scenario.get("npcs")):
            npc = CARLA.build(dict(spec), default_args={"client": self.client})
            if self._recipe is not None:
                npc.actor.set_autopilot(False, self.client.traffic_manager_port)
            self.npcs.append(npc)
            self._resources.callback(_destroy_npc, npc)
            replay["npcs"].append(_spawn_config(spec, npc, "npc_type"))
        if self._recipe is not None:
            # Source sensors need their clocks before _spawn_lead's tick, so queued captures
            # are labelled correctly. Plain resets keep the original initialization order.
            origin = self.client.world.get_snapshot()
            self.ego.initialize(origin.timestamp.elapsed_seconds, origin.frame)
        self.replay_scenario = replay
        if start is None and self._pair_scenario is not None:
            self._pair_replay = deepcopy(replay)
        if scenario.get("lead") is not None:
            self.lead = self._spawn_lead(scenario["lead"])
        self._apply_weather(scenario.get("weather"))
        snap = self.client.world.get_snapshot()
        if self._recipe is None:
            self.ego.initialize(snap.timestamp.elapsed_seconds, snap.frame)
        for npc in self.npcs:
            npc.initialize(snap.timestamp.elapsed_seconds, snap.frame)
        self._initialize_sensors()
        if self._recipe is not None:
            self._recipe_replay = RecipeReplay(self, self._recipe)
            self._recipe_replay.initialize()
            # A base-world sensor tick for runs from a scenario start, before attacks and scenario time zero.
            # Plain resets take their first tick after simulator.setup (the capture tick).
            self.client.tick()
        if self._recipe_replay is not None:
            self._trajectory_setup_ticks = self._recipe_replay.activate()
            # GPU cameras need a second base-world tick before their first aligned
            # capture. Hold the resolved origin; this fixed preparation is identical
            # for frame data generation and replay and never counts as a decision.
            self._hold_tick()
            self._recipe_replay.finish_holding()
        self._scenario_t0 = self.client.world.get_snapshot().timestamp.elapsed_seconds
        self._base_actor_ids = ({actor.id for actor in self.client.world.get_actors()}
                                if start is not None else set())

    def _initialize_sensors(self):
        """Create the aligned depth camera, if configured, before the first capture."""
        if self.depth_camera is None:
            return
        from queue import Queue

        from avsectester.scenarios.carla_gt import carla_rgb_sensor

        rgb = carla_rgb_sensor(self, self.depth_camera)
        _require_pinhole_sensor(rgb.object)
        world = self.client.world
        blueprint = world.get_blueprint_library().find("sensor.camera.depth")
        for name in ("image_size_x", "image_size_y", "fov", "lens_k", "lens_kcube",
                     "lens_circle_falloff", "lens_circle_multiplier", "lens_x_size", "lens_y_size"):
            if name in rgb.object.attributes:
                blueprint.set_attribute(name, rgb.object.attributes[name])
        blueprint.set_attribute("sensor_tick", "0")
        # Use the spawn-time relative calibration: native get_transform may still be stale
        # before the first world tick, particularly in scenes without a lead spawn tick.
        sensor = world.spawn_actor(
            blueprint, rgb.reference.as_carla_transform(local=True), attach_to=self.ego.actor,
        )
        self._resources.callback(sensor.destroy)
        self._resources.callback(sensor.stop)
        self._depth_queue = Queue()
        sensor.listen(self._depth_queue.put)

    def _replay_tick(self):
        replay = self._recipe_replay
        replay.before_step()
        self.ego.apply_control(replay.ego_command())
        self.client.tick()
        replay.after_step()

    def _hold_tick(self):
        if self._recipe_replay is not None:
            self._recipe_replay.hold_origin()
        self.client.tick()

    def _before_selected_capture(self, position):
        if position == 0:
            self._scenario_t0 = self.client.world.get_snapshot().timestamp.elapsed_seconds
        if self._recipe_replay is not None:
            self._recipe_replay.finish_holding()

    def _check_initialization(self, initialization):
        if self._start is None:
            return
        actual = self._capture_frame_data(self._start.frame)
        if initialization is not None:
            if initialization.provenance.get("deployment_ticks", 0) != self._deployment.ticks:
                raise ValueError("Reference and run must share the deployment tick plan")
        base_ids = {actor.id for actor in self.native["actors"].values()}
        self.initialization_report.update({
            "start_frame": asdict(self._start.frame), "pre_roll_steps": self._start.frame.position,
            "deployment_ticks": self._deployment.ticks,
            "resolved_scenario": deepcopy(self.replay_scenario),
            "motion": asdict(self._recipe)["motion"], "actual": actual.provenance,
            "inserted_actor_ids": [actor.id for actor in self.client.world.get_actors()
                                   if actor.id not in self._base_actor_ids and actor.id not in base_ids],
        })
        if not self._generating_frame_data:
            # Frame data generation records the prescribed ego motion; runs drive it.
            self._recipe_replay.release_ego()

    def _advance_native(self, control):
        if self._recipe_replay is not None:
            self._recipe_replay.before_step()
        self.ego.apply_control(control)
        self.client.tick()
        if self._recipe_replay is not None:
            self._recipe_replay.after_step()

    def _capture_frame_data(self, frame):
        """Capture actual states and sensor acquisition times, preserving native IDs as evidence."""
        from avsectester.scenarios.carla_gt import carla_scene_gt

        observation = self.observe()
        scenario_t0 = self._scenario_t0
        actual_time = observation.t - scenario_t0
        actors, aliases = {}, {}
        for identity, actor in self.native["actors"].items():
            actors[identity] = _actor_state(actor, actor.type_id)
            aliases[identity] = actor.id
        sensors = {}
        required_sensors = []
        for sensor in self.ego.sensors.values():
            name = sensor.source_identifier
            required_sensors.append(name)
            payload = observation.sensor_data.get(name)
            if payload is None or getattr(payload, "timestamp", None) is None:
                continue
            acquisition = float(payload.timestamp) + float(sensor.t0)
            sensors[name] = SensorDataReference(
                f"carla://capture/{observation.frame}/{name}", name, acquisition - scenario_t0,
                {"native_frame": int(payload.frame + sensor.frame0),
                 "native_timestamp": acquisition, "observation_frame": observation.frame},
            )
        # Selection geometry uses the same existing calibrated CARLA adapter as online insertion.
        rgb = [sensor for sensor in self.ego.sensors.values()
               if sensor.object.type_id == "sensor.camera.rgb"]
        scene = carla_scene_gt(self, getattr(self, "selection_camera", "front")) if rgb else None
        return FrameData(frame, scene, actors, sensors, {
            "kind": "generated", "native_frame": observation.frame,
            "native_timestamp": observation.t, "scenario_t0": scenario_t0,
            "actual_scenario_time": actual_time,
            "actor_mapping": aliases, "required_sensors": required_sensors,
            "setup_ticks": self.setup_ticks,
            "trajectory_setup_ticks": getattr(self, "_trajectory_setup_ticks", 0),
            "start_sensor_setup_ticks": 1,
        })


    def _replayed_scenario(self):
        return deepcopy(self.replay_scenario)

    def _frame_data_provenance(self):
        return {"seed": self.replay_scenario["client"].get("seed")}

    def sensor_view(self, sensor):
        """View of the current capture of an ego RGB camera, by ego sensor name or source ID."""
        from avsectester.scenarios.carla_gt import carla_rgb_sensor
        from avsectester.scenarios.carla_provider import carla_filter_context

        names = {rgb.source_identifier: name for name, rgb in self.ego.sensors.items()}
        name = names.get(sensor, sensor)
        carla_rgb_sensor(self, name)  # reject an ambiguous or missing camera
        context = carla_filter_context(self, camera=name, depth=self._view_depth(name))
        return carla_sensor_view(context, name, self.observe(), self.state().frame)

    def _view_depth(self, camera):
        """Aligned depth for ``camera``, if this runtime measures it."""
        if camera != self.depth_camera:
            return None
        if self.depth_frame != self.client.world.get_snapshot().frame:
            raise RuntimeError("Refresh the observation before requesting a view")
        return self.depth

    def selection_context(self):
        """Legacy: a same-frame filter context with the depth camera's depth."""
        from avsectester.scenarios.carla_provider import carla_filter_context

        if self.depth_camera is None:
            raise RuntimeError("selection_context requires CarlaSimulator(depth_camera=...)")
        if self.depth_frame != self.client.world.get_snapshot().frame:
            raise RuntimeError("Refresh the observation before requesting a context")
        return carla_filter_context(
            self, camera=self.depth_camera, depth=self.depth, observation=self._depth_observation,
        )

    def _observe(self) -> Observation:
        observation = self._capture_sensors()
        if self._depth_queue is not None:
            sample = self._depth_queue.get(timeout=30)
            while sample.frame < observation.frame:
                sample = self._depth_queue.get(timeout=30)
            if sample.frame != observation.frame:
                raise RuntimeError("CARLA depth and observation frames differ")
            self.depth = decode_carla_depth(sample)
            self.depth_frame = sample.frame
            self._depth_observation = observation
        return observation

    def _capture_sensors(self) -> Observation:
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
        sensor_data = (self._aligned_sensor_data(snap.frame)
                       if self._recipe_replay is not None
                       else self.ego.sensor_data_manager.pop())
        state = self.ego.get_object_state()
        return self.copy_observation(Observation(
            t=t,
            frame=snap.frame,
            sensor_data=sensor_data,
            calibration={
                key: payload.calibration for key, payload in sensor_data.items()
                if getattr(payload, "calibration", None) is not None
            },
            vehicle_state=state,
            ego_speed=float(state.velocity.norm()),
        ))

    def _aligned_sensor_data(self, frame, initial=None):
        """Wait for this tick's configured sensors, draining older queue entries without ticking."""
        manager = self.ego.sensor_data_manager
        pending = {sensor.source_identifier: sensor for sensor in self.ego.sensors.values()}
        captured, seen = {}, {}
        for name, payload in (initial or {}).items():
            if name in pending and payload is not None:
                native_frame = payload.frame + pending[name].frame0
                seen[name] = native_frame
                if native_frame > frame:
                    raise SensorAlignmentError({
                        "failures": [f"sensor {name} is ahead of world frame"],
                        "expected_native_frame": frame, "observed_sensor_frames": seen,
                    })
                if native_frame == frame:
                    captured[name] = payload
                    del pending[name]
        deadline = time.monotonic() + max(1.0, self.settle_iters * 0.005)
        while pending:
            for name, sensor in tuple(pending.items()):
                while manager.has_data(name):
                    payload = manager.pop(name)
                    native_frame = payload.frame + sensor.frame0
                    seen[name] = native_frame
                    if native_frame == frame:
                        captured[name] = payload
                        del pending[name]
                        break
                    if native_frame > frame:
                        raise SensorAlignmentError({
                            "failures": [f"sensor {name} is ahead of world frame"],
                            "expected_native_frame": frame, "observed_sensor_frames": seen,
                        })
            if pending and time.monotonic() >= deadline:
                raise SensorAlignmentError({
                    "failures": [f"missing aligned capture: {sorted(pending)}"],
                    "expected_native_frame": frame, "observed_sensor_frames": seen,
                })
            if pending:
                time.sleep(0.005)
        return captured

    def _apply_weather(self, weather: Any) -> None:
        """Set the world weather (the CARLA-native, world-level augmentation tier). ``weather`` is a
        ``carla.WeatherParameters`` preset **name** (e.g. ``"HardRainNoon"``, ``"ClearSunset"``) or a dict
        of parameters (``cloudiness``, ``precipitation``, ``sun_altitude_angle``, ``fog_density``,
        ``wetness``, …). Applied to the whole scene — including any world-level patch — so it re-renders
        under that weather. ``None`` leaves the map's default weather."""
        if weather is None:
            return
        import carla

        if isinstance(weather, str):
            params = getattr(carla.WeatherParameters, weather, None)
            if params is None:
                raise ValueError(f"unknown CARLA weather preset {weather!r} (see CARLA_WEATHER_PRESETS)")
        else:
            params = carla.WeatherParameters(**dict(weather))
        self.client.world.set_weather(params)

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
        right = ego_tf.get_right_vector()
        lateral = float(cfg.get("lateral", 0.0))  # sideways offset -> the ego sees the rear obliquely
        loc = carla.Location(
            ego_tf.location.x + fwd.x * gap + right.x * lateral,
            ego_tf.location.y + fwd.y * gap + right.y * lateral,
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
        # ExitStack attempts every callback even if one fails. Never retain stale actors for reset.
        try:
            self._resources.close()
        finally:
            self._resources = ExitStack()
            self.client = self.ego = self.lead = None
            self.npcs.clear()
            self._t0 = None
            self._in_setup = False
            self._capture = None
            self._world_initialized = False
            self._recipe_replay = None
            self._depth_queue = None
            self.depth = self.depth_frame = self._depth_observation = None


_FLIP = np.diag([1., -1., 1., 1.])


def _actor_state(actor, type_id):
    """Canonical world state of a CARLA actor or actor snapshot, with its class (``vehicle``)."""
    velocity = actor.get_velocity()
    return ActorState(_FLIP @ np.asarray(actor.get_transform().get_matrix()) @ _FLIP,
                      (velocity.x, -velocity.y, velocity.z), type_id.split(".")[0])


def _place(actor, state):
    import carla

    pose = _FLIP @ np.asarray(state.pose, dtype=float) @ _FLIP
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Prescribed CARLA pose must be a finite 4x4 matrix")
    rotation = carla.Rotation(
        pitch=math.degrees(math.asin(float(np.clip(pose[2, 0], -1, 1)))),
        yaw=math.degrees(math.atan2(pose[1, 0], pose[0, 0])),
        roll=math.degrees(math.atan2(-pose[2, 1], pose[2, 2])),
    )
    actor.set_transform(carla.Transform(carla.Location(*pose[:3, 3]), rotation))
    vx, vy, vz = state.velocity
    actor.set_target_velocity(carla.Vector3D(vx, -vy, vz))


def _trajectory_tick(actor, state, dt):
    """Apply the desired next state through a physics tick, retaining measured velocity.

    CARLA reports zero velocity for a teleported actor with physics disabled. Place the
    actor one integration interval before the requested position, then let native physics
    advance it with the prescribed velocity. References retain the actual resulting state.
    """
    import carla

    pose = np.asarray(state.pose, dtype=float).copy()
    pose[:3, 3] -= np.asarray(state.velocity) * dt
    actor.set_simulate_physics(True)
    # A prescribed 3-D path already specifies its vertical motion. Adding gravity here
    # would change its declared velocity by g*dt on every tick (even for a level path).
    actor.set_enable_gravity(False)
    actor.apply_control(carla.VehicleControl())
    actor.set_target_angular_velocity(carla.Vector3D())
    _place(actor, replace(state, pose=pose))


class RecipeReplay:
    """Prescribe background motion; release ego to driving commands after pre-roll."""

    def __init__(self, simulator_runtime, recipe):
        self.simulator_runtime, self.recipe = simulator_runtime, recipe
        self.index = 0
        self.pre_roll = True
        self.released: set[str] = set()  # actors an attack now controls

    def initialize(self):
        import carla
        for name, spec in self.recipe.motion.items():
            actor = self.simulator_runtime.native["actors"][name]
            actor.set_autopilot(False, self.simulator_runtime.client.traffic_manager_port)
            if spec.trajectory:
                actor.set_simulate_physics(False)
                _place(actor, spec.trajectory[0])
            else:
                actor.apply_control(carla.VehicleControl(brake=1.0))
        self.origins = None

    def activate(self):
        trajectories = {name: spec for name, spec in self.recipe.motion.items() if spec.trajectory}
        if trajectories:
            for name, spec in trajectories.items():
                _trajectory_tick(self.simulator_runtime.native["actors"][name], spec.trajectory[0], self.recipe.time_step)
            self.simulator_runtime.client.tick()
        self.origins = {}
        for name, actor in self.simulator_runtime.native["actors"].items():
            v = actor.get_velocity()
            self.origins[name] = ActorState(_FLIP @ np.asarray(actor.get_transform().get_matrix()) @ _FLIP,
                                           (v.x, -v.y, v.z))
        return int(bool(trajectories))

    def hold_origin(self):
        for name, state in self.origins.items():
            if name not in self.released:
                _trajectory_tick(self.simulator_runtime.native["actors"][name], state, self.recipe.time_step)

    def finish_holding(self):
        for name, spec in self.recipe.motion.items():
            if not spec.trajectory:
                self.simulator_runtime.native["actors"][name].set_enable_gravity(True)

    def before_step(self):
        if self.index >= self.recipe.decision_steps:
            raise ValueError("CARLA step exceeds terminal recipe support")
        import carla

        for identity, motion in self.recipe.motion.items():
            if identity == "ego" and not self.pre_roll or identity in self.released:
                continue
            actor = self.simulator_runtime.native["actors"][identity]
            if motion.trajectory:
                _trajectory_tick(actor, motion.trajectory[self.index + 1], self.recipe.time_step)
            elif identity != "ego":
                control = motion.controls[self.index]
                actor.apply_control(carla.VehicleControl(
                    throttle=control.throttle, brake=control.brake, steer=control.steer,
                ))

    def after_step(self):
        self.index += 1

    def ego_command(self):
        motion = self.recipe.motion["ego"]
        return motion.controls[self.index] if motion.controls else Control()

    def release(self, identity):
        """Stop prescribing an actor's motion; physics and gravity act on it again."""
        self.released.add(identity)
        motion = self.recipe.motion.get(identity)
        if motion is not None and motion.trajectory:
            actor = self.simulator_runtime.native["actors"][identity]
            actor.set_simulate_physics(True)
            actor.set_enable_gravity(True)

    def release_ego(self):
        self.pre_roll = False
        if self.recipe.motion["ego"].trajectory:
            self.simulator_runtime.ego.actor.set_simulate_physics(True)
            self.simulator_runtime.ego.actor.set_enable_gravity(True)
