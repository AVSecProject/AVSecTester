"""Configured CARLA simulations and their live native-world runtimes."""
from __future__ import annotations

import math
import time
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np

from avsectester.closed_loop.simulators.base import (
    Initialization, SensorGeometry, SensorView, Simulator, SimulatorRuntime,
)
from avsectester.datasets.base import FrameData, SensorDataReference
from avsectester.datasets.carla import CarlaRecipe, CARLA_WORLD_SETTINGS, validate_carla_world_settings
from avsectester.geometry.poses import MIRROR_Y, mirror_y_transform
from avsectester.data_types import Observation, WorldSnapshot
from avsectester.validation import require_nonnegative_integer

from .conversions import _actor_state, _traffic_light_state
from .recipe_replay import RecipeReplay, _place
from .scenario import (
    _destroy_actor, _destroy_npc, _spawn_config, prepare_scenario,
)
from .sensors import (
    SensorAlignmentError, SensorCapture, _require_pinhole_sensor, _use_pinhole_cameras,
    decode_carla_depth, register_pinhole_camera,
)


class CarlaSimulator(Simulator):
    """CARLA runtime settings, with scene content supplied by a dataset recipe.

    ``create_runtime`` allocates a runtime; reset connects its live world. ``depth_camera`` names an ego RGB camera that
    gets an aligned depth camera, so sensor views of it carry measured depth; stock RGB cameras
    are then replaced by undistorted ``PinholeRgbCamera`` sensors matching the pinhole
    projection. Custom RGB classes are kept and must expose zero lens distortion.
    """

    dataset_kinds = ("CarlaDataset",)
    capture_ticks = 1  # CARLA renders on ticks; the first capture follows simulator.setup
    can_render = True
    can_replay = True
    can_insert_objects = True
    supported_world_settings = CARLA_WORLD_SETTINGS

    def __init__(
        self, client: dict, settle_iters: int = 100, *, simulator_version="0.9.16",
        depth_camera: str | None = None,
    ) -> None:
        allowed = {"connect_ip", "connect_port", "traffic_manager_port", "rate", "synchronous",
                   "seed", "traffic_manager_seed"}
        unknown = set(client) - allowed
        if unknown:
            raise ValueError(f"CARLA client only accepts runtime settings: {sorted(unknown)}")
        self.client_config = deepcopy(client)
        self.client_config.setdefault("rate", 20.0)
        self.client_config.setdefault("synchronous", True)
        if (isinstance(self.client_config["rate"], bool)
                or not isinstance(self.client_config["rate"], (int, float))
                or not math.isfinite(self.client_config["rate"]) or self.client_config["rate"] <= 0):
            raise ValueError("CARLA client rate must be positive and finite")
        if self.client_config["synchronous"] is not True:
            raise ValueError("CARLA requires synchronous operation")
        for name in ("seed", "traffic_manager_seed"):
            if self.client_config.get(name) is not None:
                require_nonnegative_integer(self.client_config[name], f"client.{name}")
        for name in ("connect_port", "traffic_manager_port"):
            value = self.client_config.get(name, 2000 if name == "connect_port" else 8000)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"client.{name} must be an integer in [1,65535]")
        require_nonnegative_integer(settle_iters, "settle_iters")
        self.settle_iters = settle_iters
        self.simulator_version = simulator_version
        self.depth_camera = depth_camera

    def validate_world_settings(self, world: Mapping[str, Any]) -> None:
        """Validate keys, native enum names, types and ranges without connecting to CARLA.

        Map capabilities and the existence of stable actor IDs require a live world and are
        checked when the settings are applied during initialization.
        """
        super().validate_world_settings(world)
        validate_carla_world_settings(world)

    def create_runtime(self, seed: int | None = None) -> CarlaSimulatorRuntime:
        """A new live world; an explicit run seed overrides the configured client seed."""
        return CarlaSimulatorRuntime(self, seed)

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"actuator"})

    @property
    def decision_time_step(self):
        return 1 / float(self.client_config["rate"])

    @property
    def identity(self):
        return {**super().identity, "client": deepcopy(self.client_config),
                "depth_camera": self.depth_camera, "start_sensor_setup_ticks": 1}

    def validate_start(self, start):
        if start is None:
            raise ValueError("CARLA reset requires ScenarioStart")
        super().validate_start(start)
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
    *external* (the :class:`~avsectester.closed_loop.avstacks.base.AVStack`), so this runtime only senses (``_observe``)
    and actuates (``step`` -> ``apply_control``). ``reset`` also captures a strict-spawn
    ``replay_scenario`` that preparation can reuse for every treatment.
    """

    def __init__(self, simulator: CarlaSimulator, seed: int | None = None) -> None:
        super().__init__(simulator)
        self.client_config = deepcopy(simulator.client_config)
        if seed is not None:
            require_nonnegative_integer(seed, "seed")
            self.client_config["seed"] = seed
        self.settle_iters = simulator.settle_iters
        self.depth_camera = simulator.depth_camera
        self.depth = None
        self.depth_frame = None
        self._depth_queue = None
        self.lead = None
        self.client = None
        self.ego = None
        self.npcs: list = []
        self.replay_scenario: dict | None = None
        self._resources = ExitStack()
        self._recipe = None
        self._recipe_replay = None

    @property
    def world_key(self) -> tuple[str, str, int]:
        client = self.client_config
        host = client.get("connect_ip", "127.0.0.1")
        if host in {"localhost", "127.0.0.1", "::1"}:
            host = "localhost"
        return ("carla", host, client.get("connect_port", 2000))

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
                    actors[name] = _actor_state(simulated, actor.type_id, actor.bounding_box)
        return WorldSnapshot(
            snapshot.timestamp.elapsed_seconds, snapshot.frame,
            deepcopy(state), float(state.velocity.norm()), actors,
            ego=_actor_state(self.ego.actor, self.ego.actor.type_id),
            traffic_lights={
                str(actor.get_opendrive_id()): _traffic_light_state(actor.get_state())
                for actor in self.client.world.get_actors()
                if actor.type_id == "traffic.traffic_light"
            },
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

    def _replay_position(self, start):
        return start.frame.position

    def _resolve_scenario(self, start, initialization, environment):
        if initialization is None and start.frame.position:
            raise ValueError("A selected CARLA start requires materialized frame data")
        if initialization is not None:
            if not isinstance(initialization, Initialization) or initialization.expected.frame != start.frame:
                raise ValueError("Initialization evidence does not match the selected start frame")
            scenario = deepcopy(initialization.resolved_scenario)
        else:
            scenario = deepcopy(start.payload["recipe"].scenario)
        # The recipe describes scene content. Runtime connection, cadence and seeds always
        # come from the simulator before spawn resolution can contact a server.
        scenario.setdefault("client", {}).update(deepcopy(self.client_config))
        scenario["client"].setdefault("type", "CarlaClient")
        from avsectester.closed_loop.environment import merge_environments

        recipe_world = {name: scenario[name] for name in self.supported_world_settings if name in scenario}
        scenario.update(merge_environments({"world": recipe_world}, {"world": environment})["world"])
        if initialization is None:
            scenario = prepare_scenario(scenario)
        if self.depth_camera is not None:
            register_pinhole_camera()
            _use_pinhole_cameras(scenario)
        return scenario

    def _initialize_world(self, start, initialization, environment):
        from avcarla.config import CARLA

        scenario = self._resolve_scenario(start, initialization, environment)
        self.close()
        self._recipe = start.payload["recipe"]
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
        self._apply_map_layers(scenario.get("map_layers"))
        ego_cfg = deepcopy(scenario["ego"])
        ego_cfg["pipeline"] = {"type": "SerialPipeline", "modules": []}
        self.ego = CARLA.build(ego_cfg, default_args={"client": self.client})
        self._resources.callback(self.ego.destroy)
        replay["ego"] = _spawn_config(scenario["ego"], self.ego, "vehicle")
        for spec in scenario.get("npcs", []):
            npc = CARLA.build(dict(spec), default_args={"client": self.client})
            npc.actor.set_autopilot(False, self.client.traffic_manager_port)
            self.npcs.append(npc)
            self._resources.callback(_destroy_npc, npc)
            replay["npcs"].append(_spawn_config(spec, npc, "npc_type"))
        # Sensor clocks precede the optional lead-spawn tick.
        origin = self.client.world.get_snapshot()
        self.ego.initialize(origin.timestamp.elapsed_seconds, origin.frame)
        self.replay_scenario = replay
        if scenario.get("lead") is not None:
            self.lead = self._spawn_lead(scenario["lead"])
        self._apply_weather(scenario.get("weather"))
        self._apply_lighting(scenario.get("lighting"))
        self._apply_road_friction(scenario.get("road_friction"))
        snap = self.client.world.get_snapshot()
        for npc in self.npcs:
            npc.initialize(snap.timestamp.elapsed_seconds, snap.frame)
        self._initialize_sensors()
        self._recipe_replay = RecipeReplay(self, self._recipe)
        self._recipe_replay.initialize()
        # Base-world preparation ticks do not emit hooks or count as driving decisions.
        self.client.tick()
        self._trajectory_setup_ticks = self._recipe_replay.activate()
        self._hold_tick()
        self._recipe_replay.finish_holding()
        self._scenario_t0 = self.client.world.get_snapshot().timestamp.elapsed_seconds
        self._base_actor_ids = {actor.id for actor in self.client.world.get_actors()}

    def _initialize_sensors(self):
        """Create the aligned depth camera, if configured, before the first capture."""
        if self.depth_camera is None:
            return
        from queue import Queue

        rgb = self._rgb_sensor(self.depth_camera)
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
        self._recipe_replay.hold_origin()
        self.client.tick()

    def _before_selected_capture(self, position):
        if position == 0:
            self._scenario_t0 = self.client.world.get_snapshot().timestamp.elapsed_seconds
        self._recipe_replay.finish_holding()

    def _check_initialization(self, initialization):
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
        self._recipe_replay.before_step()
        self.ego.apply_control(control)
        self.client.tick()
        self._recipe_replay.after_step()

    def _capture_frame_data(self, frame):
        """Capture actual states and sensor acquisition times, preserving native IDs as evidence."""
        observation = self._capture
        if observation is None:
            raise RuntimeError("Capture the CARLA frame before requesting frame data")
        scenario_t0 = self._scenario_t0
        actual_time = observation.t - scenario_t0
        actors, aliases = {}, {}
        for identity, actor in self.native["actors"].items():
            if identity != "ego":
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
        return FrameData(frame, actors, sensors, {
            "kind": "generated", "native_frame": observation.frame,
            "native_timestamp": observation.t, "scenario_t0": scenario_t0,
            "actual_scenario_time": actual_time,
            "actor_mapping": aliases, "required_sensors": required_sensors,
            "trajectory_setup_ticks": self._trajectory_setup_ticks,
            "start_sensor_setup_ticks": 1,
        }, ego=_actor_state(self.ego.actor, self.ego.actor.type_id))


    def _replayed_scenario(self):
        return deepcopy(self.replay_scenario)

    def _frame_data_provenance(self):
        return {"seed": self.replay_scenario["client"].get("seed")}

    def _rgb_sensor(self, camera):
        """Resolve a logical RGB sensor name or source ID without selection adapters."""
        sensors = {name: sensor for name, sensor in self.ego.sensors.items()
                   if sensor.object.type_id == "sensor.camera.rgb"}
        matches = [item for name, item in sensors.items()
                   if camera in (name, item.source_identifier)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Select a specific RGB camera from {list(sensors)}")

    @property
    def sensors(self) -> frozenset[str]:
        if self.ego is None:
            return frozenset()
        return frozenset(sensor.source_identifier for sensor in self.ego.sensors.values())

    @property
    def camera_sensors(self) -> frozenset[str]:
        if self.ego is None:
            return frozenset()
        return frozenset(sensor.source_identifier for sensor in self.ego.sensors.values()
                         if sensor.object.type_id == "sensor.camera.rgb")

    def sensor_geometry(self, sensor: str) -> SensorGeometry:
        from avsectester.geometry.cameras import PinholeCamera

        if not self._world_initialized or self._capture is None:
            raise RuntimeError("Reset the CARLA simulator before requesting camera geometry")
        rgb = self._rgb_sensor(sensor)
        optical = np.eye(4)
        optical[:3, :3] = [[0, 1, 0], [0, 0, -1], [1, 0, 0]]
        native_inverse = np.asarray(rgb.object.get_transform().get_inverse_matrix(), dtype=float)
        height, width = rgb.imsize
        return SensorGeometry(rgb.source_identifier,
                              PinholeCamera(np.array(rgb.P[:, :3], copy=True), width, height),
                              optical @ native_inverse @ MIRROR_Y)

    def _sensor_view(self, sensor: str) -> SensorView:
        """Capture data and physical sensor pose; camera views also provide optical geometry.

        Native SDK measurements are copied into SensorCapture bytes, retaining their original
        coordinate convention and the native wrapper's options when written back by an action.
        """
        from avsectester.data_types import read_camera_rgb

        if not self._world_initialized or self._capture is None:
            raise RuntimeError("Reset the CARLA simulator before requesting sensor views")
        matches = [item for name, item in self.ego.sensors.items()
                   if sensor in (name, item.source_identifier)]
        if len(matches) != 1:
            raise ValueError(f"Unknown CARLA sensor {sensor!r}; available: {sorted(self.sensors)}")
        physical_sensor = matches[0]
        name = physical_sensor.source_identifier
        if name not in self._capture.sensor_data:
            raise ValueError(f"Missing CARLA capture {name!r}")
        payload = self._capture.sensor_data[name]
        truth = self.state()
        if getattr(payload, "frame", truth.frame - physical_sensor.frame0) + physical_sensor.frame0 != truth.frame:
            raise ValueError("Sensor payload and physical world must use the same frame")
        inverse = np.asarray(physical_sensor.object.get_transform().get_inverse_matrix(), dtype=float)
        sensor_from_world = mirror_y_transform(inverse)
        actors = {name: _actor_state(actor, actor.type_id)
                  for name, actor in self.native["actors"].items() if name != "ego"}
        ego = _actor_state(self.ego.actor, self.ego.actor.type_id)
        if physical_sensor.object.type_id != "sensor.camera.rgb":
            data = payload if isinstance(payload, np.ndarray) else getattr(payload, "data", payload)
            if hasattr(data, "raw_data"):
                data = SensorCapture(
                    f"{type(data).__module__}.{type(data).__qualname__}", bytes(data.raw_data),
                    int(data.frame), float(data.timestamp),
                    {field: getattr(data, field) for field in
                     ("channels", "horizontal_angle") if hasattr(data, field)},
                )
            else:
                data = deepcopy(data)
            return SensorView(name, data, None, None, actors, ego, truth.t, truth.frame,
                              sensor_from_world=sensor_from_world, native=self.native)
        geometry = self.sensor_geometry(sensor)
        rgb = physical_sensor
        captured = read_camera_rgb(self._capture, name)
        if captured is None:
            raise ValueError(f"Missing RGB camera {sensor!r}")
        depth = None
        depth_rgb = None if self.depth_camera is None else self._rgb_sensor(self.depth_camera)
        if rgb is depth_rgb:
            if self.depth_frame != truth.frame:
                raise RuntimeError("Depth and RGB capture frames differ")
            depth = np.array(self.depth, copy=True)
        return SensorView(geometry.sensor, captured[1], geometry.camera, geometry.cam_from_world,
                          actors, ego, truth.t, truth.frame, sensor_from_world=sensor_from_world,
                          depth=depth, native=self.native)

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
        self.ego.timestamp = t
        sensor_data = self._aligned_sensor_data(snap.frame)
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

    def _apply_map_layers(self, settings: Any) -> None:
        """Load/unload named layers on CARLA's optimized maps before spawning actors."""
        validate_carla_world_settings({"map_layers": settings})
        if not settings:
            return
        import carla

        world = self.client.world
        if not world.get_map().name.rsplit("/", 1)[-1].endswith("_Opt"):
            raise ValueError("map_layers requires a CARLA _Opt map")
        for action in ("unload", "load"):
            layers = settings.get(action, [])
            for name in layers:
                getattr(world, f"{action}_map_layer")(getattr(carla.MapLayer, name))

    def _apply_lighting(self, settings: Any) -> None:
        """Set native scene-light groups and the light-state flags of stable vehicle IDs."""
        validate_carla_world_settings({"lighting": settings})
        if not settings:
            return
        import carla

        scene = settings.get("street_lights")
        if scene is not None:
            group = scene.get("group", "Street")
            manager = self.client.world.get_lightmanager()
            lights = manager.get_all_lights(getattr(carla.LightGroup, group))
            # Disable CARLA's automatic sun-altitude switching when overriding light state.
            manager.set_day_night_cycle(False)
            if "enabled" in scene:
                (manager.turn_on if scene["enabled"] else manager.turn_off)(lights)
            if "intensity" in scene:
                intensity = float(scene["intensity"])
                manager.set_intensity(lights, intensity)
            if "color" in scene:
                color = scene["color"]
                manager.set_color(lights, carla.Color(*color))
        vehicles = settings.get("vehicle_lights", {})
        actors = {name: actor for name, actor in self.native["actors"].items()
                  if actor.type_id.startswith("vehicle.")}
        unknown = set(vehicles) - actors.keys() - {"all"}
        if unknown:
            raise ValueError(f"Unknown vehicle_lights actor IDs: {sorted(unknown)}")
        for actor_id, actor in actors.items():
            flags = vehicles.get(actor_id, vehicles.get("all"))
            if flags is None:
                continue
            state = carla.VehicleLightState.NONE
            for flag in flags:
                state |= getattr(carla.VehicleLightState, flag)
            actor.set_light_state(carla.VehicleLightState(state))

    def _apply_road_friction(self, settings: Any) -> None:
        """Set tire friction on selected vehicles, preserving their other physics settings."""
        validate_carla_world_settings({"road_friction": settings})
        if settings is None:
            return
        coefficient = settings["tire_friction"]
        actors = {name: actor for name, actor in self.native["actors"].items()
                  if actor.type_id.startswith("vehicle.")}
        selected = settings.get("actors", list(actors))
        if not isinstance(selected, (list, tuple)) or set(selected) - actors.keys():
            raise ValueError("road_friction.actors requires known stable actor IDs")
        for actor_id in selected:
            actor = actors[actor_id]
            physics = actor.get_physics_control()
            wheels = physics.wheels
            for wheel in wheels:
                wheel.tire_friction = float(coefficient)
            physics.wheels = wheels
            actor.apply_physics_control(physics)

    def _apply_weather(self, weather: Any) -> None:
        """Set the world weather (the CARLA-native, world-level augmentation tier). ``weather`` is a
        ``carla.WeatherParameters`` preset **name** (e.g. ``"HardRainNoon"``, ``"ClearSunset"``) or a dict
        of parameters (``cloudiness``, ``precipitation``, ``sun_altitude_angle``, ``fog_density``,
        ``wetness``, …). Applied to the whole scene — including any world-level patch — so it re-renders
        under that weather. ``None`` leaves the map's default weather."""
        validate_carla_world_settings({"weather": weather})
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
            self._in_setup = False
            self._capture = None
            self._world_initialized = False
            self._recipe_replay = None
            self._depth_queue = None
            self.depth = self.depth_frame = None
            super().close()
