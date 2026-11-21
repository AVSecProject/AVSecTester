"""Configured CARLA simulations and their live native-world runtimes."""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from avsectester.closed_loop.simulators.base import (
    ClockSnapshot, Initialization, SensorGeometry, SensorView, Simulator, SimulatorRuntime,
)
from avsectester.closed_loop.hooks import EXTERNAL_CONTROL_INPUT_ERROR
from avsectester.datasets.base import FrameData, SensorDataReference
from avsectester.datasets.carla import CarlaRecipe, CARLA_WORLD_SETTINGS, validate_carla_world_settings
from avsectester.geometry.poses import MIRROR_Y, mirror_y_transform
from avsectester.data_types import Observation, WorldSnapshot
from avsectester.validation import require_nonnegative_integer, require_nonnegative_number
from avsectester.validation import timestamp_to_microseconds
from avsectester.closed_loop.timing import (
    SensorTiming, TimingCapabilities, TimingConfiguration, initial_history_times, seconds_to_microseconds,
)

from .conversions import _actor_state, _traffic_light_state
from .recipe_replay import RecipeReplay, _place
from .scenario import (
    _destroy_actor, _destroy_npc, _loadable_map_name, _spawn_config, prepare_scenario,
)
from .sensors import (
    SensorAlignmentError, SensorCapture, _require_pinhole_sensor, _use_pinhole_cameras,
    configure_recipe_blueprints, decode_carla_depth, register_pinhole_camera, validate_native_sensor_tick,
)

_MAPS_WITHOUT_LAYERS = frozenset({"Town10HD_Opt"})


@dataclass
class _TickCapture:
    state: WorldSnapshot
    observation: Observation | None = None
    views: dict[str, tuple[SensorGeometry | None, Any]] = field(default_factory=dict)


class CarlaSimulator(Simulator):
    """CARLA runtime settings, with scene content supplied by a dataset recipe.

    ``create_runtime`` allocates a runtime. Reset connects its live world. ``depth_camera`` names
    an ego RGB camera with an aligned depth camera, so its sensor views carry measured depth.
    Stock RGB cameras
    are then replaced by undistorted ``PinholeRgbCamera`` sensors matching the pinhole
    projection. Custom RGB classes are kept and must expose zero lens distortion.
    ``camera_warmup_s`` holds actors at their initial states before the scenario origin when
    RGB cameras are present. It must span a whole number of physics ticks and is separate
    from the deployment tick budget. ``camera_warmup_ticks`` is the resolved tick count.
    ``ground_settle_s`` lets unprescribed vehicles settle under gravity before holding.
    ``ego_control='ros2'`` delegates ego control to a native ROS 2 driver and requires
    ``realtime_factor``. ``external_startup_s`` extends the held initialization period.
    """

    dataset_kinds = ("CarlaDataset",)
    capture_ticks = 1  # CARLA renders on ticks. The first capture follows simulator.setup.
    can_render = True
    can_replay = True
    can_insert_objects = True
    can_observe_past = True
    supported_world_settings = CARLA_WORLD_SETTINGS

    def __init__(
        self, client: dict, settle_iters: int = 100, *, simulator_version="1.0",
        depth_camera: str | None = None, occlusion_tolerance_m: float = 0.1,
        decision_period_us: int | None = None, camera_warmup_s: float = 4.0,
        ground_settle_s: float = 2.0, ego_control: str = "framework",
        realtime_factor: float | None = None, external_startup_s: float = 0.0,
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
        require_nonnegative_number(occlusion_tolerance_m, "occlusion_tolerance_m")
        if ego_control not in {"framework", "ros2"}:
            raise ValueError("ego_control must be 'framework' or 'ros2'")
        if realtime_factor is not None and (isinstance(realtime_factor, bool)
                or not isinstance(realtime_factor, (int, float))
                or not math.isfinite(realtime_factor) or realtime_factor <= 0):
            raise ValueError("realtime_factor must be positive and finite")
        if ego_control == "ros2" and realtime_factor is None:
            raise ValueError("ego_control='ros2' requires realtime_factor")
        self.ego_control = ego_control
        self.realtime_factor = realtime_factor
        self.settle_iters = settle_iters
        self.occlusion_tolerance_m = occlusion_tolerance_m
        self.simulator_version = simulator_version
        self.depth_camera = depth_camera
        try:
            self._base_tick_us = seconds_to_microseconds(1 / float(self.client_config["rate"]))
        except ValueError as error:
            raise ValueError("CARLA client.rate must produce an integer-microsecond base tick") from error
        self.camera_warmup_s = camera_warmup_s
        self.ground_settle_s = ground_settle_s
        self.external_startup_s = external_startup_s
        for name, duration in (("camera_warmup", camera_warmup_s), ("ground_settle", ground_settle_s),
                               ("external_startup", external_startup_s)):
            require_nonnegative_number(duration, f"{name}_s")
            ticks = duration * float(self.client_config["rate"])
            if not math.isfinite(ticks) or not math.isclose(ticks, round(ticks), rel_tol=0., abs_tol=1e-9):
                raise ValueError(f"{name}_s must be a whole number of CARLA base ticks")
            setattr(self, f"{name}_ticks", round(ticks))
        self._decision_period_us = self._base_tick_us if decision_period_us is None else decision_period_us
        if decision_period_us is not None and (isinstance(decision_period_us, bool)
                or not isinstance(decision_period_us, int) or decision_period_us <= 0):
            raise ValueError("decision_period_us must be a positive integer")
        if self._decision_period_us % self.timing_capabilities.base_tick_us:
            raise ValueError("Decision period must be a whole number of CARLA base ticks")

    def validate_world_settings(self, world: Mapping[str, Any]) -> None:
        """Validate keys, native enum names, types and ranges without connecting to CARLA.

        Map capabilities and the existence of stable actor IDs require a live world and are
        checked when the settings are applied during initialization.
        """
        super().validate_world_settings(world)
        validate_carla_world_settings(world)

    def create_runtime(self, seed: int | None = None) -> CarlaSimulatorRuntime:
        """A new live world. An explicit run seed overrides the configured client seed."""
        return CarlaSimulatorRuntime(self, seed)

    def validate_timing(self, timing: TimingConfiguration) -> None:
        """Validate native tick alignment and exclusive external ego ownership."""
        super().validate_timing(timing)
        if self.ego_control == "ros2" and timing.force_gt_duration_us:
            raise ValueError("ego_control='ros2' rejects force_gt_duration_us because the external "
                             "driver exclusively controls the ego after decision zero")

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset() if self.ego_control == "ros2" else frozenset({"actuator"})

    @property
    def can_start_runs_after_origin(self) -> bool:
        """Allow a closed-loop start after the scene origin only under framework control."""
        return self.ego_control != "ros2"

    @property
    def decision_time_step(self):
        return self._decision_period_us / 1e6

    @property
    def timing_capabilities(self):
        return TimingCapabilities(captures_between_decisions=True, motion_reports=True,
                                  recorded_ego_motion=True,
                                  base_tick_us=self._base_tick_us)

    def _recorded_end_time_us(self, start):
        recipe = start.payload["recipe"]
        return timestamp_to_microseconds(recipe.decision_steps * recipe.time_step)

    @property
    def identity(self):
        return {**super().identity,
                "client": deepcopy(self.client_config),
                "depth_camera": self.depth_camera, "start_sensor_setup_ticks": 1,
                "camera_warmup_s": self.camera_warmup_s, "ground_settle_s": self.ground_settle_s,
                "ego_control": self.ego_control, "realtime_factor": self.realtime_factor,
                "external_startup_s": self.external_startup_s}

    def validate_start(self, start):
        if start is None:
            raise ValueError("CARLA reset requires ScenarioStart")
        super().validate_start(start)
        recipe = start.payload.get("recipe")
        if not isinstance(recipe, CarlaRecipe):
            raise TypeError("CARLA ScenarioStart requires an explicit CarlaRecipe")
        if not math.isclose(recipe.time_step, self.decision_time_step, abs_tol=1e-9):
            raise ValueError("Dataset time_step differs from simulator decision time step")
        for sensor in recipe.scenario["ego"].get("sensors", ()):
            validate_native_sensor_tick(sensor.get("sensor_tick", 0.))
        if start.frame.position > recipe.decision_steps:
            raise ValueError("Selected start exceeds recipe support")
        if self.ego_control == "ros2":
            attributes = recipe.scenario["ego"].get("blueprint_attributes", {})
            if attributes.get("role_name") not in {"hero", "ego"}:
                raise ValueError("ROS 2 ego blueprint_attributes.role_name must be 'hero' or 'ego'")


class CarlaSimulatorRuntime(SimulatorRuntime):
    """One live CARLA world of a `CarlaSimulator`.

    Owns the client, ego sensors, NPC traffic and native physics. An external AVStack supplies
    driving commands. ``advance_interval`` propagates one decision interval and ``capture_at``
    selects the scheduled native-tick packets. ``reset`` captures a strict-spawn
    ``replay_scenario`` that preparation can reuse across treatments.
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
        self._recorded_ticks: dict[int, _TickCapture] = {}
        self._interval_ticks: dict[int, _TickCapture] = {}
        self._history_payload_times = {}
        self._history_pose_times = frozenset()
        self._origin_tick = 0
        self._origin_clock_us = None
        self._native_commands = []
        self._pacing_started = False
        self._last_tick_wall = None
        self._pacing_start_wall = None
        self._pacing_stats = {}

    def _queue_native_command(self, command):
        self._native_commands.append(command)

    def _flush_native_commands(self):
        if not self._native_commands:
            return
        commands, self._native_commands = self._native_commands, []
        responses = self.client.client.apply_batch_sync(commands, False)
        failures = [response.error for response in responses if response.has_error()]
        if failures:
            raise RuntimeError(f"CARLA native command failed: {'; '.join(failures)}")

    def _tick(self, *, start_pacing=False):
        factor = self.simulator.realtime_factor
        if start_pacing and self.closed_loop_run and factor is not None and not self._pacing_started:
            self._pacing_started = True
            self._pacing_start_wall = time.monotonic()
            self._pacing_stats = {"ticks": 0, "late_ticks": 0, "max_lateness_s": 0.,
                                  "elapsed_wall_s": 0., "actual_realtime_factor": None}
        if self._pacing_started and factor is not None and self._last_tick_wall is not None:
            deadline = self._last_tick_wall + self.timing_capabilities.base_tick_us / 1e6 / factor
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)
            else:
                self._pacing_stats["late_ticks"] += int(now > deadline)
                self._pacing_stats["max_lateness_s"] = max(
                    self._pacing_stats["max_lateness_s"], now - deadline)
        self._last_tick_wall = time.monotonic()
        self._flush_native_commands()
        result = self.client.world.tick()
        if self._pacing_started:
            stats = self._pacing_stats
            stats["ticks"] += 1
            stats["elapsed_wall_s"] = time.monotonic() - self._pacing_start_wall
            if stats["elapsed_wall_s"]:
                stats["actual_realtime_factor"] = (
                    stats["ticks"] * self.timing_capabilities.base_tick_us
                    / 1e6 / stats["elapsed_wall_s"])
        return result

    @property
    def world_key(self) -> tuple[str, str, int]:
        client = self.client_config
        host = client.get("connect_ip", "127.0.0.1")
        if host in {"localhost", "127.0.0.1", "::1"}:
            host = "localhost"
        return ("carla", host, client.get("connect_port", 2000))

    @property
    def can_insert_objects(self) -> bool:
        """Allow framework-composited objects outside native external-control runs."""
        return not self._native_ros2_run and super().can_insert_objects

    def _require_object_insertion(self) -> None:
        if self._native_ros2_run:
            raise NotImplementedError("Framework-composited objects do not reach the external "
                                      "driving system. " + EXTERNAL_CONTROL_INPUT_ERROR)
        super()._require_object_insertion()

    @property
    def _uses_native_ros2(self) -> bool:
        return self.simulator.ego_control == "ros2"

    @property
    def _native_ros2_run(self) -> bool:
        return self._uses_native_ros2 and self.closed_loop_run

    @property
    def termination_details(self) -> Mapping[str, Any]:
        """Record final tick pacing statistics when the run is paced."""
        details = dict(super().termination_details)
        if self._pacing_stats:
            details["realtime"] = dict(self._pacing_stats)
        return details

    @property
    def native(self) -> dict[str, Any]:
        """Borrow native resources, invalidating cached truth before possible edits.

        ``queue_command`` accepts a native ``carla.command`` for synchronous submission
        before the next framework tick. Use it for actor edits made through native resources.
        """
        self.invalidate_state()
        return self._native_resources()

    def _native_resources(self) -> dict[str, Any]:
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
            "queue_command": self._queue_native_command,
        }

    def decision_telemetry(self):
        if not self._uses_native_ros2:
            return {}
        control = self.ego.actor.get_control()
        return {"ego_control": {name: getattr(control, name) for name in (
            "throttle", "brake", "steer", "hand_brake", "reverse", "manual_gear_shift", "gear")}}

    def _native_clock(self) -> ClockSnapshot:
        snapshot = self.client.world.get_snapshot()
        return ClockSnapshot(snapshot.timestamp.elapsed_seconds, snapshot.frame)

    def _state(self) -> WorldSnapshot:
        """Snapshot the latest completed tick, or the known spawn before the first actor tick."""
        if self.client is None or self.ego is None:
            raise RuntimeError("Reset the CARLA simulator before requesting its state")
        snapshot = self.client.world.get_snapshot()
        timestamp = snapshot.timestamp.elapsed_seconds
        self.ego.timestamp = timestamp
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
            timestamp, snapshot.frame,
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
        self._queue_native_command(carla.command.ApplyVehicleControl(actor.id, carla.VehicleControl(
            throttle=control.throttle, brake=control.brake, steer=control.steer)))
        self.invalidate_state()

    def set_actor_state(self, actor_id, state):
        _place(self, self._take_over(actor_id), state)
        self.invalidate_state()

    def _take_over(self, actor_id):
        """Stop an actor's prescribed or traffic-manager motion so an attack can move it."""
        if actor_id == "ego":
            raise ValueError("The ego is controlled by the driving system")
        actors = self._native_resources()["actors"]
        if actor_id not in actors:
            raise KeyError(f"Unknown CARLA actor {actor_id!r}; available: {sorted(actors)}")
        actor = actors[actor_id]
        self._recipe_replay.release(actor_id)
        import carla

        self._queue_native_command(carla.command.SetAutopilot(actor.id, False, self.client.traffic_manager_port))
        return actor

    def copy_observation(self, observation: Observation) -> Observation:
        """Detach mutable payloads, calibration and reference graphs in one copy operation.

        Native CARLA sensor captures cannot be deep-copied and are borrowed. Their metadata
        properties are read-only, but SDK item assignment may modify capture points. Replace a
        capture with a private array or measurement to preserve the original input. The Python
        wrappers, calibration and reference graphs are copied. Capture edits do not move actors.
        """
        memo = {}
        payloads = [*observation.sensor_data.values(),
                    *(sample.data for acquisitions in observation.deliveries.values() for sample in acquisitions)]
        for payload in payloads:
            for candidate in (payload, getattr(payload, "data", None)):
                if (
                    candidate is not None
                    and type(candidate).__module__.split(".", 1)[0] == "carla"
                    and hasattr(candidate, "raw_data")
                ):
                    memo[id(candidate)] = candidate
        return deepcopy(observation, memo)

    def _replay_position(self, start):
        return start.frame.position * self.simulator.decision_ticks

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
        if "map_name" in scenario["client"]:
            scenario["client"]["map_name"] = _loadable_map_name(scenario["client"]["map_name"])
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

        self._pacing_stats = {}
        scenario = self._resolve_scenario(start, initialization, environment)
        self._close_world()
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
        replay_ego_cfg = deepcopy(ego_cfg)
        for sensor in ego_cfg.get("sensors", ()):
            # Native sensor_tick can shift its phase by a physics tick. Acquire every tick
            # and retain only the scheduler's exact source IDs and capture times.
            sensor["sensor_tick"] = 0.
        ego_cfg["pipeline"] = {"type": "SerialPipeline", "modules": []}
        configure_recipe_blueprints({"ego": ego_cfg})
        self.ego = CARLA.build(ego_cfg, default_args={"client": self.client})
        self._resources.callback(self.ego.destroy)
        replay["ego"] = _spawn_config(replay_ego_cfg, self.ego, "vehicle")
        for spec in scenario.get("npcs", []):
            npc = CARLA.build(dict(spec), default_args={"client": self.client})
            import carla

            self._queue_native_command(carla.command.SetAutopilot(
                npc.actor.id, False, self.client.traffic_manager_port))
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
        self._origin_tick = start.frame.position * self.simulator.decision_ticks
        self._origin_clock_us = None
        timings = {sensor: self.timing.sensors.get(sensor, SensorTiming()) for sensor in self.sensors}
        history_plan = initial_history_times(
            timings, self.history_requirement, self.decision_period_us,
            pose_reporting_interval_us=self.timing.pose_reporting_interval_us)
        self._history_payload_times = {sensor: frozenset(times)
                                       for sensor, times in history_plan.payload_times_us.items()}
        self._history_pose_times = frozenset(history_plan.pose_times_us)
        self._recorded_ticks = {}
        self._interval_ticks = {}
        self._recipe_replay.initialize()
        # Base-world preparation ticks do not emit hooks or count as driving decisions.
        self._tick()
        for _ in range(self.simulator.ground_settle_ticks):
            self._tick()
        self._trajectory_setup_ticks = self._recipe_replay.activate()
        self._hold_tick()
        hold_ticks = max(self.simulator.camera_warmup_ticks if self.camera_sensors else 0,
                         self.simulator.external_startup_ticks)
        for _ in range(hold_ticks):
            self._hold_tick()
        self._recipe_replay.finish_holding()
        self._scenario_t0 = self.client.world.get_snapshot().timestamp.elapsed_seconds
        self._base_actor_ids = {actor.id for actor in self.client.world.get_actors()}
        self._save_recorded_tick()

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
        expected = self._queue_ego_control(replay.ego_command())
        self._tick()
        if self._uses_native_ros2:
            actual = self.ego.actor.get_control()
            if (any(np.float32(getattr(actual, name)) != np.float32(getattr(expected, name))
                    for name in ("throttle", "brake", "steer"))
                    or actual.hand_brake != expected.hand_brake or actual.reverse != expected.reverse):
                raise RuntimeError("An external client controlled the ego during CARLA replay. "
                                   "The external system must not publish controls during "
                                   "preparation and rendering.")
        replay.after_step()
        self.invalidate_state()
        if not self._advancing:
            self._save_recorded_tick()

    def _due_sensors(self, offset_us):
        due = []
        for sensor in sorted(self.sensors):
            clock = self.timing.sensors.get(sensor, SensorTiming())
            try:
                clock.frame_index_at(offset_us, self.decision_period_us)
            except ValueError:
                continue
            due.append(sensor)
        return tuple(due)

    def _save_recorded_tick(self):
        replay = self._recipe_replay
        tick_us = self.timing_capabilities.base_tick_us
        offset_us = (replay.index - self._origin_tick) * tick_us
        payload_times, pose_times = self._history_payload_times, self._history_pose_times
        due = tuple(sorted(sensor for sensor, times in payload_times.items() if offset_us in times))
        if not due and offset_us not in pose_times:
            return
        truth = self.state()
        self._recorded_ticks[replay.index] = self._capture_tick(truth, due, freeze_views=True)

    def _capture_tick(self, truth, sensor_ids, *, freeze_views):
        tick = _TickCapture(truth)
        if sensor_ids:
            tick.observation = self._observe_sensors(sensor_ids)
            previous = self._capture
            self._capture = tick.observation
            try:
                if freeze_views:
                    tick.views = {sensor: self._freeze_native_view(sensor, tick.observation, truth)
                                  for sensor in sensor_ids}
            finally:
                self._capture = previous
        return tick

    def _recorded_tick_index(self, time_us):
        tick = self.timing_capabilities.base_tick_us
        if time_us % tick:
            raise ValueError("Recorded CARLA acquisitions must align with its base tick")
        return self._origin_tick + time_us // tick

    def _observe_recorded(self, time_us, sensors):
        index = self._recorded_tick_index(time_us)
        tick = self._recorded_ticks.get(index)
        if tick is None or tick.observation is None:
            raise ValueError("Selected CARLA start has insufficient replayed sensor history")
        observation = tick.observation
        tick.observation = None
        return observation

    def _recorded_state(self, time_us):
        index = self._recorded_tick_index(time_us)
        if index not in self._recorded_ticks:
            raise ValueError("Selected CARLA start has insufficient replayed state history")
        return self._recorded_ticks[index].state

    def _recorded_pose_state(self, time_us):
        return deepcopy(self._recorded_state(time_us))

    def _recorded_read_complete(self, time_us):
        index = self._recorded_tick_index(time_us)
        tick = self._recorded_ticks.get(index)
        if tick is not None:
            tick.views = {sensor: (geometry, None)
                          for sensor, (geometry, _) in tick.views.items()}

    def _recorded_sensor_view(self, time_us, sensor, observation):
        index = self._recorded_tick_index(time_us)
        try:
            builder = self._recorded_ticks[index].views[sensor][1]
        except KeyError as error:
            raise ValueError("Selected CARLA start has insufficient replayed sensor history") from error
        if builder is None:
            raise ValueError("Recorded CARLA acquisition has already been handed off")
        return builder()

    def _recorded_sensor_geometry(self, time_us, sensor):
        index = self._recorded_tick_index(time_us)
        try:
            return self._recorded_ticks[index].views[sensor][0]
        except KeyError as error:
            raise ValueError("Selected CARLA start has insufficient replayed sensor geometry") from error

    def _hold_tick(self):
        self._recipe_replay.hold_origin()
        self._tick(start_pacing=True)

    def _before_selected_capture(self, position):
        if position == 0:
            self._scenario_t0 = self.client.world.get_snapshot().timestamp.elapsed_seconds
        self._origin_clock_us = timestamp_to_microseconds(
            self.client.world.get_snapshot().timestamp.elapsed_seconds)
        self._recipe_replay.finish_holding()
        if self._native_ros2_run:
            import carla

            self._recipe_replay.release_ego()
            self._queue_native_command(carla.command.SetSimulatePhysics(self.ego.actor.id, True))
            self._queue_native_command(carla.command.SetEnableGravity(self.ego.actor.id, True))
            self._queue_native_command(carla.command.ApplyVehicleControl(self.ego.actor.id, carla.VehicleControl()))
            self._flush_native_commands()

    def _check_initialization(self, initialization):
        """Check the replay plan and the ROS 2 run's ego state when given a reference.

        A reset without initialization has no expected ego state to compare.
        """
        actual = self._capture_frame_data(self._start.frame)
        if initialization is not None:
            if initialization.provenance.get("deployment_ticks", 0) != self._deployment.ticks:
                raise ValueError("Reference and run must share the deployment tick plan")
            if self._native_ros2_run:
                expected_ego = initialization.expected.ego
                if expected_ego is None:
                    raise ValueError("ROS 2 run initialization requires an expected ego state")
                assert actual.ego is not None
                if (not np.allclose(actual.ego.pose, expected_ego.pose, rtol=0., atol=1e-5)
                        or not np.allclose(actual.ego.velocity, expected_ego.velocity, rtol=0., atol=1e-5)):
                    raise RuntimeError("External controls before the CARLA run changed the ego's "
                                       "initial state from the prepared reference.")
        base_ids = {actor.id for actor in self._native_resources()["actors"].values()}
        self.initialization_report.update({
            "start_frame": asdict(self._start.frame), "pre_roll_steps": self._start.frame.position,
            "deployment_ticks": self._deployment.ticks,
            "resolved_scenario": deepcopy(self.replay_scenario),
            "motion": asdict(self._recipe)["motion"], "actual": actual.provenance,
            "inserted_actor_ids": [actor.id for actor in self.client.world.get_actors()
                                   if actor.id not in self._base_actor_ids and actor.id not in base_ids],
        })
        if self._pacing_started:
            self.initialization_report["realtime"] = dict(self._pacing_stats)
        if (not self._generating_frame_data and not self.timing.force_gt_duration_us
                and not self._uses_native_ros2):
            # Frame data generation records the prescribed ego motion. Runs drive it.
            self._recipe_replay.release_ego()

    def _advance_native(self, control):
        if self._recipe_replay.pre_roll:
            self._recipe_replay.release_ego()
        self._advance_ticks(control)

    def _advance_recorded_native(self):
        self._recipe_replay.pre_roll = True
        self._advance_ticks(None)

    def _advance_ticks(self, control):
        tick = self.timing_capabilities.base_tick_us
        self._interval_ticks = {}
        origin = self._interval.start_us
        count = self.simulator.decision_ticks
        for index in range(1, count + 1):
            if control is None and self._recipe_replay.pre_roll:
                self._replay_tick()
            else:
                self._recipe_replay.before_step()
                if control is not None:
                    self._queue_ego_control(control)
                self._tick()
                self._recipe_replay.after_step()
            timestamp = origin + index * tick
            self.invalidate_state()
            truth = self.state()
            due = self._due_sensors(timestamp - self._origin_clock_us)
            # Endpoint views read truth after simulator.step.post, like a decision capture.
            self._interval_ticks[timestamp] = self._capture_tick(
                truth, due, freeze_views=index < count)

    def _queue_ego_control(self, control):
        import carla

        command = carla.VehicleControl(
            throttle=float(control.throttle), brake=float(control.brake), steer=float(control.steer),
            hand_brake=bool(getattr(control, "hand_brake", False)),
            reverse=bool(getattr(control, "reverse", False)))
        self._queue_native_command(carla.command.ApplyVehicleControl(self.ego.actor.id, command))
        return command

    def interpolate_vehicle_state(self, left: Any, right: Any, fraction: float, time_us: int) -> Any:
        """Interpolate native avstack physical vectors and attitude."""
        from avsectester.geometry.poses import (
            interpolate_transform, make_pose, quaternion_to_rotation, rotation_to_quaternion,
        )

        if not all(hasattr(left, name) for name in ("position", "attitude")):
            return super().interpolate_vehicle_state(left, right, fraction, time_us)
        if type(left) is not type(right):
            raise TypeError("Pose reports must use the same vehicle-state representation")
        result = deepcopy(left)
        for name in ("position", "velocity", "acceleration"):
            a, b = getattr(left, name, None), getattr(right, name, None)
            if a is not None and b is not None:
                getattr(result, name).x = (1 - fraction) * np.asarray(a.x) + fraction * np.asarray(b.x)
        first, second = left.attitude.q, right.attitude.q
        first_values = ((first.w, first.x, first.y, first.z) if hasattr(first, "w") else first)
        second_values = ((second.w, second.x, second.y, second.z) if hasattr(second, "w") else second)
        pose = interpolate_transform(
            make_pose(quaternion_to_rotation(*first_values), left.position.x),
            make_pose(quaternion_to_rotation(*second_values), right.position.x), fraction)
        quaternion = rotation_to_quaternion(pose[:3, :3])
        if hasattr(first, "w"):
            result.attitude.q = type(first)(*quaternion)
        elif isinstance(first, np.ndarray):
            result.attitude.q = np.array(quaternion, dtype=first.dtype)
        else:
            result.attitude.q = type(first)(quaternion)
        if hasattr(result, "t"):
            result.t = time_us / 1e6
        return result

    def _completed_state(self):
        return self._interval_ticks[self._interval.end_us].state

    def _reported_state_at(self, time_us):
        return deepcopy(self._interval_ticks[time_us].state)

    def _activate_reported_state(self, time_us, state):
        if time_us not in self._interval_ticks:
            raise ValueError("CARLA acquisitions must align with its base tick")

    def _capture_frame_data(self, frame):
        """Capture actual states and sensor acquisition times, preserving native IDs as evidence."""
        observation = self._capture
        if observation is None:
            raise RuntimeError("Capture the CARLA frame before requesting frame data")
        scenario_t0 = self._scenario_t0
        actual_time = observation.t - scenario_t0
        actors, aliases = {}, {}
        for identity, actor in self._native_resources()["actors"].items():
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
        name = self.canonical_sensor_id(camera)
        item = next(sensor for sensor in self.ego.sensors.values()
                    if sensor.source_identifier == name)
        if item.object.type_id != "sensor.camera.rgb":
            raise ValueError(f"Sensor {camera!r} is not an RGB camera")
        return item

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

    def canonical_sensor_id(self, sensor: str) -> str:
        """Resolve an ego sensor's logical name or source ID to its observation ID."""
        if self.ego is None:
            raise ValueError(f"Unknown CARLA sensor {sensor!r}")
        matches = [item.source_identifier for name, item in self.ego.sensors.items()
                   if sensor in (name, item.source_identifier)]
        if len(matches) != 1:
            raise ValueError(f"Unknown CARLA sensor {sensor!r}")
        return matches[0]

    def _frame_range_us(self, sensor: str, end_time_us: int) -> range:
        """Native CARLA LiDAR spans one physics tick, independently of packet selection."""
        interval = super()._frame_range_us(sensor, end_time_us)
        native = next(item for item in self.ego.sensors.values()
                      if item.source_identifier == sensor)
        if native.object.type_id.startswith("sensor.lidar."):
            return range(end_time_us - self.timing_capabilities.base_tick_us, end_time_us)
        return interval

    def _sensor_geometry(self, sensor: str) -> SensorGeometry:
        from avsectester.geometry.cameras import PinholeCamera

        if not self._advancing and not self._resetting:
            tick = self._interval_ticks.get(self.clock.time_us)
            entry = None if tick is None else tick.views.get(sensor)
            if entry is not None:
                return entry[0]
        if not (self._world_initialized or self._resetting or self._advancing) or self._capture is None:
            raise RuntimeError("Reset the CARLA simulator before requesting camera geometry")
        rgb = self._rgb_sensor(sensor)
        optical = np.eye(4)
        optical[:3, :3] = [[0, 1, 0], [0, 0, -1], [1, 0, 0]]
        native_inverse = np.asarray(rgb.object.get_transform().get_inverse_matrix(), dtype=float)
        height, width = rgb.imsize
        return SensorGeometry(rgb.source_identifier,
                              PinholeCamera(np.array(rgb.P[:, :3], copy=True), width, height),
                              optical @ native_inverse @ MIRROR_Y)

    def _detached_view(self, view):
        shared = {id(placed): placed for placed in view.objects.values()}
        shared.update({id(value): value for value in view.native.values()})
        return deepcopy(view, shared)

    def _capture_view_builder(self, sensor, observation, truth):
        if not self._advancing and not self._resetting:
            tick = self._interval_ticks.get(self.clock.time_us)
            entry = None if tick is None else tick.views.get(sensor)
            if entry is not None:
                return entry[1]
        return self._freeze_native_view(sensor, observation, truth)[1]

    def _freeze_native_view(self, sensor, observation, truth):
        """Capture immutable view inputs now and defer conversion until a view is requested."""
        from avsectester.data_types import read_camera_rgb

        sensor = self.canonical_sensor_id(sensor)
        physical_sensor = next(item for item in self.ego.sensors.values()
                               if item.source_identifier == sensor)
        name = physical_sensor.source_identifier
        payload = observation.sensor_data[name]
        if getattr(payload, "frame", truth.frame - physical_sensor.frame0) + physical_sensor.frame0 != truth.frame:
            raise ValueError("Sensor payload and physical world must use the same frame")
        inverse = np.asarray(physical_sensor.object.get_transform().get_inverse_matrix(), dtype=float)
        sensor_from_world = mirror_y_transform(inverse)
        native = self._native_resources()
        if physical_sensor.object.type_id != "sensor.camera.rgb":
            data = payload if isinstance(payload, np.ndarray) else getattr(payload, "data", payload)
            def build_lidar():
                converted = data
                if hasattr(data, "raw_data"):
                    converted = SensorCapture(
                        f"{type(data).__module__}.{type(data).__qualname__}", bytes(data.raw_data),
                        int(data.frame), float(data.timestamp),
                        {field: getattr(data, field) for field in ("channels", "horizontal_angle")
                         if hasattr(data, field)})
                return SensorView(name, converted, None, None, truth.actors, truth.ego,
                    truth.t, truth.frame, sensor_from_world=sensor_from_world, native=native)
            return None, build_lidar
        geometry = self._sensor_geometry(sensor)
        captured = read_camera_rgb(observation, name)
        if captured is None:
            raise ValueError(f"Missing RGB camera {sensor!r}")
        depth = None
        if self.depth_camera is not None and physical_sensor is self._rgb_sensor(self.depth_camera):
            if self.depth_frame != truth.frame:
                raise RuntimeError("Depth and RGB capture frames differ")
            depth = self.depth
        tolerance = self.simulator.occlusion_tolerance_m
        return geometry, lambda: SensorView(name, captured[1], geometry.camera, geometry.cam_from_world,
            truth.actors, truth.ego, truth.t, truth.frame, sensor_from_world=sensor_from_world,
            depth=depth, native=native, occlusion_tolerance_m=tolerance)

    def _preflight_capture(self, time_us, sensor_ids):
        super()._preflight_capture(time_us, sensor_ids)
        if not self._world_initialized:
            return
        selected = self.sensors if sensor_ids is None else set(sensor_ids)
        due = self._due_sensors(time_us - self._origin_clock_us)
        unavailable = selected - set(due)
        if unavailable:
            raise ValueError(f"CARLA sensors not due at this tick: {sorted(unavailable)}")

    def _observe(self) -> Observation:
        return self._observe_sensors(tuple(self.sensors))

    def _observe_sensors(self, sensor_ids: Sequence[str]) -> Observation:
        if not self._advancing and not self._resetting and not self._generating_frame_data:
            tick = self._interval_ticks.get(self.clock.time_us)
            cached = None if tick is None else tick.observation
            if tick is not None and cached is None and sensor_ids:
                raise ValueError(f"CARLA sensors not due at this tick: {sorted(sensor_ids)}")
            if cached is None and not sensor_ids and tick is not None:
                state = tick.state
                return Observation(t=state.t, frame=state.frame, sensor_data={}, calibration={},
                                   vehicle_state=deepcopy(state.vehicle_state), ego_speed=state.ego_speed)
            if cached is not None:
                selected = set(sensor_ids)
                if not selected <= cached.sensor_data.keys():
                    raise ValueError(f"CARLA sensors not due at this tick: {sorted(selected - cached.sensor_data.keys())}")
                return self.copy_observation(replace(cached,
                    sensor_data={key: value for key, value in cached.sensor_data.items() if key in selected},
                    calibration={key: value for key, value in cached.calibration.items() if key in selected}))
        observation = self._capture_sensors(sensor_ids)
        needs_depth = self.depth_camera is not None and self._rgb_sensor(self.depth_camera).source_identifier in sensor_ids
        if self._depth_queue is not None and needs_depth:
            sample = self._depth_queue.get(timeout=30)
            while sample.frame < observation.frame:
                sample = self._depth_queue.get(timeout=30)
            if sample.frame != observation.frame:
                raise RuntimeError("CARLA depth and observation frames differ")
            self.depth = decode_carla_depth(sample)
            self.depth_frame = sample.frame
        return observation

    def _capture_sensors(self, sensor_ids=None) -> Observation:
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
        sensor_data = self._aligned_sensor_data(snap.frame, sensor_ids=sensor_ids)
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

    def _aligned_sensor_data(self, frame, initial=None, sensor_ids=None):
        """Wait for this tick's configured sensors, draining older queue entries without ticking.

        Payloads are returned in configured sensor order, independently of arrival order.
        """
        manager = self.ego.sensor_data_manager
        pending = {sensor.source_identifier: sensor for sensor in self.ego.sensors.values()
                   if sensor_ids is None or sensor.source_identifier in sensor_ids}
        order = tuple(pending)
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
        return {name: captured[name] for name in order}

    def _apply_map_layers(self, settings: Any) -> None:
        """Load/unload named layers on CARLA's optimized maps before spawning actors."""
        validate_carla_world_settings({"map_layers": settings})
        if not settings:
            return
        import carla

        world = self.client.world
        map_name = world.get_map().name.rsplit("/", 1)[-1]
        if map_name in _MAPS_WITHOUT_LAYERS:
            raise ValueError(f"map_layers is unsupported on CARLA map {map_name}")
        if not map_name.endswith("_Opt"):
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
        actors = {name: actor for name, actor in self._native_resources()["actors"].items()
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
            self._queue_native_command(carla.command.SetVehicleLightState(actor.id, carla.VehicleLightState(state)))

    def _apply_road_friction(self, settings: Any) -> None:
        """Set wheel friction multipliers, preserving the other vehicle physics settings."""
        validate_carla_world_settings({"road_friction": settings})
        if settings is None:
            return
        multiplier = settings["friction_force_multiplier"]
        actors = {name: actor for name, actor in self._native_resources()["actors"].items()
                  if actor.type_id.startswith("vehicle.")}
        selected = settings.get("actors", list(actors))
        if not isinstance(selected, (list, tuple)) or set(selected) - actors.keys():
            raise ValueError("road_friction.actors requires known stable actor IDs")
        for actor_id in selected:
            actor = actors[actor_id]
            physics = actor.get_physics_control()
            wheels = physics.wheels
            for wheel in wheels:
                wheel.friction_force_multiplier = float(multiplier)
            physics.wheels = wheels
            import carla

            self._queue_native_command(carla.command.ApplyVehiclePhysicsControl(actor.id, physics))

    def _apply_weather(self, weather: Any) -> None:
        """Set the world weather (the CARLA-native, world-level augmentation tier). ``weather`` is a
        ``carla.WeatherParameters`` preset **name** (e.g. ``"HardRainNoon"``, ``"ClearSunset"``) or
        a mapping of parameters (``cloudiness``, ``precipitation``, ``sun_altitude_angle``,
        ``fog_density``,
        ``wetness``, …). Applied to the whole scene — including any world-level patch — so it
        re-renders under that weather. ``None`` leaves the map's default weather."""
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
        """Spawn a stationary lead vehicle ``cfg['gap']`` meters directly ahead of the ego."""
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
        self._tick()
        return lead

    def close(self) -> None:
        try:
            self._close_world()
        finally:
            super().close()

    def _close_world(self) -> None:
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
            self._history_payload_times.clear()
            self._history_pose_times = frozenset()
            self._origin_clock_us = None
            self._native_commands.clear()
            self._pacing_started = False
            self._last_tick_wall = self._pacing_start_wall = None
            for cache in (self._recorded_ticks, self._interval_ticks):
                cache.clear()


__all__ = [
    'CarlaSimulator',
    'CarlaSimulatorRuntime',
]
