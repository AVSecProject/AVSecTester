"""NuRec live worlds with independent renderer, actor state and ego dynamics."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any, TYPE_CHECKING
import zipfile
from collections import Counter
from collections.abc import Mapping

from avsectester.closed_loop.simulators.ego_dynamics import EgoDynamics, PlanarEgoState, KinematicBicycle
from avsectester.closed_loop.simulators.base import (
    SensorGeometry, SensorView, Simulator, SimulatorRuntime, UnsupportedStartError, validated_camera_aliases,
)
from avsectester.closed_loop.simulators.navigation import RouteGenerator
from avsectester.maps import ScenarioMap
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.geometry.poses import validated_transform, validated_vector
from avsectester.validation import require_nonnegative_number, require_positive_number
from avsectester.data_types import ActorState, Observation, WorldSnapshot

from .depth import _actor_depth, _scene_raycaster, _ground_raycaster
from .renderer import Renderer, planar_rig_pose
from .traffic import AlpaSimTraffic, TrafficSession, TrafficTrack, TRAFFIC_STEP_US

if TYPE_CHECKING:
    from collections.abc import Sequence

    from avsectester.closed_loop.hooks import CameraPoseRequest
    from avsectester.geometry.cameras import Camera
    from avsectester.datasets.base import ScenarioStart
    from avsectester.rendering import CompositePlan, PlacedObject
    import numpy as np


def _route_generator(kind: str, start: ScenarioStart | None) -> RouteGenerator:
    """The navigation route of a run from ``start``, built from its complete recording and map."""
    return RouteGenerator(kind, () if start is None else [state.pose[:3, 3] for state in start.recorded_ego],
                          None if start is None else start.scenario_map)


class NuRecSimulator(Simulator):
    """A NuRec neural reconstruction with in-process ego dynamics and a pluggable renderer.

    Every reset requires a ScenarioStart with the recorded clock, ego pose and scene.
    Camera aliases map dataset-native names to observation and sensor-view names.

    ``renderer`` is a configuration template that is never connected itself: every runtime
    renders through its own copy (``Renderer.copy``), with its own connection, loaded scene and
    origin. ``dynamics`` is also a template: every runtime integrates with its own copy, so a
    handler that changes it, for example ``max_brake``, changes only that runtime.

    ``object_occlusion`` chooses how inserted world objects are hidden behind the scene. The
    default ``"rendered_probes"`` renders opaque probe copies of the objects inside the
    reconstruction (``Renderer.render_scene_visibility``), so the reconstruction that produces
    the images also hides the objects; it needs no mesh. ``"geometry"`` estimates occlusion from
    the configured ``depth`` (mesh plus actor cuboids, or actor cuboids alone).
    ``occlusion_tolerance_m`` (default 0.15 m) applies to both: the reconstruction hides an
    object only when it lies more than about this distance in front of it. Reconstructed
    Gaussians extend some centimeters in front of their surfaces, so an object placed on or just
    in front of a reconstructed surface needs this margin to stay visible.
    ``validate_start`` raises ``UnsupportedStartError`` when the configured navigation route
    cannot be built from the start's recording and map; each recording is checked once.
    """

    dataset_kinds = ("NuRecDataset",)
    can_render = True
    can_observe_past = True
    can_insert_objects = True

    def __init__(
        self,
        renderer: Renderer,
        dynamics: EgoDynamics | None = None,
        *,
        time_step: float | None = None,
        camera_aliases: Mapping[str, str] | None = None,
        depth: str = "mesh",
        route_generator: str = "map",
        physics_update_mode: str = "ego_only",
        ground_mesh: str = "mesh",
        object_occlusion: str = "rendered_probes",
        occlusion_tolerance_m: float = 0.15,
        traffic: AlpaSimTraffic | None = None,
        min_track_duration_s: float = 3.,
    ) -> None:
        if time_step is None:
            raise ValueError("NuRec time_step is required; experiment configuration supplies dataset.time_step")
        require_positive_number(time_step, "time_step")
        self.time_step = float(time_step)
        if renderer is None:
            raise TypeError("NuRecSimulator requires a renderer; configure NuRecRenderer")
        if not isinstance(depth, str) or depth not in {"mesh", "none"}:
            raise ValueError("NuRec depth must be 'mesh' or 'none'")
        if physics_update_mode not in {"ego_only", "none"}:
            raise ValueError("physics_update_mode must be 'ego_only' or 'none'")
        if ground_mesh not in {"mesh", "mesh_ground"}:
            raise ValueError("ground_mesh must be 'mesh' or 'mesh_ground'")
        if object_occlusion not in {"rendered_probes", "geometry"}:
            raise ValueError("object_occlusion must be 'rendered_probes' or 'geometry'")
        require_nonnegative_number(occlusion_tolerance_m, "occlusion_tolerance_m")
        require_nonnegative_number(min_track_duration_s, "min_track_duration_s")
        if depth == "mesh" or physics_update_mode == "ego_only":
            from avsectester.geometry.mesh import MeshRaycaster

            MeshRaycaster.check_dependencies()
        self.depth = depth
        if route_generator not in {"map", "recorded", "none"}:
            raise ValueError("route_generator must be 'map', 'recorded' or 'none'")
        self.route_generator = route_generator
        self.physics_update_mode = physics_update_mode
        self.ground_mesh = ground_mesh
        self.object_occlusion = object_occlusion
        self.occlusion_tolerance_m = occlusion_tolerance_m
        self.camera_aliases = validated_camera_aliases({} if camera_aliases is None else camera_aliases)
        self.renderer = renderer
        self.dynamics = dynamics or KinematicBicycle()
        if traffic is not None and not isinstance(traffic, AlpaSimTraffic):
            raise TypeError("traffic must be AlpaSimTraffic or None")
        if traffic is not None and self.time_step != TRAFFIC_STEP_US / 1e6:
            raise ValueError(f"AlpaSim traffic requires a NuRec decision time step of {TRAFFIC_STEP_US / 1e6} s")
        self.traffic = traffic
        self.min_track_duration_s = min_track_duration_s
        self._route_failures: dict[tuple[str, str, str, bool], str | None] = {}

    def validate_start(self, start: ScenarioStart | None) -> None:
        import numpy as np

        if start is None:
            raise ValueError("NuRec requires a ScenarioStart for the recorded clock, ego pose and scene")
        super().validate_start(start)
        if start.frame.position < self.past_steps:
            raise UnsupportedStartError(f"AlpaSim traffic requires {self.past_steps} past decision steps")
        if self.route_generator == "none":
            return
        # The route depends only on the complete recording and the map, not on the start frame.
        # Within one configured simulator a scenario's recording and map are fixed.
        positions = np.asarray([state.pose[:3, 3] for state in start.recorded_ego], dtype=float)
        key = (start.scenario_id, self.route_generator, hashlib.sha256(positions.tobytes()).hexdigest(),
               isinstance(start.scenario_map, ScenarioMap))
        if key not in self._route_failures:
            try:
                _route_generator(self.route_generator, start)
                self._route_failures[key] = None
            except ValueError as error:
                self._route_failures[key] = str(error)
        failure = self._route_failures[key]
        if failure is not None:
            raise UnsupportedStartError(f"Navigation route cannot be built: {failure}")

    def create_runtime(self, seed: int | None = None) -> NuRecSimulatorRuntime:
        """A new world; the optional traffic session receives seed for its log."""
        return NuRecSimulatorRuntime(self, seed)

    @property
    def past_steps(self) -> int:
        """Recorded decision history needed by the optional traffic model."""
        return 0 if self.traffic is None else self.traffic.history_steps - 1

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset({
            "simulator.setup", "simulator.step.pre", "simulator.step.post", "render.pre", "render.post",
        })

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Delegate command capabilities to the selected ego dynamics implementation."""
        return getattr(self.dynamics, "command_modes", None)

    @property
    def decision_time_step(self):
        return self.time_step

    @property
    def identity(self):
        identity = {**super().identity, "camera_aliases": dict(self.camera_aliases),
                "time_step": self.time_step,
                "depth": self.depth,
                "route_generator": self.route_generator,
                "physics_update_mode": self.physics_update_mode,
                "ground_mesh": self.ground_mesh,
                "object_occlusion": self.object_occlusion,
                "min_track_duration_s": self.min_track_duration_s,
                "renderer": type(self.renderer).__name__,
                "dynamics": {"type": f"{type(self.dynamics).__module__}:{type(self.dynamics).__qualname__}",
                             "options": asdict(self.dynamics) if dataclasses.is_dataclass(self.dynamics) else {}},
                "endpoint": getattr(self.renderer, "endpoint", None)}
        if self.traffic is not None:
            identity["traffic"] = {
                "type": f"{type(self.traffic).__module__}:{type(self.traffic).__qualname__}",
                "options": {name: getattr(self.traffic, name) for name in (
                    "endpoint", "history_steps", "rpc_timeout_s")}}
        return identity


class NuRecSimulatorRuntime(SimulatorRuntime):
    """One live NuRec world: the ego pose and dynamics loop, and its own renderer connection."""

    def __init__(self, simulator: NuRecSimulator, seed: int | None = None) -> None:
        super().__init__(simulator)
        self.seed = seed
        self._traffic_session: TrafficSession | None = None
        self._recorded_tracks: dict[str, dict[str, Any]] = {}
        self._filtered_tracks: set[str] = set()
        self._traffic_states: dict[str, ActorState] = {}
        self._traffic_timestamp_us: int | None = None
        self._capture_dynamic_objects: dict[str, tuple] = {}
        # Duck-typed renderers without ``copy`` are copied shallowly, like Renderer.copy.
        copy_renderer = getattr(simulator.renderer, "copy", None)
        self.renderer = copy_renderer() if callable(copy_renderer) else copy.copy(simulator.renderer)
        self.dynamics = deepcopy(simulator.dynamics)
        self.dt = simulator.time_step
        self.depth = simulator.depth
        self.pose = PlanarEgoState()
        self.frame = 0
        self._render_requests: dict[str, CameraPoseRequest] = {}
        # Recorded actor tracks, by archive; recorded files are immutable.
        self._track_datasets: dict[tuple, Any] = {}
        self._actor_states: tuple[int, dict] | None = None
        self._actor_overrides: dict[str, tuple[int, ActorState]] = {}
        self._capture_actor_states: dict[str, dict[str, ActorState]] = {}
        self._capture_depth: dict[str, np.ndarray] = {}
        self._recorded_end: float | None = None
        self._route = RouteGenerator("none", (), None)
        self._world_from_ego: Any = None
        self._ground_contact = None
        self._ground_contact_failures: Counter[str] = Counter()

    @property
    def native(self) -> dict[str, Any]:
        """Native resources for explicit world changes or custom renderer operations."""
        return {
            "renderer": self.renderer,
            "dynamics": self.dynamics,
            "render_requests": deepcopy(self._render_requests),
        }

    def render_request(self, camera: str) -> CameraPoseRequest:
        """Return the effective request that produced this camera's current observation.

        Both native camera names and configured output aliases are accepted. The returned
        request is independent of the stored geometry, including its ego pose.
        """
        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == camera), camera)
        if native_camera not in self.renderer.cameras:
            raise ValueError(f"Unknown NuRec camera {camera!r}")
        if native_camera not in self._render_requests:
            raise RuntimeError(f"No current render request for camera {camera!r}")
        return deepcopy(self._render_requests[native_camera])

    def _ego_state(self):
        """Physical three-dimensional rig pose, owned by this runtime."""
        import numpy as np

        pose = self._world_from_ego.copy()
        recorded = self._start.payload["ego"]
        velocity = pose[:3, :3] @ np.array([self.pose.speed, self.pose.lateral_speed, 0.])
        return ActorState(pose, tuple(float(value) for value in velocity), "vehicle",
                          recorded.extent, recorded.box_center, recorded.box_rotation)

    @property
    def termination_details(self) -> dict[str, Any]:
        return {"ground_contact_failures": dict(self._ground_contact_failures)}

    def _state(self) -> WorldSnapshot:
        """Physical ego and current traffic, detached from model-visible observations."""
        return WorldSnapshot(self.pose.t, self.frame, deepcopy(self.pose), self.pose.speed,
                             deepcopy(self._actors_at(self._timestamp_us(self.pose))),
                             ego=self._ego_state())

    def _timestamp_us(self, pose: PlanarEgoState) -> int:
        origin = self._start.frame.timestamp
        return round((origin + pose.t) * 1e6)

    def _recorded_planar_state(self, position: int, ego: ActorState | None = None) -> PlanarEgoState:
        """Rig-frame motion and finite differences from the actual recording."""
        import numpy as np

        start = self._start
        recorded = start.recorded_ego
        current = recorded[position] if ego is None else ego
        if current.velocity is None:
            raise ValueError(f"Recorded ego state at position {position} has no velocity")
        pose = np.asarray(current.pose)
        velocity = pose[:3, :3].T @ validated_vector(current.velocity, "recorded ego velocity")
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        yaw_rate, acceleration = 0., 0.
        neighbor = position - 1 if position else 1
        if 0 <= neighbor < len(recorded):
            other = recorded[neighbor]
            if other.velocity is None:
                raise ValueError(f"Recorded ego state at position {neighbor} has no velocity")
            other_velocity = np.asarray(other.pose)[:3, :3].T @ validated_vector(
                other.velocity, "recorded ego velocity")
            interval = (position - neighbor) * self.dt
            other_yaw = math.atan2(other.pose[1, 0], other.pose[0, 0])
            difference = yaw - other_yaw
            yaw_rate = math.atan2(math.sin(difference), math.cos(difference)) / interval
            acceleration = (velocity[0] - other_velocity[0]) / interval
        return PlanarEgoState(float(pose[0, 3]), float(pose[1, 3]), yaw, float(velocity[0]),
                              (position - start.frame.position) * self.dt,
                              lateral_speed=float(velocity[1]), yaw_rate=float(yaw_rate),
                              acceleration=float(acceleration))

    def _tracks(self):
        from avsectester.datasets.nurec import NuRecDataset

        payload = self._start.payload
        key = (payload["usdz_path"], payload["camera_id"], payload["time_step"])
        if key not in self._track_datasets:
            self._track_datasets[key] = NuRecDataset([key[0]], camera_id=key[1], time_step=key[2])
        return self._track_datasets[key]

    def _recorded_actors(self, timestamp_us: int) -> dict:
        if self._start is None or "usdz_path" not in self._start.payload:
            return {}
        if self._actor_states is None or self._actor_states[0] != timestamp_us:
            dataset = self._tracks()
            self._actor_states = (timestamp_us, dataset.actor_states(
                self._start.payload["scene_id"], timestamp_us))
        return self._actor_states[1]

    def _actors_at(self, timestamp_us: int) -> dict[str, ActorState]:
        import numpy as np

        actors = {key: state for key, state in self._recorded_actors(timestamp_us).items()
                  if key not in self._filtered_tracks}
        if timestamp_us == self._traffic_timestamp_us:
            actors.update(self._traffic_states)
        for actor_id, (origin_us, state) in self._actor_overrides.items():
            pose = state.pose.copy()
            pose[:3, 3] += np.asarray(state.velocity) * ((timestamp_us - origin_us) / 1e6)
            actors[actor_id] = dataclasses.replace(state, pose=pose)
        return actors

    def set_actor_state(self, actor_id: str, state: ActorState) -> None:
        """Override a current reconstructed actor's pose and world velocity until reset.

        The given pose is its physical state at the current clock. Subsequent captures
        extrapolate translation with the supplied velocity, retaining recorded appearance
        and dimensions. Parked or duration-filtered tracks cannot be moved. A previous
        capture and its sensor-view geometry are unchanged. For reactive traffic the
        first override backfills constant-velocity history only in the service request.
        """
        if actor_id == "ego":
            raise ValueError("ego is controlled by the driving system")
        if actor_id in self._filtered_tracks:
            raise ValueError(f"Filtered NuRec track {actor_id!r} cannot be overridden")
        track = self._recorded_tracks.get(actor_id)
        if track is not None and not track["controllable"]:
            raise ValueError(f"NRE cannot move parked NuRec track {actor_id!r}; it is baked into the background")
        actors = self.state().actors
        if actor_id not in actors:
            raise KeyError(f"Unknown NuRec actor {actor_id!r}; available: {sorted(actors)}")
        if not isinstance(state, ActorState):
            raise TypeError("NuRec actor state must be an ActorState")
        if state.velocity is None:
            raise ValueError("NuRec actor overrides require a finite world velocity")
        velocity = tuple(float(value) for value in validated_vector(state.velocity, "velocity"))
        replacement = dataclasses.replace(actors[actor_id],
            pose=validated_transform(state.pose), velocity=velocity)
        timestamp_us = self._timestamp_us(self.pose)
        self._actor_overrides[actor_id] = (timestamp_us, replacement)
        session = self._traffic_session
        if session is not None and actor_id in session.track_ids:
            session.record_override(actor_id, timestamp_us, replacement.box_transform(), velocity)

    def _close_traffic_session(self) -> None:
        session, self._traffic_session = self._traffic_session, None
        if session is not None:
            session.close()

    def _open_traffic_session(self, short_tracks: set[str]) -> None:
        import numpy as np

        traffic = self.simulator.traffic
        assert traffic is not None
        start = self._start
        if start is None:
            raise ValueError("AlpaSim traffic requires a recorded NuRec ScenarioStart")
        ego = self._ego_state()
        if ego.extent is None:
            raise ValueError("AlpaSim traffic requires the ego bounding-box extent")
        rig_from_box = np.linalg.inv(ego.pose) @ ego.box_transform()
        first_time = start.payload["first_timestamp"]
        time_step = start.payload["time_step"]
        trajectory = tuple((round((first_time + index * time_step) * 1e6), state.pose @ rig_from_box)
                           for index, state in enumerate(start.recorded_ego))
        tracks = [TrafficTrack(identity, track["trajectory"], track["extent"],
                               track["label_class"], not track["controllable"])
                  for identity, track in self._recorded_tracks.items() if identity not in short_tracks]
        self._traffic_session = traffic.open_session(
            scene_id=start.payload["scene_id"], random_seed=0 if self.seed is None else self.seed,
            start_us=self._timestamp_us(self.pose), end_us=round(start.payload["terminal_timestamp"] * 1e6),
            ego_extent=ego.extent, ego_trajectory=trajectory, tracks=tracks)
        self.initialization_report["traffic"] = {
            "session_uuid": self._traffic_session.uuid,
            "handover_time_us": self._traffic_session.handover_time_us,
            "session_tracks": sorted(self._traffic_session.track_ids),
        }

    def _dynamic_objects(self, timestamp_us: int, actors: dict[str, ActorState]) -> tuple:
        """The exact actor edits for ordinary captures and inserted-object probes."""
        objects = [(identity, state.pose) for identity, state in actors.items()]
        for identity, state in self._recorded_actors(timestamp_us).items():
            if identity in self._filtered_tracks:
                pose = state.pose.copy()
                pose[2, 3] -= 100.
                objects.append((identity, pose))
        return tuple(objects)

    def _simulate_traffic(self, previous: dict[str, ActorState]) -> None:
        timestamp_us = self._timestamp_us(self.pose)
        session = self._traffic_session
        if session is None:
            raise RuntimeError("AlpaSim traffic session is closed; reset the runtime before stepping")
        current = self._actors_at(timestamp_us)
        overrides = {identity: current[identity].box_transform()
                     for identity in self._actor_overrides if identity in session.track_ids}
        returned = session.simulate(timestamp_us, self._ego_state().box_transform(), overrides)
        states = {}
        for identity, pose in returned.items():
            velocity = None if identity not in previous else tuple(
                float(value) for value in ((pose[:3, 3] - previous[identity].pose[:3, 3]) / self.dt))
            track = self._recorded_tracks[identity]
            states[identity] = ActorState(pose, velocity, track["category"], track["extent"])
        self._traffic_states = states
        self._traffic_timestamp_us = timestamp_us

    def _initialize_world(self, start, initialization, environment):
        self._close_traffic_session()
        self._recorded_tracks.clear()
        self._filtered_tracks.clear()
        self._traffic_states.clear()
        self._traffic_timestamp_us = None
        self._capture_dynamic_objects.clear()
        if initialization is not None:
            raise ValueError("CARLA replay initialization does not apply to NuRec")
        if environment:
            raise ValueError(f"Unsupported NuRec environment overrides: {sorted(environment)}")
        if self.depth == "mesh":
            path = start.payload["usdz_path"]
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
            missing = [name for name in ("mesh.ply", "mesh_ground.ply") if name not in members]
            if missing:
                raise ValueError(f"NuRec mesh depth archive {str(path)!r} is missing: {', '.join(missing)}")
        self._render_requests.clear()
        self._actor_states = None
        self._actor_overrides.clear()
        self._capture_actor_states.clear()
        self._capture_depth.clear()
        self._recorded_end = None
        self._ground_contact_failures.clear()
        self._ground_contact = None
        ego = start.payload["ego"]
        scene_id = start.payload["scene_id"]
        initial_state = self._recorded_planar_state(start.frame.position, ego)
        self._recorded_end = start.payload["terminal_timestamp"] - start.frame.timestamp
        self.initialization_report = {
            "kind": "recorded", "start_frame": asdict(start.frame),
            "time_step": start.payload.get("time_step"),
            "interpolation": start.payload.get("interpolation"),
            "terminal_timestamp": start.payload["terminal_timestamp"], "pre_roll_steps": 0,
        }
        reset_origin = getattr(self.renderer, "reset_origin", None)
        self._route = _route_generator(self.simulator.route_generator, start)
        if self.simulator.physics_update_mode == "ego_only":
            if start.payload["ego"].extent is None:
                raise ValueError("NuRec ground contact requires ego bounding-box extent")
            self._ground_contact = _ground_raycaster(
                start.payload["usdz_path"], self.simulator.ground_mesh + ".ply")
        if callable(reset_origin):
            reset_origin(start)
        else:
            raise TypeError("Source initialization requires renderer.reset_origin(ScenarioStart)")
        self.pose = initial_state
        self._world_from_ego = ego.pose.copy()
        reset_dynamics = getattr(self.dynamics, "reset", None)
        if callable(reset_dynamics):
            reset_dynamics(deepcopy(self.pose))
        self.renderer.load_scene(scene_id)
        self.frame = 0
        # Short tracks are measured over the whole recording: movable ones leave the world,
        # and no short track joins a traffic session.
        short_tracks: set[str] = set()
        if "usdz_path" in start.payload:
            self._recorded_tracks = self._tracks().track_trajectories(start.payload["scene_id"])
            minimum_us = round(self.simulator.min_track_duration_s * 1e6)
            short_tracks = {
                identity for identity, track in self._recorded_tracks.items()
                if not track["trajectory"]
                or track["trajectory"][-1][0] - track["trajectory"][0][0] < minimum_us
            }
            self._filtered_tracks = {identity for identity in short_tracks
                                     if self._recorded_tracks[identity]["controllable"]}
            self.initialization_report["filtered_tracks"] = sorted(self._filtered_tracks)
        if self.simulator.traffic is not None:
            self._open_traffic_session(short_tracks)

    def _advance_native(self, control):
        if self._recorded_end is not None and round((self.pose.t + self.dt) * 1e6) > round(self._recorded_end * 1e6):
            raise ValueError("NuRec step exceeds terminal recorded support")
        self._render_requests.clear()
        uses_traffic = self.simulator.traffic is not None
        previous_actors = self._actors_at(self._timestamp_us(self.pose)) if uses_traffic else {}
        previous_yaw = self.pose.yaw
        self.pose = self.dynamics.step(self.pose, control, self.dt)
        world_from_ego = planar_rig_pose(self.pose.x, self.pose.y, self.pose.yaw,
                                         self._world_from_ego[2, 3])
        world_from_ego[:3, :3] = (world_from_ego[:3, :3]
            @ planar_rig_pose(0, 0, previous_yaw)[:3, :3].T @ self._world_from_ego[:3, :3])
        self._world_from_ego = world_from_ego
        if self._ground_contact is not None:
            from avsectester.closed_loop.simulators.ground_contact import constrain_to_ground
            import numpy as np

            ego = self._ego_state()
            corrected, status = constrain_to_ground(self._ground_contact, ego.box_transform(), ego.extent)
            if status != "successful_update":
                self._ground_contact_failures[status] += 1
            self._world_from_ego = corrected @ np.linalg.inv(
                np.linalg.inv(ego.pose) @ ego.box_transform())
            self.pose.x, self.pose.y = self._world_from_ego[:2, 3]
            self.pose.yaw = math.atan2(self._world_from_ego[1, 0], self._world_from_ego[0, 0])
        self.frame += 1
        if uses_traffic:
            self._simulate_traffic(previous_actors)

    @property
    def sensors(self) -> frozenset[str]:
        return self.camera_sensors

    @property
    def camera_sensors(self) -> frozenset[str]:
        aliases = self.simulator.camera_aliases
        return frozenset(aliases.get(camera, camera) for camera in self.renderer.cameras
                         if self.renderer.camera_model(camera) is not None)

    def sensor_geometry(self, sensor: str) -> SensorGeometry:
        if not self._world_initialized:
            raise RuntimeError("Reset the NuRec simulator before requesting camera geometry")
        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == sensor), sensor)
        if native_camera not in self.renderer.cameras:
            raise ValueError(f"Unknown NuRec camera {sensor!r}")
        request = self.render_request(native_camera)
        camera = self.renderer.camera_model(request.camera)
        if camera is None:
            raise ValueError(f"NuRec camera {sensor!r} has no camera geometry")
        return SensorGeometry(aliases.get(native_camera, native_camera), camera,
                              self.renderer.cam_from_world(request.pose, request.camera))

    def _sensor_view(self, sensor: str) -> SensorView:
        """View of the current capture, including its recorded or overridden actor poses."""
        if self._start is None:
            raise ValueError("NuRec sensor views require a scenario start with recorded actors")
        from avsectester.data_types import read_camera_rgb

        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == sensor), sensor)
        output_key = aliases.get(native_camera, native_camera)
        request = self.render_request(native_camera)
        truth = self.state()
        timestamp = self._timestamp_us(self.pose)
        captured = read_camera_rgb(self._capture, output_key)
        if captured is None:
            raise ValueError(f"Missing NuRec camera {output_key!r}")
        cam_from_world = self.renderer.cam_from_world(request.pose, request.camera)
        camera = self.renderer.camera_model(request.camera)
        if camera is None:
            raise ValueError(f"NuRec camera {output_key!r} has no camera geometry")
        return SensorView(
            output_key, captured[1], camera, cam_from_world,
            self._capture_actor_states[native_camera], truth.ego, truth.t, truth.frame,
            sensor_from_world=BODY_FROM_OPTICAL @ cam_from_world,
            depth=self._camera_depth(native_camera, camera, cam_from_world),
            native={"render_request": request, "timestamp_us": timestamp,
                    "dynamic_objects": deepcopy(self._capture_dynamic_objects[native_camera])},
            object_occlusion=self.simulator.object_occlusion,
            occlusion_tolerance_m=self.simulator.occlusion_tolerance_m,
        )

    def _camera_depth(self, camera_name: str, camera: Camera,
                      cam_from_world: np.ndarray) -> np.ndarray | None:
        if self.depth == "none":
            return None
        if camera_name not in self._capture_depth:
            from avsectester.geometry.mesh import camera_depth

            depth = camera_depth(
                camera, cam_from_world, [_scene_raycaster(self._start.payload["usdz_path"])],
            )
            self._capture_depth[camera_name] = _actor_depth(
                camera, cam_from_world, self._capture_actor_states[camera_name], depth,
            )
        return self._capture_depth[camera_name].copy()

    def _recorded_capture(self, position: int) -> tuple[np.ndarray, int, dict[str, ActorState]]:
        """Recorded rig pose, scene timestamp and actor states of a past decision position."""
        start = self._start
        if start is None:
            raise RuntimeError("Past NuRec observations require a scenario start")
        pose = validated_transform(start.recorded_ego[position].pose)
        relative_time = (position - start.frame.position) * self.dt
        timestamp_us = round((start.frame.timestamp + relative_time) * 1e6)
        actors = self._tracks().actor_states(start.payload["scene_id"], timestamp_us)
        actors = {key: state for key, state in actors.items() if key not in self._filtered_tracks}
        return pose, timestamp_us, actors

    def _recorded_sensor_geometry(self, position: int, sensor: str) -> SensorGeometry:
        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == sensor), sensor)
        if native_camera not in self.renderer.cameras:
            raise ValueError(f"Unknown NuRec camera {sensor!r}")
        camera = self.renderer.camera_model(native_camera)
        if camera is None:
            raise ValueError(f"NuRec camera {sensor!r} has no camera geometry")
        pose, _, _ = self._recorded_capture(position)
        return SensorGeometry(aliases.get(native_camera, native_camera), camera,
                              self.renderer.cam_from_world(pose, native_camera))

    def _recorded_sensor_view(self, position: int, sensor: str, observation: Observation) -> SensorView:
        """View of a recorded past capture with its recorded actors and, when enabled, mesh depth."""
        from avsectester.closed_loop.hooks import CameraPoseRequest
        from avsectester.data_types import read_camera_rgb

        geometry = self._recorded_sensor_geometry(position, sensor)
        pose, timestamp_us, actors = self._recorded_capture(position)
        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == sensor), sensor)
        captured = read_camera_rgb(observation, geometry.sensor)
        if captured is None:
            raise ValueError(f"Missing NuRec camera {geometry.sensor!r}")
        depth = None
        if self.depth != "none":
            from avsectester.geometry.mesh import camera_depth

            depth = _actor_depth(geometry.camera, geometry.cam_from_world, actors, camera_depth(
                geometry.camera, geometry.cam_from_world, [_scene_raycaster(self._start.payload["usdz_path"])]))
        return SensorView(
            geometry.sensor, captured[1], geometry.camera, geometry.cam_from_world, actors,
            self._start.recorded_ego[position], observation.t, observation.frame,
            sensor_from_world=BODY_FROM_OPTICAL @ geometry.cam_from_world, depth=depth,
            native={"render_request": CameraPoseRequest(pose, native_camera), "timestamp_us": timestamp_us,
                    "dynamic_objects": self._dynamic_objects(timestamp_us, actors)},
            object_occlusion=self.simulator.object_occlusion,
            occlusion_tolerance_m=self.simulator.occlusion_tolerance_m,
        )

    def _composite_plan(self, view: SensorView, objects: Sequence[PlacedObject]) -> CompositePlan:
        """Plan with the visibility of rendered probes at the view's own capture, when configured."""
        if self.simulator.object_occlusion == "geometry":
            return super()._composite_plan(view, objects)
        from avsectester.rendering import plan_composite

        assert view.camera is not None and view.cam_from_world is not None
        request = view.native["render_request"]
        visibility = self.renderer.render_scene_visibility(
            request.pose, view.native["timestamp_us"], request.camera, objects,
            dynamic_objects=view.native["dynamic_objects"],
            occlusion_tolerance_m=self.simulator.occlusion_tolerance_m)
        return plan_composite(view.camera, view.cam_from_world, objects, scene_visibility=visibility)

    def _observe_recorded(self, position: int) -> Observation:
        pose, timestamp_us, actors = self._recorded_capture(position)
        start = self._start
        assert start is not None
        relative_frame = position - start.frame.position
        relative_time = relative_frame * self.dt
        aliases = self.simulator.camera_aliases
        sensor_data = {
            aliases.get(camera, camera): self.renderer.render(
                pose, timestamp_us, camera,
                dynamic_objects=self._dynamic_objects(timestamp_us, actors))
            for camera in self.renderer.cameras
        }
        vehicle_state = self._recorded_planar_state(position)
        # Route generation may extend its stored polyline. A past query must not change the
        # live runtime's navigation state or any later model-visible route.
        route = deepcopy(self._route).route(pose)
        return Observation(
            t=relative_time, frame=relative_frame, sensor_data=sensor_data,
            calibration={aliases.get(key, key): deepcopy(value)
                         for key, value in self.renderer.calibration.items()},
            vehicle_state=vehicle_state, ego_speed=vehicle_state.speed, route=route,
        )

    def _observe(self) -> Observation:
        from avsectester.closed_loop.hooks import CameraPoseRequest

        aliases = self.simulator.camera_aliases
        sensor_data = {}
        self._capture_actor_states.clear()
        self._capture_depth.clear()
        calibration = {aliases.get(key, key): value for key, value in self.renderer.calibration.items()}
        for camera in self.renderer.cameras:
            # A render request may use a false viewpoint without moving the physical ego.
            request = self._emit(
                "render.pre", CameraPoseRequest(self._world_from_ego.copy(), camera), sensor=camera,
            )
            self._render_requests[camera] = deepcopy(request)
            if request.camera != camera:
                output_key = aliases.get(camera, camera)
                calibration.pop(output_key, None)
                if request.camera in self.renderer.calibration:
                    calibration[output_key] = self.renderer.calibration[request.camera]
            actors = self._actors_at(self._timestamp_us(self.pose))
            dynamic_objects = self._dynamic_objects(self._timestamp_us(self.pose), actors)
            image = self.renderer.render(request.pose, self._timestamp_us(self.pose), request.camera,
                                         dynamic_objects=dynamic_objects)
            self._capture_dynamic_objects[camera] = deepcopy(dynamic_objects)
            self._capture_actor_states[camera] = deepcopy(actors)
            sensor_data[aliases.get(camera, camera)] = self._emit(
                "render.post", image, sensor=camera,
            )
        return self.copy_observation(Observation(
            t=self.pose.t,
            frame=self.frame,
            sensor_data=sensor_data,
            calibration=calibration,
            vehicle_state=self.pose,
            ego_speed=self.pose.speed,
            route=self._route.route(self._ego_state().pose),
        ))

    def close(self) -> None:
        self._capture_depth.clear()
        try:
            self._close_traffic_session()
        finally:
            try:
                self.renderer.close()
            finally:
                super().close()

    # In-process world checkpoints retain frozen assets and loaded harmonizer references.

    def checkpoint(self) -> dict:
        """Capture a replay checkpoint; active reactive-traffic sessions cannot be checkpointed."""
        if self.simulator.traffic is not None:
            raise RuntimeError("AlpaSim traffic sessions cannot be checkpointed")
        return {"pose": asdict(self.pose), "frame": self.frame,
                "world_from_ego": self._world_from_ego.copy(),
                "ground_contact_failures": dict(self._ground_contact_failures),
                "actor_overrides": deepcopy(self._actor_overrides),
                "objects": dict(self._objects)}

    def restore(self, ckpt: dict) -> None:
        """Restore a saved replay checkpoint into this runtime."""
        if self.simulator.traffic is not None:
            raise RuntimeError("AlpaSim traffic sessions cannot be restored from checkpoints")
        self.pose = PlanarEgoState(**ckpt["pose"])
        reset_dynamics = getattr(self.dynamics, "reset", None)
        if callable(reset_dynamics):
            reset_dynamics(deepcopy(self.pose))
        self.frame = ckpt["frame"]
        self._world_from_ego = ckpt["world_from_ego"].copy()
        self._ground_contact_failures = Counter(ckpt["ground_contact_failures"])
        self._actor_overrides = deepcopy(ckpt["actor_overrides"])
        # Frozen placed assets and harmonizer configurations are shared; copying a
        # harmonizer's loaded model would duplicate heavyweight native resources.
        self._objects = dict(ckpt["objects"])
        self._render_requests.clear()
        self._world_initialized = False
        self._capture = None
        self._take_capture()
        self._world_initialized = True


def _compose_pose(a, b, pb):
    """SE(3) compose two protobuf poses: ``a @ b``. AlpaSim builds the camera's world pose as
    ``world_rig @ rig_to_camera`` (rig_to_camera = the camera's pose in the rig)."""
    aw, ax, ay, az = a.quat.w, a.quat.x, a.quat.y, a.quat.z
    bw, bx, by, bz = b.quat.w, b.quat.x, b.quat.y, b.quat.z
    q = pb.Quat(  # quaternion product a*b
        w=aw * bw - ax * bx - ay * by - az * bz,
        x=aw * bx + ax * bw + ay * bz - az * by,
        y=aw * by - ax * bz + ay * bw + az * bx,
        z=aw * bz + ax * by - ay * bx + az * bw,
    )
    tx, ty, tz = (
        b.vec.x,
        b.vec.y,
        b.vec.z,
    )  # rotate b's translation by a's rotation: v + 2w(q×v)+2(q×(q×v))
    cx, cy, cz = ay * tz - az * ty, az * tx - ax * tz, ax * ty - ay * tx
    ccx, ccy, ccz = ay * cz - az * cy, az * cx - ax * cz, ax * cy - ay * cx
    vec = pb.Vec3(
        x=a.vec.x + tx + 2 * (aw * cx + ccx),
        y=a.vec.y + ty + 2 * (aw * cy + ccy),
        z=a.vec.z + tz + 2 * (aw * cz + ccz),
    )
    return pb.Pose(vec=vec, quat=q)


__all__ = [
    'NuRecSimulator',
    'NuRecSimulatorRuntime',
]
