"""NuRec live worlds with independent renderer, actor state and ego dynamics."""

from __future__ import annotations

import copy
import dataclasses
import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any, TYPE_CHECKING
import zipfile
from collections import Counter

from avsectester.closed_loop.simulators.ego_dynamics import EgoDynamics, PlanarEgoState, KinematicBicycle
from avsectester.closed_loop.simulators.base import SensorGeometry, SensorView, Simulator, SimulatorRuntime
from avsectester.closed_loop.simulators.navigation import RouteGenerator
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.geometry.poses import validated_transform, validated_vector
from avsectester.data_types import ActorState, Observation, WorldSnapshot

from .depth import _actor_depth, _scene_raycaster, _ground_raycaster
from .renderer import Renderer, planar_rig_pose

if TYPE_CHECKING:
    from avsectester.closed_loop.hooks import CameraPoseRequest
    from avsectester.geometry.cameras import Camera
    import numpy as np


class NuRecSimulator(Simulator):
    """A NuRec neural reconstruction with in-process ego dynamics and a pluggable renderer.

    ``renderer`` is a configuration template that is never connected itself: every runtime
    renders through its own copy (``Renderer.copy``), with its own connection, loaded scene and
    origin. ``dynamics`` is also a template: every runtime integrates with its own copy, so a
    handler that changes it, for example ``max_brake``, changes only that runtime.
    """

    dataset_kinds = ("NuRecDataset",)
    can_render = True
    can_observe_past = True
    can_insert_objects = True

    def __init__(
        self,
        config: dict,
        renderer: Renderer,
        dynamics: EgoDynamics | None = None,
        *,
        depth: str = "mesh",
        route_generator: str = "map",
        physics_update_mode: str = "ego_only",
        ground_mesh: str = "mesh",
    ) -> None:
        if renderer is None:
            raise TypeError("NuRecSimulator requires a renderer; configure NuRecRenderer")
        if not isinstance(depth, str) or depth not in {"mesh", "none"}:
            raise ValueError("NuRec depth must be 'mesh' or 'none'")
        if physics_update_mode not in {"ego_only", "none"}:
            raise ValueError("physics_update_mode must be 'ego_only' or 'none'")
        if ground_mesh not in {"mesh", "mesh_ground"}:
            raise ValueError("ground_mesh must be 'mesh' or 'mesh_ground'")
        if depth == "mesh" or physics_update_mode == "ego_only":
            from avsectester.geometry.mesh import MeshRaycaster

            MeshRaycaster.check_dependencies()
        self.depth = depth
        if route_generator not in {"map", "recorded", "none"}:
            raise ValueError("route_generator must be 'map', 'recorded' or 'none'")
        self.route_generator = route_generator
        self.physics_update_mode = physics_update_mode
        self.ground_mesh = ground_mesh
        self.config = deepcopy(config)
        self.simulator_version = str(self.config.get("simulator_version", "1"))
        self.renderer = renderer
        self.dynamics = dynamics or KinematicBicycle(**self.config.get("dynamics", {}))
        self.dt = float(self.config.get("dt", 0.05))
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("NuRec decision time step must be positive and finite")

    def create_runtime(self, seed: int | None = None) -> NuRecSimulatorRuntime:
        """A new live world; NuRec rendering and ego dynamics are deterministic, so ``seed`` is unused."""
        return NuRecSimulatorRuntime(self)

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
        return self.dt

    @property
    def identity(self):
        return {**super().identity, "config": deepcopy(self.config),
                "depth": self.depth,
                "route_generator": self.route_generator,
                "physics_update_mode": self.physics_update_mode,
                "ground_mesh": self.ground_mesh,
                "renderer": type(self.renderer).__name__,
                "resolution_hw": deepcopy(getattr(self.renderer, "resolution_hw", {})),
                "dynamics": {"type": f"{type(self.dynamics).__module__}:{type(self.dynamics).__qualname__}",
                             "options": asdict(self.dynamics) if dataclasses.is_dataclass(self.dynamics) else {}},
                "endpoint": getattr(self.renderer, "endpoint", None)}


class NuRecSimulatorRuntime(SimulatorRuntime):
    """One live NuRec world: the ego pose and dynamics loop, and its own renderer connection."""

    def __init__(self, simulator: NuRecSimulator) -> None:
        super().__init__(simulator)
        self.config = deepcopy(simulator.config)
        # Duck-typed renderers without ``copy`` are copied shallowly, like Renderer.copy.
        copy_renderer = getattr(simulator.renderer, "copy", None)
        self.renderer = copy_renderer() if callable(copy_renderer) else copy.copy(simulator.renderer)
        self.dynamics = deepcopy(simulator.dynamics)
        self.dt = simulator.dt
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
        aliases = self.config.get("camera_aliases", {})
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
        recorded = None if self._start is None else self._start.payload["ego"]
        velocity = pose[:3, :3] @ np.array([self.pose.speed, self.pose.lateral_speed, 0.])
        return ActorState(pose, tuple(float(value) for value in velocity), "vehicle",
                          None if recorded is None else recorded.extent,
                          (0., 0., 0.) if recorded is None else recorded.box_center,
                          (0., 0., 0.) if recorded is None else recorded.box_rotation)

    @property
    def termination_details(self) -> dict[str, Any]:
        return {"ground_contact_failures": dict(self._ground_contact_failures)}

    def _state(self) -> WorldSnapshot:
        """Physical ego and current traffic, detached from model-visible observations."""
        return WorldSnapshot(self.pose.t, self.frame, deepcopy(self.pose), self.pose.speed,
                             deepcopy(self._actors_at(self._timestamp_us(self.pose))),
                             ego=self._ego_state())

    def _timestamp_us(self, pose: PlanarEgoState) -> int:
        origin = 0. if self._start is None else self._start.frame.timestamp
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

        actors = dict(self._recorded_actors(timestamp_us))
        for actor_id, (origin_us, state) in self._actor_overrides.items():
            pose = state.pose.copy()
            pose[:3, 3] += np.asarray(state.velocity) * ((timestamp_us - origin_us) / 1e6)
            actors[actor_id] = dataclasses.replace(state, pose=pose)
        return actors

    def set_actor_state(self, actor_id: str, state: ActorState) -> None:
        """Override a current reconstructed actor's pose and world velocity until reset.

        The given pose is its physical state at the current clock. Subsequent captures
        extrapolate translation with the supplied velocity, retaining recorded appearance
        and dimensions. A previous capture and its sensor-view geometry are unchanged.
        """
        if actor_id == "ego":
            raise ValueError("ego is controlled by the driving system")
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
        self._actor_overrides[actor_id] = (self._timestamp_us(self.pose), replacement)

    def _initialize_world(self, start, initialization, environment):
        if initialization is not None:
            raise ValueError("CARLA replay initialization does not apply to NuRec")
        if environment:
            raise ValueError(f"Unsupported NuRec environment overrides: {sorted(environment)}")
        if self.depth == "mesh" and start is not None:
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
        config = deepcopy(self.config)
        if start is not None:
            ego = start.payload["ego"]
            config["scene"] = start.payload.get("scene_id")
            config["ego0"] = asdict(self._recorded_planar_state(start.frame.position, ego))
            self._recorded_end = start.payload["terminal_timestamp"] - start.frame.timestamp
            self.initialization_report = {
                "kind": "recorded", "start_frame": asdict(start.frame),
                "time_step": start.payload.get("time_step"),
                "interpolation": start.payload.get("interpolation"),
                "terminal_timestamp": start.payload["terminal_timestamp"], "pre_roll_steps": 0,
            }
        reset_origin = getattr(self.renderer, "reset_origin", None)
        self._route = RouteGenerator(
            self.simulator.route_generator,
            () if start is None else [state.pose[:3, 3] for state in start.recorded_ego],
            None if start is None else start.scenario_map,
        )
        if self.simulator.physics_update_mode == "ego_only":
            if start is None:
                raise ValueError("NuRec ground contact requires a ScenarioStart; "
                                 "select physics_update_mode='none' without a recording")
            if start.payload["ego"].extent is None:
                raise ValueError("NuRec ground contact requires ego bounding-box extent")
            self._ground_contact = _ground_raycaster(
                start.payload["usdz_path"], self.simulator.ground_mesh + ".ply")
        if callable(reset_origin):
            reset_origin(start)
        elif start is not None:
            raise TypeError("Source initialization requires renderer.reset_origin(ScenarioStart)")
        self.pose = PlanarEgoState(**config.get("ego0", {}))
        self._world_from_ego = (start.payload["ego"].pose.copy() if start is not None else
            planar_rig_pose(self.pose.x, self.pose.y, self.pose.yaw))
        reset_dynamics = getattr(self.dynamics, "reset", None)
        if callable(reset_dynamics):
            reset_dynamics(deepcopy(self.pose))
        self.renderer.load_scene(config.get("scene"))
        self.frame = 0

    def _advance_native(self, control):
        if self._recorded_end is not None and round((self.pose.t + self.dt) * 1e6) > round(self._recorded_end * 1e6):
            raise ValueError("NuRec step exceeds terminal recorded support")
        self._render_requests.clear()
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

    @property
    def sensors(self) -> frozenset[str]:
        return self.camera_sensors

    @property
    def camera_sensors(self) -> frozenset[str]:
        aliases = self.config.get("camera_aliases", {})
        return frozenset(aliases.get(camera, camera) for camera in self.renderer.cameras
                         if self.renderer.camera_model(camera) is not None)

    def sensor_geometry(self, sensor: str) -> SensorGeometry:
        if not self._world_initialized:
            raise RuntimeError("Reset the NuRec simulator before requesting camera geometry")
        aliases = self.config.get("camera_aliases", {})
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

        aliases = self.config.get("camera_aliases", {})
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
            native={"render_request": request, "timestamp_us": timestamp},
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
        return pose, timestamp_us, self._tracks().actor_states(start.payload["scene_id"], timestamp_us)

    def _recorded_sensor_geometry(self, position: int, sensor: str) -> SensorGeometry:
        aliases = self.config.get("camera_aliases", {})
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
        from avsectester.data_types import read_camera_rgb

        geometry = self._recorded_sensor_geometry(position, sensor)
        _, timestamp_us, actors = self._recorded_capture(position)
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
            native={"timestamp_us": timestamp_us},
        )

    def _observe_recorded(self, position: int) -> Observation:
        pose, timestamp_us, actors = self._recorded_capture(position)
        start = self._start
        assert start is not None
        relative_frame = position - start.frame.position
        relative_time = relative_frame * self.dt
        aliases = self.config.get("camera_aliases", {})
        sensor_data = {
            aliases.get(camera, camera): self.renderer.render(
                pose, timestamp_us, camera,
                dynamic_objects=tuple((identity, state.pose) for identity, state in actors.items()))
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

        aliases = self.config.get("camera_aliases", {})
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
            image = self.renderer.render(request.pose, self._timestamp_us(self.pose), request.camera,
                dynamic_objects=tuple((identity, state.pose) for identity, state in actors.items()))
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
            self.renderer.close()
        finally:
            super().close()

    # In-process world checkpoints retain frozen assets and loaded harmonizer references.

    def checkpoint(self) -> dict:
        return {"pose": asdict(self.pose), "frame": self.frame,
                "world_from_ego": self._world_from_ego.copy(),
                "ground_contact_failures": dict(self._ground_contact_failures),
                "actor_overrides": deepcopy(self._actor_overrides),
                "objects": dict(self._objects)}

    def restore(self, ckpt: dict) -> None:
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
