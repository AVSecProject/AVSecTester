"""NuRec live worlds with independent renderer, actor state and ego dynamics."""

from __future__ import annotations

import copy
import bisect
import dataclasses
import math
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, TYPE_CHECKING
import zipfile
from collections import Counter, OrderedDict, deque
from collections.abc import Mapping

from avsectester.closed_loop.simulators.ego_dynamics import (
    EgoDynamics, EgoMotion, KinematicBicycle, PlanarEgoState, _advance_dynamics,
)
from avsectester.validation import timestamp_to_microseconds
from avsectester.closed_loop.timing import (
    SensorTiming, TimingCapabilities, TimingConfiguration, seconds_to_microseconds,
)
from avsectester.closed_loop.simulators.base import (
    ClockSnapshot, SensorGeometry, SensorView, Simulator, SimulatorRuntime, UnsupportedStartError, validated_camera_aliases,
)
from avsectester.closed_loop.simulators.navigation import RouteGenerator
from avsectester.geometry.cameras import BODY_FROM_OPTICAL
from avsectester.geometry.lidars import LidarCalibration
from avsectester.geometry.poses import interpolate_transform, validated_transform, validated_vector
from avsectester.validation import require_nonnegative_number, require_positive_number
from avsectester.data_types import ActorState, Control, Observation, WorldSnapshot

from .depth import _actor_depth, _scene_raycaster, _ground_raycaster
from .renderer import Renderer, planar_rig_pose
from .traffic import AlpaSimTraffic, TrafficSession, TrafficTrack, TRAFFIC_STEP_US

if TYPE_CHECKING:
    from collections.abc import Sequence

    from avsectester.closed_loop.hooks import CameraPoseRequest
    from avsectester.datasets.base import ScenarioStart
    from avsectester.rendering import (
        CompositePlan, CompositeResult, MeshAsset, ObjectAsset, ObjectComposition, PlacedObject,
    )
    import numpy as np


# Include soft edges below 0.5 coverage, matching composite harmonizers' full owned pixels.
_HARMONIZER_COVERAGE_THRESHOLD = 8 / 255
FORCE_GT_REFERENCE_HORIZON_US = 5_000_000


@dataclasses.dataclass(frozen=True)
class _RenderTracks:
    """Actor and inserted-object entries used for one sensor request."""

    actor_tracks: tuple[tuple[str, np.ndarray], ...]
    object_tracks: Mapping[str, tuple[str, np.ndarray]]

    @property
    def dynamic_objects(self) -> tuple[tuple[str, np.ndarray], ...]:
        """Actor and inserted-object entries in render-request order."""
        return self.actor_tracks + tuple(self.object_tracks.values())

    def native(self) -> dict[str, Any]:
        """Detached request entries for a sensor view."""
        return deepcopy({"dynamic_objects": self.dynamic_objects, "actor_tracks": self.actor_tracks,
                         "object_tracks": dict(self.object_tracks)})


@dataclasses.dataclass(frozen=True)
class _CaptureRequest:
    tracks: _RenderTracks
    actors: Mapping[str, ActorState]
    exposure: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _MotionInterval:
    start_us: int
    end_us: int
    motion: EgoMotion
    initial: WorldSnapshot
    final: WorldSnapshot | None
    actor_overrides: dict[str, tuple[int, ActorState]]
    follows_recording: bool = False
    recorded_poses: tuple[np.ndarray, ...] = ()


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

    ``object_insertion="composite"`` blends inserted objects into camera captures. With
    ``"scene"``, NRE renders objects together with the reconstruction in every NRE sensor.
    Scene insertion requires the renderer's asset directory. ``object_occlusion`` and
    ``occlusion_tolerance_m`` apply only to composite insertion.

    ``object_occlusion`` chooses how composited world objects are hidden behind the scene. The
    default ``"rendered_probes"`` renders opaque probe copies of the objects inside the
    reconstruction (``Renderer.render_scene_visibility``), so the reconstruction that produces
    the images also hides the objects. It needs no mesh. ``"geometry"`` estimates occlusion from
    the configured ``depth`` (mesh plus actor cuboids, or actor cuboids alone).
    ``occlusion_tolerance_m`` (default 0.15 m) applies to both: the reconstruction hides an
    object only when it lies more than about this distance in front of it. Reconstructed
    Gaussians extend some centimeters in front of their surfaces, so an object placed on or just
    in front of a reconstructed surface needs this margin to stay visible.
    ``validate_start`` raises ``UnsupportedStartError`` when the configured navigation route
    cannot be built from the start's recording and map. Each recording is checked once.
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
        object_insertion: str = "composite",
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
        if not isinstance(object_insertion, str) or object_insertion not in {"composite", "scene"}:
            raise ValueError("object_insertion must be 'composite' or 'scene'")
        if object_insertion == "scene" and not getattr(renderer, "asset_directory", None):
            raise ValueError("Scene object insertion requires renderer.asset_directory")
        self.object_insertion = object_insertion
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
        if traffic is not None and seconds_to_microseconds(self.time_step) % TRAFFIC_STEP_US:
            raise ValueError("NuRec decision periods must be whole multiples of the traffic step")
        self.traffic = traffic
        self.min_track_duration_s = min_track_duration_s
        self._route_failures: dict[str, str | None] = {}

    def validate_start(self, start: ScenarioStart | None) -> None:
        if start is None:
            raise ValueError("NuRec requires a ScenarioStart for the recorded clock, ego pose and scene")
        super().validate_start(start)
        if start.frame.position < self.past_steps:
            raise UnsupportedStartError(
                f"AlpaSim traffic requires {self.past_steps} past decision steps")
        if start.frame.position < self.observation_lead_steps:
            raise UnsupportedStartError(
                "NuRec LiDAR requires 1 recorded decision step before the start")
        if self.route_generator == "none":
            return
        # The route depends only on the complete recording and the map, not on the start frame.
        # Within one configured simulator a scenario's recording and map are fixed.
        if start.scenario_id not in self._route_failures:
            try:
                _route_generator(self.route_generator, start)
                self._route_failures[start.scenario_id] = None
            except ValueError as error:
                self._route_failures[start.scenario_id] = str(error)
        failure = self._route_failures[start.scenario_id]
        if failure is not None:
            raise UnsupportedStartError(f"Navigation route cannot be built: {failure}")

    def validate_start_timing(self, start: ScenarioStart | None, timing: TimingConfiguration) -> None:
        """Reject starts without enough source history for the selected LiDAR frame clock."""
        self.validate_start(start)
        lead = self.observation_lead_steps_for(timing)
        if start.frame.position < lead:
            raise UnsupportedStartError(f"NuRec LiDAR requires {lead} recorded decision steps "
                                        "before the start for its scheduled scan window")

    def create_runtime(self, seed: int | None = None) -> NuRecSimulatorRuntime:
        """A new world. The optional traffic session receives seed for its log."""
        return NuRecSimulatorRuntime(self, seed)

    @property
    def timing_capabilities(self) -> TimingCapabilities:
        """NuRec can report intermediate motion using the configured ego dynamics."""
        return TimingCapabilities(captures_between_decisions=True, motion_reports=True,
                                  recorded_ego_motion=True, camera_exposure=True)

    def validate_timing(self, timing: TimingConfiguration) -> None:
        super().validate_timing(timing)
        lidar = self.renderer.lidar
        if lidar is not None and timing.sensors.get(lidar, SensorTiming()).shutter_duration_us:
            raise ValueError("NuRec LiDAR does not accept shutter_duration_us")

    def observation_lead_steps_for(self, timing: TimingConfiguration) -> int:
        if self.renderer.lidar is None:
            return 0
        period = seconds_to_microseconds(self.time_step)
        interval = timing.sensors.get(self.renderer.lidar, SensorTiming()).resolved_frame_interval_us(period)
        return (interval + period - 1) // period

    @property
    def past_steps(self) -> int:
        """Recorded decision history needed by the optional traffic model."""
        return 0 if self.traffic is None else self.traffic.history_steps - 1

    @property
    def observation_lead_steps(self) -> int:
        """One recorded step supplies the start pose for a LiDAR scan, or zero without LiDAR."""
        return self.observation_lead_steps_for(TimingConfiguration())

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
                             "options": {name: value for name, value in asdict(self.dynamics).items()
                                         if name in {item.name for item in dataclasses.fields(self.dynamics)
                                                     if item.init}}
                             if dataclasses.is_dataclass(self.dynamics) else {}},
                "endpoint": getattr(self.renderer, "endpoint", None)}
        if self.renderer.lidar is not None:
            identity["lidar"] = self.renderer.lidar
        if self.object_insertion != "composite":
            identity["object_insertion"] = self.object_insertion
        if self.traffic is not None:
            identity["traffic"] = {
                "type": f"{type(self.traffic).__module__}:{type(self.traffic).__qualname__}",
                "options": {name: getattr(self.traffic, name) for name in (
                    "endpoint", "history_steps", "rpc_timeout_s")}}
        return identity


class NuRecSimulatorRuntime(SimulatorRuntime):
    """One live NuRec world: the ego pose and dynamics loop, and its own renderer connection.

    Segment selection reserves source history through ``observation_lead_steps_for(timing)``.
    Runtime recording-coverage checks are a final safeguard for the complete LiDAR scan
    before each queried timestamp.
    """

    def __init__(self, simulator: NuRecSimulator, seed: int | None = None) -> None:
        super().__init__(simulator)
        self.seed = seed
        self._traffic_session: TrafficSession | None = None
        self._recorded_tracks: dict[str, dict[str, Any]] = {}
        self._filtered_tracks: set[str] = set()
        self._traffic_states: dict[str, ActorState] = {}
        self._traffic_timestamp_us: int | None = None
        self._traffic_history: deque[tuple[int, dict[str, ActorState]]] = deque()
        self._capture_requests: dict[str, _CaptureRequest] = {}
        self._nominal_assets: dict[int, tuple[ObjectAsset, ObjectAsset]] = {}
        self._mesh_raycasters: dict[str, tuple[MeshAsset, np.ndarray, Any]] = {}
        # Duck-typed renderers without ``copy`` are copied shallowly, like Renderer.copy.
        copy_renderer = getattr(simulator.renderer, "copy", None)
        self.renderer = copy_renderer() if callable(copy_renderer) else copy.copy(simulator.renderer)
        self.dynamics = deepcopy(simulator.dynamics)
        self._recording_interval_us = seconds_to_microseconds(simulator.time_step)
        self.depth = simulator.depth
        self.pose = PlanarEgoState()
        self._state_time_us = 0
        self._motion_interval: _MotionInterval | None = None
        self._motion_history: deque[_MotionInterval] = deque()
        self.frame = 0
        self._render_requests: dict[str, CameraPoseRequest] = {}
        self._capture_geometries: dict[str, SensorGeometry] = {}
        self._recorded_geometry_cache: dict[tuple[int, str], SensorGeometry] = {}
        self._last_recorded_capture = None
        self._force_gt_poses = ()
        self._force_gt_corrected_poses = {}
        # Recorded actor tracks, by archive. Recorded files are immutable.
        self._track_datasets: dict[tuple, Any] = {}
        self._actor_states: OrderedDict[int, dict[str, ActorState]] = OrderedDict()
        self._actor_overrides: dict[str, tuple[int, ActorState]] = {}
        self._recorded_end_us: int | None = None
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
            try:
                return self._latest_render_request(aliases.get(native_camera, native_camera))
            except KeyError:
                raise RuntimeError(f"No current render request for camera {camera!r}") from None
        return deepcopy(self._render_requests[native_camera])

    def _ego_state(self):
        """Physical three-dimensional rig pose, owned by this runtime."""
        return self._ego_state_at(self.pose, self._world_from_ego)

    def _ego_state_at(self, planar: PlanarEgoState, world_from_ego: np.ndarray) -> ActorState:
        """Compute rig geometry and velocity without modifying the live physical pose."""
        import numpy as np

        pose = world_from_ego.copy()
        recorded = self._start.payload["ego"]
        velocity = pose[:3, :3] @ np.array([planar.speed, planar.lateral_speed, 0.])
        return ActorState(pose, tuple(float(value) for value in velocity), "vehicle",
                          recorded.extent, recorded.box_center, recorded.box_rotation)

    @property
    def termination_details(self) -> dict[str, Any]:
        return {"ground_contact_failures": dict(self._ground_contact_failures)}

    def _native_clock(self) -> ClockSnapshot:
        return ClockSnapshot(self.pose.t, self.frame)

    def _state(self) -> WorldSnapshot:
        """Physical ego and current traffic, detached from model-visible observations."""
        return WorldSnapshot(self.pose.t, self.frame, deepcopy(self.pose), self.pose.speed,
                             deepcopy(self._actors_at(self._timestamp_us())),
                             ego=self._ego_state())

    def _timestamp_us(self) -> int:
        return timestamp_to_microseconds(self._start.frame.timestamp) + self._state_time_us

    def _recorded_planar_state(self, time_us: int, ego: ActorState | None = None) -> PlanarEgoState:
        """Rig-frame motion with finite differences over the actual source time interval."""
        import numpy as np

        start = self._start
        recorded = start.recorded_ego
        position, remainder = divmod(start.frame.position * self._recording_interval_us + time_us,
                                    self._recording_interval_us)
        current = self._recorded_ego(time_us) if ego is None else ego
        if current.velocity is None:
            raise ValueError(f"Recorded ego state at {time_us} us has no velocity")
        pose = np.asarray(current.pose)
        velocity = pose[:3, :3].T @ validated_vector(current.velocity, "recorded ego velocity")
        yaw = math.atan2(pose[1, 0], pose[0, 0])
        yaw_rate, acceleration = 0., 0.
        neighbor = position if remainder else (position - 1 if position else 1)
        if 0 <= neighbor < len(recorded):
            other = recorded[neighbor]
            if other.velocity is None:
                raise ValueError(f"Recorded ego state at position {neighbor} has no velocity")
            other_velocity = np.asarray(other.pose)[:3, :3].T @ validated_vector(
                other.velocity, "recorded ego velocity")
            interval = (time_us - (neighbor - start.frame.position) * self._recording_interval_us) / 1e6
            other_yaw = math.atan2(other.pose[1, 0], other.pose[0, 0])
            difference = yaw - other_yaw
            yaw_rate = math.atan2(math.sin(difference), math.cos(difference)) / interval
            acceleration = (velocity[0] - other_velocity[0]) / interval
        return PlanarEgoState(float(pose[0, 3]), float(pose[1, 3]), yaw, float(velocity[0]),
                              time_us / 1e6, lateral_speed=float(velocity[1]),
                              yaw_rate=float(yaw_rate), acceleration=float(acceleration))

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
        # Keep both scan endpoints without reinterpolating immutable recorded tracks.
        if timestamp_us not in self._actor_states:
            self._actor_states[timestamp_us] = self._tracks().actor_states(
                self._start.payload["scene_id"], timestamp_us)
        self._actor_states.move_to_end(timestamp_us)
        while len(self._actor_states) > 2:
            self._actor_states.popitem(last=False)
        return self._actor_states[timestamp_us]

    def _actors_at(self, timestamp_us: int, *,
                   actor_overrides: Mapping[str, tuple[int, ActorState]] | None = None) -> dict[str, ActorState]:
        actors = self._recorded_scene_actors(timestamp_us)
        self._apply_recorded_interval(actors, timestamp_us)
        self._apply_traffic_states(actors, timestamp_us)
        self._apply_actor_overrides(actors, timestamp_us, actor_overrides)
        return actors

    def _recorded_scene_actors(self, timestamp_us: int) -> dict[str, ActorState]:
        return {key: state for key, state in self._recorded_actors(timestamp_us).items()
                  if key not in self._filtered_tracks}

    def _apply_recorded_interval(self, actors: dict[str, ActorState], timestamp_us: int) -> None:
        if self._traffic_session is None and self._start is not None:
            offset = timestamp_us - timestamp_to_microseconds(self._start.frame.timestamp)
            intervals = ([self._motion_interval] if self._motion_interval is not None else [])
            intervals += list(reversed(self._motion_history))
            interval = next((item for item in intervals if item.follows_recording and
                             item.start_us < offset < item.end_us), None)
            if interval is not None:
                right_us = timestamp_to_microseconds(self._start.frame.timestamp) + interval.end_us
                right = self._recorded_actors(right_us)
                fraction = (offset - interval.start_us) / (interval.end_us - interval.start_us)
                for identity, first in interval.initial.actors.items():
                    if identity in right and self._recorded_tracks.get(identity, {}).get("controllable", True):
                        actors[identity] = replace(right[identity],
                            pose=interpolate_transform(first.pose, right[identity].pose, fraction))

    def _apply_traffic_states(self, actors: dict[str, ActorState], timestamp_us: int) -> None:
        import numpy as np

        if timestamp_us == self._traffic_timestamp_us:
            actors.update(self._traffic_states)
        elif self._traffic_session is not None:
            exact = next((states for time, states in self._traffic_history if time == timestamp_us), None)
            if exact is not None:
                actors.update({key: value for key, value in exact.items()
                               if key in self._traffic_session.track_ids
                               and self._recorded_tracks[key]["controllable"]})
            for (left_us, left), (right_us, right) in zip(self._traffic_history, tuple(self._traffic_history)[1:]):
                if exact is None and left_us <= timestamp_us <= right_us:
                    fraction = (timestamp_us - left_us) / (right_us - left_us)
                    for identity in self._traffic_session.track_ids:
                        if (self._recorded_tracks[identity]["controllable"]
                                and identity in left and identity in right):
                            actors[identity] = replace(right[identity],
                                pose=interpolate_transform(left[identity].pose, right[identity].pose, fraction),
                                velocity=(None if left[identity].velocity is None or right[identity].velocity is None
                                          else tuple((1 - fraction) * np.asarray(left[identity].velocity)
                                                     + fraction * np.asarray(right[identity].velocity))))
                    break

    def _apply_actor_overrides(self, actors: dict[str, ActorState], timestamp_us: int,
                               actor_overrides) -> None:
        import numpy as np

        overrides = self._actor_overrides if actor_overrides is None else actor_overrides
        for actor_id, (origin_us, state) in overrides.items():
            pose = state.pose.copy()
            pose[:3, 3] += np.asarray(state.velocity) * ((timestamp_us - origin_us) / 1e6)
            actors[actor_id] = dataclasses.replace(state, pose=pose)

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
        timestamp_us = self._timestamp_us()
        self._actor_overrides[actor_id] = (timestamp_us, replacement)
        self.invalidate_state()
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
        first_us = timestamp_to_microseconds(start.payload["first_timestamp"])
        trajectory = tuple((first_us + index * self._recording_interval_us, state.pose @ rig_from_box)
                           for index, state in enumerate(start.recorded_ego))
        tracks = [TrafficTrack(identity, track["trajectory"], track["extent"],
                               track["label_class"], not track["controllable"])
                  for identity, track in self._recorded_tracks.items() if identity not in short_tracks]
        self._traffic_session = traffic.open_session(
            scene_id=start.payload["scene_id"], random_seed=0 if self.seed is None else self.seed,
            start_us=self._timestamp_us(),
            end_us=timestamp_to_microseconds(start.payload["terminal_timestamp"]),
            ego_extent=ego.extent, ego_trajectory=trajectory, tracks=tracks,
            handover_us=self._timestamp_us() + self.timing.force_gt_duration_us
                        + self.decision_period_us)
        self.initialization_report["traffic"] = {
            "session_uuid": self._traffic_session.uuid,
            "handover_time_us": self._traffic_session.handover_time_us,
            "session_tracks": sorted(self._traffic_session.track_ids),
        }
        timestamp = self._timestamp_us()
        self._traffic_history.append((timestamp, deepcopy(self._actors_at(timestamp))))

    def _dynamic_objects(self, timestamp_us: int, actors: dict[str, ActorState]) -> tuple:
        """The exact actor edits for ordinary captures and inserted-object probes."""
        edits = [(identity, state.pose) for identity, state in actors.items()]
        for identity, state in self._recorded_actors(timestamp_us).items():
            if identity in self._filtered_tracks:
                pose = state.pose.copy()
                pose[2, 3] -= 100.
                edits.append((identity, pose))
        return tuple(edits)

    def _frame_range_us(self, sensor: str, end_time_us: int) -> range:
        if sensor == getattr(self.renderer, "lidar", None):
            clock = self.timing.sensors.get(sensor, SensorTiming())
            interval = clock.resolved_frame_interval_us(self.decision_period_us)
            return range(end_time_us - interval, end_time_us)
        return super()._frame_range_us(sensor, end_time_us)

    def _object_tracks(self, objects: Sequence[PlacedObject]) -> dict[str, tuple[str, np.ndarray]]:
        """Map inserted-object IDs to the explicit scene tracks used for rendering."""
        if self.simulator.object_insertion == "composite":
            return {}
        return dict(zip(
            (placed.object_id for placed in objects),
            self.renderer.scene_object_tracks(objects), strict=True))

    def _supports_asset(self, asset: ObjectAsset | MeshAsset) -> bool:
        """Accept mesh world objects only when NRE renders scene insertion."""
        from avsectester.rendering import MeshAsset, ObjectAsset

        return isinstance(asset, ObjectAsset) or (
            self.simulator.object_insertion == "scene" and isinstance(asset, MeshAsset))

    def _nominal_asset(self, asset: ObjectAsset | MeshAsset) -> ObjectAsset | MeshAsset:
        """The asset showing its nominal textures, built once so renderer caches can reuse it."""
        from avsectester.rendering import MeshAsset

        if isinstance(asset, MeshAsset):
            return asset
        if all(surface.nominal_texture is None for surface in asset.surfaces):
            return asset
        if id(asset) not in self._nominal_assets:
            surfaces = tuple(
                replace(surface, texture=surface.nominal_texture)
                if surface.nominal_texture is not None else surface
                for surface in asset.surfaces)
            # The cached source keeps its id from being reused while the entry exists.
            self._nominal_assets[id(asset)] = (asset, replace(asset, surfaces=surfaces))
        return self._nominal_assets[id(asset)][1]

    def _view_occlusion(self) -> dict[str, Any]:
        """Occlusion source and tolerance shared by current and recorded views."""
        if self.simulator.object_insertion == "scene":
            return {"object_occlusion": "scene_render", "occlusion_tolerance_m": 0.}
        return {
            "object_occlusion": self.simulator.object_occlusion,
            "occlusion_tolerance_m": self.simulator.occlusion_tolerance_m,
        }

    def _simulate_traffic(self, timestamp_us: int, ego: ActorState,
                          previous: dict[str, ActorState], *, force_gt: bool,
                          is_endpoint: bool) -> None:
        session = self._traffic_session
        if session is None:
            raise RuntimeError("AlpaSim traffic session is closed; reset the runtime before stepping")
        current = self._actors_at(timestamp_us)
        overrides = {identity: current[identity].box_transform()
                     for identity in self._actor_overrides if identity in session.track_ids}
        returned = session.simulate(timestamp_us, ego.box_transform(), overrides)
        if force_gt:
            states = {identity: state for identity, state in self._recorded_actors(timestamp_us).items()
                      if identity in session.track_ids and self._recorded_tracks[identity]["controllable"]}
        else:
            states = {}
            for identity, pose in returned.items():
                velocity = None if identity not in previous else tuple(
                    float(value) for value in ((pose[:3, 3] - previous[identity].pose[:3, 3]) / (TRAFFIC_STEP_US / 1e6)))
                track = self._recorded_tracks[identity]
                states[identity] = ActorState(pose, velocity, track["category"], track["extent"])
        self._traffic_states = states
        self._traffic_timestamp_us = timestamp_us
        if not force_gt or is_endpoint:
            self._traffic_history.append((timestamp_us, deepcopy(self._actors_at(timestamp_us))))
        earliest = (timestamp_to_microseconds(self._start.frame.timestamp) + self._interval.start_us
                    - self._sensor_motion_lookback_us())
        while len(self._traffic_history) > 2 and self._traffic_history[1][0] < earliest:
            self._traffic_history.popleft()

    def _initialize_world(self, start, initialization, environment):
        self._close_traffic_session()
        self._force_gt_corrected_poses.clear()
        self._recorded_tracks.clear()
        self._filtered_tracks.clear()
        self._traffic_states.clear()
        self._traffic_timestamp_us = None
        self._traffic_history.clear()
        self._capture_requests.clear()
        self._capture_geometries.clear()
        self._recorded_geometry_cache.clear()
        self._last_recorded_capture = None
        self._force_gt_poses = ()
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
        self._state_time_us = 0
        self._render_requests.clear()
        self._actor_states.clear()
        self._actor_overrides.clear()
        self._recorded_end_us = None
        self._ground_contact_failures.clear()
        self._ground_contact = None
        ego = start.payload["ego"]
        scene_id = start.payload["scene_id"]
        self._recorded_end_us = timestamp_to_microseconds(start.payload["terminal_timestamp"]) - timestamp_to_microseconds(start.frame.timestamp)
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
        self._build_force_gt_trajectory()
        if self._force_gt_poses:
            ego = self._recorded_ego(0)
        self.pose = self._recorded_planar_state(0, ego)
        self._motion_interval = None
        self._motion_history.clear()
        self._world_from_ego = ego.pose.copy()
        reset_dynamics = getattr(self.dynamics, "reset", None)
        if callable(reset_dynamics):
            reset_dynamics(deepcopy(self.pose))
        lidar = self.renderer.lidar
        scene = (scene_id if lidar is None
                 else {"scene_id": scene_id, "usdz_path": start.payload["usdz_path"]})
        self.renderer.load_scene(scene)
        if lidar is not None:
            calibration = self.renderer.calibration[lidar]
            if not isinstance(calibration, LidarCalibration):
                raise TypeError("NuRec LiDAR requires LidarCalibration")
            validated_transform(calibration.rig_from_lidar)
            require_positive_number(calibration.spin_period_s, "LiDAR spin period")
            sensor_timing = self.timing.sensors.get(lidar, SensorTiming())
            duration = sensor_timing.resolved_frame_interval_us(self.decision_period_us)
            if not math.isclose(calibration.spin_period_s, duration / 1e6,
                                rel_tol=0., abs_tol=duration / 1e6 * .01):
                raise ValueError("NuRec LiDAR calibration spin period must match its scheduled "
                                 "acquisition period within 1%")
            camera_names = {self.simulator.camera_aliases.get(camera, camera)
                            for camera in self.renderer.cameras}
            if lidar in camera_names:
                raise ValueError("NuRec LiDAR name must differ from camera output names")
        self.frame = 0
        # Short tracks are measured over the whole recording: movable ones leave the world,
        # and no short track joins a traffic session.
        short_tracks: set[str] = set()
        if "usdz_path" in start.payload:
            self._recorded_tracks = self._tracks().track_trajectories(start.payload["scene_id"])
            minimum_us = timestamp_to_microseconds(self.simulator.min_track_duration_s)
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

    def _coerce_recorded_dynamics(self):
        recorded = self._recorded_planar_state(self._interval.start_us)
        self.pose.speed, self.pose.lateral_speed = recorded.speed, recorded.lateral_speed
        self.pose.yaw_rate, self.pose.acceleration = recorded.yaw_rate, recorded.acceleration
        coerce = getattr(self.dynamics, "coerce_dynamic_state", None)
        if callable(coerce):
            coerce(deepcopy(self.pose))

    def _advance_native(self, control):
        if self._interval.coerce_dynamic_state:
            self._coerce_recorded_dynamics()
        self._advance_motion(control)

    def _advance_force_gt_native(self, control):
        self._coerce_recorded_dynamics()
        self._advance_motion(control, recorded=True)

    def recorded_reference(self, time_us: int, end_us: int) -> Control:
        import numpy as np
        from avsectester.geometry.poses import rotation_to_quaternion

        if (isinstance(time_us, bool) or not isinstance(time_us, int) or time_us < 0
                or isinstance(end_us, bool) or not isinstance(end_us, int) or end_us <= time_us):
            raise ValueError("Recorded reference requires increasing nonnegative integer times")
        if self._recorded_end_us is None or end_us > self._recorded_end_us:
            raise ValueError("Recorded reference exceeds source support")
        assert time_us == self.clock.time_us
        physical = self.state()
        assert physical.ego is not None
        anchor = np.linalg.inv(physical.ego.pose)
        horizon = min(self._recorded_end_us, time_us + FORCE_GT_REFERENCE_HORIZON_US)
        times = sorted({time_us, horizon, *(timestamp for timestamp, _pose in self._force_gt_poses
                                            if time_us < timestamp <= horizon)})
        points = []
        for timestamp in times:
            pose = anchor @ self._recorded_ego(timestamp).pose
            points.append((tuple(pose[:3, 3]), tuple(rotation_to_quaternion(pose[:3, :3])), timestamp))
        return Control(trajectory=points)

    def _advance_motion(self, control, *, recorded=False):
        duration_us = self.decision_period_us
        if self._recorded_end_us is not None and self._interval.end_us > self._recorded_end_us:
            raise ValueError("NuRec step exceeds terminal recorded support")
        self._render_requests.clear()
        uses_traffic = self.simulator.traffic is not None
        previous_actors = self._actors_at(self._timestamp_us()) if uses_traffic else {}
        previous_yaw = self.pose.yaw
        traffic_times = (tuple(range(self._interval.start_us + TRAFFIC_STEP_US,
                                    self._interval.end_us + 1, TRAFFIC_STEP_US)) if uses_traffic else ())
        report_times = tuple(sorted({self._interval.start_us, *self._interval.report_times_us,
                                     self._interval.end_us}))
        if control is None:
            states = []
            for timestamp in report_times:
                state = self._recorded_planar_state(timestamp)
                states.append(state)
            motion = EgoMotion(tuple(states))
        else:
            motion = _advance_dynamics(
                self.dynamics, self.pose, control, duration_us, self._interval.report_times_us)
            if recorded:
                motion = EgoMotion(tuple(self._recorded_planar_state(timestamp)
                                                   for timestamp in report_times))
        if not isinstance(motion, EgoMotion):
            raise TypeError("Ego dynamics advance must return EgoMotion")
        self.pose = deepcopy(motion.states[-1])
        self._state_time_us = self._interval.end_us
        if recorded or control is None:
            world_from_ego = self._recorded_ego(self._interval.end_us).pose.copy()
        else:
            world_from_ego = planar_rig_pose(self.pose.x, self.pose.y, self.pose.yaw,
                                         self._world_from_ego[2, 3])
            world_from_ego[:3, :3] = (world_from_ego[:3, :3]
                @ planar_rig_pose(0, 0, previous_yaw)[:3, :3].T @ self._world_from_ego[:3, :3])
        self._world_from_ego = world_from_ego
        if self._ground_contact is not None and control is not None and not recorded:
            from avsectester.closed_loop.simulators.ground_contact import constrain_to_ground
            import numpy as np

            ego = self._ego_state()
            corrected, status = constrain_to_ground(self._ground_contact, ego.box_transform(), ego.extent)
            if status != "successful_update":
                self._ground_contact_failures[status] += 1
            corrected_rig = corrected @ np.linalg.inv(np.linalg.inv(ego.pose) @ ego.box_transform())
            self._world_from_ego = corrected_rig
            self.pose.x, self.pose.y = self._world_from_ego[:2, 3]
            self.pose.yaw = math.atan2(self._world_from_ego[1, 0], self._world_from_ego[0, 0])
        self.frame += 1
        motion = _MotionInterval(self._interval.start_us, self._interval.end_us, motion,
                                 self._interval.start_state, None,
                                 deepcopy(self._actor_overrides), follows_recording=recorded or control is None,
                                 recorded_poses=tuple(self._recorded_ego(timestamp).pose
                                                      for timestamp in report_times)
                                 if recorded or control is None else ())
        self._motion_interval = motion
        if uses_traffic:
            final_pose = self.pose
            final_ego = self._ego_state()
            for timestamp in traffic_times:
                ego = (self._motion_ego(motion, timestamp, final_ego, final_pose)[1]
                       if timestamp != self._interval.end_us else final_ego)
                timestamp_us = timestamp_to_microseconds(self._start.frame.timestamp) + timestamp
                self._simulate_traffic(timestamp_us, ego, previous_actors,
                    force_gt=recorded, is_endpoint=timestamp == self._interval.end_us)
                previous_actors = self._actors_at(timestamp_us)

    def _reported_state_at(self, time_us: int) -> WorldSnapshot:
        return self._motion_state(self._motion_history[-1], time_us)

    def _completed_interval(self, end_state: WorldSnapshot) -> None:
        assert self._interval is not None
        assert self._motion_interval is not None
        self._motion_interval.final = end_state
        self._motion_history.append(self._motion_interval)
        earliest = self._interval.end_us - self._sensor_motion_lookback_us() - self.decision_period_us
        while len(self._motion_history) > 1 and self._motion_history[1].end_us < earliest:
            self._motion_history.popleft()

    def _motion_ego(self, interval, time_us, final_ego, final_planar):
        import numpy as np

        first = interval.initial
        assert first.ego is not None
        fraction = (time_us - interval.start_us) / (interval.end_us - interval.start_us)
        planar = interval.motion.at(time_us)
        if interval.follows_recording:
            times = [timestamp_to_microseconds(state.t) for state in interval.motion.states]
            right = min(len(times) - 1, bisect.bisect_left(times, time_us))
            left = max(0, right - 1)
            if times[right] == time_us:
                pose = interval.recorded_poses[right].copy()
            else:
                pose = interpolate_transform(interval.recorded_poses[left],
                    interval.recorded_poses[right], (time_us - times[left]) / (times[right] - times[left]))
            planar.x, planar.y = pose[:2, 3]
            planar.yaw = math.atan2(pose[1, 0], pose[0, 0])
        else:
            integrated_end = interval.motion.states[-1]
            planar.x += fraction * (final_planar.x - integrated_end.x)
            planar.y += fraction * (final_planar.y - integrated_end.y)
            yaw_correction = final_planar.yaw - integrated_end.yaw
            planar.yaw += fraction * math.atan2(math.sin(yaw_correction), math.cos(yaw_correction))
            pose = interpolate_transform(first.ego.pose, final_ego.pose, fraction)
            yaw = math.atan2(pose[1, 0], pose[0, 0])
            pose[:3, :3] = planar_rig_pose(0., 0., planar.yaw - yaw)[:3, :3] @ pose[:3, :3]
            pose[:2, 3] = (planar.x, planar.y)
        velocity = pose[:3, :3] @ np.array([planar.speed, planar.lateral_speed, 0.])
        ego = replace(final_ego, pose=pose, velocity=tuple(float(value) for value in velocity))
        return planar, ego

    def _motion_state(self, interval: _MotionInterval, time_us: int) -> WorldSnapshot:
        first, last = interval.initial, interval.final
        assert last is not None and last.ego is not None
        if time_us == interval.start_us:
            return deepcopy(first)
        if time_us == interval.end_us:
            return deepcopy(last)
        planar, ego = self._motion_ego(interval, time_us, last.ego, last.vehicle_state)
        actors = deepcopy(self._actors_at(timestamp_to_microseconds(self._start.frame.timestamp) + time_us,
                                         actor_overrides=interval.actor_overrides))
        return WorldSnapshot(planar.t, first.frame, planar, planar.speed, actors, ego=ego,
                             traffic_lights=deepcopy(first.traffic_lights), objects=deepcopy(last.objects))

    def _activate_reported_state(self, time_us: int, state: WorldSnapshot) -> None:
        self.pose = deepcopy(state.vehicle_state)
        self._state_time_us = time_us
        self._world_from_ego = state.ego.pose.copy()
        self.frame = state.frame
        self._render_requests.clear()
        self._capture_geometries.clear()

    def _sensor_motion_lookback_us(self) -> int:
        scan = 0
        if self.renderer.lidar is not None:
            clock = self.timing.sensors.get(self.renderer.lidar, SensorTiming())
            scan = clock.resolved_frame_interval_us(self.decision_period_us)
        return max([scan, *(clock.shutter_duration_us for clock in self.timing.sensors.values())])

    @property
    def sensors(self) -> frozenset[str]:
        aliases = self.simulator.camera_aliases
        cameras = frozenset(aliases.get(camera, camera) for camera in self.renderer.cameras)
        lidar = self.renderer.lidar
        return cameras | ({lidar} if lidar is not None else set())

    @property
    def camera_sensors(self) -> frozenset[str]:
        aliases = self.simulator.camera_aliases
        return frozenset(aliases.get(camera, camera) for camera in self.renderer.cameras
                         if self.renderer.camera_model(camera) is not None)

    def _sensor_geometry(self, sensor: str) -> SensorGeometry:
        if not (self._world_initialized or self._resetting or self._advancing):
            raise RuntimeError("Reset the NuRec simulator before requesting camera geometry")
        if sensor == self.renderer.lidar:
            raise ValueError(f"NuRec sensor {sensor!r} has no camera geometry")
        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == sensor), sensor)
        if native_camera not in self.renderer.cameras:
            raise ValueError(f"Unknown NuRec camera {sensor!r}")
        if native_camera in self._capture_geometries:
            return self._capture_geometries[native_camera]
        request = self.render_request(native_camera)
        camera = self.renderer.camera_model(request.camera)
        if camera is None:
            raise ValueError(f"NuRec camera {sensor!r} has no camera geometry")
        geometry = SensorGeometry(aliases.get(native_camera, native_camera), camera,
                                  self.renderer.cam_from_world(request.pose, request.camera))
        self._capture_geometries[native_camera] = geometry
        return geometry

    def _capture_render_request(self, sensor):
        if sensor == self.renderer.lidar:
            return None
        native = next((key for key, alias in self.simulator.camera_aliases.items() if alias == sensor), sensor)
        return self._render_requests[native]

    def _capture_view_builder(self, sensor, observation, truth):
        """Retain capture inputs for lazy geometry and depth without reading later world state."""
        import numpy as np
        from avsectester.data_types import read_camera_rgb

        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == sensor), sensor)
        is_lidar = sensor == self.renderer.lidar
        key = sensor if is_lidar else native_camera
        captured = self._capture_requests[key]
        actors = captured.actors
        native = {"timestamp_us": self._timestamp_us(),
                  "dynamic_objects": captured.tracks.dynamic_objects,
                  "actor_tracks": captured.tracks.actor_tracks,
                  "object_tracks": captured.tracks.object_tracks,
                  "exposure": captured.exposure}
        occlusion = self._view_occlusion()
        if is_lidar:
            calibration = self.renderer.calibration[sensor]
            sensor_from_world = np.linalg.inv(truth.ego.pose @ calibration.rig_from_lidar)
            return lambda: SensorView(sensor, observation.sensor_data[sensor], None, None,
                actors, truth.ego, truth.t, truth.frame, sensor_from_world=sensor_from_world,
                native=native, **occlusion)
        request = self._render_requests[native_camera]
        native["render_request"] = request
        geometry = self._capture_geometries.get(native_camera)
        camera = None if geometry is None else geometry.camera
        captured = read_camera_rgb(observation, sensor)
        if captured is None:
            raise ValueError(f"Missing NuRec camera {sensor!r}")
        cam_from_world = None if geometry is None else geometry.cam_from_world
        depth_enabled = self.depth != "none" and camera is not None
        path = self._start.payload["usdz_path"]

        def build():
            depth = None
            if depth_enabled:
                from avsectester.geometry.mesh import camera_depth

                depth = _actor_depth(camera, cam_from_world, actors, camera_depth(
                    camera, cam_from_world, [_scene_raycaster(path)]))
            return SensorView(sensor, captured[1], camera, cam_from_world,
                actors, truth.ego, truth.t, truth.frame,
                sensor_from_world=None if cam_from_world is None else BODY_FROM_OPTICAL @ cam_from_world,
                depth=depth, native=native, **occlusion)
        return build

    def _source_recorded_ego(self, time_us: int) -> ActorState:
        position, remainder = divmod(self._start.frame.position * self._recording_interval_us + time_us,
                                    self._recording_interval_us)
        if not 0 <= position < len(self._start.recorded_ego):
            raise ValueError("Rig timestamp exceeds recorded support")
        if not remainder:
            return self._start.recorded_ego[position]
        timestamp = timestamp_to_microseconds(self._start.frame.timestamp) + time_us
        return self._tracks().ego_state(self._start.payload["scene_id"], timestamp)

    def _build_force_gt_trajectory(self) -> None:
        from avsectester.closed_loop.timing import initial_history_times

        duration = self.timing.force_gt_duration_us
        if not duration:
            return
        history = initial_history_times(
            {sensor: self.timing.sensors.get(sensor, SensorTiming()) for sensor in self.sensors},
            self.history_requirement, self.decision_period_us,
            pose_reporting_interval_us=self.timing.pose_reporting_interval_us)
        origin = timestamp_to_microseconds(self._start.frame.timestamp)
        begin = min(history.source_start_us, -self._sensor_motion_lookback_us())
        blend_end = min(duration, self._recorded_end_us)
        end = min(self._recorded_end_us, blend_end + FORCE_GT_REFERENCE_HORIZON_US)
        if "usdz_path" in self._start.payload:
            native_times = (timestamp_to_microseconds(value) - origin for value in
                            self._tracks().camera_timestamps(self._start.scenario_id))
        else:
            native_times = ((index - self._start.frame.position) * self._recording_interval_us
                            for index in range(len(self._start.recorded_ego)))
        times = sorted({begin, 0, blend_end, end,
                        *(time for time in native_times if begin <= time <= end),
                        *range(0, end + 1, self.decision_period_us)})
        self._force_gt_poses = tuple((timestamp, self._source_recorded_ego(timestamp).pose.copy())
                                     for timestamp in times)
        self._force_gt_corrected_poses.clear()

    def _force_gt_pose(self, index: int) -> np.ndarray:
        import numpy as np
        from avsectester.closed_loop.simulators.ground_contact import constrain_to_ground

        timestamp, pose = self._force_gt_poses[index]
        if timestamp <= 0 or self._ground_contact is None:
            return pose
        if timestamp not in self._force_gt_corrected_poses:
            ego = self._source_recorded_ego(timestamp)
            corrected, status = constrain_to_ground(
                self._ground_contact, ego.box_transform(), ego.extent)
            if status != "successful_update":
                self._ground_contact_failures[status] += 1
            corrected_rig = corrected @ np.linalg.inv(np.linalg.inv(ego.pose) @ ego.box_transform())
            alpha = float(np.float32(min(1., timestamp / self.timing.force_gt_duration_us)))
            self._force_gt_corrected_poses[timestamp] = interpolate_transform(pose, corrected_rig, alpha)
        return self._force_gt_corrected_poses[timestamp]

    def _recorded_ego(self, time_us: int) -> ActorState:
        import bisect

        ego = self._source_recorded_ego(time_us)
        if not self._force_gt_poses or self._ground_contact is None:
            return ego
        times = tuple(timestamp for timestamp, _pose in self._force_gt_poses)
        if not times[0] <= time_us <= times[-1]:
            return ego
        index = bisect.bisect_left(times, time_us)
        pose = self._force_gt_pose(index) if times[index] == time_us else interpolate_transform(
            self._force_gt_pose(index - 1), self._force_gt_pose(index),
            (time_us - times[index - 1]) / (times[index] - times[index - 1]))
        return replace(ego, pose=pose.copy())

    def _recorded_pose_state(self, time_us):
        ego = self._recorded_ego(time_us)
        state = self._recorded_planar_state(time_us, ego)
        return WorldSnapshot(time_us / 1e6, time_us // self._recording_interval_us,
                             state, state.speed, ego=ego)

    def _recorded_state(self, time_us: int) -> WorldSnapshot:
        ego = self._recorded_ego(time_us)
        _, _, actors = self._recorded_capture(time_us)
        state = self._recorded_planar_state(time_us, ego)
        return WorldSnapshot(state.t, time_us // self._recording_interval_us, state,
                             state.speed, actors, ego=ego)

    def _recorded_capture(self, time_us: int) -> tuple[np.ndarray, int, dict[str, ActorState]]:
        """Recorded rig pose and actors at an integer offset from the selected origin."""
        start = self._start
        if start is None:
            raise RuntimeError("Past NuRec observations require a scenario start")
        if self._last_recorded_capture is not None and self._last_recorded_capture[0] == time_us:
            return self._last_recorded_capture[1:]
        pose = validated_transform(self._recorded_ego(time_us).pose)
        timestamp_us = timestamp_to_microseconds(start.frame.timestamp) + time_us
        actors = self._tracks().actor_states(start.payload["scene_id"], timestamp_us)
        actors = {key: state for key, state in actors.items() if key not in self._filtered_tracks}
        self._last_recorded_capture = (time_us, pose, timestamp_us, actors)
        return pose, timestamp_us, actors

    def _recorded_sensor_geometry(self, time_us: int, sensor: str) -> SensorGeometry:
        cached = self._recorded_geometry_cache.get((time_us, sensor))
        if cached is not None:
            return cached
        if sensor == self.renderer.lidar:
            raise ValueError(f"NuRec sensor {sensor!r} has no camera geometry")
        aliases = self.simulator.camera_aliases
        native_camera = next((key for key, alias in aliases.items() if alias == sensor), sensor)
        if native_camera not in self.renderer.cameras:
            raise ValueError(f"Unknown NuRec camera {sensor!r}")
        camera = self.renderer.camera_model(native_camera)
        if camera is None:
            raise ValueError(f"NuRec camera {sensor!r} has no camera geometry")
        pose, _, _ = self._recorded_capture(time_us)
        geometry = SensorGeometry(aliases.get(native_camera, native_camera), camera,
                                  self.renderer.cam_from_world(pose, native_camera))
        self._recorded_geometry_cache[time_us, sensor] = geometry
        return geometry

    def _recorded_sensor_view(self, time_us: int, sensor: str, observation: Observation) -> SensorView:
        """
        View of a recorded past capture with its recorded actors and, when enabled, mesh depth.
        """
        from avsectester.closed_loop.hooks import CameraPoseRequest
        from avsectester.data_types import read_camera_rgb

        pose, timestamp_us, actors = self._recorded_capture(time_us)
        tracks = _RenderTracks(
            self._dynamic_objects(timestamp_us, actors),
            self._object_tracks(tuple(self._start_objects.values())))
        lidar = self.renderer.lidar
        if sensor == lidar:
            import numpy as np

            calibration = self.renderer.calibration[lidar]
            return SensorView(
                lidar, observation.sensor_data[lidar], None, None, actors,
                self._recorded_ego(time_us), observation.t, observation.frame,
                sensor_from_world=np.linalg.inv(pose @ calibration.rig_from_lidar),
                native={"timestamp_us": timestamp_us,
                        **tracks.native()},
                **self._view_occlusion())
        geometry = self._recorded_sensor_geometry(time_us, sensor)
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
            self._recorded_ego(time_us), observation.t, observation.frame,
            sensor_from_world=BODY_FROM_OPTICAL @ geometry.cam_from_world, depth=depth,
            native={"render_request": CameraPoseRequest(pose, native_camera), "timestamp_us": timestamp_us,
                    **tracks.native(),
                    "exposure": self._camera_exposure(sensor, time_us, pose, tracks, recorded=True)},
            **self._view_occlusion(),
        )

    def _composite_plan(self, view: SensorView, objects: Sequence[PlacedObject]) -> CompositePlan:
        """
        Plan with the visibility of rendered probes at the view's own capture, when configured.
        """
        if self.simulator.object_occlusion == "geometry":
            return super()._composite_plan(view, objects)
        from avsectester.rendering import plan_composite

        assert view.camera is not None and view.cam_from_world is not None
        request = view.native["render_request"]
        visibility = self.renderer.render_scene_visibility(
            request.pose, view.native["timestamp_us"], request.camera, objects,
            dynamic_objects=view.native["dynamic_objects"],
            occlusion_tolerance_m=self.simulator.occlusion_tolerance_m,
            **view.native.get("exposure", {}))
        return plan_composite(view.camera, view.cam_from_world, objects, scene_visibility=visibility)

    def _compose_objects(
        self, view: SensorView, objects: Sequence[PlacedObject],
    ) -> tuple[CompositeResult, ObjectComposition | None]:
        """Compose camera overlays or describe objects already rendered by NRE."""
        if self.simulator.object_insertion == "composite":
            return super()._compose_objects(view, objects)
        import numpy as np

        from avsectester.rendering import CompositeResult, ObjectVisibility

        assert view.camera is not None and view.cam_from_world is not None
        request = view.native["render_request"]
        timestamp_us = view.native["timestamp_us"]
        actor_tracks = view.native["actor_tracks"]
        tracks = view.native["object_tracks"]
        coverage = {}
        for placed in objects:
            others = tuple(track for identity, track in tracks.items()
                           if identity != placed.object_id)
            coverage[placed.object_id] = self.renderer.render_object_coverage(
                request.pose, timestamp_us, request.camera, placed,
                dynamic_objects=(*actor_tracks, *others), **view.native.get("exposure", {}))
        masks = {identity: values >= .5 for identity, values in coverage.items()}
        visible = np.zeros(view.data.shape[:2], dtype=bool)
        for mask in masks.values():
            visible |= mask
        image = np.array(view.data, copy=True)
        harmonized_objects = [placed for placed in objects if placed.harmonizer is not None]
        if harmonized_objects:
            background = self.renderer.render(
                request.pose, timestamp_us, request.camera, dynamic_objects=actor_tracks,
                **view.native.get("exposure", {}))
            # As in CompositePlan, only objects with differentiable harmonizers show nominal textures.
            nominal_objects = [replace(placed, asset=self._nominal_asset(placed.asset))
                               if getattr(placed.harmonizer, "differentiable", False) else placed
                               for placed in objects]
            nominal = view.data
            if any(nominal_placed.asset is not placed.asset
                   for nominal_placed, placed in zip(nominal_objects, objects, strict=True)):
                candidates = {placed.object_id for placed in objects}
                others = tuple(track for identity, track in tracks.items() if identity not in candidates)
                nominal_tracks = tuple(self._object_tracks(nominal_objects).values())
                exposure = dict(view.native.get("exposure", {}))
                if exposure:
                    nominal_ids = {track[0] for track in nominal_tracks}
                    exposure["dynamic_objects_start"] = tuple(
                        track for track in exposure["dynamic_objects_start"]
                        if track[0] not in nominal_ids) + nominal_tracks
                nominal = self.renderer.render(
                    request.pose, timestamp_us, request.camera,
                    dynamic_objects=(*actor_tracks, *others, *nominal_tracks), **exposure)
            for placed in harmonized_objects:
                active = coverage[placed.object_id] > _HARMONIZER_COVERAGE_THRESHOLD
                if not active.any():
                    continue
                harmonizer = placed.harmonizer
                if getattr(harmonizer, "differentiable", False):
                    import torch

                    def tensor(rgb: np.ndarray) -> torch.Tensor:
                        return torch.as_tensor(rgb.copy(), dtype=torch.float32).permute(2, 0, 1) / 255

                    with torch.no_grad():
                        adjusted = harmonizer.harmonize_tensor(
                            tensor(view.data), torch.as_tensor(active),
                            tensor(background), tensor(nominal))
                        adjusted_rgb = adjusted.permute(1, 2, 0).cpu().numpy()
                    image[active] = np.rint(np.clip(adjusted_rgb[active], 0, 1) * 255).astype(np.uint8)
                else:
                    adjusted = harmonizer(
                        view.data.copy(), active.astype(np.uint8) * 255, background.copy())
                    image[active] = adjusted[active]
        depth = np.full(view.data.shape[:2], np.inf, dtype=np.float32)
        if view.depth is not None:
            for placed in objects:
                mask = masks[placed.object_id]
                candidate = self._object_depth(view, placed, mask)
                depth[mask] = np.minimum(depth[mask], candidate[mask])
        return CompositeResult(
            image=image, depth=depth, visible_mask=visible, occlusion_source="scene_render",
            object_visibility={identity: ObjectVisibility("scene_render", (), int(mask.sum()))
                               for identity, mask in masks.items()},
            object_masks=masks), None

    def _object_depth(self, view: SensorView, placed: PlacedObject, mask: np.ndarray) -> np.ndarray:
        """Camera z depth of one world asset, computed only for captures with scene depth."""
        import numpy as np

        from avsectester.geometry.mesh import MeshRaycaster, camera_depth
        from avsectester.rendering import MeshAsset, plan_composite

        assert view.camera is not None and view.cam_from_world is not None
        if not isinstance(placed.asset, MeshAsset):
            return plan_composite(
                view.camera, view.cam_from_world, (placed,), scene_visibility=mask).depth
        cached = self._mesh_raycasters.get(placed.object_id)
        if (cached is None or cached[0] is not placed.asset
                or not np.array_equal(cached[1], placed.world_from_object)):
            vertices, faces = placed.world_mesh()
            raycaster = MeshRaycaster(vertices, faces)
            cached = (placed.asset, placed.world_from_object.copy(), raycaster)
            self._mesh_raycasters[placed.object_id] = cached
        return camera_depth(view.camera, view.cam_from_world, (cached[2],))

    def _observe_recorded(self, time_us: int, sensors: Sequence[str] | None) -> Observation:
        pose, timestamp_us, actors = self._recorded_capture(time_us)
        start = self._start
        assert start is not None
        relative_frame = time_us // self._recording_interval_us
        relative_time = time_us / 1e6
        aliases = self.simulator.camera_aliases
        selected = self.sensors if sensors is None else set(sensors)
        object_tracks = self._object_tracks(tuple(self._start_objects.values()))
        tracks = _RenderTracks(self._dynamic_objects(timestamp_us, actors), object_tracks)
        sensor_data = {}
        for camera in self.renderer.cameras:
            sensor = aliases.get(camera, camera)
            if sensor not in selected:
                continue
            exposure = self._camera_exposure(sensor, time_us, pose, tracks, recorded=True)
            sensor_data[sensor] = self.renderer.render(pose, timestamp_us, camera,
                dynamic_objects=tracks.dynamic_objects, **exposure)
            lens = self.renderer.camera_model(camera)
            if lens is not None:
                self._recorded_geometry_cache[time_us, sensor] = SensorGeometry(sensor, lens,
                    self.renderer.cam_from_world(pose, camera))
        lidar = self.renderer.lidar
        if lidar is not None and lidar in selected:
            start_pose, start_us, start_actors = self._recorded_capture(
                self.frame_range_us(lidar, time_us).start)
            start_tracks = _RenderTracks(self._dynamic_objects(start_us, start_actors), object_tracks)
            sensor_data[lidar] = self.renderer.render_lidar(
                start_pose, pose, start_us, timestamp_us,
                start_tracks.dynamic_objects, tracks.dynamic_objects)
        vehicle_state = self._recorded_planar_state(int(time_us), self._recorded_ego(time_us))
        vehicle_state.t = relative_time
        # Route generation may extend its stored polyline. A past query must not change the
        # live runtime's navigation state or any later model-visible route.
        route = deepcopy(self._route).route(pose)
        return Observation(
            t=relative_time, frame=int(relative_frame), sensor_data=sensor_data,
            calibration={(aliases.get(key, key) if key in self.renderer.cameras else key): deepcopy(value)
                         for key, value in self.renderer.calibration.items()},
            vehicle_state=vehicle_state, ego_speed=vehicle_state.speed, route=route,
        )

    def _trajectory_capture(self, time_us):
        absolute = timestamp_to_microseconds(self._start.frame.timestamp) + time_us
        if time_us < 0:
            return self._recorded_capture(time_us)
        for interval in reversed(self._motion_history):
            if interval.start_us <= time_us <= interval.end_us:
                state = self._motion_state(interval, time_us)
                return state.ego.pose, absolute, dict(state.actors)
        if time_us == self._state_time_us:
            return self._world_from_ego.copy(), absolute, self._actors_at(absolute)
        raise ValueError("No physical trajectory covers the camera shutter start")

    def _camera_exposure(self, sensor, time_us, end_pose, end_tracks, *, recorded=False):
        import numpy as np

        duration = self.timing.sensors.get(sensor, SensorTiming()).shutter_duration_us
        if not duration:
            return {}
        start_us = time_us - duration
        pose, absolute, actors = (self._recorded_capture(start_us) if recorded
                                   else self._trajectory_capture(start_us))
        true_end = self._recorded_ego(time_us).pose if recorded else self._world_from_ego
        pose = pose @ np.linalg.inv(true_end) @ end_pose
        tracks = _RenderTracks(self._dynamic_objects(absolute, actors), end_tracks.object_tracks)
        return {"frame_start_us": absolute, "world_from_rig_start": pose,
                "dynamic_objects_start": tracks.dynamic_objects}

    def _observe(self) -> Observation:
        return self._observe_sensors(tuple(self.sensors))

    def _observe_sensors(self, sensor_ids: Sequence[str]) -> Observation:
        from avsectester.closed_loop.hooks import CameraPoseRequest

        aliases = self.simulator.camera_aliases
        requested = set(sensor_ids)
        sensor_data = {}
        object_tracks = self._object_tracks(tuple(self._capture_objects.values()))
        self._capture_requests.clear()
        calibration = {(aliases.get(key, key) if key in self.renderer.cameras else key): value
                       for key, value in self.renderer.calibration.items()
                       if aliases.get(key, key) in requested}
        for camera in self.renderer.cameras:
            if aliases.get(camera, camera) not in requested:
                continue
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
            actors = self._actors_at(self._timestamp_us())
            tracks = _RenderTracks(
                self._dynamic_objects(self._timestamp_us(), actors), object_tracks)
            exposure = self._camera_exposure(aliases.get(camera, camera), self._state_time_us,
                                             request.pose, tracks)
            lens = self.renderer.camera_model(request.camera)
            if lens is not None:
                self._capture_geometries[camera] = SensorGeometry(aliases.get(camera, camera), lens,
                    self.renderer.cam_from_world(request.pose, request.camera))
            image = self.renderer.render(request.pose, self._timestamp_us(), request.camera,
                                         dynamic_objects=tracks.dynamic_objects, **exposure)
            self._capture_requests[camera] = _CaptureRequest(tracks, actors, exposure)
            sensor_data[aliases.get(camera, camera)] = self._emit(
                "render.post", image, sensor=camera,
            )
        lidar = self.renderer.lidar
        if lidar is not None and lidar in requested:
            timestamp_us = self._timestamp_us()
            start_pose, start_us, start_actors = self._trajectory_capture(
                self.frame_range_us(lidar, self._state_time_us).start)
            actors = self._actors_at(timestamp_us)
            tracks = _RenderTracks(self._dynamic_objects(timestamp_us, actors), object_tracks)
            start_tracks = _RenderTracks(
                self._dynamic_objects(start_us, start_actors), object_tracks)
            cloud = self.renderer.render_lidar(
                start_pose, self._world_from_ego, start_us, timestamp_us,
                start_tracks.dynamic_objects, tracks.dynamic_objects)
            self._capture_requests[lidar] = _CaptureRequest(tracks, actors)
            sensor_data[lidar] = self._emit("render.post", cloud, sensor=lidar)
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
        self._motion_interval = None
        self._motion_history.clear()
        self._traffic_history.clear()
        self._actor_states.clear()
        self._capture_geometries.clear()
        self._recorded_geometry_cache.clear()
        self._last_recorded_capture = None
        self._force_gt_poses = ()
        self._force_gt_corrected_poses.clear()
        self._capture_requests.clear()
        self._traffic_states.clear()
        self._render_requests.clear()
        self._actor_overrides.clear()
        self._recorded_tracks.clear()
        try:
            self._close_traffic_session()
        finally:
            try:
                self.renderer.close()
            finally:
                super().close()

    # In-process world checkpoints retain frozen assets and loaded harmonizer references.

    def checkpoint(self) -> dict:
        """Capture a replay checkpoint. Active reactive-traffic sessions cannot be checkpointed."""
        if self._interval is not None:
            raise RuntimeError("Publish the pending motion interval before checkpointing")
        if self.simulator.traffic is not None:
            raise RuntimeError("AlpaSim traffic sessions cannot be checkpointed")
        return {"pose": asdict(self.pose), "frame": self.frame, "time_us": self.clock.time_us,
                "world_from_ego": self._world_from_ego.copy(),
                "ground_contact_failures": dict(self._ground_contact_failures),
                "actor_overrides": deepcopy(self._actor_overrides),
                "objects": dict(self._objects), "motion_history": deepcopy(self._motion_history),
                "force_gt_poses": deepcopy(self._force_gt_poses),
                "force_gt_corrected_poses": deepcopy(self._force_gt_corrected_poses)}

    def restore(self, ckpt: dict) -> None:
        """Restore a saved replay checkpoint into this runtime."""
        if self.simulator.traffic is not None:
            raise RuntimeError("AlpaSim traffic sessions cannot be restored from checkpoints")
        self.pose = PlanarEgoState(**ckpt["pose"])
        self._state_time_us = ckpt["time_us"]
        self._restore_clock(ckpt.get("time_us", timestamp_to_microseconds(self.pose.t)))
        self._motion_interval = None
        self._motion_history = deepcopy(ckpt.get("motion_history", deque()))
        self._force_gt_poses = deepcopy(ckpt.get("force_gt_poses", ()))
        self._force_gt_corrected_poses = deepcopy(ckpt.get("force_gt_corrected_poses", {}))
        self.invalidate_state()
        reset_dynamics = getattr(self.dynamics, "reset", None)
        if callable(reset_dynamics):
            reset_dynamics(deepcopy(self.pose))
        self.frame = ckpt["frame"]
        self._world_from_ego = ckpt["world_from_ego"].copy()
        self._ground_contact_failures = Counter(ckpt["ground_contact_failures"])
        self._actor_overrides = deepcopy(ckpt["actor_overrides"])
        # Frozen placed assets and harmonizer configurations are shared. Copying a
        # harmonizer's loaded model would duplicate heavyweight native resources.
        self._objects = dict(ckpt["objects"])
        self._render_requests.clear()
        try:
            self._take_capture()
        except BaseException:
            self._world_initialized = False
            raise


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
