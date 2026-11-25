"""NuRec reconstruction with planar ego dynamics and optional reactive traffic."""

from __future__ import annotations

import bisect
import dataclasses
import math
from collections import Counter, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, TYPE_CHECKING

from avsectester.closed_loop.simulators.base import UnsupportedStartError
from avsectester.closed_loop.simulators.ego_dynamics import (
    EgoDynamics, EgoMotion, KinematicBicycle, PlanarEgoState, _advance_dynamics,
)
from avsectester.closed_loop.timing import seconds_to_microseconds
from avsectester.data_types import ActorState, Control
from avsectester.geometry.poses import interpolate_transform
from avsectester.validation import timestamp_to_microseconds

from .depth import _ground_raycaster
from .renderer import Renderer, planar_rig_pose
from .scene import EgoPropagation, NuRecSceneRuntime, NuRecSceneSimulator
from .traffic import AlpaSimTraffic, TrafficSession, TrafficTrack, TRAFFIC_STEP_US

if TYPE_CHECKING:
    import numpy as np

    from avsectester.datasets.base import ScenarioStart


class NuRecSimulator(NuRecSceneSimulator):
    """A NuRec neural reconstruction with in-process ego dynamics and a pluggable renderer.

    The renderer and dynamics are configuration templates. Each runtime uses its own copies.
    With ``physics_update_mode="ego_only"``, ground contact adjusts the ego pose against the
    selected ground mesh. An optional traffic model replaces the recorded motion of reactive
    actors after its startup window.
    """

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
        super().__init__(
            renderer, time_step=time_step, camera_aliases=camera_aliases, depth=depth,
            route_generator=route_generator, object_occlusion=object_occlusion,
            object_insertion=object_insertion, occlusion_tolerance_m=occlusion_tolerance_m,
            min_track_duration_s=min_track_duration_s)
        if physics_update_mode not in {"ego_only", "none"}:
            raise ValueError("physics_update_mode must be 'ego_only' or 'none'")
        if ground_mesh not in {"mesh", "mesh_ground"}:
            raise ValueError("ground_mesh must be 'mesh' or 'mesh_ground'")
        if depth != "mesh" and physics_update_mode == "ego_only":
            from avsectester.geometry.mesh import MeshRaycaster

            MeshRaycaster.check_dependencies()
        self.physics_update_mode = physics_update_mode
        self.ground_mesh = ground_mesh
        self.dynamics = dynamics or KinematicBicycle()
        if traffic is not None and not isinstance(traffic, AlpaSimTraffic):
            raise TypeError("traffic must be AlpaSimTraffic or None")
        if traffic is not None and seconds_to_microseconds(self.time_step) % TRAFFIC_STEP_US:
            raise ValueError("NuRec decision periods must be whole multiples of the traffic step")
        self.traffic = traffic

    def _validate_start_history(self, start: ScenarioStart) -> None:
        if start.frame.position < self.past_steps:
            raise UnsupportedStartError(
                f"AlpaSim traffic requires {self.past_steps} past decision steps")
        super()._validate_start_history(start)

    def create_runtime(self, seed: int | None = None) -> NuRecSimulatorRuntime:
        """A new world. The optional traffic session receives seed for its log."""
        return NuRecSimulatorRuntime(self, seed)

    @property
    def past_steps(self) -> int:
        """Recorded decision history needed by the optional traffic model."""
        return 0 if self.traffic is None else self.traffic.history_steps - 1

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Delegate command capabilities to the selected ego dynamics implementation."""
        return getattr(self.dynamics, "command_modes", None)

    @property
    def identity(self) -> Mapping[str, Any]:
        """Include compatible dynamics, ground contact and reactive traffic settings."""
        identity = dict(super().identity)
        identity.update({
            "physics_update_mode": self.physics_update_mode,
            "ground_mesh": self.ground_mesh,
            "dynamics": {
                "type": f"{type(self.dynamics).__module__}:{type(self.dynamics).__qualname__}",
                "options": {name: value for name, value in asdict(self.dynamics).items()
                            if name in {item.name for item in dataclasses.fields(self.dynamics)
                                        if item.init}}
                if dataclasses.is_dataclass(self.dynamics) else {},
            },
        })
        if self.traffic is not None:
            identity["traffic"] = {
                "type": f"{type(self.traffic).__module__}:{type(self.traffic).__qualname__}",
                "options": {name: getattr(self.traffic, name) for name in (
                    "endpoint", "history_steps", "rpc_timeout_s")}}
        return identity


class NuRecSimulatorRuntime(NuRecSceneRuntime):
    """One live NuRec world with planar ego dynamics and its own renderer and traffic sessions."""

    def __init__(self, simulator: NuRecSimulator, seed: int | None = None) -> None:
        super().__init__(simulator, seed)
        self.dynamics = deepcopy(simulator.dynamics)
        self._traffic_session: TrafficSession | None = None
        self._traffic_states: dict[str, ActorState] = {}
        self._traffic_timestamp_us: int | None = None
        self._traffic_history: deque[tuple[int, dict[str, ActorState]]] = deque()
        self._ground_contact = None
        self._ground_contact_failures: Counter[str] = Counter()
        self._force_gt_corrected_poses: dict[int, np.ndarray] = {}

    @property
    def native(self) -> dict[str, Any]:
        """Native renderer, dynamics and render requests for this runtime."""
        resources = super().native
        return {"renderer": resources["renderer"], "dynamics": self.dynamics,
                "render_requests": resources["render_requests"]}

    @property
    def termination_details(self) -> dict[str, Any]:
        """Counts of unsuccessful ego ground contact updates by status."""
        return {"ground_contact_failures": dict(self._ground_contact_failures)}

    def _apply_recorded_interval(self, actors: dict[str, ActorState], timestamp_us: int) -> None:
        if self._traffic_session is None:
            super()._apply_recorded_interval(actors, timestamp_us)

    def _apply_dynamic_actor_states(self, actors: dict[str, ActorState], timestamp_us: int) -> None:
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

    def set_actor_state(self, actor_id: str, state: ActorState) -> None:
        """Override an actor and backfill constant-velocity history in a reactive service request."""
        super().set_actor_state(actor_id, state)
        timestamp_us, replacement = self._actor_overrides[actor_id]
        session = self._traffic_session
        if session is not None and actor_id in session.track_ids:
            session.record_override(actor_id, timestamp_us, replacement.box_transform(), replacement.velocity)

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

    def _simulate_traffic(self, timestamp_us: int, ego: ActorState,
                          previous: dict[str, ActorState], *, force_gt: bool,
                          is_endpoint: bool) -> None:
        session = self._traffic_session
        if session is None:
            raise RuntimeError("AlpaSim traffic session is closed. Reset the runtime before stepping")
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
        self._traffic_states.clear()
        self._traffic_timestamp_us = None
        self._traffic_history.clear()
        self._ground_contact_failures.clear()
        self._ground_contact = None
        super()._initialize_world(start, initialization, environment)

    def _initialize_physics(self, start: ScenarioStart, short_tracks: set[str]) -> None:
        if self.simulator.physics_update_mode == "ego_only":
            if start.payload["ego"].extent is None:
                raise ValueError("NuRec ground contact requires ego bounding-box extent")
            self._ground_contact = _ground_raycaster(
                start.payload["usdz_path"], self.simulator.ground_mesh + ".ply")
        reset_dynamics = getattr(self.dynamics, "reset", None)
        if callable(reset_dynamics):
            reset_dynamics(deepcopy(self.pose))
        if self.simulator.traffic is not None:
            self._open_traffic_session(short_tracks)

    def _coerce_recorded_dynamics(self):
        super()._coerce_recorded_dynamics()
        coerce = getattr(self.dynamics, "coerce_dynamic_state", None)
        if callable(coerce):
            coerce(deepcopy(self.pose))

    def _propagate_ego(self, control: Control | None, report_times: tuple[int, ...], *,
                       recorded: bool) -> EgoPropagation:
        assert self._interval is not None
        duration_us = self.decision_period_us
        previous_yaw = self.pose.yaw
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
        pose = deepcopy(motion.states[-1])
        if recorded or control is None:
            world_from_ego = self._recorded_ego(self._interval.end_us).pose.copy()
        else:
            world_from_ego = planar_rig_pose(pose.x, pose.y, pose.yaw,
                                         self._world_from_ego[2, 3])
            world_from_ego[:3, :3] = (world_from_ego[:3, :3]
                @ planar_rig_pose(0, 0, previous_yaw)[:3, :3].T @ self._world_from_ego[:3, :3])
        if self._ground_contact is not None and control is not None and not recorded:
            from avsectester.closed_loop.simulators.ground_contact import constrain_to_ground
            import numpy as np

            ego = self._ego_state_at(pose, world_from_ego)
            corrected, status = constrain_to_ground(self._ground_contact, ego.box_transform(), ego.extent)
            if status != "successful_update":
                self._ground_contact_failures[status] += 1
            corrected_rig = corrected @ np.linalg.inv(np.linalg.inv(ego.pose) @ ego.box_transform())
            world_from_ego = corrected_rig
            pose.x, pose.y = world_from_ego[:2, 3]
            pose.yaw = math.atan2(world_from_ego[1, 0], world_from_ego[0, 0])
        poses = (tuple(self._recorded_ego(timestamp).pose for timestamp in report_times)
                 if recorded or control is None else ())
        return EgoPropagation(motion, pose, world_from_ego, poses)

    def _advance_motion(self, control, *, recorded=False):
        assert self._interval is not None
        uses_traffic = self.simulator.traffic is not None
        previous_actors = self._actors_at(self._timestamp_us()) if uses_traffic else {}
        super()._advance_motion(control, recorded=recorded)
        motion = self._motion_interval
        assert motion is not None
        if uses_traffic:
            traffic_times = range(self._interval.start_us + TRAFFIC_STEP_US,
                                  self._interval.end_us + 1, TRAFFIC_STEP_US)
            final_pose = self.pose
            final_ego = self._ego_state()
            for timestamp in traffic_times:
                ego = (self._motion_ego(motion, timestamp, final_ego, final_pose)[1]
                       if timestamp != self._interval.end_us else final_ego)
                timestamp_us = timestamp_to_microseconds(self._start.frame.timestamp) + timestamp
                self._simulate_traffic(timestamp_us, ego, previous_actors,
                    force_gt=recorded, is_endpoint=timestamp == self._interval.end_us)
                previous_actors = self._actors_at(timestamp_us)

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
        ego = super()._recorded_ego(time_us)
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

    def close(self) -> None:
        """Close the traffic session and the shared NuRec scene."""
        self._traffic_history.clear()
        self._traffic_states.clear()
        self._force_gt_corrected_poses.clear()
        try:
            self._close_traffic_session()
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
                "objects": dict(self._objects), "start_objects": dict(self._start_objects),
                "object_origin_time_us": self._object_origin_time_us,
                "motion_history": deepcopy(self._motion_history),
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
        self._start_objects = dict(ckpt["start_objects"])
        self._object_origin_time_us = ckpt["object_origin_time_us"]
        self._render_requests.clear()
        try:
            self._take_capture()
        except BaseException:
            self._world_initialized = False
            raise


__all__ = [
    'NuRecSimulator',
    'NuRecSimulatorRuntime',
]
