"""CARLA ego dynamics and recorded obstacles in a NuRec reconstruction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from avsectester.closed_loop.simulators.base import UnsupportedStartError
from avsectester.closed_loop.timing import SensorTiming, TimingCapabilities, seconds_to_microseconds
from avsectester.data_types import ActorState, Control
from avsectester.datasets.base import ScenarioStart
from avsectester.datasets.nurec import NuRecDataset
from avsectester.maps.opendrive import check_opendrive, opendrive_geo_origin
from avsectester.validation import require_positive_number, timestamp_to_microseconds

from ..carla.ego_vehicle import CarlaEgoVehicle
from ..carla.trajectory_tracking import TrajectoryTracker
from ..carla.vehicles import BlueprintGeometry, best_fit_blueprint, probe_blueprints
from ..carla.world import (CarlaWorldSession, carla_world_key, ticks_for_duration,
                           validated_client_settings)
from ..ego_dynamics import EgoMotion, PlanarEgoState
from ..nurec.scene import EgoPropagation, NuRecSceneRuntime, NuRecSceneSimulator
from .alignment import SceneAlignment
from .proxies import RecordedTrackProxies


_CLIENT_OPTIONS = frozenset({"connect_ip", "connect_port", "rate", "synchronous",
                             "substepping", "max_substep_delta_time", "max_substeps"})


def _read_map_alignment(start: ScenarioStart, archive: str,
                        timeout_s: float) -> tuple[str, SceneAlignment]:
    source = NuRecDataset([archive], camera_id=start.payload["camera_id"],
                         time_step=start.payload["time_step"])
    xodr = source.opendrive(start.scenario_id)
    origin = None if xodr is None else opendrive_geo_origin(xodr)
    if xodr is None or origin is None:
        raise UnsupportedStartError("CARLA + NuRec requires map.xodr with a complete geoReference origin")
    try:
        alignment = SceneAlignment.from_scene(origin, source.ecef_from_scene(start.scenario_id))
    except ValueError as error:
        raise UnsupportedStartError(f"Invalid NuRec scene alignment: {error}") from error
    rejection = check_opendrive(xodr, timeout_s)
    if rejection is not None:
        raise UnsupportedStartError(f"CARLA rejected the NuRec OpenDRIVE map: {rejection}")
    return xodr, alignment


class CarlaNuRecSimulator(NuRecSceneSimulator):
    """Use CARLA vehicle physics with NRE rendering and recorded traffic as solid obstacles."""

    def __init__(
        self, renderer: Any, *, client: Mapping[str, Any], time_step: float | None = None,
        camera_aliases: Mapping[str, str] | None = None, depth: str = "mesh", route_generator: str = "map",
        object_occlusion: str = "rendered_probes", object_insertion: str = "composite",
        occlusion_tolerance_m: float = 0.15, min_track_duration_s: float = 3.0,
        ego_blueprint: str | None = None, settle_s: float = 1.0,
        tracker: TrajectoryTracker | None = None, timeout_s: float = 60.0,
    ) -> None:
        super().__init__(
            renderer, time_step=time_step, camera_aliases=camera_aliases, depth=depth,
            route_generator=route_generator, object_occlusion=object_occlusion,
            object_insertion=object_insertion, occlusion_tolerance_m=occlusion_tolerance_m,
            min_track_duration_s=min_track_duration_s,
        )
        self.client_config, self._base_tick_us = validated_client_settings(dict(client), allowed=_CLIENT_OPTIONS)
        if seconds_to_microseconds(self.time_step) % self._base_tick_us:
            raise ValueError("Decision period must be a whole number of CARLA base ticks")
        if ego_blueprint is not None and (not isinstance(ego_blueprint, str) or not ego_blueprint):
            raise ValueError("ego_blueprint must be a nonempty CARLA blueprint name or None")
        if tracker is not None and not isinstance(tracker, TrajectoryTracker):
            raise TypeError("tracker must be a TrajectoryTracker or None")
        require_positive_number(timeout_s, "timeout_s")
        self.timeout_s = timeout_s
        self.ego_blueprint = ego_blueprint
        self.settle_s = settle_s
        self.settle_ticks = ticks_for_duration(settle_s, float(self.client_config["rate"]), "settle_s")
        self.tracker = TrajectoryTracker() if tracker is None else tracker
        self._map_validation: tuple[str, tuple[str, SceneAlignment] | str] | None = None

    @property
    def command_modes(self) -> frozenset[str]:
        """Accept CARLA actuator commands and rig-frame trajectories."""
        return frozenset({"actuator", "trajectory"})

    @property
    def timing_capabilities(self) -> TimingCapabilities:
        """Align NuRec timing to the CARLA physics tick."""
        return replace(super().timing_capabilities, base_tick_us=self._base_tick_us)

    def validate_start(self, start: ScenarioStart | None) -> None:
        """Require recorded ego dimensions and an aligned, loadable OpenDRIVE map."""
        super().validate_start(start)
        assert start is not None
        ego = start.payload.get("ego")
        if not isinstance(ego, ActorState) or ego.extent is None or min(ego.extent) <= 0:
            raise UnsupportedStartError("CARLA + NuRec requires recorded ego dimensions")
        self._map_alignment(start)

    def _map_alignment(self, start: ScenarioStart) -> tuple[str, SceneAlignment]:
        archive = start.payload.get("usdz_path")
        if archive is None:
            raise UnsupportedStartError("CARLA + NuRec requires a NuRec archive")
        archive = str(archive)
        previous = self._map_validation
        if previous is None or previous[0] != archive:
            try:
                result: tuple[str, SceneAlignment] | str = _read_map_alignment(
                    start, archive, self.timeout_s)
            except UnsupportedStartError as error:
                result = str(error)
            previous = self._map_validation = (archive, result)
        result = previous[1]
        if isinstance(result, str):
            raise UnsupportedStartError(result)
        return result

    @property
    def identity(self) -> dict[str, Any]:
        """Record the scene, world settings and controller configuration for preparation."""
        return {**super().identity, "client": dict(self.client_config), "ego_blueprint": self.ego_blueprint,
                "settle_s": self.settle_s, "tracker": self.tracker.identity}

    def create_runtime(self, seed: int | None = None) -> CarlaNuRecSimulatorRuntime:
        """Create a detached NRE renderer and CARLA runtime for one experiment run."""
        return CarlaNuRecSimulatorRuntime(self, seed)


class CarlaNuRecSimulatorRuntime(NuRecSceneRuntime):
    """Render NuRec at physical CARLA poses and use recordings before the run origin."""

    def __init__(self, simulator: CarlaNuRecSimulator, seed: int | None = None) -> None:
        super().__init__(simulator, seed)
        self.simulator: CarlaNuRecSimulator = simulator
        self._session: CarlaWorldSession | None = None
        self._ego: CarlaEgoVehicle | None = None
        self._alignment: SceneAlignment | None = None
        self._proxies: RecordedTrackProxies | None = None
        self._geometry: BlueprintGeometry | None = None
        self._tracker: TrajectoryTracker | None = None
        self._collisions: set[str] = set()

    @property
    def world_key(self) -> tuple[str, str, int]:
        """Identify the native world so concurrent CARLA runtimes cannot change it."""
        client = self.simulator.client_config
        return carla_world_key(client.get("connect_ip", "localhost"), client.get("connect_port", 2000))

    @property
    def termination_details(self) -> dict[str, Any]:
        """List any recorded tracks that could not be represented by a CARLA proxy."""
        return {} if self._proxies is None else {"missing_proxies": sorted(self._proxies.missing)}

    def interval_telemetry(self) -> dict[str, Any]:
        """List recorded tracks whose CARLA body box touched the ego body box in the interval."""
        collisions, self._collisions = sorted(self._collisions), set()
        return {"collisions": collisions}

    def _initialize_world(self, start, initialization, environment):
        self._close_session()
        self._ego = self._alignment = self._proxies = self._geometry = self._tracker = None
        self._collisions.clear()
        try:
            super()._initialize_world(start, initialization, environment)
        except BaseException as error:
            try:
                self._close_session()
            except Exception as cleanup:
                error.add_note(f"Could not close the CARLA world: {cleanup}")
            raise

    def _initialize_physics(self, start: ScenarioStart, short_tracks: set[str]) -> None:
        simulator = self.simulator
        xodr, self._alignment = simulator._map_alignment(start)
        session = CarlaWorldSession.connect(simulator.client_config, simulator.timeout_s)
        self._session = session
        session.load_opendrive_world(xodr)
        settings = session.configure(deterministic_collisions=True)
        vehicles = probe_blueprints(session, "vehicle.*")
        walkers = probe_blueprints(session, "walker.pedestrian.*")
        eligible = tuple(item for item in vehicles if item.rear_axle_x is not None
                         and item.wheelbase and item.max_steer_angle)
        if simulator.ego_blueprint is not None:
            eligible = tuple(item for item in eligible if item.id == simulator.ego_blueprint)
            if not eligible:
                raise RuntimeError(f"CARLA ego blueprint {simulator.ego_blueprint!r} has no valid wheel geometry")
        recorded = self._recorded_ego(0)
        assert recorded.extent is not None
        geometry = best_fit_blueprint(eligible, recorded.extent)
        self._geometry = geometry
        self._tracker = replace(simulator.tracker)
        carla_pose = self._alignment.to_carla(recorded.pose)
        self._ego = CarlaEgoVehicle.spawn(session, geometry, carla_pose)
        self._proxies = RecordedTrackProxies(session, self._alignment, vehicles, walkers)
        actors = self._actors_at(self._timestamp_us())
        self._proxies.sync(actors)
        settled = recorded.pose.copy()
        if self.timing.force_gt_duration_us == 0:
            self._ego.hold()
            if simulator.settle_ticks:
                for _ in range(simulator.settle_ticks):
                    self._proxies.sync(actors)
                    session.tick()
                settled = self._alignment.from_carla(self._ego.read().world_from_rig)
            elif self._ego.spawn_lift_m:
                carla_pose[2, 3] += self._ego.spawn_lift_m
                settled = self._alignment.from_carla(carla_pose)
        drift = settled[:3, 3] - recorded.pose[:3, 3]
        self._world_from_ego = settled
        self.pose.x, self.pose.y = settled[:2, 3]
        self.pose.yaw = math.atan2(settled[1, 0], settled[0, 0])
        self.initialization_report.update({
            "world_settings": settings,
            "ego_blueprint": geometry.id,
            "ego_geometry": asdict(geometry),
            "proxy_count": self._proxies.count,
            "missing_proxies": sorted(self._proxies.missing),
            "settle_horizontal_drift_m": float(np.linalg.norm(drift[:2])),
            "settle_height_change_m": float(drift[2]),
        })

    def _recorded_velocity(self, time_us: int) -> np.ndarray:
        assert self._alignment is not None
        velocity = self._recorded_ego(time_us).velocity
        if velocity is None:
            raise ValueError("CARLA + NuRec requires recorded ego velocity")
        return self._alignment.vector_to_carla(np.asarray(velocity))

    def _before_selected_capture(self, position: int) -> None:
        super()._before_selected_capture(position)
        assert self._ego is not None and self._alignment is not None
        # Contacts before the selected capture belong to world setup, not to a decision interval.
        self._collisions.clear()
        if self.timing.force_gt_duration_us:
            self._world_from_ego = self._recorded_ego(0).pose.copy()
            self.pose = self._recorded_planar_state(0)
            self._ego.set_kinematic(self._alignment.to_carla(self._world_from_ego))
        else:
            self._ego.release()
            self._ego.enable_physics(self._recorded_velocity(0))

    def _coerce_recorded_dynamics(self) -> None:
        super()._coerce_recorded_dynamics()
        assert self._ego is not None
        if self._ego.physics_enabled:
            self._ego.enable_physics(self._recorded_velocity(self._interval.start_us))

    def _sensor_motion_lookback_us(self) -> int:
        # A plan can retain the latest delivered camera pose as its origin.
        clocks = (self.timing.sensors.get(sensor, SensorTiming()) for sensor in self.sensors)
        acquisition_period = max((clock.resolved_frame_interval_us(self.decision_period_us)
                                  for clock in clocks), default=0)
        return max(super()._sensor_motion_lookback_us(), acquisition_period + self.timing.planner_delay_us)

    def _trajectory(self, control: Control, start: int, tick: int):
        xyz, _, timestamps = zip(*control.trajectory)
        anchor_time = min(start, timestamps[0])
        points, times = np.asarray(xyz), np.asarray(timestamps)
        if timestamps[0] > start:
            points = np.vstack((np.zeros(3), points))
            times = np.r_[start, times]
        if len(times) == 1:
            points = np.vstack((points, points[-1]))
            times = np.r_[times, times[-1] + tick]
        origin, _, _ = self._trajectory_capture(anchor_time)
        return points, times, np.linalg.inv(origin)

    def _propagate_ego(self, control: Control | None, report_times: tuple[int, ...], *,
                       recorded: bool) -> EgoPropagation:
        session = self._session
        tick = self.timing_capabilities.base_tick_us
        assert session is not None and tick is not None and self._start is not None
        ego, proxies, alignment = self._ego, self._proxies, self._alignment
        geometry, tracker = self._geometry, self._tracker
        assert ego is not None and proxies is not None and alignment is not None
        assert geometry is not None and tracker is not None
        wheelbase, max_steer = geometry.wheelbase, geometry.max_steer_angle
        assert wheelbase is not None and max_steer is not None
        start, end = report_times[0], report_times[-1]
        state, pose = replace(self.pose), self._world_from_ego
        states, poses = [state], [pose.copy()]
        trajectory = None
        if not recorded:
            assert control is not None
            if not ego.physics_enabled:
                ego.enable_physics(self._recorded_velocity(start))
            if control.trajectory is not None and len(control.trajectory):
                trajectory = self._trajectory(control, start, tick)
        if trajectory is None:
            tracker.reset()
        absolute_start = timestamp_to_microseconds(self._start.frame.timestamp)
        for target in range(start + tick, end + 1, tick):
            proxies.sync(self._actors_at(absolute_start + target))
            if recorded:
                pose = self._recorded_ego(target).pose
                ego.set_kinematic(alignment.to_carla(pose))
            else:
                assert control is not None
                command = control
                if trajectory is not None:
                    points, times, anchor = trajectory
                    local = anchor @ pose
                    yaw = math.atan2(local[1, 0], local[0, 0])
                    command = tracker.command(
                        points, times, local[:3, 3], yaw, abs(state.speed), target - tick, tick / 1e6,
                        wheelbase_m=wheelbase, max_wheel_angle_deg=max_steer,
                    )
                elif control.trajectory is not None:
                    command = Control(brake=1.0)
                ego.apply_control(command.throttle, command.brake, command.steer)
            session.tick()
            if recorded:
                state = self._recorded_planar_state(target)
                carla_rig = alignment.to_carla(pose)
            else:
                measured = ego.read()
                carla_rig = measured.world_from_rig
                pose = alignment.from_carla(carla_rig)
                carla_rotation = measured.world_from_rig[:3, :3]
                velocity = carla_rotation.T @ measured.velocity
                acceleration = carla_rotation.T @ measured.acceleration
                state = PlanarEgoState(
                    x=float(pose[0, 3]), y=float(pose[1, 3]), yaw=math.atan2(pose[1, 0], pose[0, 0]),
                    speed=float(velocity[0]), lateral_speed=float(velocity[1]),
                    yaw_rate=measured.yaw_rate, acceleration=float(acceleration[0]), t=target / 1e6,
                )
            self._collisions.update(proxies.touching(ego.box_transform(carla_rig), geometry.dimensions))
            states.append(state)
            poses.append(pose.copy())
        return EgoPropagation(EgoMotion(tuple(states)), state, pose, tuple(poses))

    def _close_session(self) -> None:
        if self._session is not None:
            session, self._session = self._session, None
            session.close()

    def close(self) -> None:
        """Destroy owned CARLA actors, restore asynchronous operation and close NRE."""
        try:
            self._close_session()
        finally:
            super().close()


__all__ = ["CarlaNuRecSimulator", "CarlaNuRecSimulatorRuntime"]
