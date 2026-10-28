"""Small deterministic worlds and drivers shared by public-API tests.

These are alternative implementations of the framework's extension interfaces. Tests use the
real experiment, simulator lifecycle, hooks, recorder and services around them. Native boundary
doubles below replace only software and hardware outside this repository.
"""
from __future__ import annotations

from copy import deepcopy
from collections import deque
from types import ModuleType, SimpleNamespace
import sys
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from avsectester.attack import Attack
from avsectester.attack.runtime import AttackRuntime
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import ActorState, Control, Detection, Observation, WorldSnapshot
from avsectester.closed_loop.simulators.base import SensorGeometry, SensorView, Simulator, SimulatorRuntime
from avsectester.closed_loop.simulators.nurec.renderer import Renderer
from avsectester.datasets import Dataset, FrameData, FrameReference, ScenarioDescriptor, ScenarioStart
from avsectester.geometry.cameras import BODY_FROM_OPTICAL, PinholeCamera
from avsectester.maps import Lane, ScenarioMap, StopLine, TrafficControl


CAMERA = PinholeCamera(np.array([[50., 0., 50.], [0., 50., 40.], [0., 0., 1.]]), 100, 80)
CAM_FROM_WORLD = np.linalg.inv(BODY_FROM_OPTICAL)


def actor_state(position=(10., 0., 1.), speed=0.):
    pose = np.eye(4)
    pose[:3, 3] = position
    return ActorState(pose, (speed, 0., 0.), "vehicle", (4., 2., 2.))


class ResetRecordingDynamics:
    """A user dynamics implementation that exposes its recorded initial state."""

    command_modes = frozenset({"actuator"})

    def reset(self, state):
        self.initial_state = deepcopy(state)

    def step(self, state, command, dt):
        from avsectester.closed_loop.simulators.ego_dynamics import KinematicBicycle

        return KinematicBicycle().step(state, command, dt)


class MemoryDataset(Dataset):
    """A calibrated camera, one stationary lead vehicle and an evenly spaced timeline."""

    def __init__(self, scenario_ids=("scene",), decision_steps=6, time_step=.1,
                 maps=None, scenarios_error=None, ego_position=(0., 0., 0.)):
        self.scenario_ids = tuple(scenario_ids)
        self.decision_steps, self.time_step = decision_steps, time_step
        self.maps = dict(maps or {})
        self.scenarios_error = scenarios_error
        self.ego_position = np.asarray(ego_position, dtype=float)

    def scenario_map(self, scenario_id):
        if scenario_id not in self.scenario_ids:
            raise KeyError(scenario_id)
        return self.maps[scenario_id] if scenario_id in self.maps else super().scenario_map(scenario_id)

    def scenarios(self):
        if self.scenarios_error == "interrupt":
            raise KeyboardInterrupt()
        if self.scenarios_error is not None:
            raise RuntimeError(self.scenarios_error)
        return iter(ScenarioDescriptor(name, "memory") for name in self.scenario_ids)

    def frames(self, scenario_id):
        if scenario_id not in self.scenario_ids:
            raise KeyError(scenario_id)
        return tuple(FrameReference(scenario_id, i, i, i * self.time_step)
                     for i in range(self.decision_steps + 1))

    def frame_data(self, scenario_id, frame):
        frame = self.checked_frame(scenario_id, frame)
        return FrameData(frame, actors={"lead": actor_state()},
                         ego=actor_state(self.ego_position + [frame.timestamp, 0., 0.], speed=1.))

    def scenario_start(self, scenario_id, frame):
        frame = self.checked_frame(scenario_id, frame)
        return ScenarioStart(
            "memory", scenario_id, frame, {"time_step": self.time_step},
            recorded_ego=tuple(self.frame_data(scenario_id, item).ego
                               for item in self.frames(scenario_id)),
            scenario_map=self.scenario_map(scenario_id),
        )

    def native(self, scenario_id):
        return {"scenario_id": scenario_id}


class StopApproachDataset(MemoryDataset):
    """A real Dataset extension with a straight approach and a recorded pause."""

    def __init__(self, scenario_map, stopped=True):
        self.positions = np.r_[np.arange(59.), [58., 58.], np.arange(59., 81.)]
        self.stopped = stopped
        super().__init__(decision_steps=len(self.positions) - 1, maps={"scene": scenario_map})

    def frame_data(self, scenario_id, frame):
        frame = self.checked_frame(scenario_id, frame)
        speed = 10.
        if self.stopped and 58 <= frame.position <= 60:
            speed = 0.
        return FrameData(frame, ego=actor_state((self.positions[frame.position], 0., 0.), speed))


def stop_approach_map(*, line_kind="stop_sign", facing=True, line_offset_y=0.,
                      linked_sign=True, sign_x=60.):
    """Map geometry for a straight recorded stop approach."""
    pose = np.diag([-1., -1., 1., 1.])
    pose[:3, 3] = [sign_x, -3., 2.]
    if not facing:
        pose[:3, :3] = np.eye(3)
    control = TrafficControl("sign", "stop_sign", "MUTCD_R1_STOP", pose, (.05, .8, .8),
                             "face_center", lane_ids=("lane",))
    lane = Lane("lane", np.array([[0., 0., 0.], [80., 0., 0.]]),
                np.array([[0., 3., 0.], [80., 3., 0.]]),
                np.array([[0., -3., 0.], [80., -3., 0.]]))
    line = StopLine("line", np.array([[60., -3. + line_offset_y, 0.],
                                     [60., 3. + line_offset_y, 0.]]), "ENTRY",
                    kind=line_kind, lane_ids=("lane",),
                    control_id="sign" if linked_sign else None)
    return ScenarioMap(lanes={"lane": lane}, traffic_controls={"sign": control},
                       stop_lines={"line": line})


class MemorySimulator(Simulator):
    """An image-producing actuator world, with explicit optional external failures."""

    dataset_kinds = ("memory",)
    can_render = True
    can_replay = True
    can_observe_past = True
    can_insert_objects = True
    command_modes = frozenset({"actuator"})
    supported_world_settings = frozenset({"weather", "lighting", "fail", "interrupt"})

    def __init__(self, time_step=.1, capture_ticks=0, world_key=None,
                 extra_sensors=None, camera_geometries=None, sensor_depths=None, actors=None, runtime_error=None, start_error=None,
                 can_replay=True, can_observe_past=True, can_insert_objects=True, can_render=True):
        self.time_step, self.capture_ticks = time_step, capture_ticks
        self.world_key = world_key
        self.runtime_error = runtime_error
        self.start_error = start_error
        self.can_render = can_render
        self.can_replay = can_replay
        self.can_observe_past = can_observe_past
        self.can_insert_objects = can_insert_objects
        self.extra_sensors = deepcopy(extra_sensors or {})
        self.camera_geometries = deepcopy(camera_geometries if camera_geometries is not None else {
            "front": SensorGeometry("front", CAMERA, CAM_FROM_WORLD),
        })
        self.sensor_depths = deepcopy(sensor_depths or {})
        self.initial_actors = deepcopy(actors if actors is not None else {"lead": actor_state()})

    @property
    def decision_time_step(self):
        return self.time_step

    @property
    def identity(self):
        return {**super().identity, "world_key": self.world_key}

    def validate_start(self, start):
        super().validate_start(start)
        if self.start_error is not None:
            raise RuntimeError(self.start_error)

    def create_runtime(self, seed=None):
        if self.runtime_error is not None:
            raise RuntimeError(self.runtime_error)
        return MemorySimulatorRuntime(self, seed)


class MemorySimulatorRuntime(SimulatorRuntime):
    """Native operations for the shared world; the production base owns reset and step."""

    def __init__(self, simulator, seed=None):
        super().__init__(simulator)
        self._native = {"objects": {}, "world": {}}

    @property
    def world_key(self):
        return self.simulator.world_key if self.simulator.world_key is not None else ("memory", id(self))

    @property
    def native(self):
        return self._native

    def _initialize_world(self, start, initialization, environment):
        self.position, self.distance, self.speed = 0, 0., 1.
        self.actors = deepcopy(self.simulator.initial_actors)
        self._native = {"objects": {}, "world": deepcopy(environment)}

    def _replay_position(self, start):
        return 0 if start is None else start.frame.position

    def _replay_tick(self):
        self.position += 1
        self.distance += self.simulator.time_step

    def _hold_tick(self):
        pass

    def _check_initialization(self, initialization):
        # Memory frame data and replay follow the same exact prescribed motion.
        pass

    def _observe(self):
        color = 20 + self.position
        images = {sensor: np.full((geometry.camera.height, geometry.camera.width, 3), color, np.uint8)
                  for sensor, geometry in self.simulator.camera_geometries.items()}
        return Observation(self.position * self.simulator.time_step, self.position,
                           {**images, **deepcopy(self.simulator.extra_sensors)}, ego_speed=self.speed)

    def _observe_recorded(self, position):
        from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState

        start = self._start
        ego = start.recorded_ego[position]
        relative_position = position - start.frame.position
        images = {sensor: np.full((geometry.camera.height, geometry.camera.width, 3),
                                  20 + position, np.uint8)
                  for sensor, geometry in self.simulator.camera_geometries.items()}
        velocity = np.asarray(ego.velocity, dtype=float)
        return Observation(
            relative_position * self.simulator.time_step, relative_position,
            {**images, **deepcopy(self.simulator.extra_sensors)},
            vehicle_state=PlanarEgoState(x=float(ego.pose[0, 3]), y=float(ego.pose[1, 3]),
                                  yaw=float(np.arctan2(ego.pose[1, 0], ego.pose[0, 0])),
                                  speed=float(np.linalg.norm(velocity)),
                                  t=relative_position * self.simulator.time_step),
            ego_speed=float(np.linalg.norm(velocity)),
        )

    def _recorded_sensor_geometry(self, position, sensor):
        if sensor not in self.camera_sensors:
            raise ValueError(f"Unknown memory camera: {sensor}")
        geometry = self.simulator.camera_geometries[sensor]
        transform = geometry.cam_from_world.copy()
        transform[:, 3] -= transform[:, 0] * self._start.recorded_ego[position].pose[0, 3]
        return SensorGeometry(sensor, geometry.camera, transform)

    def _recorded_sensor_view(self, position, sensor, observation):
        geometry = self._recorded_sensor_geometry(position, sensor)
        return SensorView(sensor, observation.sensor_data[sensor], geometry.camera, geometry.cam_from_world,
                          {}, self._start.recorded_ego[position], observation.t, observation.frame,
                          depth=deepcopy(self.simulator.sensor_depths.get(sensor)),
                          sensor_from_world=BODY_FROM_OPTICAL @ geometry.cam_from_world)

    def _state(self):
        return WorldSnapshot(self.position * self.simulator.time_step, self.position,
                             ego_speed=self.speed, actors=deepcopy(self.actors),
                             ego=actor_state((self.distance, 0., 0.), self.speed))

    def _advance_native(self, control):
        world = self.native["world"]
        if world.get("interrupt"):
            raise KeyboardInterrupt("memory simulation interrupted")
        if world.get("fail"):
            raise RuntimeError("memory simulation failed")
        rain = .5 if world.get("weather") == "rain" else 1.
        self.speed = max(0., self.speed + (control.throttle * rain - control.brake) * self.simulator.time_step)
        self.distance += self.speed * self.simulator.time_step
        self.position += 1

    @property
    def sensors(self):
        return frozenset({*self.simulator.camera_geometries, *self.simulator.extra_sensors})

    @property
    def camera_sensors(self):
        return frozenset(self.simulator.camera_geometries)

    def sensor_geometry(self, sensor):
        if sensor not in self.camera_sensors:
            raise ValueError(f"Unknown memory camera: {sensor}")
        geometry = self.simulator.camera_geometries[sensor]
        transform = geometry.cam_from_world.copy()
        transform[:, 3] -= transform[:, 0] * self.distance
        return SensorGeometry(sensor, geometry.camera, transform)

    def _sensor_view(self, sensor):
        if sensor not in self.sensors:
            raise ValueError(f"Unknown memory sensor: {sensor}")
        observation, state = self._capture, self.state()
        data = observation.sensor_data[sensor]
        body_from_world = np.eye(4)
        body_from_world[0, 3] = -self.distance
        camera, cam_from_world = None, None
        if sensor in self.camera_sensors:
            geometry = self.sensor_geometry(sensor)
            camera, cam_from_world = geometry.camera, geometry.cam_from_world
            body_from_world = BODY_FROM_OPTICAL @ cam_from_world
        elif not isinstance(data, np.ndarray):
            data = getattr(data, "data", data)
        return SensorView(sensor, data, camera, cam_from_world,
                          state.actors,
                          state.ego, state.t, state.frame,
                          depth=deepcopy(self.simulator.sensor_depths.get(sensor)),
                          sensor_from_world=body_from_world)

    def set_actor_state(self, actor_id, state):
        if actor_id not in self.actors:
            raise KeyError(actor_id)
        self.actors[actor_id] = deepcopy(state)

    def control_actor(self, actor_id, control):
        state = self.actors[actor_id]
        self.actors[actor_id] = replace(state, velocity=(float(control.throttle), 0., 0.))

    def _capture_frame_data(self, frame):
        state = self.state()
        return FrameData(frame, actors=state.actors, ego=state.ego)

    def close(self):
        super().close()
        self.native["objects"].clear()


class ConstantAVStack(AVStack):
    """A deterministic driver exposing one actual component-stage boundary."""

    command_modes = frozenset({"actuator"})
    supported_stages = frozenset({"perception.post"})

    def __init__(self, command=None, perception_output=None,
                 fail_on_position=None, interrupt_on_position=None):
        self.command = Control(throttle=.25) if command is None else command
        self.perception_output = [] if perception_output is None else perception_output
        self.fail_on_position, self.interrupt_on_position = fail_on_position, interrupt_on_position
        self.output = None

    def reset(self, observation, past=()):
        self.output = None

    def __call__(self, observation):
        if observation.frame == self.interrupt_on_position:
            raise KeyboardInterrupt("memory driver interrupted")
        if observation.frame == self.fail_on_position:
            raise RuntimeError("memory driver failed")
        self.output = self._emit("perception.post", deepcopy(self.perception_output))
        return replace(self.command)

    def component_log(self):
        return {"perception": self.output}

    def detections(self):
        if isinstance(self.output, (tuple, list)) and all(isinstance(item, Detection) for item in self.output):
            return tuple(self.output)
        return None


class HistoryAVStack(ConstantAVStack):
    """A driver that records its actual warmup inputs in public component telemetry."""

    past_steps = 2

    def reset(self, observation, past=()):
        super().reset(observation, past=past)
        self.past_observations = tuple(deepcopy(item) for item in past)
        self.initial_observation = deepcopy(observation)

    def component_log(self):
        return {**super().component_log(), "past_observations": self.past_observations,
                "initial_observation": self.initial_observation}


def decelerating_command(speed, t, acceleration=-2.):
    """A local trajectory with constant acceleration and 0.1 s samples over five seconds."""
    times = np.arange(51) * .1
    positions = float(speed) * times + acceleration * times ** 2 / 2
    return Control(trajectory=[
        ((float(position), 0., 0.), (1., 0., 0., 0.), round((t + time) * 1e6))
        for time, position in zip(times, positions)
    ])


class DeceleratingAVStack(AVStack):
    """Replan a -2 m/s² trajectory from the current observed speed every decision."""

    command_modes = frozenset({"trajectory"})
    past_steps = 15

    def __call__(self, observation):
        return decelerating_command(observation.ego_speed, observation.t)


class ImageForwardAVStack(DeceleratingAVStack):
    """Plan farther when the captured image is darker, for black-box transport tests."""

    def __call__(self, observation):
        image = np.asarray(next(iter(observation.sensor_data.values())), dtype=float)
        forward = 5. + 20. * (1. - image.mean() / 255.)
        return Control(trajectory=[((0., 0., 0.), (1., 0., 0., 0.), round(observation.t * 1e6)),
                                   ((forward, 0., 0.), (1., 0., 0., 0.), round((observation.t + 5.) * 1e6))])


class MetadataOnlyAttack(Attack):
    """A user attack that accepts supplied segments and has no scenario selector."""

    def __init__(self, required_key=None):
        self.required_key = required_key

    def validate_segment(self, segment):
        if self.required_key is not None and self.required_key not in segment.user_data:
            raise ValueError(f"Missing required user_data key {self.required_key!r}")


class MinimalAttack(Attack):
    """Select deterministic consecutive windows and record per-run progress."""

    def __init__(self, decision_steps=2, max_segments=2, assets=None, fail_scenarios=(), stop_marker_path=None,
                 variants_per_window=1):
        self.decision_steps, self.max_segments, self.assets = decision_steps, max_segments, assets
        self.fail_scenarios, self.stop_marker_path = tuple(fail_scenarios), stop_marker_path
        self.variants_per_window = variants_per_window

    def build_metadata(self, scenario):
        if scenario.scenario_id in self.fail_scenarios:
            raise RuntimeError("synthetic selection failed")
        from avsectester.attack.segments import split_segments
        windows = split_segments(scenario.frames, lambda frame: True,
                                 decision_steps=self.decision_steps)[:self.max_segments]
        if self.variants_per_window == 1:
            return windows
        return tuple(replace(segment, user_data={"variant": variant})
                     for segment in windows for variant in range(self.variants_per_window))

    def run_offline(self, context):
        return self.assets

    def create_runtime(self, segment, artifacts):
        return MinimalAttackRuntime(segment, self.stop_marker_path)


class MinimalAttackRuntime(AttackRuntime):
    """Record output through the real recorder, with optional durable cleanup evidence."""

    def __init__(self, segment, stop_marker_path=None):
        self.segment, self.stop_marker_path = segment, stop_marker_path
        self.decision = 0

    def step(self, context):
        context.record_output("progress", {"decision": self.decision, "frame": context.frame})
        self.decision += 1
        return ()

    def stop(self, context):
        if self.stop_marker_path is not None:
            path = Path(self.stop_marker_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps({"scenario": self.segment.scenario_id,
                                         "start": self.segment.start.position}) + "\n")


# Native dependency boundaries used by the real simulator implementations.


def proto_world_pose(pose):
    """Read the external RPC pose fields for independent boundary expectations."""
    quaternion = np.array([pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z])
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ]
    result[:3, 3] = [pose.vec.x, pose.vec.y, pose.vec.z]
    return result


class NuRecRpcError(Exception):
    """A gRPC status error raised by the NuRec service boundary."""

    def __init__(self, code, details):
        super().__init__(details)
        self._code, self._details = code, details

    def code(self):
        return getattr(sys.modules["grpc"].StatusCode, self._code, self._code)

    def details(self):
        return self._details


def container_running(state_path):
    """Whether a Docker fake container state is absent (no container check) or running."""
    if state_path is None:
        return True
    try:
        state = json.loads(state_path.read_text())
    except FileNotFoundError:
        return False
    return state.get("running", False) if isinstance(state, dict) else state is True


class NuRecRpcBoundary:
    """A SensorsimService boundary; NuRecRenderer itself remains real.

    Inserted Gaussian assets are drawn at their listed poses, mapped into track frames as the
    measured NRE convention does, except at pixels in ``hidden``, where the scene covers them.
    ``asset_directories`` maps server asset paths to local ones. With ``scene_tracks`` set, a
    render naming any other track that was not inserted fails as NRE does.
    ``returned_resolution_hw`` is None for the native resolution, an (H, W) tuple, or a
    callable mapping the integer frame-start timestamp in microseconds to an (H, W) tuple. With
    ``service_container`` set, scene queries answer UNAVAILABLE unless that Docker fake state
    reports a running container.
    """

    TRACK_FROM_PLY = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    TRACK_PADDING_M = np.array([1., 1., .25])

    def __init__(self, returned_resolution_hw=None, *, service_container=None):
        self.returned_resolution_hw = returned_resolution_hw
        self.service_container = service_container
        self.last_request = None
        self.requests = []
        self.available_scenes = ("scene",)
        self.scene_query_error = None
        self.rpc_error = NuRecRpcError
        self.scene_query_timeouts = []
        self.tracks = {}
        self.scene_tracks = None
        self.hidden = None
        self.asset_directories = {}
        self.mount = SimpleNamespace(vec=SimpleNamespace(x=0., y=0., z=1.5),
            quat=SimpleNamespace(w=.5, x=-.5, y=.5, z=-.5))
        self.spec = SimpleNamespace(resolution_h=12, resolution_w=16, ftheta_param=SimpleNamespace(
            principal_point_x=8., principal_point_y=6., angle_to_pixeldist_poly=[0., 8.],
            pixeldist_to_angle_poly=[0., .125], max_angle=1.5,
            linear_cde=SimpleNamespace(linear_c=1., linear_d=0., linear_e=0.)),
            WhichOneof=lambda group: None)

    def get_available_scenes(self, request, timeout=None):
        self.requests.append(request)
        self.scene_query_timeouts.append(timeout)
        if not container_running(self.service_container):
            raise self.rpc_error("UNAVAILABLE", "renderer is not running")
        if self.scene_query_error is not None:
            raise self.rpc_error(*self.scene_query_error)
        return SimpleNamespace(scene_ids=list(self.available_scenes))

    def get_available_cameras(self, request):
        self.requests.append(request)
        return SimpleNamespace(available_cameras=[SimpleNamespace(logical_id="front",
            intrinsics=self.spec, rig_to_camera=self.mount)])

    def get_available_trajectories(self, request):
        self.requests.append(request)
        pose = SimpleNamespace(vec=SimpleNamespace(x=0., y=0., z=.3),
                               quat=SimpleNamespace(w=1., x=0., y=0., z=0.))
        return SimpleNamespace(available_trajectories=[SimpleNamespace(trajectory=SimpleNamespace(
            poses=[SimpleNamespace(timestamp_us=0, pose=pose)]))])

    def render_rgb(self, request):
        from io import BytesIO
        from PIL import Image
        from avsectester.geometry.cameras import FThetaCamera
        self.requests.append(request)
        self.last_request = request
        returned = self.returned_resolution_hw
        if callable(returned):
            returned = returned(request.frame_start_us)
        height, width = returned or (
            request.resolution_h,
            round(request.resolution_h * self.spec.resolution_w / self.spec.resolution_h),
        )
        listed = [item.track_id for item in request.dynamic_objects]
        unknown = [track for track in listed if track not in self.tracks
                   and self.scene_tracks is not None and track not in self.scene_tracks]
        if unknown:
            raise self.rpc_error("INVALID_ARGUMENT", f"{unknown[0]!r} is not in list")
        image = np.full((height, width, 3), 30, np.uint8)
        camera = FThetaCamera(8.5, 6.5, (0., 8.), 16, 12, (0., .125), 1.5)
        cam_from_world = np.linalg.inv(proto_world_pose(request.sensor_pose.start_pose))
        for item in request.dynamic_objects:
            if item.track_id not in self.tracks:
                continue
            points, colours = self.tracks[item.track_id]
            cam_from_track = cam_from_world @ proto_world_pose(item.pose_pair.start_pose)
            optical = points @ cam_from_track[:3, :3].T + cam_from_track[:3, 3]
            front = optical[:, 2] > 0
            pixels = np.floor(camera.project(optical[front])).astype(int)
            inside = (pixels >= 0).all(axis=1) & (pixels < [width, height]).all(axis=1)
            for (x, y), colour in zip(pixels[inside], colours[front][inside]):
                if self.hidden is None or not self.hidden[y, x]:
                    image[y, x] = colour
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        return SimpleNamespace(image_bytes=stream.getvalue())

    def edit_assets(self, request):
        self.requests.append(request)
        for track in request.insert:
            directory, name = track.asset_id.rsplit("/", 1)
            data = (Path(self.asset_directories.get(directory, directory)) / name).read_bytes()
            header, body = data.split(b"end_header\n", 1)
            rows = np.frombuffer(body, "<f4").reshape(-1, 14)
            size = np.array([track.object_size.size_x, track.object_size.size_y, track.object_size.size_z])
            scale = (size - self.TRACK_PADDING_M).max()
            colours = np.clip((rows[:, 3:6] * .28209479177387814 + .5) * 255, 0, 255).round().astype(np.uint8)
            self.tracks[track.id] = (rows[:, :3] @ self.TRACK_FROM_PLY.T * scale, colours)
        return SimpleNamespace(success=True, message="Success")

    def _unary_unary(self, method, request_serializer, response_deserializer):
        if not method.endswith("/edit_assets"):
            raise NotImplementedError(method)
        return lambda request, timeout=None: self.edit_assets(request)

    def install(self, monkeypatch, *, passthrough_endpoints=()):
        boundary = self
        native_grpc = None
        native_common = None
        if passthrough_endpoints:
            import grpc as native_grpc
            from alpasim_grpc.v0 import common_pb2 as native_common
            native_channel = native_grpc.insecure_channel
            self.rpc_error = type("SensorsimRpcError", (NuRecRpcError, native_grpc.RpcError), {})

        class Channel:
            def close(self):
                pass

            def unary_unary(self, *args, **kwargs):
                return boundary._unary_unary(*args, **kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        common = native_common or SimpleNamespace(Pose=SimpleNamespace, Quat=SimpleNamespace,
                                                  Vec3=SimpleNamespace, Empty=SimpleNamespace)
        sensor = SimpleNamespace(AvailableCamerasRequest=SimpleNamespace,
            AvailableTrajectoriesRequest=SimpleNamespace, RGBRenderRequest=SimpleNamespace,
            PosePair=SimpleNamespace, DynamicObject=SimpleNamespace,
            ImageFormat=SimpleNamespace(JPEG=1))
        modules = {
            "grpc": {"insecure_channel": lambda endpoint: Channel(),
                     "RpcError": NuRecRpcError,
                     "StatusCode": SimpleNamespace(INVALID_ARGUMENT="INVALID_ARGUMENT",
                         UNAVAILABLE="UNAVAILABLE", DEADLINE_EXCEEDED="DEADLINE_EXCEEDED",
                         PERMISSION_DENIED="PERMISSION_DENIED")},
            "alpasim_grpc": {}, "alpasim_grpc.v0": {"common_pb2": common,
                "sensorsim_pb2": sensor,
                "sensorsim_pb2_grpc": SimpleNamespace(SensorsimServiceStub=lambda channel: self)},
        }
        if native_grpc is not None:
            grpc_values = dict(vars(native_grpc))
            grpc_values.update(
                insecure_channel=lambda endpoint: (
                    native_channel(endpoint) if endpoint in passthrough_endpoints else Channel()),
            )
            modules = {"grpc": grpc_values}
            import alpasim_grpc.v0 as protocol
            monkeypatch.setattr(protocol, "sensorsim_pb2", sensor, raising=False)
            monkeypatch.setattr(protocol, "sensorsim_pb2_grpc",
                                SimpleNamespace(SensorsimServiceStub=lambda channel: self), raising=False)
        for name, values in modules.items():
            module = ModuleType(name)
            vars(module).update(values)
            monkeypatch.setitem(sys.modules, name, module)
        return self


class NuRecImageBoundary(Renderer):
    """A deterministic image service boundary with a chosen optical lens model."""

    cameras = ["front"]

    def __init__(self, camera, camera_id="front", rig_from_camera=None):
        self.camera = camera
        self.cameras = [camera_id]
        self.rig_from_camera = rig_from_camera

    def camera_model(self, camera=None):
        if camera is not None and camera not in self.cameras:
            raise ValueError(f"Unknown camera {camera!r}")
        return self.camera

    def cam_from_world(self, world_from_rig, camera=None):
        rig_from_camera = BODY_FROM_OPTICAL.copy()
        rig_from_camera[2, 3] = 1.5
        if self.rig_from_camera is not None:
            rig_from_camera = self.rig_from_camera
        return np.linalg.inv(world_from_rig @ rig_from_camera)

    def render(self, world_from_rig, timestamp_us, camera, dynamic_objects=()):
        return np.full((self.camera.height, self.camera.width, 3), 30, np.uint8)


class NuRecTrafficImageBoundary(NuRecImageBoundary):
    """Record native image/probe requests while using a deterministic image service."""

    def __init__(self, camera):
        super().__init__(camera)
        self.rendered = []
        self.probes = []
        self.fail_render = False

    def render(self, world_from_rig, timestamp_us, camera, dynamic_objects=()):
        if self.fail_render:
            raise RuntimeError("image service failed")
        self.rendered.append((timestamp_us, {key: pose.copy() for key, pose in dynamic_objects}))
        return super().render(world_from_rig, timestamp_us, camera, dynamic_objects)

    def render_scene_visibility(self, world_from_rig, timestamp_us, camera, objects,
                                dynamic_objects=(), *, occlusion_tolerance_m=0.):
        self.probes.append((timestamp_us, {key: pose.copy() for key, pose in dynamic_objects}))
        return np.ones((self.camera.height, self.camera.width), dtype=bool)


class CarlaVector:
    def __init__(self, x=0., y=0., z=0.):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def norm(self):
        return float(np.linalg.norm((self.x, self.y, self.z)))


class CarlaTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location or CarlaVector()
        self.rotation = rotation or SimpleNamespace(pitch=0., yaw=0., roll=0.)

    def get_matrix(self):
        pitch, yaw, roll = np.deg2rad([self.rotation.pitch, self.rotation.yaw, self.rotation.roll])
        cosine_pitch, cosine_yaw, cosine_roll = np.cos([pitch, yaw, roll])
        sine_pitch, sine_yaw, sine_roll = np.sin([pitch, yaw, roll])
        pose = np.eye(4)
        pose[:3, :3] = [
            [cosine_pitch * cosine_yaw,
             cosine_yaw * sine_pitch * sine_roll - sine_yaw * cosine_roll,
             -cosine_yaw * sine_pitch * cosine_roll - sine_yaw * sine_roll],
            [cosine_pitch * sine_yaw,
             sine_yaw * sine_pitch * sine_roll + cosine_yaw * cosine_roll,
             -sine_yaw * sine_pitch * cosine_roll + cosine_yaw * sine_roll],
            [sine_pitch, -cosine_pitch * sine_roll, cosine_pitch * cosine_roll],
        ]
        pose[:3, 3] = (self.location.x, self.location.y, self.location.z)
        return pose

    def get_inverse_matrix(self):
        return np.linalg.inv(self.get_matrix())

    def get_forward_vector(self):
        return CarlaVector(1., 0., 0.)

    def get_right_vector(self):
        return CarlaVector(0., 1., 0.)


class CarlaActor:
    def __init__(self, world, actor_id, vehicle="vehicle.test", transform=None):
        self.world, self.id, self.type_id = world, actor_id, vehicle
        self.transform = transform or CarlaTransform()
        self.velocity = CarlaVector()
        self.control = SimpleNamespace(throttle=0., brake=0., steer=0.)
        self.bounding_box = SimpleNamespace(extent=CarlaVector(2., 1., .7),
            location=CarlaVector(.2, 0., .7), rotation=SimpleNamespace(roll=0., pitch=0., yaw=0.))
        self.physics = SimpleNamespace(wheels=[SimpleNamespace(tire_friction=2.)])
        self.destroyed = False
        self.light_state = None
        self.attached_to = None
        self.textures = {}
        self.attributes = {}
        self.capture_callback = None

    def listen(self, callback):
        self.capture_callback = callback

    def stop(self):
        self.capture_callback = None

    @property
    def is_alive(self):
        return not self.destroyed

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity

    def set_transform(self, transform):
        self.transform = transform

    def set_target_velocity(self, velocity):
        self.velocity = velocity

    def set_target_angular_velocity(self, velocity):
        pass

    def set_simulate_physics(self, enabled):
        pass

    def set_enable_gravity(self, enabled):
        pass

    def set_autopilot(self, enabled, port):
        pass

    def apply_control(self, control):
        self.control = control

    def get_physics_control(self):
        return self.physics

    def apply_physics_control(self, physics):
        self.physics = physics

    def set_light_state(self, state):
        self.light_state = state

    def destroy(self):
        self.destroyed = True


class CarlaTrafficLight(CarlaActor):
    def __init__(self, world, actor_id, signal_id, state="Red"):
        super().__init__(world, actor_id, "traffic.traffic_light")
        self.signal_id = signal_id
        self.state = state

    def get_opendrive_id(self):
        return self.signal_id

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class CarlaSensorManager:
    def __init__(self):
        self.data = {"front": deque()}

    def empty(self):
        return not any(self.data.values())

    def has_data(self, name):
        return bool(self.data[name])

    def pop(self, name):
        return self.data[name].popleft()


class CarlaLidarCapture:
    """The native SDK buffer can be read as bytes but cannot be deep-copied."""

    __module__ = "carla.libcarla"

    def __init__(self, frame, timestamp):
        self.frame, self.timestamp = frame, timestamp
        self.raw_data = memoryview(np.array([[5., -2., .5, .8]], np.float32).tobytes())

    def __deepcopy__(self, memo):
        raise TypeError("Native CARLA capture cannot be deep-copied")


class CarlaImageData(SimpleNamespace):
    """ImageData's RGB accessor follows its replaceable pixel buffer."""

    @property
    def rgb_image(self):
        return self.data


class CarlaMobileActor:
    def __init__(self, client, specification):
        self.client = client
        self.specification = deepcopy(specification)
        spawn = specification.get("spawn", 0)
        if "spawn_transform" in specification:
            spec = specification["spawn_transform"]
            transform = CarlaTransform(CarlaVector(**spec["location"]), SimpleNamespace(**spec["rotation"]))
        else:
            transform = CarlaTransform(CarlaVector(float(spawn) * 3, 0., 0.))
        self.actor = CarlaActor(client.world, len(client.world.actors) + 1,
            specification.get("vehicle", specification.get("npc_type", "vehicle.test")), transform)
        client.world.actors.append(self.actor)
        self.spawn_transform = transform
        self.reference = SimpleNamespace(x=np.zeros(3), q=np.array([1., 0., 0., 0.]))
        self.timestamp = 0.
        self.sensor_data_manager = CarlaSensorManager()
        native_camera = SimpleNamespace(type_id="sensor.camera.rgb",
            get_transform=self.actor.get_transform,
            attributes={"image_size_x": "16", "image_size_y": "12", "fov": "90",
                        "lens_k": "0", "lens_kcube": "0", "lens_circle_multiplier": "0"})
        self.sensors = {"front": SimpleNamespace(source_identifier="front", frame0=0, t0=0.,
            object=native_camera, reference=SimpleNamespace(as_carla_transform=lambda local=True: CarlaTransform()),
            imsize=(12, 16), P=np.array([[8., 0., 8., 0.],
             [0., 8., 6., 0.], [0., 0., 1., 0.]]))}
        for sensor_config in specification.get("sensors", ()):
            if sensor_config["type"] == "CarlaLidar":
                def lidar_transform():
                    location = self.actor.get_transform().location
                    return CarlaTransform(CarlaVector(location.x + 1., location.y + 2., 1.5),
                                          SimpleNamespace(pitch=0., yaw=90., roll=0.))

                self.sensors["lidar"] = SimpleNamespace(source_identifier="lidar", frame0=0, t0=0.,
                    object=SimpleNamespace(type_id="sensor.lidar.ray_cast", get_transform=lidar_transform))
                self.sensor_data_manager.data["lidar"] = deque()
        client.world.vehicles.append(self)

    def initialize(self, timestamp, frame):
        for sensor in self.sensors.values():
            sensor.frame0, sensor.t0 = frame, timestamp

    def apply_control(self, control):
        self.actor.apply_control(control)

    def get_pose(self):
        vector = self.actor.transform.location
        return SimpleNamespace(position=SimpleNamespace(x=np.array([vector.x, -vector.y, vector.z])),
                               attitude=SimpleNamespace(q=np.array([1., 0., 0., 0.])))

    def get_object_state(self):
        pose = self.get_pose()
        return SimpleNamespace(position=pose.position, attitude=pose.attitude, velocity=self.actor.velocity)

    def capture(self, frame, timestamp, frame_offset=0):
        # Identical physical scenes have identical pixels even when reset setup ticks
        # give their captures different native clock/frame metadata.
        color = int(round(self.actor.transform.location.x * 1000)) % 255
        sensor = self.sensors["front"]
        payload = CarlaImageData(frame=frame - sensor.frame0 + frame_offset,
            timestamp=timestamp - sensor.t0,
            data=np.full((12, 16, 3), color, np.uint8),
            calibration=SimpleNamespace(reference=SimpleNamespace(reference=self.reference)))
        self.sensor_data_manager.data["front"].append(payload)
        if "lidar" in self.sensors:
            lidar = self.sensors["lidar"]
            payload = SimpleNamespace(frame=frame - lidar.frame0 + frame_offset,
                timestamp=timestamp - lidar.t0,
                data=CarlaLidarCapture(frame, timestamp), flipy=True, n_features=4)
            self.sensor_data_manager.data["lidar"].append(payload)

    def destroy(self):
        self.actor.destroy()


class CarlaBlueprint:
    """CARLA actor blueprint fields consumed by sensor construction."""

    def __init__(self, identity):
        self.id = identity
        self.attributes = {}

    def set_attribute(self, name, value):
        self.attributes[name] = str(value)


class CarlaWorld:
    def __init__(self, config, frame_offset=0):
        self.frame, self.time = 0, 0.
        self.dt = 1 / config.get("rate", 10.)
        self.actors, self.vehicles = [], []
        self.actors.append(CarlaTrafficLight(self, 10000, "signal-1"))
        self.config, self.frame_offset = config, frame_offset
        self.weather = None
        self.layers = []

    def get_snapshot(self):
        return SimpleNamespace(frame=self.frame,
            timestamp=SimpleNamespace(elapsed_seconds=self.time),
            find=lambda identity: next((actor for actor in self.actors if actor.id == identity), None))

    def tick(self):
        self.frame += 1
        self.time += self.dt
        for actor in self.actors:
            speed = max(0., actor.velocity.x + (actor.control.throttle - actor.control.brake) * self.dt)
            actor.velocity = CarlaVector(speed, actor.velocity.y, actor.velocity.z)
            actor.transform.location.x += speed * self.dt
        for actor in self.vehicles:
            actor.capture(self.frame, self.time, self.frame_offset)
        for actor in self.actors:
            if actor.type_id == "sensor.camera.depth" and actor.capture_callback is not None:
                width = int(actor.attributes.get("image_size_x", 16))
                height = int(actor.attributes.get("image_size_y", 12))
                encoded = round(10. / 1000. * (256 ** 3 - 1))
                bgra = np.empty((height, width, 4), np.uint8)
                bgra[:] = [(encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255, 255]
                actor.capture_callback(SimpleNamespace(
                    frame=self.frame, timestamp=self.time, width=width, height=height,
                    raw_data=bgra.tobytes()))
        return self.frame

    def get_actors(self):
        return list(self.actors)

    def get_blueprint_library(self):
        vehicles = [SimpleNamespace(id="vehicle.a"), SimpleNamespace(id="vehicle.b")]
        return SimpleNamespace(filter=lambda pattern: (
            [CarlaBlueprint(pattern)] if pattern.startswith("static.prop") else vehicles),
            find=CarlaBlueprint)

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        actor = CarlaActor(self, len(self.actors) + 1, blueprint.id, transform)
        actor.attributes = dict(getattr(blueprint, "attributes", {}))
        actor.attached_to = attach_to
        self.actors.append(actor)
        return actor

    def try_spawn_actor(self, blueprint, transform):
        return self.spawn_actor(blueprint, transform)

    def get_names_of_all_objects(self):
        return [f"object-{actor.id}" for actor in self.actors
                if actor.type_id.startswith("static.prop") and actor.is_alive]

    def apply_color_texture_to_object(self, name, parameter, texture):
        actor = next(actor for actor in self.actors if name == f"object-{actor.id}")
        actor.textures[parameter] = texture.pixels.copy()

    def apply_float_color_texture_to_object(self, name, parameter, texture):
        self.apply_color_texture_to_object(name, parameter, texture)

    def set_weather(self, weather):
        self.weather = weather

    def get_map(self):
        return SimpleNamespace(name="Town03_Opt")

    def unload_map_layer(self, layer):
        self.layers.append(("unload", layer))

    def load_map_layer(self, layer):
        self.layers.append(("load", layer))


class CarlaNativeClient:
    def __init__(self, config, frame_offset=0):
        self.config = config
        self.world = CarlaWorld(config, frame_offset)
        self.spawn_points = [CarlaTransform(CarlaVector(i * 3, 0, 0)) for i in range(6)]
        self.traffic_manager_port = config.get("traffic_manager_port", 8000)

    def scene_settings(self):
        return {}

    def tick(self):
        return self.world.tick()

    def close(self):
        pass


class CarlaTexture:
    def __init__(self, width, height):
        self.pixels = np.zeros((height, width, 4), np.float32)

    def set(self, x, y, color):
        self.pixels[y, x] = color


class CarlaNativeBoundary:
    """Registry/client boundary used by the real CARLA simulator, without a server."""

    def __init__(self, frame_offset=0):
        self.frame_offset = frame_offset
        self.module_dict = {}

    def get(self, name):
        return self.module_dict.get(name)

    def register_module(self, name=None, force=False):
        def register(component):
            key = name or component.__name__
            if key in self.module_dict and not force:
                raise KeyError(key)
            self.module_dict[key] = component
            return component
        return register

    def build(self, config, default_args=None):
        if default_args is None:
            return CarlaNativeClient(config, self.frame_offset)
        return CarlaMobileActor(default_args["client"], config)

    def install(self, monkeypatch, rgb_camera):
        if rgb_camera is not None:
            self.module_dict["CarlaRgbCamera"] = rgb_camera
        carla = ModuleType("carla")
        carla.Vector3D = carla.Location = CarlaVector
        carla.LidarMeasurement = CarlaLidarCapture
        carla.Transform = CarlaTransform
        carla.Rotation = lambda **values: SimpleNamespace(**values)
        carla.TextureColor = carla.TextureFloatColor = CarlaTexture
        carla.Color = carla.FloatColor = lambda r, g, b, a: (r, g, b, a)
        carla.MaterialParameter = SimpleNamespace(Diffuse="Diffuse", Emissive="Emissive",
            AO_Roughness_Metallic_Emissive="material")
        carla.AttachmentType = SimpleNamespace(Rigid="Rigid")
        from enum import IntFlag
        carla.VehicleLightState = IntFlag("VehicleLightState", {
            "NONE": 0, "Position": 1, "LowBeam": 2, "HighBeam": 4, "Brake": 8})
        carla.VehicleControl = lambda throttle=0., brake=0., steer=0.: SimpleNamespace(
            throttle=throttle, brake=brake, steer=steer)
        carla.WeatherParameters = lambda **values: SimpleNamespace(**values)
        carla.MapLayer = SimpleNamespace(Buildings="Buildings")
        carla.TrafficLightState = SimpleNamespace(Red="Red", Yellow="Yellow", Green="Green", Off="Off",
                                                 Unknown="Unknown")
        monkeypatch.setitem(sys.modules, "carla", carla)
        monkeypatch.setitem(sys.modules, "avcarla", ModuleType("avcarla"))
        monkeypatch.setitem(sys.modules, "avcarla.config", SimpleNamespace(CARLA=self))
        return self


def recorded_policy_observations(count=16, start_time=-1.5, time_step=.1, speed=4.,
                                 acceleration=0., yaw=0., route=None):
    """A small synthetic recording supplied to policy tests as actual observed history."""
    from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState

    observations = []
    for index in range(count):
        elapsed = index * time_step
        time = start_time + elapsed
        distance = speed * elapsed + .5 * acceleration * elapsed ** 2
        state = PlanarEgoState(x=distance * float(np.cos(yaw)), y=distance * float(np.sin(yaw)),
                        yaw=yaw, speed=speed + acceleration * elapsed, t=time,
                        acceleration=acceleration)
        observations.append(Observation(
            time, index, {"front": np.full((4, 6, 3), index, np.uint8)},
            vehicle_state=state, ego_speed=state.speed, route=deepcopy(route)))
    return tuple(observations)


def yolox_detection_model(predictions=None, *, image_weight=0.):
    """Create raw YOLOX predictions with optional image-dependent objectness."""
    import torch

    if predictions is None:
        probabilities = [0.] * 80
        probabilities[11] = 1.
        predictions = [[50., 40., 4., 4., .9, *probabilities]]

    class DetectionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(float(image_weight)))
            self.last_input = None

        def forward(self, images):
            self.last_input = images.detach().clone()
            rows = images.new_tensor(predictions).reshape(1, -1, 85)
            objectness = rows[..., 4:5] + images.mean() / 255 * self.weight
            return torch.cat((rows[..., :4], objectness, rows[..., 5:]), dim=-1)

    return DetectionModel()


class TrafficServiceBoundary:
    """A real in-process gRPC TrafficService with deterministic responses and request records.

    Session, simulate and close requests are copied into sessions, requests and
    closed. Tests may set simulate_response(request), rpc_errors[method] =
    (grpc.StatusCode, details), or delays[method] to replace this native boundary. With
    ``service_container`` set, scene queries answer UNAVAILABLE unless that Docker fake state
    reports a running container.
    """

    def __init__(self, scene_ids=("scene",), minimum_history_length_us=1_600_000,
                 simulate_response=None, service_container=None):
        self.scene_ids = tuple(scene_ids)
        self.service_container = service_container
        self.minimum_history_length_us = minimum_history_length_us
        self.simulate_response = simulate_response
        self.sessions = []
        self.requests = []
        self.closed = []
        self.rpc_errors = {}
        self.delays = {}
        self.endpoint = None
        self._server = None

    def capture_channels(self, monkeypatch):
        """Observe actual gRPC channel cleanup at the native transport boundary."""
        import grpc

        create = grpc.insecure_channel
        self.channels = []

        class Channel:
            def __init__(self, *args, **kwargs):
                self.native = create(*args, **kwargs)
                self.closed = False
                owner.channels.append(self)

            def __getattr__(self, name):
                return getattr(self.native, name)

            def close(self):
                self.closed = True
                self.native.close()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        owner = self
        monkeypatch.setattr(grpc, "insecure_channel", Channel)

    def __enter__(self):
        from concurrent.futures import ThreadPoolExecutor
        import time

        import grpc
        from alpasim_grpc.v0 import common_pb2, traffic_pb2, traffic_pb2_grpc

        owner = self

        def before(method, context):
            time.sleep(owner.delays.get(method, 0.))
            if method in owner.rpc_errors:
                code, details = owner.rpc_errors[method]
                context.abort(code, details)

        class Service(traffic_pb2_grpc.TrafficServiceServicer):
            def get_metadata(self, request, context):
                before("get_metadata", context)
                return traffic_pb2.TrafficModuleMetadata(
                    minimum_history_length_us=owner.minimum_history_length_us)

            def get_available_scenes(self, request, context):
                before("get_available_scenes", context)
                if not container_running(owner.service_container):
                    context.abort(grpc.StatusCode.UNAVAILABLE, "traffic is not running")
                return common_pb2.AvailableScenesReturn(scene_ids=owner.scene_ids)

            def start_session(self, request, context):
                owner.sessions.append(deepcopy(request))
                before("start_session", context)
                return common_pb2.SessionRequestStatus()

            def close_session(self, request, context):
                owner.closed.append(deepcopy(request))
                before("close_session", context)
                return common_pb2.Empty()

            def simulate(self, request, context):
                owner.requests.append(deepcopy(request))
                before("simulate", context)
                if owner.simulate_response is not None:
                    return owner.simulate_response(request)
                # A deterministic replay response at the requested instant. The
                # runtime tests can replace it with controlled reactive poses.
                session = next(item for item in owner.sessions if item.session_uuid == request.session_uuid)
                updates = []
                for obj in session.logged_object_trajectories:
                    points = obj.trajectory.poses
                    if not points:
                        continue
                    timestamp = request.time_query_us
                    if not obj.is_static and not (points[0].timestamp_us <= timestamp <= points[-1].timestamp_us):
                        continue
                    pose = min(points, key=lambda point: abs(point.timestamp_us - timestamp)).pose
                    updates.append(traffic_pb2.ObjectTrajectoryUpdate(
                        object_id=obj.object_id, trajectory=common_pb2.Trajectory(poses=[
                            common_pb2.PoseAtTime(timestamp_us=timestamp, pose=pose)])))
                return traffic_pb2.TrafficReturn(object_trajectory_updates=updates)

        self._server = grpc.server(ThreadPoolExecutor(max_workers=4))
        traffic_pb2_grpc.add_TrafficServiceServicer_to_server(Service(), self._server)
        port = self._server.add_insecure_port("127.0.0.1:0")
        self.endpoint = f"127.0.0.1:{port}"
        self._server.start()
        return self

    def __exit__(self, *exc):
        if self._server is not None:
            self._server.stop(0).wait()


def replica_detector():
    """Create differentiable YOLOX outputs for replica pixels on a constant background."""
    import torch

    class ReplicaDetector(torch.nn.Module):
        """Model boundary whose confidence differentiates the patch's red channel."""

        def forward(self, images):
            image = images[0]
            padding = (image - 114).abs().amax(dim=0) < 1e-6
            rows = torch.nonzero(~padding.all(dim=1), as_tuple=True)[0]
            columns = torch.nonzero(~padding.all(dim=0), as_tuple=True)[0]
            if not len(rows) or not len(columns):
                return image.new_empty((1, 0, 85))
            height, width = int(rows[-1]) + 1, int(columns[-1]) + 1
            image = image[:, :height, :width]
            changed = (image - 30).abs().amax(dim=0) > 1e-6
            ys, xs = torch.nonzero(changed, as_tuple=True)
            if not len(xs):
                return image.new_empty((1, 0, 85))
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            score = .5 + .4 * image[2, y0:y1, x0:x1].mean() / 255
            box = image.new_tensor([[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]])
            classes = image.new_zeros((1, 80))
            classes[0, 11] = 1.
            return torch.cat((box, score.reshape(1, 1), classes), dim=-1)[None]

    return ReplicaDetector()


def docker_process_boundary(tmp_path, monkeypatch):
    """A deterministic Docker process boundary.

    DOCKER_MODE selects ready, race, run-fail, exit, timeout, interrupt or stop-fail.
    Container state and every command argument are stored under tmp_path.
    Interrupt mode assumes the Docker subprocess's parent is the test process.
    A DOCKER_STATE marker ensures that only its first log request sends SIGINT.
    """
    import os
    import sys

    directory = tmp_path / "bin"
    directory.mkdir()
    state = tmp_path / "containers"
    state.mkdir()
    calls = tmp_path / "docker.jsonl"
    docker = directory / "docker"
    docker.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys, uuid
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["DOCKER_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\n")
root = Path(os.environ["DOCKER_STATE"])
mode = os.environ.get("DOCKER_MODE", "ready")
def read(p):
    try:return json.loads(p.read_text())
    except (ValueError, FileNotFoundError):return None
def locate(key):
    direct=root/key
    if direct.exists():return direct
    for p in root.iterdir():
        data=read(p)
        if isinstance(data,dict) and data.get("id")==key:return p
    return direct
if args[0] == "run":
    if mode == "run-fail":
        print("injected docker run failure", file=sys.stderr)
        raise SystemExit(1)
    name=args[args.index("--name")+1]
    label=args[args.index("--label")+1].split("=",1)[1] if "--label" in args else ""
    identity="id-"+uuid.uuid4().hex
    (root/name).write_text(json.dumps({"id":identity,"owner":"someone-else" if mode=="race" else label,"running":mode!="exit"}))
    print(identity)
    raise SystemExit(1 if mode == "race" else 0)
p=locate(args[-1]);data=read(p)
if isinstance(data,bool):data={"id":"seed-"+p.name,"owner":"","running":data}
if args[0] == "inspect":
    if data is None:raise SystemExit(1)
    fmt=args[args.index("--format")+1] if "--format" in args else ""
    if ".Config.Labels" in fmt:print(data["id"]+" "+data["owner"])
    elif ".State.Running" in fmt:print(str(data["running"]).lower())
    else:print(json.dumps(data))
elif args[0] == "logs":
    marker = root / ".interrupted"
    if mode == "interrupt" and not marker.exists():
        import signal
        marker.touch()
        os.kill(os.getppid(), signal.SIGINT)
    print("Serving on\nStarting CATK traffic gRPC server on" if mode == "ready" else "startup diagnostic")
elif args[0] == "stop":
    if data is None:
        print("No such container", file=sys.stderr)
        raise SystemExit(1)
    data["running"]=False;p.write_text(json.dumps(data))
    if mode == "stop-fail":
        print("stop failed", file=sys.stderr)
        raise SystemExit(1)
elif args[0] == "rm":
    if data is None:raise SystemExit(1)
    p.unlink()
''')
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("DOCKER_STATE", str(state))
    monkeypatch.setenv("DOCKER_CALLS", str(calls))
    return state, calls
