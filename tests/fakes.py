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
from typing import Any

import numpy as np

from avsectester.attack import Attack
from avsectester.attack.runtime import AttackRuntime
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.data_types import Control, Detection, Observation, WorldSnapshot
from avsectester.closed_loop.simulators.base import SensorGeometry, SensorView, Simulator, SimulatorRuntime
from avsectester.closed_loop.simulators.nurec.renderer import Renderer
from avsectester.data_types import ActorState
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

    def __init__(self, scenario_ids=("scene",), decision_steps=6, time_step=.1, dataroot=None,
                 maps=None):
        self.scenario_ids = tuple(scenario_ids)
        self.decision_steps, self.time_step, self.dataroot = decision_steps, time_step, dataroot
        self.maps = dict(maps or {})

    def scenario_map(self, scenario_id):
        if scenario_id not in self.scenario_ids:
            raise KeyError(scenario_id)
        return self.maps[scenario_id] if scenario_id in self.maps else super().scenario_map(scenario_id)

    def scenarios(self):
        return iter(ScenarioDescriptor(name, "memory") for name in self.scenario_ids)

    def frames(self, scenario_id):
        if scenario_id not in self.scenario_ids:
            raise KeyError(scenario_id)
        return tuple(FrameReference(scenario_id, i, i, i * self.time_step)
                     for i in range(self.decision_steps + 1))

    def frame_data(self, scenario_id, frame):
        frame = self.checked_frame(scenario_id, frame)
        return FrameData(frame, actors={"lead": actor_state()},
                         ego=actor_state((frame.timestamp, 0., 0.), speed=1.))

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

    def __init__(self, time_step=.1, capture_ticks=0, world_key=None, fail_position=None,
                 extra_sensors=None, camera_geometries=None, sensor_depths=None, actors=None):
        self.time_step, self.capture_ticks = time_step, capture_ticks
        self.world_key, self.fail_position = world_key, fail_position
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

    def create_runtime(self, seed=None):
        return MemorySimulatorRuntime(self, seed)


class MemorySimulatorRuntime(SimulatorRuntime):
    """Native operations for the shared world; the production base owns reset and step."""

    def __init__(self, simulator, seed=None):
        super().__init__(simulator)
        self.seed = seed
        self.closed = False
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
        self.closed = False

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
        if world.get("fail") or self.position == self.simulator.fail_position:
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
        self.closed = True
        self._world_initialized = False
        self._capture = None
        self.native["objects"].clear()


class ConstantAVStack(AVStack):
    """A deterministic driver exposing one actual component-stage boundary."""

    command_modes = frozenset({"actuator"})
    supported_stages = frozenset({"perception.post"})

    def __init__(self, command=None, perception_output=None, read_speed=False,
                 fail_on_position=None, interrupt_on_position=None, checkpoint_path=None):
        self.command = Control(throttle=.25) if command is None else command
        self.perception_output = [] if perception_output is None else perception_output
        self.read_speed = read_speed
        self.fail_on_position, self.interrupt_on_position = fail_on_position, interrupt_on_position
        self.checkpoint_path = checkpoint_path
        self.closed = False
        self.output = None

    def reset(self, observation, past=()):
        self.output = None

    def __call__(self, observation):
        if observation.frame == self.interrupt_on_position:
            raise KeyboardInterrupt("memory driver interrupted")
        if observation.frame == self.fail_on_position:
            raise RuntimeError("memory driver failed")
        self.output = self._emit("perception.post", deepcopy(self.perception_output))
        return (Control(throttle=float(np.clip(observation.ego_speed / 10, 0, 1)))
                if self.read_speed else replace(self.command))

    def component_log(self):
        return {"perception": self.output}

    def detections(self):
        if isinstance(self.output, (tuple, list)) and all(isinstance(item, Detection) for item in self.output):
            return tuple(self.output)
        return None

    def close(self):
        self.closed = True


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


class DeceleratingAVStack(AVStack):
    """Replan a -2 m/s² trajectory from the current observed speed every decision."""

    command_modes = frozenset({"trajectory"})
    past_steps = 15

    def __call__(self, observation):
        times = np.arange(51) * .1
        positions = float(observation.ego_speed) * times - times ** 2
        return Control(trajectory=[
            ((float(position), 0., 0.), (1., 0., 0., 0.), round((observation.t + time) * 1e6))
            for time, position in zip(times, positions)
        ])


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


class NuRecRpcError(Exception):
    """A gRPC status error raised by the NuRec service boundary."""

    def __init__(self, code, details):
        super().__init__(details)
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


class NuRecRpcBoundary:
    """A SensorsimService boundary; NuRecRenderer itself remains real.

    Inserted Gaussian assets are drawn at their listed poses, mapped into track frames as the
    measured NRE convention does, except at pixels in ``hidden``, where the scene covers them.
    ``asset_directories`` maps server asset paths to local ones. With ``scene_tracks`` set, a
    render naming any other track that was not inserted fails as NRE does.
    """

    TRACK_FROM_PLY = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    TRACK_PADDING_M = np.array([1., 1., .25])

    def __init__(self, returned_resolution_hw=None):
        self.returned_resolution_hw = returned_resolution_hw
        self.last_request = None
        self.requests = []
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

    def get_available_scenes(self, request):
        self.requests.append(request)
        return SimpleNamespace(scene_ids=["scene"])

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
        from avsectester.closed_loop.simulators.nurec.renderer import ftheta_camera_from_spec, pose_from_proto
        self.requests.append(request)
        self.last_request = request
        height, width = self.returned_resolution_hw or (
            request.resolution_h,
            round(request.resolution_h * self.spec.resolution_w / self.spec.resolution_h),
        )
        listed = [item.track_id for item in request.dynamic_objects]
        unknown = [track for track in listed if track not in self.tracks
                   and self.scene_tracks is not None and track not in self.scene_tracks]
        if unknown:
            raise NuRecRpcError("INVALID_ARGUMENT", f"{unknown[0]!r} is not in list")
        image = np.full((height, width, 3), 30, np.uint8)
        camera = ftheta_camera_from_spec(self.spec)
        cam_from_world = np.linalg.inv(pose_from_proto(request.sensor_pose.start_pose))
        for item in request.dynamic_objects:
            if item.track_id not in self.tracks:
                continue
            points, colours = self.tracks[item.track_id]
            cam_from_track = cam_from_world @ pose_from_proto(item.pose_pair.start_pose)
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

    def install(self, monkeypatch):
        common = SimpleNamespace(Pose=SimpleNamespace, Quat=SimpleNamespace,
                                 Vec3=SimpleNamespace, Empty=SimpleNamespace)
        sensor = SimpleNamespace(AvailableCamerasRequest=SimpleNamespace,
            AvailableTrajectoriesRequest=SimpleNamespace, RGBRenderRequest=SimpleNamespace,
            PosePair=SimpleNamespace, DynamicObject=SimpleNamespace,
            ImageFormat=SimpleNamespace(JPEG=1))
        modules = {
            "grpc": {"insecure_channel": lambda endpoint: SimpleNamespace(
                         close=lambda: None, unary_unary=self._unary_unary),
                     "RpcError": NuRecRpcError,
                     "StatusCode": SimpleNamespace(INVALID_ARGUMENT="INVALID_ARGUMENT")},
            "alpasim_grpc": {}, "alpasim_grpc.v0": {"common_pb2": common,
                "sensorsim_pb2": sensor,
                "sensorsim_pb2_grpc": SimpleNamespace(SensorsimServiceStub=lambda channel: self)},
        }
        for name, values in modules.items():
            module = ModuleType(name)
            vars(module).update(values)
            monkeypatch.setitem(sys.modules, name, module)
        return self


class NuRecImageBoundary(Renderer):
    """A deterministic image service boundary with a chosen optical lens model."""

    cameras = ["front"]

    def __init__(self, camera):
        self.camera = camera

    def camera_model(self, camera=None):
        if camera not in {None, "front"}:
            raise ValueError(f"Unknown camera {camera!r}")
        return self.camera

    def cam_from_world(self, world_from_rig, camera=None):
        rig_from_camera = BODY_FROM_OPTICAL.copy()
        rig_from_camera[2, 3] = 1.5
        return np.linalg.inv(world_from_rig @ rig_from_camera)

    def render(self, world_from_rig, timestamp_us, camera, dynamic_objects=()):
        return np.full((self.camera.height, self.camera.width, 3), 30, np.uint8)


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
            object=native_camera, imsize=(12, 16), P=np.array([[8., 0., 8., 0.],
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
        return self.frame

    def get_actors(self):
        return list(self.actors)

    def get_blueprint_library(self):
        vehicles = [SimpleNamespace(id="vehicle.a"), SimpleNamespace(id="vehicle.b")]
        return SimpleNamespace(filter=lambda pattern: (
            [SimpleNamespace(id=pattern)] if pattern.startswith("static.prop") else vehicles))

    def spawn_actor(self, blueprint, transform, attach_to=None, attachment_type=None):
        actor = CarlaActor(self, len(self.actors) + 1, blueprint.id, transform)
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
        self.closed = False

    def scene_settings(self):
        return {}

    def tick(self):
        return self.world.tick()

    def close(self):
        self.closed = True


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

    def build(self, config, default_args=None):
        if default_args is None:
            return CarlaNativeClient(config, self.frame_offset)
        return CarlaMobileActor(default_args["client"], config)

    def install(self, monkeypatch):
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


class StubRenderer(Renderer):
    """A placeholder renderer — no scene, no server: returns a black HWC uint8 frame. Lets the whole
    loop run and be debugged, and gives image-consuming stacks (e.g. Alpamayo) a valid tensor. Swap
    in :class:`NuRecRenderer` for real imagery without touching the simulator or the AVStack.
    """

    def __init__(
        self, cameras: list[str] | None = None, height: int = 480, width: int = 640
    ) -> None:
        self.cameras = list(cameras) if cameras else ["camera_front"]
        self.height, self.width = height, width

    def render(self, world_from_rig: Any, timestamp_us: int, camera: str, dynamic_objects=()):
        import numpy as np

        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


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
